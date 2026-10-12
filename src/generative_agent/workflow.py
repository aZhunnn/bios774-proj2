"""End-to-end tabular synthetic-data workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

from generative_agent.agent import AgentFoundation
from generative_agent.artifacts import LocalArtifactStore
from generative_agent.classical import (
    CopulaConfig,
    GmmConfig,
    fit_copula,
    fit_gmm,
    propose_copula_configs,
    propose_gmm_configs,
)
from generative_agent.decisioning import DecisionAction, DeterministicDecisionEngine
from generative_agent.eligibility import select_eligible_method
from generative_agent.evaluation import bootstrap_baseline, evaluate_synthetic
from generative_agent.flow import FlowConfig
from generative_agent.flow_search import propose_flow_configs, run_flow_search
from generative_agent.generation import generate_classical_synthetic, generate_flow_synthetic, generate_synthetic
from generative_agent.search import propose_vae_configs, run_vae_search
from generative_agent.splitting import DatasetSplitter, SplitConfig
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles, load_csv_dataset
from generative_agent.tabular_preprocessing import TabularVaePreprocessor
from generative_agent.vae import VaeConfig


@dataclass(frozen=True)
class WorkflowResult:
    artifacts_dir: Path
    report_path: Path
    synthetic_path: Path
    checkpoint_path: Path | None
    training_failed: bool


def run_tabular_generation_workflow(
    *,
    csv_path: str | Path,
    objective_text: str,
    output_dir: str | Path,
    roles: ColumnRoles | None = None,
    budget: ExperimentBudget | None = None,
    n_synthetic: int | None = None,
) -> WorkflowResult:
    output = Path(output_dir)
    store = LocalArtifactStore(output)
    frame, handle = load_csv_dataset(csv_path, roles=roles)
    selected_budget = budget or ExperimentBudget(max_models=2, max_epochs=20, max_wall_minutes=30)
    state = AgentFoundation().prepare(handle, objective_text, budget=selected_budget)
    if state.feasibility is None or not state.feasibility.feasible:
        raise ValueError("Dataset/objective is not feasible: " + "; ".join(state.feasibility.reasons))

    decision = DeterministicDecisionEngine().decide(
        profile=state.profile,
        objective=state.objective,
        feasibility=state.feasibility,
        budget=selected_budget,
    )
    if decision.action is DecisionAction.REJECT:
        raise ValueError("; ".join(decision.reasons))

    manifest = DatasetSplitter().manifest(
        frame.to_dict(orient="records"),
        SplitConfig(seed=selected_budget.seed, stratify_column=(roles.target if roles and roles.target else None)),
    )
    preprocessor = TabularVaePreprocessor(roles=roles).fit(frame, manifest)
    train_array = preprocessor.transform(frame, manifest.indices.train)
    validation_array = preprocessor.transform(frame, manifest.indices.validation)
    train_frame = frame.iloc[list(manifest.indices.train)].reset_index(drop=True)
    test_frame = frame.iloc[list(manifest.indices.test)].reset_index(drop=True)
    metadata = preprocessor._metadata()

    sample_count = n_synthetic or len(frame)
    synthetic_path = output / "synthetic_samples.csv"
    baseline = bootstrap_baseline(train_frame[list(metadata.schema.model_columns)], sample_count, seed=selected_budget.seed)
    baseline.insert(0, "synthetic_record", True)
    baseline_eval = evaluate_synthetic(
        train_real=train_frame,
        test_real=test_frame,
        synthetic=baseline,
        schema=metadata.schema,
        method="training_row_bootstrap",
    )

    selected_config: VaeConfig | FlowConfig | GmmConfig | CopulaConfig | None = None
    checkpoint_path: Path | None = None
    training_failed = False
    search_payload: dict[str, object] | None = None
    family_searches: dict[str, object] = {}
    validation_comparison: dict[str, object] = {}
    selected_family = "baseline"

    if decision.action is DecisionAction.USE_BASELINE:
        baseline.to_csv(synthetic_path, index=False)
        generation_metadata = {
            "n_samples": sample_count,
            "seed": selected_budget.seed,
            "method": "training_row_bootstrap",
            "output_path": str(synthetic_path),
        }
        evaluation_payload = {"selected": baseline_eval.to_dict(), "baseline": baseline_eval.to_dict()}
    else:
        continuous_only = len(metadata.schema.binary) == 0 and len(metadata.schema.continuous) == train_array.shape[1]
        family_budget = _allocate_training_budget(total_candidates=selected_budget.max_models, continuous_only=continuous_only)
        if sum(family_budget.values()) <= 0:
            baseline.to_csv(synthetic_path, index=False)
            generation_metadata = {
                "n_samples": sample_count,
                "seed": selected_budget.seed,
                "method": "training_row_bootstrap_no_training_budget",
                "output_path": str(synthetic_path),
            }
            evaluation_payload = {"selected": baseline_eval.to_dict(), "baseline": baseline_eval.to_dict()}
            search_payload = {"budget": {"max_trained_candidates": selected_budget.max_models, "trained_candidates": 0}}
        else:
            validation_real = frame.iloc[list(manifest.indices.validation)].reset_index(drop=True)
            validation_baseline = bootstrap_baseline(
                train_frame[list(metadata.schema.model_columns)],
                len(validation_real),
                seed=selected_budget.seed + 7,
            )
            validation_baseline.insert(0, "synthetic_record", True)
            validation_scores = {
                "baseline": _validation_score(
                    train_real=train_frame,
                    validation_real=validation_real,
                    synthetic=validation_baseline,
                    schema=metadata.schema,
                    method="validation_bootstrap",
                )
            }
            models = {}
            trained_candidates = 0
            vae_search = None
            flow_search = None
            gmm_search = None
            copula_search = None

            if family_budget["vae"] > 0:
                vae_configs = propose_vae_configs(
                    input_dim=train_array.shape[1],
                    continuous_dim=len(metadata.schema.continuous),
                    binary_dim=len(metadata.schema.binary),
                    n_train=len(train_array),
                    budget=replace(selected_budget, max_models=family_budget["vae"]),
                )
                vae_search, vae_model = run_vae_search(
                    train_array=train_array,
                    validation_array=validation_array,
                    configs=vae_configs,
                    checkpoint_dir=output,
                )
                family_searches["vae"] = vae_search.to_dict()
                trained_candidates += len(vae_search.candidates)
                models["vae"] = vae_model
                if not vae_search.selected.failed:
                    validation_vae = generate_synthetic(
                        vae_model,
                        preprocessor,
                        n_samples=len(validation_real),
                        seed=selected_budget.seed + 11,
                    )
                    validation_scores["vae"] = _validation_score(
                        train_real=train_frame,
                        validation_real=validation_real,
                        synthetic=validation_vae.samples,
                        schema=metadata.schema,
                        method="validation_tabular_vae",
                    )
            else:
                family_searches["vae"] = {"skipped": True, "reason": "No remaining trained-candidate budget."}

            if continuous_only and family_budget["flow"] > 0:
                flow_configs = propose_flow_configs(
                    input_dim=train_array.shape[1],
                    n_train=len(train_array),
                    budget=replace(selected_budget, max_models=family_budget["flow"]),
                )
                flow_search, flow_model = run_flow_search(
                    train_array=train_array,
                    validation_array=validation_array,
                    configs=flow_configs,
                    checkpoint_dir=output,
                )
                family_searches["flow"] = flow_search.to_dict()
                trained_candidates += len(flow_search.candidates)
                models["flow"] = flow_model
                if not flow_search.selected.failed:
                    validation_flow = generate_flow_synthetic(
                        flow_model,
                        preprocessor,
                        n_samples=len(validation_real),
                        seed=selected_budget.seed + 13,
                    )
                    validation_scores["flow"] = _validation_score(
                        train_real=train_frame,
                        validation_real=validation_real,
                        synthetic=validation_flow.samples,
                        schema=metadata.schema,
                        method="validation_tabular_flow",
                    )
            elif continuous_only:
                family_searches["flow"] = {"skipped": True, "reason": "No remaining trained-candidate budget."}
            else:
                family_searches["flow"] = {
                    "skipped": True,
                    "reason": "Normalizing flow supports continuous-only tabular features in Phase 3A.",
                }

            if continuous_only and family_budget["gmm"] > 0:
                gmm_search, gmm_model = _run_gmm_search(
                    train_array=train_array,
                    validation_array=validation_array,
                    configs=propose_gmm_configs(
                        n_train=len(train_array),
                        budget_count=family_budget["gmm"],
                        seed=selected_budget.seed + 31,
                    ),
                    checkpoint_dir=output,
                )
                family_searches["gmm"] = gmm_search
                trained_candidates += len(gmm_search["candidates"])
                models["gmm"] = gmm_model
                if not gmm_search["selected"]["failed"]:
                    validation_gmm = generate_classical_synthetic(
                        gmm_model,
                        preprocessor,
                        n_samples=len(validation_real),
                        seed=selected_budget.seed + 17,
                        method="validation_tabular_gmm",
                    )
                    validation_scores["gmm"] = _validation_score(
                        train_real=train_frame,
                        validation_real=validation_real,
                        synthetic=validation_gmm.samples,
                        schema=metadata.schema,
                        method="validation_tabular_gmm",
                    )
            elif continuous_only:
                family_searches["gmm"] = {"skipped": True, "reason": "No remaining trained-candidate budget."}
            else:
                family_searches["gmm"] = {
                    "skipped": True,
                    "reason": "Gaussian mixture model supports continuous-only tabular features in Phase 6A.",
                }

            if continuous_only and family_budget["copula"] > 0:
                copula_search, copula_model = _run_copula_search(
                    train_array=train_array,
                    validation_array=validation_array,
                    configs=propose_copula_configs(
                        budget_count=family_budget["copula"],
                        seed=selected_budget.seed + 41,
                    ),
                    checkpoint_dir=output,
                )
                family_searches["copula"] = copula_search
                trained_candidates += len(copula_search["candidates"])
                models["copula"] = copula_model
                if not copula_search["selected"]["failed"]:
                    validation_copula = generate_classical_synthetic(
                        copula_model,
                        preprocessor,
                        n_samples=len(validation_real),
                        seed=selected_budget.seed + 19,
                        method="validation_tabular_copula",
                    )
                    validation_scores["copula"] = _validation_score(
                        train_real=train_frame,
                        validation_real=validation_real,
                        synthetic=validation_copula.samples,
                        schema=metadata.schema,
                        method="validation_tabular_copula",
                    )
            elif continuous_only:
                family_searches["copula"] = {"skipped": True, "reason": "No remaining trained-candidate budget."}
            else:
                family_searches["copula"] = {
                    "skipped": True,
                    "reason": "Gaussian copula supports continuous-only tabular features in Phase 6A.",
                }
            family_searches["budget"] = {
                "semantics": "max_models is the total number of trained generative candidates across all model families. Bootstrap baseline does not consume a training slot.",
                "max_trained_candidates": selected_budget.max_models,
                "trained_candidates": trained_candidates,
                "allocation": family_budget,
            }
            if trained_candidates > selected_budget.max_models:
                raise RuntimeError("Internal budget accounting error: trained candidates exceeded max_models.")

            validation_comparison = {
                "decision_rule": (
                    "Select the method with the highest validation selection_score from synthetic-vs-validation "
                    "evaluation. VAE ELBO and flow NLL are used only within their own family searches."
                ),
                "metric_direction": "higher selection_score is better",
                "metric_definition": (
                    "selection_score = 0.45 train-scale-normalized marginal fidelity + 0.35 dependence preservation "
                    "+ 0.20 legacy marginal fidelity when correlations are valid; without correlations, 0.75 normalized "
                    "marginal + 0.25 legacy marginal."
                ),
                "sample_count": len(validation_real),
                "seeds": {
                    "baseline": selected_budget.seed + 7,
                    "vae": selected_budget.seed + 11,
                    "flow": selected_budget.seed + 13,
                    "gmm": selected_budget.seed + 17,
                    "copula": selected_budget.seed + 19,
                },
                "scores": validation_scores,
            }
            eligibility_selection = select_eligible_method(
                validation_scores,
                use_case=state.objective.constraints["use_case"],
                ranking_metric="selection_score",
            )
            validation_comparison["eligibility"] = eligibility_selection.to_dict()
            if eligibility_selection.selected_method is None:
                selected_family = "rejected"
            else:
                selected_family = eligibility_selection.selected_method
            final_evaluations = {"baseline": baseline_eval.to_dict()}
            final_generations = {"baseline": baseline}
            if "vae" in validation_scores:
                final_vae_generation = generate_synthetic(
                    models["vae"],
                    preprocessor,
                    n_samples=sample_count,
                    seed=selected_budget.seed + 1,
                )
                final_generations["vae"] = final_vae_generation.samples
                final_evaluations["vae"] = evaluate_synthetic(
                    train_real=train_frame,
                    test_real=test_frame,
                    synthetic=final_vae_generation.samples,
                    schema=metadata.schema,
                    method="tabular_vae",
                ).to_dict()
            if "flow" in validation_scores:
                final_flow_generation = generate_flow_synthetic(
                    models["flow"],
                    preprocessor,
                    n_samples=sample_count,
                    seed=selected_budget.seed + 1,
                )
                final_generations["flow"] = final_flow_generation.samples
                final_evaluations["flow"] = evaluate_synthetic(
                    train_real=train_frame,
                    test_real=test_frame,
                    synthetic=final_flow_generation.samples,
                    schema=metadata.schema,
                    method="tabular_flow",
                ).to_dict()
            if "gmm" in validation_scores:
                final_gmm_generation = generate_classical_synthetic(
                    models["gmm"],
                    preprocessor,
                    n_samples=sample_count,
                    seed=selected_budget.seed + 1,
                    method="tabular_gmm",
                )
                final_generations["gmm"] = final_gmm_generation.samples
                final_evaluations["gmm"] = evaluate_synthetic(
                    train_real=train_frame,
                    test_real=test_frame,
                    synthetic=final_gmm_generation.samples,
                    schema=metadata.schema,
                    method="tabular_gmm",
                ).to_dict()
            if "copula" in validation_scores:
                final_copula_generation = generate_classical_synthetic(
                    models["copula"],
                    preprocessor,
                    n_samples=sample_count,
                    seed=selected_budget.seed + 1,
                    method="tabular_copula",
                )
                final_generations["copula"] = final_copula_generation.samples
                final_evaluations["copula"] = evaluate_synthetic(
                    train_real=train_frame,
                    test_real=test_frame,
                    synthetic=final_copula_generation.samples,
                    schema=metadata.schema,
                    method="tabular_copula",
                ).to_dict()

            if selected_family == "rejected":
                baseline.to_csv(synthetic_path, index=False)
                generation_metadata = {
                    "n_samples": sample_count,
                    "seed": selected_budget.seed,
                    "method": "rejected_no_eligible_synthetic_release",
                    "output_path": str(synthetic_path),
                }
                evaluation_payload = {"selected": {"method": "rejected", "reason": eligibility_selection.selection_reason}, **final_evaluations}
                search_payload = family_searches
            elif selected_family == "baseline":
                final_generations["baseline"].to_csv(synthetic_path, index=False)
                generation_metadata = {
                    "n_samples": sample_count,
                    "seed": selected_budget.seed,
                    "method": "training_row_bootstrap_selected_by_validation",
                    "output_path": str(synthetic_path),
                }
                evaluation_payload = {"selected": final_evaluations["baseline"], **final_evaluations}
                search_payload = family_searches
            else:
                if selected_family == "vae":
                    selected_config = vae_search.selected.config
                    checkpoint_path = vae_search.selected.checkpoint_path
                    training_failed = vae_search.selected.failed
                    final_generations["vae"].to_csv(synthetic_path, index=False)
                    generation_metadata = {
                        "n_samples": sample_count,
                        "seed": selected_budget.seed + 1,
                        "synthetic_record_column": "synthetic_record",
                        "method": "tabular_vae",
                        "output_path": str(synthetic_path),
                    }
                elif selected_family == "flow":
                    selected_config = flow_search.selected.config
                    checkpoint_path = flow_search.selected.checkpoint_path
                    training_failed = flow_search.selected.failed
                    final_generations["flow"].to_csv(synthetic_path, index=False)
                    generation_metadata = {
                        "n_samples": sample_count,
                        "seed": selected_budget.seed + 1,
                        "synthetic_record_column": "synthetic_record",
                        "method": "tabular_flow",
                        "output_path": str(synthetic_path),
                    }
                elif selected_family == "gmm":
                    selected_config = GmmConfig(**gmm_search["selected"]["config"])
                    checkpoint_path = Path(gmm_search["selected"]["checkpoint_path"])
                    training_failed = gmm_search["selected"]["failed"]
                    final_generations["gmm"].to_csv(synthetic_path, index=False)
                    generation_metadata = {
                        "n_samples": sample_count,
                        "seed": selected_budget.seed + 1,
                        "synthetic_record_column": "synthetic_record",
                        "method": "tabular_gmm",
                        "output_path": str(synthetic_path),
                    }
                else:
                    selected_config = CopulaConfig(**copula_search["selected"]["config"])
                    checkpoint_path = Path(copula_search["selected"]["checkpoint_path"])
                    training_failed = copula_search["selected"]["failed"]
                    final_generations["copula"].to_csv(synthetic_path, index=False)
                    generation_metadata = {
                        "n_samples": sample_count,
                        "seed": selected_budget.seed + 1,
                        "synthetic_record_column": "synthetic_record",
                        "method": "tabular_copula",
                        "output_path": str(synthetic_path),
                    }
                evaluation_payload = {"selected": final_evaluations[selected_family], **final_evaluations}
                search_payload = family_searches

    decisions = {
        "feasible": state.feasibility.feasible,
        "reasons": state.feasibility.reasons,
        "warnings": state.feasibility.warnings,
        "constraints": state.feasibility.constraints,
        "objective_decision": decision.to_dict(),
        "selection": (
            f"Selected {selected_family} using shared validation selection_score; held-out test was not used."
            if validation_comparison
            else "Baseline selected before generative training."
        ),
        "validation_comparison": validation_comparison,
        "eligibility": validation_comparison.get("eligibility", {}),
        "model_limitations": {
            "vae": "Supports continuous and binary tabular features with Gaussian/Bernoulli observation assumptions.",
            "flow": "Phase 3A flow supports continuous-only tabular features with no missing values.",
            "gmm": "Gaussian mixtures support continuous-only tabular data and can be sensitive to component count and covariance regularization.",
            "copula": "Gaussian copula supports continuous-only tabular data with empirical marginals and linear latent dependence; it does not capture arbitrary nonlinear or tail dependence.",
            "baseline": "Bootstrap resamples training rows and is non-generative.",
        },
        "planner": "deterministic rules plus within-family searches and shared validation comparison; no LLM reasoning or autonomous optimization",
    }

    store.save_json("dataset_profile", asdict(state.profile), kind="profile")
    store.save_json("objective", asdict(state.objective), kind="objective")
    store.save_json("split_manifest", asdict(manifest), kind="split")
    store.save_json("preprocessing", metadata.to_dict(), kind="preprocessing")
    store.save_json("model_config", selected_config.to_dict() if selected_config else {"method": "baseline"}, kind="model")
    if search_payload is not None:
        store.save_json("candidate_search", search_payload, kind="search")
        store.save_json("validation_comparison", validation_comparison, kind="selection")
    store.save_json("generation_metadata", generation_metadata, kind="generation")
    store.save_json("evaluation_results", evaluation_payload, kind="evaluation")
    store.save_json("agent_decisions", decisions, kind="decisions")

    report_path = output / "final_report.md"
    report_path.write_text(
        _render_report(selected_config, search_payload, generation_metadata, evaluation_payload, decisions),
        encoding="utf-8",
    )
    return WorkflowResult(
        artifacts_dir=output,
        report_path=report_path,
        synthetic_path=synthetic_path,
        checkpoint_path=checkpoint_path,
        training_failed=training_failed,
    )


def _render_report(
    config: VaeConfig | FlowConfig | GmmConfig | CopulaConfig | None,
    search: dict[str, object] | None,
    generation: dict[str, object],
    evaluation: dict[str, object],
    decisions: dict[str, object],
) -> str:
    return "\n".join(
        [
            "# Phase 2.5 Tabular VAE Report",
            "",
            "## Scope",
            "This report is generated from stored workflow results for unconditional tabular synthetic data generation.",
            "",
            "## Modeling Assumptions",
            "- Continuous features use a unit-variance Gaussian observation model after training-only standardization.",
            "- Binary features use a Bernoulli likelihood parameterized by decoder logits.",
            "- Within-family selection uses validation loss or validation likelihood only inside that family.",
            "- max_models is total trained generative candidates across all families; bootstrap baseline does not consume a training slot.",
            "- Eligibility is applied before performance ranking; hard eligibility constraints cannot be overridden by high fidelity.",
            "- Cross-family selection uses the predeclared validation selection_score among eligible methods; held-out test data is used only for final evaluation.",
            "- RealNVP flow is available only for continuous tabular features with no missing values.",
            "- GMM and Gaussian copula are available only for continuous tabular features with no missing values.",
            "- Conditional generation, class balancing, image generation, GANs, and diffusion are not implemented.",
            "",
            "## Selected Configuration",
            f"```json\n{json.dumps(config.to_dict() if config else {'method': 'training_row_bootstrap'}, indent=2)}\n```",
            "",
            "## Candidate Search",
            f"```json\n{json.dumps(search or {'selected': 'baseline'}, indent=2)}\n```",
            "",
            "## Generation",
            f"Generated samples: {generation['n_samples']}",
            f"Method: {generation.get('method', 'tabular_vae')}",
            "",
            "## Evaluation",
            f"```json\n{json.dumps(evaluation, indent=2)}\n```",
            "",
            "## Agent Decisions",
            f"```json\n{json.dumps(decisions, indent=2)}\n```",
            "",
            "## Limitations",
            "- Nearest-neighbor diagnostics are proxies, not privacy guarantees.",
            "- The decision layer is deterministic and does not perform LLM-based reasoning.",
            "- No downstream predictive utility is reported unless a separate valid protocol is implemented.",
            "- Cross-family validation scores are finite-sample heuristics, not statistical uncertainty estimates.",
            "",
        ]
    )


def _validation_score(*, train_real, validation_real, synthetic, schema, method: str) -> dict[str, float]:
    result = evaluate_synthetic(
        train_real=train_real,
        test_real=validation_real,
        synthetic=synthetic,
        schema=schema,
        method=method,
    )
    return {
        "data_generation_fidelity": result.objective_metrics["data_generation_fidelity"],
        "train_scale_normalized_marginal_fidelity": result.objective_metrics[
            "train_scale_normalized_marginal_fidelity"
        ],
        "dependence_preservation": result.objective_metrics["dependence_preservation"],
        "selection_score": result.objective_metrics["selection_score"],
        "duplicate_fraction": result.duplicate_fraction,
        "nearest_neighbor_mean": result.nearest_neighbor_mean,
    }


def _allocate_training_budget(*, total_candidates: int, continuous_only: bool) -> dict[str, int]:
    remaining = total_candidates
    allocation = {"vae": 0, "flow": 0, "gmm": 0, "copula": 0}
    if remaining > 0:
        allocation["vae"] = 1
        remaining -= 1
    if continuous_only:
        for family in ("flow", "gmm", "copula"):
            if remaining > 0:
                allocation[family] = 1
                remaining -= 1
    while remaining > 0:
        eligible_families = ("vae", "flow", "gmm", "copula") if continuous_only else ("vae",)
        family = min(eligible_families, key=lambda item: allocation[item])
        allocation[family] += 1
        remaining -= 1
    return allocation


def _run_gmm_search(*, train_array, validation_array, configs: tuple[GmmConfig, ...], checkpoint_dir: Path):
    candidates = []
    models = []
    for index, config in enumerate(configs):
        checkpoint = checkpoint_dir / f"tabular_gmm_candidate_{index}.pkl"
        result, model = fit_gmm(train_array, config, checkpoint)
        validation_score = None if result.failed else model.validation_score(validation_array)
        candidate = {
            "config": config.to_dict(),
            "diagnostics": result.diagnostics,
            "checkpoint_path": str(result.checkpoint_path),
            "wall_seconds": result.wall_seconds,
            "failed": result.failed,
            "failure_reason": result.failure_reason,
            "validation_log_likelihood": validation_score,
        }
        candidates.append(candidate)
        models.append(model)
    successful = [(idx, item) for idx, item in enumerate(candidates) if not item["failed"]]
    if not successful:
        selected_index = 0
        reason = "All GMM candidates failed; no GMM candidate selected."
    else:
        selected_index = max(successful, key=lambda pair: pair[1]["validation_log_likelihood"])[0]
        reason = "Selected highest validation average log likelihood within GMM family; test data was not used."
    return {
        "candidates": candidates,
        "selected_index": selected_index,
        "selected": candidates[selected_index],
        "selection_reason": reason,
    }, models[selected_index]


def _run_copula_search(*, train_array, validation_array, configs: tuple[CopulaConfig, ...], checkpoint_dir: Path):
    candidates = []
    models = []
    for index, config in enumerate(configs):
        checkpoint = checkpoint_dir / f"tabular_copula_candidate_{index}.pkl"
        result, model = fit_copula(train_array, config, checkpoint)
        validation_score = None if result.failed else model.validation_score(validation_array)
        candidate = {
            "config": config.to_dict(),
            "diagnostics": result.diagnostics,
            "checkpoint_path": str(result.checkpoint_path),
            "wall_seconds": result.wall_seconds,
            "failed": result.failed,
            "failure_reason": result.failure_reason,
            "validation_dependence_score": validation_score,
        }
        candidates.append(candidate)
        models.append(model)
    successful = [(idx, item) for idx, item in enumerate(candidates) if not item["failed"]]
    if not successful:
        selected_index = 0
        reason = "All Gaussian copula candidates failed; no copula candidate selected."
    else:
        selected_index = max(successful, key=lambda pair: pair[1]["validation_dependence_score"])[0]
        reason = "Selected highest validation dependence proxy within copula family; test data was not used."
    return {
        "candidates": candidates,
        "selected_index": selected_index,
        "selected": candidates[selected_index],
        "selection_reason": reason,
    }, models[selected_index]


def run_tabular_vae_workflow(**kwargs) -> WorkflowResult:
    """Backward-compatible alias for the model-agnostic workflow."""

    return run_tabular_generation_workflow(**kwargs)
