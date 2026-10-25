"""End-to-end tabular synthetic-data workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import json

from generative_agent.agent import AgentFoundation
from generative_agent.ai_analysis import FinalAnalysisConfig, generate_final_analysis
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
from generative_agent.tabular_planning import build_tabular_training_plan
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
    report_name: str = "final_report.md",
    roles: ColumnRoles | None = None,
    budget: ExperimentBudget | None = None,
    n_synthetic: int | None = None,
    ai_analysis_mode: str = "auto",
    ai_model: str = "gpt-4o-mini",
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
        training_plan = build_tabular_training_plan(
            train_frame=train_frame,
            schema=metadata.schema,
            objective_type=state.objective.objective_type,
            use_case=state.objective.constraints["use_case"],
            budget=selected_budget,
        )
        family_budget = training_plan.allocation
        if sum(family_budget.values()) <= 0:
            baseline.to_csv(synthetic_path, index=False)
            no_candidate_reason = (
                "No implemented model is eligible for the requested objective/use case."
                if not any(candidate.eligible for candidate in training_plan.candidates)
                else "No trained model was allocated under the requested budget."
            )
            rejected = not any(candidate.eligible for candidate in training_plan.candidates)
            generation_metadata = {
                "n_samples": sample_count,
                "seed": selected_budget.seed,
                "method": "rejected_no_eligible_method" if rejected else "training_row_bootstrap_no_training_budget",
                "output_path": str(synthetic_path),
            }
            evaluation_payload = (
                {"selected": {"method": "rejected", "reason": no_candidate_reason}, "baseline": baseline_eval.to_dict()}
                if rejected
                else {"selected": baseline_eval.to_dict(), "baseline": baseline_eval.to_dict()}
            )
            search_payload = {
                "planning": training_plan.to_dict(),
                "budget": {"max_trained_candidates": selected_budget.max_models, "trained_candidates": 0},
            }
            selected_family = "rejected" if rejected else "baseline"
            validation_comparison = {
                "planning": training_plan.to_dict(),
                "objective_recommendation": _objective_recommendation(
                    selected_family=selected_family,
                    validation_scores={},
                    eligibility={"selected_method": None, "selection_reason": no_candidate_reason},
                    use_case=state.objective.constraints["use_case"],
                ),
            }
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
            family_searches["planning"] = training_plan.to_dict()
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
            objective_recommendation = _objective_recommendation(
                selected_family=selected_family,
                validation_scores=validation_scores,
                eligibility=eligibility_selection.to_dict(),
                use_case=state.objective.constraints["use_case"],
            )
            validation_comparison["objective_recommendation"] = objective_recommendation
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
        "objective_recommendation": validation_comparison.get("objective_recommendation", {}),
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

    ai_analysis = generate_final_analysis(
        {
            "modality": "continuous_tabular",
            "profile": asdict(state.profile),
            "objective": asdict(state.objective),
            "planning": search_payload,
            "generation": generation_metadata,
            "evaluation": evaluation_payload,
            "decisions": decisions,
        },
        FinalAnalysisConfig(mode=ai_analysis_mode, model=ai_model),
    )
    store.save_json("ai_assisted_analysis", ai_analysis, kind="analysis")

    report_path = output / report_name
    report_path.write_text(
        _render_report(
            profile=asdict(state.profile),
            objective=asdict(state.objective),
            config=selected_config,
            search=search_payload,
            generation=generation_metadata,
            evaluation=evaluation_payload,
            decisions=decisions,
            report_name=report_name,
            ai_analysis=ai_analysis,
        ),
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
    *,
    profile: dict[str, object],
    objective: dict[str, object],
    config: VaeConfig | FlowConfig | GmmConfig | CopulaConfig | None,
    search: dict[str, object] | None,
    generation: dict[str, object],
    evaluation: dict[str, object],
    decisions: dict[str, object],
    report_name: str,
    ai_analysis: dict[str, object] | None = None,
) -> str:
    validation = decisions.get("validation_comparison", {})
    recommendation = decisions.get("objective_recommendation", {})
    eligibility = decisions.get("eligibility", {})
    budget = (search or {}).get("budget", {})
    planning = (search or {}).get("planning", {})
    rows = ["| Method | Validation score | Runtime (s) | Duplicate fraction |", "|---|---:|---:|---:|"]
    for method, metrics in validation.get("scores", {}).items():
        runtime = "not recorded"
        family = (search or {}).get(method, {}) if isinstance(search, dict) else {}
        candidates = family.get("candidates", []) if isinstance(family, dict) else []
        if candidates and isinstance(candidates[0], dict):
            runtime = candidates[0].get("wall_seconds", "not recorded")
        rows.append(
            f"| {method} | {metrics.get('selection_score', 'not evaluated'):.4f} | {runtime} | "
            f"{metrics.get('duplicate_fraction', 'not evaluated')} |"
            if isinstance(metrics.get("selection_score"), (int, float))
            else f"| {method} | not evaluated | {runtime} | not evaluated |"
        )
    allocated = planning.get("allocated_models", [])
    skipped = planning.get("skipped_models", {})
    config_summary = config.to_dict() if config else {"method": "training_row_bootstrap"}
    selected_eval = evaluation.get("selected", {}) if isinstance(evaluation, dict) else {}
    lines = [
        "# Generated Report 1: UCI Wine Quality",
        "",
        "## 1. Dataset and objective",
        f"The agent inspected **{profile.get('dataset_name', 'the supplied tabular dataset')}** with "
        f"{profile.get('n_samples')} rows and {profile.get('n_features')} input columns. "
        f"The user objective was: *{objective.get('raw_text', 'data generation')}*",
        "",
        "The 11 physicochemical variables were modeled as continuous features. The `quality` column was excluded from unconditional generation and retained as metadata/target information. No missing values were detected.",
        "",
        "## 2. Agent decision trace",
        "1. **Inspect:** profiled feature types, missingness, roles, and dataset size.",
        "2. **Interpret:** classified the request as unconditional continuous-tabular generation for internal testing.",
        f"3. **Eligibility:** eligible methods were `{', '.join(eligibility.get('eligible_methods', []))}`. Eligibility was checked before ranking.",
        f"4. **Plan:** global budget was `{budget.get('max_trained_candidates', 'not recorded')}` trained candidates; allocated families were `{', '.join(allocated) if allocated else 'none'}`. Bootstrap did not consume a training slot.",
        "5. **Execute/Evaluate:** preprocessing was fit on training data; candidates were compared on validation data and the selected output was evaluated once on held-out data.",
        "6. **Reflect/Recommend:** the recommendation is relative and heuristic; it does not establish privacy or release suitability.",
        "",
        "## 3. Training decisions",
        f"Selected generated-output configuration: `{config_summary.get('method', 'training_row_bootstrap')}`.",
        f"Recorded training budget: `{budget.get('max_trained_candidates', 'not recorded')}` candidates and `{decisions.get('constraints', ['max_epochs not recorded'])[1] if len(decisions.get('constraints', [])) > 1 else 'max_epochs not recorded'}`.",
        f"Candidate allocation was determined by the profile-aware planner, not registry order. Skipped models: `{', '.join(skipped) if skipped else 'none recorded'}`.",
        "",
        "## 4. Validation comparison",
        "Higher `selection_score` is better. The score combines train-scale-normalized marginal fidelity, dependence preservation, and the legacy marginal-fidelity component. VAE ELBO and flow NLL were not compared across families.",
        "",
        *rows,
        "",
        f"The validation winner was **{eligibility.get('selected_method', 'not recorded')}**. The stored decision was: {str(eligibility.get('selection_reason', 'not recorded')).rstrip('.') }.",
        "",
        "## 5. Generated output and recommendation",
        f"- Generated samples: `{generation.get('n_samples', 'not recorded')}`",
        f"- Generation method: `{generation.get('method', 'not recorded')}`",
        f"- Output file: `{generation.get('output_path', 'synthetic_samples.csv')}`",
        f"- Recommendation: **{recommendation.get('recommendation', 'not recorded')}**",
        f"- Reason: {' '.join(recommendation.get('reasons', []))}",
        f"- Selected-output test diagnostics: selection score `{selected_eval.get('selection_score', 'not recorded')}`, duplicate fraction `{selected_eval.get('duplicate_fraction', 'not recorded')}`, nearest-neighbor mean `{selected_eval.get('nearest_neighbor_mean', 'not recorded')}`.",
        "",
        "For this internal-testing objective, bootstrap ranked first. Because it resamples exact training rows, it is not eligible for external synthetic-data sharing. No current method provides a formal privacy guarantee.",
        "",
        "## 6. Limitations and artifacts",
        "- Metrics are finite-sample diagnostics, not uncertainty estimates or proof of utility.",
        "- The deterministic planner uses explainable rules; reflection does not trigger replanning or retraining.",
        "",
        "## 7. AI-assisted final analysis",
        _ai_analysis_markdown(ai_analysis),
        "- No conditional generation, imputation, formal privacy mechanism, or downstream predictive-utility protocol is implemented.",
        "- Detailed artifacts are stored beside this report: `dataset_profile.json`, `objective.json`, `split_manifest.json`, `preprocessing.json`, `candidate_search.json`, `validation_comparison.json`, `evaluation_results.json`, `agent_decisions.json`, model checkpoints, and `synthetic_samples.csv`.",
        f"- This report was generated as `{report_name}` from stored workflow results.",
        "",
    ]
    return "\n".join(lines)


def _ai_analysis_markdown(analysis: dict[str, object] | None) -> str:
    analysis = analysis or {}
    status = analysis.get("status", "not recorded")
    if status == "generated":
        return f"Status: `{status}`; provider: `{analysis.get('provider')}`; model: `{analysis.get('model')}`.\n\n{analysis.get('text', '')}"
    return f"Status: `{status}`. {analysis.get('reason', 'No AI-assisted narrative was recorded.')}"


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


def _objective_recommendation(*, selected_family: str, validation_scores: dict[str, dict[str, float]], eligibility: dict[str, object], use_case: str) -> dict[str, object]:
    sorted_scores = sorted(
        validation_scores.items(),
        key=lambda item: item[1]["selection_score"],
        reverse=True,
    )
    selected_score = validation_scores.get(selected_family, {}).get("selection_score")
    adequacy_threshold = 0.85
    if use_case == "formal_privacy" or selected_family == "rejected":
        recommendation = "reject"
        reasons = ["No implemented method provides a formal privacy guarantee."]
    elif selected_score is not None and selected_score >= adequacy_threshold:
        recommendation = "consider_selected_with_limitations"
        reasons = ["Selected method ranked highest among eligible validation candidates."]
    else:
        recommendation = "insufficient_evidence"
        reasons = ["Best eligible validation score is below the heuristic adequacy-warning threshold."]
    if use_case == "external_sharing":
        reasons.append("External sharing excludes exact-record bootstrap and still requires disclosure that no formal privacy guarantee is implemented.")
    return {
        "use_case": use_case,
        "selected_family": selected_family,
        "relative_ranking": [name for name, _score in sorted_scores],
        "selected_validation_score": selected_score,
        "adequacy_warning_threshold": adequacy_threshold,
        "adequacy_claim": False,
        "recommendation": recommendation,
        "reasons": reasons,
        "eligibility_summary": eligibility,
        "language_guardrail": "Eligibility and high relative fidelity do not establish privacy, safety, clinical utility, or external-release suitability.",
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
