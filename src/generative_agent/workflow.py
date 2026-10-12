"""End-to-end Phase 2.5 tabular VAE workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from generative_agent.agent import AgentFoundation
from generative_agent.artifacts import LocalArtifactStore
from generative_agent.decisioning import DecisionAction, DeterministicDecisionEngine
from generative_agent.evaluation import bootstrap_baseline, evaluate_synthetic
from generative_agent.generation import generate_synthetic
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


def run_tabular_vae_workflow(
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

    selected_config: VaeConfig | None = None
    checkpoint_path: Path | None = None
    training_failed = False
    search_payload: dict[str, object] | None = None

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
        configs = propose_vae_configs(
            input_dim=train_array.shape[1],
            continuous_dim=len(metadata.schema.continuous),
            binary_dim=len(metadata.schema.binary),
            n_train=len(train_array),
            budget=selected_budget,
        )
        search, model = run_vae_search(
            train_array=train_array,
            validation_array=validation_array,
            configs=configs,
            checkpoint_dir=output,
        )
        search_payload = search.to_dict()
        selected_config = search.selected.config
        checkpoint_path = search.selected.checkpoint_path
        training_failed = search.selected.failed
        if training_failed:
            baseline.to_csv(synthetic_path, index=False)
            generation_metadata = {
                "n_samples": sample_count,
                "seed": selected_budget.seed,
                "method": "training_row_bootstrap_after_vae_failure",
                "output_path": str(synthetic_path),
            }
            evaluation_payload = {"selected": baseline_eval.to_dict(), "baseline": baseline_eval.to_dict()}
        else:
            generation = generate_synthetic(
                model,
                preprocessor,
                n_samples=sample_count,
                seed=selected_budget.seed + 1,
                output_path=synthetic_path,
            )
            vae_eval = evaluate_synthetic(
                train_real=train_frame,
                test_real=test_frame,
                synthetic=generation.samples,
                schema=metadata.schema,
                method="tabular_vae",
            )
            generation_metadata = generation.metadata
            evaluation_payload = {
                "selected": vae_eval.to_dict(),
                "vae": vae_eval.to_dict(),
                "baseline": baseline_eval.to_dict(),
            }

    decisions = {
        "feasible": state.feasibility.feasible,
        "reasons": state.feasibility.reasons,
        "warnings": state.feasibility.warnings,
        "constraints": state.feasibility.constraints,
        "objective_decision": decision.to_dict(),
        "selection": search_payload["selection_reason"] if search_payload else "Baseline selected before VAE training.",
        "planner": "deterministic rules plus validation-only candidate search; no LLM reasoning or autonomous optimization",
    }

    store.save_json("dataset_profile", asdict(state.profile), kind="profile")
    store.save_json("objective", asdict(state.objective), kind="objective")
    store.save_json("split_manifest", asdict(manifest), kind="split")
    store.save_json("preprocessing", metadata.to_dict(), kind="preprocessing")
    store.save_json("model_config", selected_config.to_dict() if selected_config else {"method": "baseline"}, kind="model")
    if search_payload is not None:
        store.save_json("candidate_search", search_payload, kind="search")
        store.save_json("training_history", search.selected.history, kind="training")
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
    config: VaeConfig | None,
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
            "- Candidate selection uses validation loss only; held-out test data is used only for final evaluation.",
            "- Conditional generation, class balancing, image generation, and non-VAE model families are not implemented.",
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
            "",
        ]
    )
