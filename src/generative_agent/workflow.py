"""End-to-end Phase 2 tabular VAE workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd

from generative_agent.agent import AgentFoundation
from generative_agent.artifacts import LocalArtifactStore
from generative_agent.evaluation import bootstrap_baseline, evaluate_synthetic
from generative_agent.generation import generate_synthetic
from generative_agent.objectives import ObjectiveType
from generative_agent.splitting import DatasetSplitter, SplitConfig
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles, load_csv_dataset
from generative_agent.tabular_preprocessing import TabularVaePreprocessor
from generative_agent.vae import TabularVAE, VaeConfig, VaeTrainer


@dataclass(frozen=True)
class WorkflowResult:
    artifacts_dir: Path
    report_path: Path
    synthetic_path: Path
    checkpoint_path: Path
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
    selected_budget = budget or ExperimentBudget(max_models=1, max_epochs=20, max_wall_minutes=30)
    state = AgentFoundation().prepare(handle, objective_text, budget=selected_budget)
    if state.objective is None or state.objective.objective_type is not ObjectiveType.DATA_GENERATION:
        raise ValueError("Phase 2 tabular VAE workflow supports unconditional data_generation objectives only.")
    if state.feasibility is None or not state.feasibility.feasible:
        raise ValueError("Dataset/objective is not feasible: " + "; ".join(state.feasibility.reasons))

    splitter = DatasetSplitter()
    manifest = splitter.manifest(
        frame.to_dict(orient="records"),
        SplitConfig(seed=selected_budget.seed, stratify_column=(roles.target if roles and roles.target else None)),
    )
    preprocessor = TabularVaePreprocessor(roles=roles).fit(frame, manifest)
    train_array = preprocessor.transform(frame, manifest.indices.train)
    validation_array = preprocessor.transform(frame, manifest.indices.validation)
    test_frame = frame.iloc[list(manifest.indices.test)].reset_index(drop=True)
    train_frame = frame.iloc[list(manifest.indices.train)].reset_index(drop=True)
    metadata = preprocessor._metadata()

    config = VaeConfig(
        input_dim=train_array.shape[1],
        continuous_dim=len(metadata.schema.continuous),
        binary_dim=len(metadata.schema.binary),
        latent_dim=min(4, max(1, train_array.shape[1])),
        epochs=selected_budget.max_epochs,
        batch_size=min(32, max(2, len(train_array))),
        seed=selected_budget.seed,
        early_stopping_patience=5,
    )
    model = TabularVAE(config)
    checkpoint_path = output / "tabular_vae_checkpoint.pt"
    training = VaeTrainer().train(model, train_array, validation_array, checkpoint_path)
    if training.failed:
        raise RuntimeError(f"VAE training failed: {training.failure_reason}")

    sample_count = n_synthetic or len(frame)
    synthetic_path = output / "synthetic_samples.csv"
    generation = generate_synthetic(
        model,
        preprocessor,
        n_samples=sample_count,
        seed=selected_budget.seed + 1,
        output_path=synthetic_path,
    )
    baseline = bootstrap_baseline(train_frame[list(metadata.schema.model_columns)], sample_count, seed=selected_budget.seed)
    baseline.insert(0, "synthetic_record", True)
    vae_eval = evaluate_synthetic(
        train_real=train_frame,
        test_real=test_frame,
        synthetic=generation.samples,
        schema=metadata.schema,
        method="tabular_vae",
    )
    baseline_eval = evaluate_synthetic(
        train_real=train_frame,
        test_real=test_frame,
        synthetic=baseline,
        schema=metadata.schema,
        method="training_row_bootstrap",
    )

    store.save_json("dataset_profile", asdict(state.profile), kind="profile")
    store.save_json("objective", asdict(state.objective), kind="objective")
    store.save_json("split_manifest", asdict(manifest), kind="split")
    store.save_json("preprocessing", metadata.to_dict(), kind="preprocessing")
    store.save_json("model_config", config.to_dict(), kind="model")
    store.save_json("training_history", asdict(training), kind="training")
    store.save_json("generation_metadata", generation.metadata, kind="generation")
    store.save_json(
        "evaluation_results",
        {"vae": vae_eval.to_dict(), "baseline": baseline_eval.to_dict()},
        kind="evaluation",
    )
    decisions = {
        "feasible": state.feasibility.feasible,
        "reasons": state.feasibility.reasons,
        "warnings": state.feasibility.warnings,
        "constraints": state.feasibility.constraints,
        "planner": "deterministic MockPlanner; no LLM reasoning or autonomous optimization",
    }
    store.save_json("agent_decisions", decisions, kind="decisions")
    report_path = output / "final_report.md"
    report_path.write_text(
        _render_report(config, training, generation.metadata, vae_eval.to_dict(), baseline_eval.to_dict(), decisions),
        encoding="utf-8",
    )
    return WorkflowResult(
        artifacts_dir=output,
        report_path=report_path,
        synthetic_path=synthetic_path,
        checkpoint_path=checkpoint_path,
        training_failed=training.failed,
    )


def _render_report(
    config: VaeConfig,
    training: object,
    generation: dict[str, object],
    vae_eval: dict[str, object],
    baseline_eval: dict[str, object],
    decisions: dict[str, object],
) -> str:
    return "\n".join(
        [
            "# Phase 2 Tabular VAE Report",
            "",
            "## Scope",
            "This report is generated from stored Phase 2 workflow results for unconditional tabular synthetic data generation.",
            "",
            "## Modeling Assumptions",
            "- Continuous features use a unit-variance Gaussian observation model after training-only standardization.",
            "- Binary features use a Bernoulli likelihood parameterized by decoder logits.",
            "- Conditional generation, class balancing, image generation, and non-VAE model families are not implemented in Phase 2.",
            "",
            "## Configuration",
            f"```json\n{json.dumps(config.to_dict(), indent=2)}\n```",
            "",
            "## Training",
            f"Best validation loss: {training.best_validation_loss}",
            f"Epochs run: {training.epochs_ran}",
            "",
            "## Generation",
            f"Generated samples: {generation['n_samples']}",
            "",
            "## Evaluation",
            f"```json\n{json.dumps({'vae': vae_eval, 'baseline': baseline_eval}, indent=2)}\n```",
            "",
            "## Agent Decisions",
            f"```json\n{json.dumps(decisions, indent=2)}\n```",
            "",
            "## Limitations",
            "- Nearest-neighbor diagnostics are proxies, not privacy guarantees.",
            "- The planner is deterministic and hardcoded; it does not perform LLM-based reasoning.",
            "- No downstream predictive utility is reported unless a separate valid protocol is implemented.",
            "",
        ]
    )
