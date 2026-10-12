"""Modality-aware image generation workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

from generative_agent.artifacts import LocalArtifactStore
from generative_agent.eligibility import select_eligible_method
from generative_agent.image_data import ImagePreprocessConfig, SplitNpzImageAdapter
from generative_agent.image_evaluation import evaluate_image_vae
from generative_agent.image_vae import ConvImageVAE, ImageVaeConfig, ImageVaeTrainer
from generative_agent.objectives import ObjectiveInterpreter, ObjectiveType
from generative_agent.state import ExperimentBudget


@dataclass(frozen=True)
class ImageWorkflowResult:
    artifacts_dir: Path
    report_path: Path
    checkpoint_path: Path
    generated_grid_path: Path
    training_failed: bool


def run_image_generation_workflow(
    *,
    npz_path: str | Path,
    output_dir: str | Path,
    objective_text: str,
    budget: ExperimentBudget | None = None,
    preprocess: ImagePreprocessConfig | None = None,
    latent_dim: int = 16,
    n_generated: int = 32,
) -> ImageWorkflowResult:
    workflow_start = time.perf_counter()
    selected_budget = budget or ExperimentBudget(max_models=1, max_epochs=3, max_wall_minutes=30)
    objective = ObjectiveInterpreter().interpret(objective_text)
    if objective.objective_type is not ObjectiveType.DATA_GENERATION:
        raise ValueError(f"Unsupported image objective: {objective.objective_type.value}.")
    output = Path(output_dir)
    store = LocalArtifactStore(output)
    config = preprocess or ImagePreprocessConfig(
        max_train_samples=256,
        max_validation_samples=64,
        max_test_samples=64,
        seed=selected_budget.seed,
    )
    bundle = SplitNpzImageAdapter().load(npz_path, config=config)
    eligibility = select_eligible_method(
        {"image_vae": {"selection_score": 0.0}},
        use_case=objective.constraints["use_case"],
        ranking_metric="selection_score",
    )
    if eligibility.selected_method is None:
        raise ValueError(eligibility.selection_reason)
    image_shape = bundle.profile.image_shape
    model_config = ImageVaeConfig(
        image_shape=image_shape,
        latent_dim=latent_dim,
        epochs=selected_budget.max_epochs,
        batch_size=64,
        seed=selected_budget.seed,
        reconstruction_assumption=config.reconstruction_assumption,
    )
    model = ConvImageVAE(model_config)
    checkpoint = output / "image_vae_checkpoint.pt"
    training = ImageVaeTrainer().train(model, bundle.train, bundle.validation, checkpoint)
    if training.failed:
        raise RuntimeError(f"Image VAE training failed: {training.failure_reason}")
    grid_path = output / "generated_grid.png"
    evaluation = evaluate_image_vae(
        model=model,
        train_dataset=bundle.train,
        validation_dataset=bundle.validation,
        n_generated=n_generated,
        seed=selected_budget.seed + 1,
        grid_path=grid_path,
    )
    workflow_wall_seconds = time.perf_counter() - workflow_start
    store.save_json("dataset_profile", bundle.profile.to_dict(), kind="profile")
    store.save_json("preprocessing", bundle.preprocessing, kind="preprocessing")
    store.save_json("objective", {"text": objective_text, "type": objective.objective_type.value, "constraints": dict(objective.constraints)}, kind="objective")
    store.save_json("model_config", model_config.to_dict(), kind="model")
    store.save_json("training_history", asdict(training), kind="training")
    store.save_json("evaluation_results", evaluation.to_dict(), kind="evaluation")
    decisions = {
        "architecture": "Inspect -> Interpret -> Check Eligibility -> Plan -> Execute -> Evaluate -> Reflect -> Report",
        "eligibility": eligibility.to_dict(),
        "labels": "metadata_only_not_conditioning",
        "reconstruction_assumption": config.reconstruction_assumption,
        "workflow_wall_seconds": workflow_wall_seconds,
        "limitations": [
            "Smoke test uses a small subset and is not evidence of production generative quality.",
            "No privacy guarantee is implemented.",
            "Image VAE is unconditional; labels are not used for conditioning.",
        ],
    }
    store.save_json("agent_decisions", decisions, kind="decisions")
    report = output / "final_report.md"
    report.write_text(_render_report(bundle.profile.to_dict(), model_config.to_dict(), asdict(training), evaluation.to_dict(), decisions), encoding="utf-8")
    return ImageWorkflowResult(output, report, checkpoint, grid_path, training.failed)


def _render_report(profile, config, training, evaluation, decisions) -> str:
    return "\n".join(
        [
            "# Phase 5A Image VAE Smoke Report",
            "",
            "## Dataset Profile",
            f"```json\n{json.dumps(profile, indent=2)}\n```",
            "",
            "## Model Configuration",
            f"```json\n{json.dumps(config, indent=2)}\n```",
            "",
            "## Training",
            f"```json\n{json.dumps(training, indent=2, default=str)}\n```",
            "",
            "## Evaluation",
            f"```json\n{json.dumps(evaluation, indent=2)}\n```",
            "",
            "## Agent Decisions",
            f"```json\n{json.dumps(decisions, indent=2)}\n```",
            "",
        ]
    )
