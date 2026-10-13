"""Modality-aware image generation workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import numpy as np

from generative_agent.artifacts import LocalArtifactStore
from generative_agent.data import DatasetKind
from generative_agent.eligibility import method_eligibility, select_eligible_method
from generative_agent.image_data import ImagePreprocessConfig, SplitNpzImageAdapter
from generative_agent.image_evaluation import ImageEvaluationResult, evaluate_image_gan, evaluate_image_vae, image_metric_specification
from generative_agent.image_gan import ImageGAN, ImageGanConfig, ImageGanTrainer
from generative_agent.image_vae import ConvImageVAE, ImageVaeConfig, ImageVaeTrainer
from generative_agent.objectives import ObjectiveInterpreter, ObjectiveType
from generative_agent.registry import CapabilityRegistry
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
    gan_latent_dim: int = 32,
    n_generated: int = 32,
    repeated_generation_seeds: int = 3,
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
    image_shape = bundle.profile.image_shape
    candidates = _plan_candidates(image_shape=image_shape, budget=selected_budget, latent_dim=latent_dim, gan_latent_dim=gan_latent_dim)
    if not candidates:
        raise ValueError("No feasible image-generation candidates fit the requested budget.")
    planning = {
        "planning_rule": "Rank implemented feasible image candidates by deterministic capability rules; allocate up to max_models total trained candidates; evaluate validation before held-out test.",
        "modality": "image",
        "dataset_profile": bundle.profile.to_dict(),
        "candidate_priorities": [
            {
                "model": candidate["name"],
                "priority_score": candidate["priority_score"],
                "feasible": True,
                "eligible_before_objective_filter": True,
                "reasons": candidate["reasons"],
            }
            for candidate in candidates
        ],
        "allocated_models": [candidate["name"] for candidate in candidates[: selected_budget.max_models]],
        "budget": {
            "max_trained_candidates": selected_budget.max_models,
            "semantics": "max_models is the total number of trained image candidates across model families.",
        },
    }

    candidate_records: dict[str, dict[str, object]] = {}
    validation_scores: dict[str, dict[str, float]] = {}
    trained_models: dict[str, object] = {}
    trained = 0
    for candidate in candidates:
        name = candidate["name"]
        if trained >= selected_budget.max_models:
            candidate_records[name] = {
                "status": "skipped",
                "reason": f"Total trained-model budget exhausted at {selected_budget.max_models}.",
            }
            continue
        eligibility = method_eligibility(name, objective.constraints["use_case"])
        if not eligibility.eligible:
            candidate_records[name] = {"status": "rejected", "eligibility": eligibility.to_dict()}
            continue
        record, model, evaluation = _train_and_evaluate_candidate(
            name=name,
            config=candidate["config"],
            train_dataset=bundle.train,
            validation_dataset=bundle.validation,
            output=output,
            n_generated=n_generated,
            seeds=tuple(selected_budget.seed + 100 * trained + offset + 1 for offset in range(repeated_generation_seeds)),
        )
        candidate_records[name] = record
        trained += 1
        if model is not None and evaluation is not None:
            trained_models[name] = model
            validation_scores[name] = {"selection_score": float(record["validation_reliability"]["selection_score_mean"])}

    eligibility = select_eligible_method(
        validation_scores,
        use_case=objective.constraints["use_case"],
        ranking_metric="selection_score",
    )
    if eligibility.selected_method is None:
        raise ValueError(eligibility.selection_reason)
    selected_score = max(validation_scores[method]["selection_score"] for method in validation_scores)
    adequacy_warning_threshold = 0.85
    absolute_evidence = _absolute_evidence(candidate_records, selected_score, adequacy_warning_threshold)
    selected_name = _selected_candidate_name(eligibility.selected_method, candidate_records)
    selected_record = candidate_records[selected_name]
    selected_model = trained_models[selected_name]
    final_grid_path = output / f"{selected_name}_test_grid.png"
    final_test_evaluation = _evaluate_selected_on_test(
        name=selected_name,
        model=selected_model,
        train_dataset=bundle.train,
        test_dataset=bundle.test,
        n_generated=n_generated,
        seed=selected_budget.seed + 1000,
        grid_path=final_grid_path,
    )
    workflow_wall_seconds = time.perf_counter() - workflow_start
    store.save_json("dataset_profile", bundle.profile.to_dict(), kind="profile")
    store.save_json("preprocessing", bundle.preprocessing, kind="preprocessing")
    store.save_json("objective", {"text": objective_text, "type": objective.objective_type.value, "constraints": dict(objective.constraints)}, kind="objective")
    store.save_json("planning", planning, kind="planning")
    store.save_json("candidate_results", candidate_records, kind="candidate_results")
    store.save_json("image_metric_specification", image_metric_specification(), kind="metric_specification")
    store.save_json("model_config", selected_record["config"], kind="model")
    store.save_json("training_history", selected_record["training"], kind="training")
    store.save_json("evaluation_results", selected_record["validation_evaluation"], kind="validation_evaluation")
    store.save_json("final_test_evaluation", final_test_evaluation.to_dict(), kind="test_evaluation")
    decisions = {
        "architecture": "Inspect -> Interpret -> Check Eligibility -> Plan -> Execute -> Evaluate -> Reflect -> Report",
        "planning": planning,
        "eligibility": eligibility.to_dict(),
        "labels": "metadata_only_not_conditioning",
        "reconstruction_assumption": config.reconstruction_assumption,
        "selection_rule": "Train up to max_models total feasible eligible image candidates; select highest shared validation selection_score; evaluate held-out test only after selection.",
        "selection_metric": "1 / (1 + pixel mean error + pixel std error + 0.5 gradient mean error + 0.5 gradient std error + local patch std error + diversity error + 0.25 handcrafted feature MMD + duplicate fraction)",
        "adequacy_warning_threshold": adequacy_warning_threshold,
        "absolute_evidence": absolute_evidence,
        "selected_candidate": selected_name,
        "workflow_wall_seconds": workflow_wall_seconds,
        "limitations": [
            "Small image experiments are not evidence of clinically meaningful generative quality.",
            "No privacy guarantee is implemented.",
            "Image VAE and GAN are unconditional; labels are not used for conditioning.",
            "The handcrafted feature MMD is not KID/FID and is not a validated pathology quality metric.",
        ],
    }
    store.save_json("agent_decisions", decisions, kind="decisions")
    report = output / "final_report.md"
    report.write_text(_render_report(bundle.profile.to_dict(), selected_record["config"], selected_record["training"], selected_record["validation_evaluation"], final_test_evaluation.to_dict(), decisions, candidate_records), encoding="utf-8")
    return ImageWorkflowResult(output, report, Path(str(selected_record["checkpoint_path"])), final_grid_path, False)


def _plan_candidates(*, image_shape: tuple[int, int, int], budget: ExperimentBudget, latent_dim: int, gan_latent_dim: int) -> list[dict[str, object]]:
    registry = CapabilityRegistry()
    capabilities = registry.candidates(DatasetKind.IMAGE, ObjectiveType.DATA_GENERATION, implemented_only=True)
    planned: list[dict[str, object]] = []
    for capability in capabilities:
        if capability.name == "image_vae":
            planned.append(
                {
                    "name": "image_vae",
                    "priority_score": 0.65,
                    "reasons": (
                        "Image VAE supports small-image unconditional generation and reconstruction diagnostics.",
                        "VAE is prioritized for CPU smoke tests because training is stable and deterministic.",
                    ),
                    "reason": "VAE supports unconditional small-image generation and reconstruction diagnostics.",
                    "config": ImageVaeConfig(
                        image_shape=image_shape,
                        latent_dim=latent_dim,
                        epochs=budget.max_epochs,
                        batch_size=64,
                        seed=budget.seed,
                    ),
                }
            )
        elif capability.name == "image_gan":
            planned.append(
                {
                    "name": "image_gan",
                    "priority_score": 0.55,
                    "reasons": (
                        "GAN supports small-image unconditional generation.",
                        "GAN is lower priority under small CPU budgets because adversarial quality evidence is less stable.",
                    ),
                    "reason": "GAN supports unconditional small-image generation with adversarial training.",
                    "config": ImageGanConfig(
                        image_shape=image_shape,
                        latent_dim=gan_latent_dim,
                        epochs=budget.max_epochs,
                        batch_size=64,
                        seed=budget.seed + 17,
                    ),
                }
            )
    return sorted(planned, key=lambda item: (-item["priority_score"], item["name"]))


def _train_and_evaluate_candidate(
    *,
    name: str,
    config,
    train_dataset,
    validation_dataset,
    output: Path,
    n_generated: int,
    seeds: tuple[int, ...],
) -> tuple[dict[str, object], object | None, ImageEvaluationResult | None]:
    if name == "image_vae":
        model = ConvImageVAE(config)
        checkpoint = output / "image_vae_checkpoint.pt"
        training = ImageVaeTrainer().train(model, train_dataset, validation_dataset, checkpoint)
        if training.failed:
            return _failed_record(name, config, training), None, None
        evaluations = [
            evaluate_image_vae(
                model=model,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                n_generated=n_generated,
                seed=seed,
                grid_path=output / ("image_vae_validation_grid.png" if index == 0 else f"image_vae_validation_grid_seed{index}.png"),
            )
            for index, seed in enumerate(seeds)
        ]
        return _success_record(config, training, evaluations), model, evaluations[0]
    if name == "image_gan":
        model = ImageGAN(config)
        checkpoint = output / "image_gan_checkpoint.pt"
        training = ImageGanTrainer().train(model, train_dataset, validation_dataset, checkpoint)
        if training.failed:
            return _failed_record(name, config, training), None, None
        evaluations = [
            evaluate_image_gan(
                model=model,
                train_dataset=train_dataset,
                validation_dataset=validation_dataset,
                n_generated=n_generated,
                seed=seed,
                grid_path=output / ("image_gan_validation_grid.png" if index == 0 else f"image_gan_validation_grid_seed{index}.png"),
            )
            for index, seed in enumerate(seeds)
        ]
        return _success_record(config, training, evaluations), model, evaluations[0]
    raise ValueError(f"Unsupported image candidate: {name}.")


def _success_record(config, training, evaluations: list[ImageEvaluationResult]) -> dict[str, object]:
    scores = [item.selection_score for item in evaluations]
    return {
        "status": "trained",
        "config": config.to_dict(),
        "training": asdict(training),
        "validation_evaluation": evaluations[0].to_dict(),
        "validation_repeated_evaluations": [item.to_dict() for item in evaluations],
        "validation_reliability": {
            "selection_score_mean": float(sum(scores) / len(scores)),
            "selection_score_std": float(np.std(scores, ddof=0)),
            "selection_score_min": float(min(scores)),
            "selection_score_max": float(max(scores)),
            "n_generation_seeds": len(scores),
            "practical_difference_threshold": 0.01,
        },
        "checkpoint_path": str(training.checkpoint_path),
    }


def _failed_record(name: str, config, training) -> dict[str, object]:
    return {
        "status": "failed",
        "name": name,
        "config": config.to_dict(),
        "training": asdict(training),
        "reason": training.failure_reason,
    }


def _absolute_evidence(candidate_records: dict[str, dict[str, object]], selected_score: float, warning_threshold: float) -> dict[str, object]:
    trained = {name: record for name, record in candidate_records.items() if record.get("status") == "trained"}
    ranked = sorted(
        (
            (name, float(record["validation_reliability"]["selection_score_mean"]), float(record["validation_reliability"]["selection_score_std"]))
            for name, record in trained.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if len(ranked) >= 2:
        margin = ranked[0][1] - ranked[1][1]
        practical_difference = margin >= float(trained[ranked[0][0]]["validation_reliability"]["practical_difference_threshold"])
    else:
        margin = None
        practical_difference = False
    return {
        "selected_validation_score_mean": selected_score,
        "warning_threshold": warning_threshold,
        "adequacy_claim": False,
        "quality_warning": selected_score < warning_threshold,
        "top_two_score_margin": margin,
        "difference_practically_meaningful_by_heuristic": practical_difference,
        "interpretation": "Ranking is relative and heuristic; no score threshold establishes clinical realism, privacy, or external-release suitability.",
    }


def _selected_candidate_name(selected_method: str, candidate_records: dict[str, dict[str, object]]) -> str:
    if selected_method == "vae" and "image_vae" in candidate_records:
        return "image_vae"
    if selected_method == "gan" and "image_gan" in candidate_records:
        return "image_gan"
    if selected_method in candidate_records:
        return selected_method
    raise ValueError(f"Selected method {selected_method} does not match trained image candidates.")


def _evaluate_selected_on_test(*, name: str, model, train_dataset, test_dataset, n_generated: int, seed: int, grid_path: Path) -> ImageEvaluationResult:
    if name == "image_vae":
        return evaluate_image_vae(model=model, train_dataset=train_dataset, validation_dataset=test_dataset, n_generated=n_generated, seed=seed, grid_path=grid_path)
    if name == "image_gan":
        return evaluate_image_gan(model=model, train_dataset=train_dataset, validation_dataset=test_dataset, n_generated=n_generated, seed=seed, grid_path=grid_path)
    raise ValueError(f"Unsupported selected image model: {name}.")


def _render_report(profile, config, training, validation_evaluation, test_evaluation, decisions, candidates) -> str:
    return "\n".join(
        [
            "# Project 2 Image Generative Agent Report",
            "",
            "## Dataset Profile",
            f"```json\n{json.dumps(profile, indent=2)}\n```",
            "",
            "## Selected Model Configuration",
            f"```json\n{json.dumps(config, indent=2)}\n```",
            "",
            "## Selected Model Training",
            f"```json\n{json.dumps(training, indent=2, default=str)}\n```",
            "",
            "## Validation Evaluation",
            f"```json\n{json.dumps(validation_evaluation, indent=2)}\n```",
            "",
            "## Held-Out Test Evaluation",
            f"```json\n{json.dumps(test_evaluation, indent=2)}\n```",
            "",
            "## Candidate Results",
            f"```json\n{json.dumps(candidates, indent=2, default=str)}\n```",
            "",
            "## Agent Decisions",
            f"```json\n{json.dumps(decisions, indent=2)}\n```",
            "",
        ]
    )
