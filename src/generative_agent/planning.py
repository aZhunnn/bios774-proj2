"""Structured mock planning for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

from generative_agent.data import DatasetProfile
from generative_agent.feasibility import FeasibilityReport
from generative_agent.objectives import ObjectiveSpec, ObjectiveType
from generative_agent.state_types import BudgetView


@dataclass(frozen=True)
class PlannedExperiment:
    model_name: str
    family: str
    training_framework: str
    rationale: str
    planned_hyperparameters: dict[str, int | float | str]
    implemented: bool


@dataclass(frozen=True)
class ExperimentPlan:
    objective_type: ObjectiveType
    experiments: tuple[PlannedExperiment, ...]
    evaluation_criteria: tuple[str, ...]
    downstream_recommendation_focus: tuple[str, ...]
    training_enabled: bool = False


class MockPlanner:
    """Creates deterministic plans without invoking model training."""

    def plan(
        self,
        profile: DatasetProfile,
        objective: ObjectiveSpec,
        feasibility: FeasibilityReport,
        budget: BudgetView,
    ) -> ExperimentPlan:
        chosen = feasibility.candidates[: budget.max_models] if feasibility.feasible else tuple()
        experiments = tuple(
            PlannedExperiment(
                model_name=candidate.name,
                family=candidate.family,
                training_framework=candidate.training_framework,
                rationale=(
                    f"{candidate.family} is planned for {profile.kind.value} data and "
                    f"{objective.objective_type.value} objectives."
                ),
                planned_hyperparameters={
                    "epochs": min(25, budget.max_epochs),
                    "seed": budget.seed,
                    "batch_size": _batch_size(profile.n_samples),
                },
                implemented=candidate.implemented,
            )
            for candidate in chosen
        )
        return ExperimentPlan(
            objective_type=objective.objective_type,
            experiments=experiments,
            evaluation_criteria=_evaluation_criteria(objective.objective_type, profile),
            downstream_recommendation_focus=_recommendation_focus(objective.objective_type),
            training_enabled=False,
        )


def _batch_size(n_samples: int) -> int:
    if n_samples < 64:
        return 16
    if n_samples < 512:
        return 32
    return 64


def _evaluation_criteria(
    objective_type: ObjectiveType,
    profile: DatasetProfile,
) -> tuple[str, ...]:
    criteria = ["distribution_similarity", "privacy_risk_screen"]
    if profile.is_labeled:
        criteria.append("label_distribution_preservation")
    if objective_type is ObjectiveType.IMPUTATION:
        criteria.append("heldout_missingness_reconstruction")
    if objective_type is ObjectiveType.CLASS_BALANCING:
        criteria.append("minority_class_utility")
    return tuple(criteria)


def _recommendation_focus(objective_type: ObjectiveType) -> tuple[str, ...]:
    mapping = {
        ObjectiveType.DATA_GENERATION: ("synthetic_data_release", "downstream_model_augmentation"),
        ObjectiveType.IMPUTATION: ("completed_dataset_quality", "sensitivity_analysis"),
        ObjectiveType.CLASS_BALANCING: ("training_set_augmentation", "class_specific_validation"),
        ObjectiveType.ANOMALY_DETECTION: ("candidate_outlier_review",),
        ObjectiveType.REPRESENTATION_LEARNING: ("latent_space_analysis",),
        ObjectiveType.UNCERTAINTY_QUANTIFICATION: ("uncertainty_intervals",),
        ObjectiveType.EXPLORATORY: ("model_applicability_review",),
    }
    return mapping[objective_type]
