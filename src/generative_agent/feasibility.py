"""Deterministic feasibility checks before planning experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from generative_agent.data import DatasetKind, DatasetProfile, FeatureType
from generative_agent.objectives import ObjectiveSpec, ObjectiveType
from generative_agent.registry import CapabilityRegistry, ModelCapability
from generative_agent.state_types import BudgetView


@dataclass(frozen=True)
class FeasibilityReport:
    feasible: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = field(default_factory=tuple)
    candidates: tuple[ModelCapability, ...] = field(default_factory=tuple)
    constraints: tuple[str, ...] = field(default_factory=tuple)


class FeasibilityChecker:
    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self.registry = registry or CapabilityRegistry()

    def check(
        self,
        profile: DatasetProfile,
        objective: ObjectiveSpec,
        budget: BudgetView | None = None,
    ) -> FeasibilityReport:
        reasons: list[str] = []
        warnings: list[str] = []
        constraints: list[str] = []
        candidates = self.registry.candidates(profile.kind, objective.objective_type)

        if profile.n_samples < 10:
            reasons.append("At least 10 samples are required before generative modeling is considered.")
        if profile.kind is DatasetKind.UNKNOWN:
            reasons.append("Dataset kind is unknown; adapter-specific inspection is required.")
        if objective.objective_type is ObjectiveType.CLASS_BALANCING and not profile.is_labeled:
            reasons.append("Class balancing requires a labeled dataset.")
        if objective.objective_type is ObjectiveType.IMPUTATION and not profile.has_missing_values:
            warnings.append("No missing values were detected; imputation may not be needed.")
        unsupported = _unsupported_features(profile)
        if unsupported:
            reasons.append(f"Unsupported feature types detected: {', '.join(unsupported)}.")
        fully_missing = [
            feature.name for feature in profile.feature_schema if feature.missing_count >= profile.n_samples
        ]
        if fully_missing:
            reasons.append(f"Features with no observed values cannot be modeled: {', '.join(fully_missing)}.")
        if not candidates:
            reasons.append("No planned model capability matches this dataset/objective pair.")
        if profile.n_samples < 100:
            warnings.append("Small sample size may limit generative model quality.")
        if budget is not None:
            constraints.extend(
                (
                    f"max_models={budget.max_models}",
                    f"max_epochs={budget.max_epochs}",
                    f"max_wall_minutes={budget.max_wall_minutes}",
                    f"seed={budget.seed}",
                )
            )
            if budget.max_wall_minutes < 5:
                reasons.append("At least 5 wall-clock minutes are required for planned experiments.")

        return FeasibilityReport(
            feasible=not reasons,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            candidates=candidates,
            constraints=tuple(constraints),
        )


def _unsupported_features(profile: DatasetProfile) -> tuple[str, ...]:
    if profile.kind is not DatasetKind.TABULAR:
        return tuple()
    supported = {FeatureType.NUMERIC, FeatureType.CATEGORICAL, FeatureType.BOOLEAN}
    return tuple(
        feature.name
        for feature in profile.feature_schema
        if feature.feature_type not in supported and feature.name not in profile.role_exclusions
    )
