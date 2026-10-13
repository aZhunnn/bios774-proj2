"""Deterministic profile-aware planning for tabular generative models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from generative_agent.eligibility import method_eligibility
from generative_agent.objectives import ObjectiveType, ObjectiveUseCase
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_preprocessing import TabularVaeSchema


@dataclass(frozen=True)
class TabularProfileStats:
    n_samples: int
    n_features: int
    continuous_features: int
    binary_features: int
    continuous_only: bool
    mean_abs_skewness: float
    mean_abs_correlation: float
    max_abs_correlation: float
    sample_feature_ratio: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidatePriority:
    model: str
    feasible: bool
    eligible: bool
    priority_score: float
    reasons: tuple[str, ...]
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TabularTrainingPlan:
    profile_stats: TabularProfileStats
    candidates: tuple[CandidatePriority, ...]
    allocation: dict[str, int]
    allocated_models: tuple[str, ...]
    skipped_models: dict[str, str]
    planning_rule: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["profile_stats"] = self.profile_stats.to_dict()
        payload["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return payload


def build_tabular_training_plan(
    *,
    train_frame: pd.DataFrame,
    schema: TabularVaeSchema,
    objective_type: ObjectiveType,
    use_case: ObjectiveUseCase | str,
    budget: ExperimentBudget,
    prior_validation_scores: Mapping[str, float] | None = None,
) -> TabularTrainingPlan:
    if isinstance(use_case, str):
        use_case = ObjectiveUseCase(use_case)
    stats = profile_tabular_training_data(train_frame, schema)
    priors = dict(prior_validation_scores or {})
    candidates = tuple(
        sorted(
            (
                _candidate_priority("vae", stats, objective_type, use_case, priors),
                _candidate_priority("flow", stats, objective_type, use_case, priors),
                _candidate_priority("gmm", stats, objective_type, use_case, priors),
                _candidate_priority("copula", stats, objective_type, use_case, priors),
            ),
            key=lambda item: (-item.priority_score, item.model),
        )
    )
    allocated: list[str] = []
    skipped: dict[str, str] = {}
    trainable = [candidate for candidate in candidates if candidate.feasible and candidate.eligible]
    for candidate in candidates:
        if not candidate.feasible:
            skipped[candidate.model] = candidate.skip_reason or "Model is infeasible for this dataset profile."
            continue
        if not candidate.eligible:
            skipped[candidate.model] = candidate.skip_reason or "Model is ineligible for the requested objective."
            continue
        if len(allocated) < budget.max_models:
            allocated.append(candidate.model)
        else:
            skipped[candidate.model] = f"Training budget exhausted after {budget.max_models} allocated candidates."
    while trainable and len(allocated) < budget.max_models:
        allocated.append(trainable[(len(allocated) - len(trainable)) % len(trainable)].model)
    allocation = {model: allocated.count(model) for model in ("vae", "flow", "gmm", "copula")}
    return TabularTrainingPlan(
        profile_stats=stats,
        candidates=candidates,
        allocation=allocation,
        allocated_models=tuple(allocated),
        skipped_models=skipped,
        planning_rule=(
            "Rank feasible and eligible tabular candidates using deterministic dataset-profile scores "
            "from modality, feature types, sample size, dimensionality, skewness, correlation, objective, "
            "budget, eligibility, and optional prior validation evidence; allocate max_models total "
            "trained candidates by priority."
        ),
    )


def profile_tabular_training_data(train_frame: pd.DataFrame, schema: TabularVaeSchema) -> TabularProfileStats:
    continuous = list(schema.continuous)
    n_features = len(schema.model_columns)
    continuous_frame = train_frame[continuous].astype(float) if continuous else pd.DataFrame(index=train_frame.index)
    if continuous:
        skewness = continuous_frame.skew(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        mean_abs_skewness = float(skewness.abs().mean())
    else:
        mean_abs_skewness = 0.0
    if len(continuous) >= 2:
        corr = continuous_frame.corr().fillna(0.0).to_numpy()
        mask = ~np.eye(corr.shape[0], dtype=bool)
        offdiag = np.abs(corr[mask])
        mean_abs_correlation = float(offdiag.mean()) if len(offdiag) else 0.0
        max_abs_correlation = float(offdiag.max()) if len(offdiag) else 0.0
    else:
        mean_abs_correlation = 0.0
        max_abs_correlation = 0.0
    return TabularProfileStats(
        n_samples=int(len(train_frame)),
        n_features=int(n_features),
        continuous_features=len(schema.continuous),
        binary_features=len(schema.binary),
        continuous_only=len(schema.binary) == 0 and len(schema.continuous) == n_features,
        mean_abs_skewness=mean_abs_skewness,
        mean_abs_correlation=mean_abs_correlation,
        max_abs_correlation=max_abs_correlation,
        sample_feature_ratio=float(len(train_frame) / max(n_features, 1)),
    )


def _candidate_priority(
    model: str,
    stats: TabularProfileStats,
    objective_type: ObjectiveType,
    use_case: ObjectiveUseCase,
    priors: Mapping[str, float],
) -> CandidatePriority:
    feasible, reasons = _feasibility(model, stats)
    eligibility = method_eligibility(model, use_case)
    score = _base_score(model, stats)
    skip_reason = None
    if not feasible:
        return CandidatePriority(
            model=model,
            feasible=False,
            eligible=eligibility.eligible,
            priority_score=float("-inf"),
            reasons=tuple(reasons),
            skip_reason=reasons[-1],
        )
    if not eligibility.eligible:
        return CandidatePriority(
            model=model,
            feasible=True,
            eligible=False,
            priority_score=float("-inf"),
            reasons=tuple(reasons),
            skip_reason="; ".join(eligibility.reasons),
        )
    if objective_type is ObjectiveType.DATA_GENERATION:
        score += 0.05
        reasons.append("Objective is unconditional synthetic data generation.")
    if use_case is ObjectiveUseCase.EXTERNAL_SHARING:
        score += 0.03
        reasons.append("External sharing excludes exact bootstrap but permits model-generated candidates without privacy claims.")
    if model in priors:
        adjustment = max(-0.05, min(0.05, 0.1 * (priors[model] - 0.75)))
        score += adjustment
        reasons.append(f"Prior validation evidence adjusted priority by {adjustment:.3f}.")
    return CandidatePriority(
        model=model,
        feasible=feasible,
        eligible=eligibility.eligible,
        priority_score=float(score),
        reasons=tuple(reasons),
        skip_reason=skip_reason,
    )


def _feasibility(model: str, stats: TabularProfileStats) -> tuple[bool, list[str]]:
    reasons = [f"n={stats.n_samples}, p={stats.n_features}, continuous_only={stats.continuous_only}."]
    if model in {"flow", "gmm", "copula"} and not stats.continuous_only:
        reasons.append(f"{model} currently supports continuous-only tabular features.")
        return False, reasons
    if model == "flow" and stats.n_samples < 40:
        reasons.append("RealNVP is skipped for very small datasets.")
        return False, reasons
    if model == "gmm" and stats.n_samples < max(20, 2 * stats.n_features):
        reasons.append("GMM is skipped when sample size is too small relative to dimensionality.")
        return False, reasons
    if model == "copula" and stats.n_samples < max(20, stats.n_features + 2):
        reasons.append("Copula is skipped when empirical marginals would be too sparse.")
        return False, reasons
    reasons.append("Model is feasible under deterministic Phase 6B rules.")
    return True, reasons


def _base_score(model: str, stats: TabularProfileStats) -> float:
    skew = min(stats.mean_abs_skewness, 3.0) / 3.0
    corr = min(stats.mean_abs_correlation, 1.0)
    ratio = min(stats.sample_feature_ratio / 100.0, 1.0)
    low_dim_bonus = 0.08 if stats.n_features <= 20 else 0.0
    score = 0.25 + 0.15 * ratio + low_dim_bonus
    if model == "vae":
        score += 0.10
        if stats.binary_features:
            score += 0.10
        if stats.n_samples >= 200:
            score += 0.05
        if stats.n_samples < 60:
            score += 0.25
    elif model == "flow":
        score += 0.15 + 0.08 * corr + 0.04 * skew
        if stats.n_features <= 30:
            score += 0.05
    elif model == "gmm":
        score += 0.12 + 0.06 * corr
        if stats.n_features <= 15:
            score += 0.06
        if stats.mean_abs_skewness < 1.0:
            score += 0.04
    elif model == "copula":
        score += 0.12 + 0.14 * corr + 0.08 * skew
        if stats.n_samples < 500:
            score += 0.04
    return score
