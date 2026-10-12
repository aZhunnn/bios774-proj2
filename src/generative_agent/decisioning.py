"""Deterministic objective-aware decisions for Phase 2.5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from generative_agent.data import DatasetProfile
from generative_agent.feasibility import FeasibilityReport
from generative_agent.objectives import ObjectiveSpec, ObjectiveType
from generative_agent.state import ExperimentBudget


class DecisionAction(str, Enum):
    USE_VAE = "use_vae"
    USE_BASELINE = "use_baseline"
    REJECT = "reject"


@dataclass(frozen=True)
class AgentDecision:
    action: DecisionAction
    reasons: tuple[str, ...]
    constraints: tuple[str, ...]
    supported_objective: bool
    uses_generative_model: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["action"] = self.action.value
        return payload


class DeterministicDecisionEngine:
    """Offline-testable decision rules; future LLM reasoning can wrap this."""

    def decide(
        self,
        *,
        profile: DatasetProfile,
        objective: ObjectiveSpec,
        feasibility: FeasibilityReport,
        budget: ExperimentBudget,
    ) -> AgentDecision:
        reasons: list[str] = []
        constraints = (
            f"objective={objective.objective_type.value}",
            f"n_samples={profile.n_samples}",
            f"n_features={profile.n_features}",
            f"max_models={budget.max_models}",
            f"max_epochs={budget.max_epochs}",
            f"max_wall_minutes={budget.max_wall_minutes}",
            f"seed={budget.seed}",
        )
        if objective.objective_type is not ObjectiveType.DATA_GENERATION:
            return AgentDecision(
                action=DecisionAction.REJECT,
                reasons=(f"Unsupported Phase 2.5 objective: {objective.objective_type.value}.",),
                constraints=constraints,
                supported_objective=False,
                uses_generative_model=False,
            )
        if not feasibility.feasible:
            return AgentDecision(
                action=DecisionAction.REJECT,
                reasons=feasibility.reasons,
                constraints=constraints,
                supported_objective=True,
                uses_generative_model=False,
            )
        if profile.n_samples < 20:
            reasons.append("Dataset is too small for a VAE comparison; use bootstrap baseline only.")
        if budget.max_epochs < 3 or budget.max_wall_minutes < 5:
            reasons.append("Budget is too small for a meaningful VAE fit; use bootstrap baseline only.")
        if reasons:
            return AgentDecision(
                action=DecisionAction.USE_BASELINE,
                reasons=tuple(reasons),
                constraints=constraints,
                supported_objective=True,
                uses_generative_model=False,
            )
        return AgentDecision(
            action=DecisionAction.USE_VAE,
            reasons=("Unconditional tabular data_generation is supported and budget permits VAE training.",),
            constraints=constraints,
            supported_objective=True,
            uses_generative_model=True,
        )
