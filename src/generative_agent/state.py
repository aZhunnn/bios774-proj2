"""Agent state and experiment budget objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from generative_agent.data import DatasetProfile
from generative_agent.feasibility import FeasibilityReport
from generative_agent.objectives import ObjectiveSpec
from generative_agent.planning import ExperimentPlan


class AgentPhase(str, Enum):
    INSPECTED = "inspected"
    INTERPRETED = "interpreted"
    FEASIBILITY_CHECKED = "feasibility_checked"
    PLANNED = "planned"


@dataclass(frozen=True)
class ExperimentBudget:
    max_models: int = 2
    max_epochs: int = 25
    max_wall_minutes: int = 30
    seed: int = 2026

    def __post_init__(self) -> None:
        if min(self.max_models, self.max_epochs, self.max_wall_minutes) <= 0:
            raise ValueError("Experiment budgets must be positive.")


@dataclass
class AgentState:
    run_id: str = field(default_factory=lambda: str(uuid4()))
    phase: AgentPhase | None = None
    profile: DatasetProfile | None = None
    objective: ObjectiveSpec | None = None
    feasibility: FeasibilityReport | None = None
    plan: ExperimentPlan | None = None
    budget: ExperimentBudget = field(default_factory=ExperimentBudget)

