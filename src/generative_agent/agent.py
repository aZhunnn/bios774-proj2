"""Objective-aware Phase 1 agent foundation."""

from __future__ import annotations

from generative_agent.data import DatasetHandle
from generative_agent.feasibility import FeasibilityChecker
from generative_agent.inspection import DatasetInspector
from generative_agent.objectives import ObjectiveInterpreter
from generative_agent.planning import MockPlanner
from generative_agent.state import AgentPhase, AgentState, ExperimentBudget


class AgentFoundation:
    """Coordinates inspection, interpretation, feasibility, and mock planning."""

    def __init__(
        self,
        *,
        inspector: DatasetInspector | None = None,
        interpreter: ObjectiveInterpreter | None = None,
        feasibility_checker: FeasibilityChecker | None = None,
        planner: MockPlanner | None = None,
    ) -> None:
        self.inspector = inspector or DatasetInspector()
        self.interpreter = interpreter or ObjectiveInterpreter()
        self.feasibility_checker = feasibility_checker or FeasibilityChecker()
        self.planner = planner or MockPlanner()

    def prepare(
        self,
        dataset: DatasetHandle,
        objective_text: str,
        *,
        budget: ExperimentBudget | None = None,
    ) -> AgentState:
        state = AgentState(budget=budget or ExperimentBudget())
        state.profile = self.inspector.inspect(dataset)
        state.phase = AgentPhase.INSPECTED
        state.objective = self.interpreter.interpret(objective_text)
        state.phase = AgentPhase.INTERPRETED
        state.feasibility = self.feasibility_checker.check(state.profile, state.objective, state.budget)
        state.phase = AgentPhase.FEASIBILITY_CHECKED
        state.plan = self.planner.plan(state.profile, state.objective, state.feasibility, state.budget)
        state.phase = AgentPhase.PLANNED
        return state
