"""Re-evaluate stored experiment artifacts under new objective eligibility rules."""

from __future__ import annotations

import json
from pathlib import Path

from generative_agent.eligibility import select_eligible_method
from generative_agent.objectives import ObjectiveInterpreter


def reevaluate_experiment_artifacts(
    artifact_dir: str | Path,
    *,
    objective_text: str,
) -> dict[str, object]:
    root = Path(artifact_dir)
    objective = ObjectiveInterpreter().interpret(objective_text)
    validation = json.loads((root / "validation_comparison.json").read_text(encoding="utf-8"))
    selection = select_eligible_method(
        validation["scores"],
        use_case=objective.constraints["use_case"],
        ranking_metric="selection_score",
    )
    return {
        "objective_text": objective_text,
        "objective": {
            "type": objective.objective_type.value,
            "use_case": objective.constraints["use_case"],
        },
        "selection": selection.to_dict(),
        "validation_decision_rule": validation["decision_rule"],
        "validation_scores_used_for_ranking": validation["scores"],
        "heldout_test_used_for_selection": False,
    }
