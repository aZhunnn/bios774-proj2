import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from generative_agent.objectives import ObjectiveType, ObjectiveUseCase
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles
from generative_agent.tabular_planning import build_tabular_training_plan
from generative_agent.tabular_preprocessing import TabularVaeSchema
from generative_agent.workflow import run_tabular_generation_workflow


def schema(columns=("x1", "x2")):
    return TabularVaeSchema(continuous=tuple(columns), binary=tuple(), excluded=tuple())


class Phase6BPlanningTests(unittest.TestCase):
    def test_planning_is_deterministic_and_profile_sensitive(self):
        rng = np.random.default_rng(10)
        gaussian = pd.DataFrame({"x1": rng.normal(size=120), "x2": rng.normal(size=120)})
        base = rng.exponential(size=120)
        skewed_corr = pd.DataFrame({"x1": base, "x2": base * 0.8 + rng.normal(scale=0.1, size=120)})

        first = build_tabular_training_plan(
            train_frame=gaussian,
            schema=schema(),
            objective_type=ObjectiveType.DATA_GENERATION,
            use_case=ObjectiveUseCase.INTERNAL_TESTING,
            budget=ExperimentBudget(max_models=1),
        )
        second = build_tabular_training_plan(
            train_frame=gaussian,
            schema=schema(),
            objective_type=ObjectiveType.DATA_GENERATION,
            use_case=ObjectiveUseCase.INTERNAL_TESTING,
            budget=ExperimentBudget(max_models=1),
        )
        skewed = build_tabular_training_plan(
            train_frame=skewed_corr,
            schema=schema(),
            objective_type=ObjectiveType.DATA_GENERATION,
            use_case=ObjectiveUseCase.INTERNAL_TESTING,
            budget=ExperimentBudget(max_models=1),
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first.allocated_models, skewed.allocated_models)

    def test_planning_respects_objective_eligibility_and_budget(self):
        frame = pd.DataFrame({"x1": np.arange(50, dtype=float), "x2": np.arange(50, dtype=float) ** 0.5})
        plan = build_tabular_training_plan(
            train_frame=frame,
            schema=schema(),
            objective_type=ObjectiveType.DATA_GENERATION,
            use_case=ObjectiveUseCase.FORMAL_PRIVACY,
            budget=ExperimentBudget(max_models=2),
        )

        self.assertEqual(plan.allocated_models, tuple())
        self.assertTrue(all(not candidate.eligible for candidate in plan.candidates))

    def test_workflow_budget_one_two_four_records_planning_and_costs(self):
        rng = np.random.default_rng(11)
        frame = pd.DataFrame(
            {
                "x1": rng.normal(size=96),
                "x2": rng.exponential(size=96),
                "x3": rng.normal(size=96),
                "quality": np.arange(96) % 4,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "toy.csv"
            frame.to_csv(csv_path, index=False)
            for budget_size in (1, 2, 4):
                output = Path(directory) / f"budget_{budget_size}"
                run_tabular_generation_workflow(
                    csv_path=csv_path,
                    objective_text="Generate synthetic tabular data for internal testing.",
                    output_dir=output,
                    roles=ColumnRoles(target="quality"),
                    budget=ExperimentBudget(max_models=budget_size, max_epochs=3, seed=77),
                    n_synthetic=24,
                )
                search = json.loads((output / "candidate_search.json").read_text())
                self.assertEqual(search["budget"]["trained_candidates"], budget_size)
                self.assertEqual(sum(search["budget"]["allocation"].values()), budget_size)
                self.assertIn("planning", search)
                self.assertTrue(any("wall_seconds" in item for family in ("vae", "flow", "gmm", "copula") for item in search.get(family, {}).get("candidates", [])))

    def test_external_sharing_filters_bootstrap_after_validation_ranking(self):
        rng = np.random.default_rng(12)
        frame = pd.DataFrame({"x1": rng.normal(size=80), "x2": rng.normal(size=80), "quality": np.arange(80) % 2})
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "toy.csv"
            output = Path(directory) / "external"
            frame.to_csv(csv_path, index=False)
            run_tabular_generation_workflow(
                csv_path=csv_path,
                objective_text="Generate synthetic tabular data for external sharing without a formal privacy guarantee.",
                output_dir=output,
                roles=ColumnRoles(target="quality"),
                budget=ExperimentBudget(max_models=2, max_epochs=3, seed=88),
                n_synthetic=20,
            )
            decisions = json.loads((output / "agent_decisions.json").read_text())
            generation = json.loads((output / "generation_metadata.json").read_text())

            self.assertNotEqual(generation["method"], "training_row_bootstrap_selected_by_validation")
            self.assertIn("baseline", decisions["eligibility"]["rejected_methods"])
            self.assertIn("no formal privacy guarantee", " ".join(decisions["objective_recommendation"]["reasons"]))

    def test_formal_privacy_request_is_rejected_with_honest_language(self):
        frame = pd.DataFrame({"x1": np.linspace(0, 1, 60), "x2": np.linspace(1, 2, 60), "quality": np.arange(60) % 3})
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "toy.csv"
            output = Path(directory) / "privacy"
            frame.to_csv(csv_path, index=False)
            run_tabular_generation_workflow(
                csv_path=csv_path,
                objective_text="Generate synthetic tabular data with a formal privacy guarantee.",
                output_dir=output,
                roles=ColumnRoles(target="quality"),
                budget=ExperimentBudget(max_models=2, max_epochs=3, seed=99),
                n_synthetic=20,
            )
            decisions = json.loads((output / "agent_decisions.json").read_text())
            evaluation = json.loads((output / "evaluation_results.json").read_text())

            self.assertEqual(evaluation["selected"]["method"], "rejected")
            self.assertEqual(decisions["objective_recommendation"]["recommendation"], "reject")
            self.assertIn("do not establish privacy", decisions["objective_recommendation"]["language_guardrail"])


if __name__ == "__main__":
    unittest.main()
