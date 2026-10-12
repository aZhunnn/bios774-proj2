import json
import tempfile
import unittest
from pathlib import Path

from generative_agent.eligibility import method_eligibility, select_eligible_method
from generative_agent.objectives import ObjectiveInterpreter, ObjectiveUseCase
from generative_agent.reevaluate import reevaluate_experiment_artifacts


class Phase4BEligibilityTests(unittest.TestCase):
    def test_eligibility_is_hard_gate_before_ranking(self):
        scores = {
            "baseline": {"selection_score": 0.99},
            "flow": {"selection_score": 0.80},
            "vae": {"selection_score": 0.70},
        }

        selection = select_eligible_method(scores, use_case=ObjectiveUseCase.EXTERNAL_SHARING)

        self.assertEqual(selection.selected_method, "flow")
        self.assertIn("baseline", selection.rejected_methods)
        self.assertIn("flow", selection.eligible_methods)

    def test_objective_dependent_decisions_on_same_scores(self):
        scores = {
            "baseline": {"selection_score": 0.95},
            "flow": {"selection_score": 0.90},
        }

        internal = select_eligible_method(scores, use_case=ObjectiveUseCase.INTERNAL_TESTING)
        external = select_eligible_method(scores, use_case=ObjectiveUseCase.EXTERNAL_SHARING)

        self.assertEqual(internal.selected_method, "baseline")
        self.assertEqual(external.selected_method, "flow")

    def test_bootstrap_rejected_for_external_release(self):
        eligibility = method_eligibility("training_row_bootstrap", ObjectiveUseCase.EXTERNAL_SHARING)

        self.assertFalse(eligibility.eligible)
        self.assertIn("Exact-record bootstrap", eligibility.reasons[0])

    def test_formal_privacy_request_rejects_all_current_methods(self):
        scores = {
            "baseline": {"selection_score": 0.99},
            "flow": {"selection_score": 0.80},
            "vae": {"selection_score": 0.70},
        }

        selection = select_eligible_method(scores, use_case=ObjectiveUseCase.FORMAL_PRIVACY)

        self.assertIsNone(selection.selected_method)
        self.assertEqual(set(selection.rejected_methods), {"baseline", "flow", "vae"})

    def test_interpreter_records_use_case_constraints(self):
        interpreter = ObjectiveInterpreter()

        internal = interpreter.interpret("Generate synthetic data for internal pipeline stress testing.")
        external = interpreter.interpret("Generate synthetic data for external sharing.")
        private = interpreter.interpret("Generate synthetic data with a differential privacy guarantee.")
        external_without_guarantee = interpreter.interpret(
            "Generate synthetic data for external sharing without a formal privacy guarantee."
        )

        self.assertEqual(internal.constraints["use_case"], "internal_testing")
        self.assertEqual(external.constraints["use_case"], "external_sharing")
        self.assertEqual(private.constraints["use_case"], "formal_privacy")
        self.assertEqual(external_without_guarantee.constraints["use_case"], "external_sharing")

    def test_artifact_reevaluation_uses_validation_scores_not_test_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "validation_comparison.json").write_text(
                json.dumps(
                    {
                        "decision_rule": "validation only",
                        "scores": {
                            "baseline": {"selection_score": 0.99},
                            "flow": {"selection_score": 0.80},
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = reevaluate_experiment_artifacts(
                root,
                objective_text="Generate synthetic data for external sharing.",
            )

            self.assertFalse(result["heldout_test_used_for_selection"])
            self.assertEqual(result["selection"]["selected_method"], "flow")


if __name__ == "__main__":
    unittest.main()
