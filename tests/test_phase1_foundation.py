import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from generative_agent.agent import AgentFoundation
from generative_agent.artifacts import LocalArtifactStore
from generative_agent.data import DatasetHandle, DatasetKind, FeatureType
from generative_agent.feasibility import FeasibilityChecker
from generative_agent.inspection import DatasetInspector
from generative_agent.objectives import ObjectiveInterpreter, ObjectiveType
from generative_agent.preprocessing import TabularPreprocessor
from generative_agent.registry import CapabilityRegistry
from generative_agent.splitting import DatasetSplitter, SplitConfig
from generative_agent.state import AgentPhase, ExperimentBudget


def toy_rows():
    return (
        {"x": "1.0", "group": "A", "label": "case"},
        {"x": "2.0", "group": "A", "label": "case"},
        {"x": "", "group": "B", "label": "control"},
        {"x": "4.0", "group": "B", "label": "control"},
        {"x": "5.0", "group": "B", "label": "control"},
        {"x": "6.0", "group": "B", "label": "control"},
        {"x": "7.0", "group": "C", "label": "control"},
        {"x": "8.0", "group": "C", "label": "control"},
        {"x": "9.0", "group": "C", "label": "control"},
        {"x": "10.0", "group": "C", "label": "control"},
        {"x": "11.0", "group": "C", "label": "control"},
        {"x": "12.0", "group": "C", "label": "control"},
    )


def image_rows(count=12):
    return tuple({"path": f"image_{index}.png", "label": "rare" if index < 2 else "common"} for index in range(count))


def long_text_rows(count=12):
    return tuple(
        {
            "x": str(index),
            "free_text": f"This is a long unstructured clinical note number {index} with narrative content.",
            "label": "case",
        }
        for index in range(count)
    )


def fully_missing_rows(count=12):
    return tuple({"x": "", "group": "A", "label": "case"} for _ in range(count))


class Phase1FoundationTests(unittest.TestCase):
    def test_dataset_inspection_profiles_records(self):
        dataset = DatasetHandle.from_rows("toy", toy_rows(), target_column="label")
        profile = DatasetInspector().inspect(dataset)

        self.assertEqual(profile.dataset_name, "toy")
        self.assertEqual(profile.n_samples, 12)
        self.assertEqual(profile.n_features, 3)
        self.assertEqual(profile.missing_values["x"], 1)
        self.assertEqual(profile.class_balance["case"], 2)
        self.assertEqual(profile.class_balance["control"], 10)
        x_schema = next(feature for feature in profile.feature_schema if feature.name == "x")
        self.assertEqual(x_schema.feature_type, FeatureType.NUMERIC)
        self.assertIn("Class imbalance detected.", profile.notes)

    def test_objective_interpreter_maps_common_requests(self):
        spec = ObjectiveInterpreter().interpret("Generate synthetic samples to balance the minority class.")

        self.assertEqual(spec.objective_type, ObjectiveType.CLASS_BALANCING)
        self.assertGreater(spec.confidence, 0.5)
        self.assertIn("synthetic_or_completed_data", spec.requested_outputs)

    def test_splitter_is_deterministic_and_complete(self):
        records = toy_rows()
        config = SplitConfig(seed=123, stratify_column="label")
        first = DatasetSplitter().split(records, config)
        second = DatasetSplitter().split(records, config)

        self.assertEqual(first, second)
        self.assertEqual(sorted(first.all_indices), list(range(len(records))))
        self.assertTrue(set(first.train).isdisjoint(first.validation))
        self.assertTrue(set(first.train).isdisjoint(first.test))
        self.assertTrue(set(first.validation).isdisjoint(first.test))

    def test_preprocessor_fits_only_training_rows(self):
        records = toy_rows()
        train_indices = (0, 1)
        preprocessor = TabularPreprocessor().fit(records, train_indices)
        transformed = preprocessor.transform(records, (2,))

        self.assertEqual(preprocessor.fitted_on, train_indices)
        self.assertEqual(transformed[0]["x"], 1.5)
        self.assertEqual(transformed[0]["group"], "B")

    def test_registry_exposes_planned_pytorch_capabilities(self):
        candidates = CapabilityRegistry().candidates(DatasetKind.TABULAR, ObjectiveType.DATA_GENERATION)

        self.assertTrue(candidates)
        self.assertTrue(all(candidate.training_framework == "pytorch" for candidate in candidates))
        self.assertTrue(any(candidate.name == "tabular_vae" and candidate.implemented for candidate in candidates))
        self.assertTrue(any(candidate.name == "tabular_flow" and not candidate.implemented for candidate in candidates))

    def test_feasibility_is_deterministic(self):
        dataset = DatasetHandle.from_rows("toy", toy_rows(), target_column="label")
        profile = DatasetInspector().inspect(dataset)
        objective = ObjectiveInterpreter().interpret("Use synthetic data for class balancing.")

        first = FeasibilityChecker().check(profile, objective)
        second = FeasibilityChecker().check(profile, objective)

        self.assertEqual(first, second)
        self.assertTrue(first.feasible)
        self.assertTrue(first.candidates)

    def test_agent_foundation_produces_mock_plan_without_training(self):
        dataset = DatasetHandle.from_rows("toy", toy_rows(), target_column="label")
        state = AgentFoundation().prepare(
            dataset,
            "Generate synthetic data for class balancing.",
            budget=ExperimentBudget(max_models=1, max_epochs=10),
        )

        self.assertEqual(state.phase, AgentPhase.PLANNED)
        self.assertIsNotNone(state.plan)
        self.assertFalse(state.plan.training_enabled)
        self.assertEqual(len(state.plan.experiments), 1)
        self.assertEqual(state.plan.experiments[0].planned_hyperparameters["epochs"], 10)
        self.assertIn("max_models=1", state.feasibility.constraints)

    def test_same_dataset_different_objectives_change_decisions(self):
        dataset = DatasetHandle.from_rows("toy", toy_rows(), target_column="label")
        agent = AgentFoundation()

        balancing = agent.prepare(dataset, "Generate synthetic data to balance the minority class.")
        uncertainty = agent.prepare(dataset, "Quantify uncertainty with posterior-like samples.")

        self.assertEqual(balancing.objective.objective_type, ObjectiveType.CLASS_BALANCING)
        self.assertEqual(uncertainty.objective.objective_type, ObjectiveType.UNCERTAINTY_QUANTIFICATION)
        self.assertNotEqual(
            balancing.plan.downstream_recommendation_focus,
            uncertainty.plan.downstream_recommendation_focus,
        )
        self.assertEqual(
            tuple(experiment.model_name for experiment in balancing.plan.experiments),
            ("tabular_vae",),
        )
        self.assertEqual(
            tuple(experiment.model_name for experiment in uncertainty.plan.experiments),
            ("tabular_flow",),
        )

    def test_dataset_kind_changes_model_selection(self):
        tabular = DatasetHandle.from_rows("tabular", toy_rows(), target_column="label")
        image = DatasetHandle.from_rows(
            "images",
            image_rows(),
            kind=DatasetKind.IMAGE,
            target_column="label",
        )
        agent = AgentFoundation()

        tabular_state = agent.prepare(tabular, "Generate synthetic samples.")
        image_state = agent.prepare(image, "Generate synthetic samples.")

        self.assertEqual(tabular_state.profile.kind, DatasetKind.TABULAR)
        self.assertEqual(image_state.profile.kind, DatasetKind.IMAGE)
        self.assertEqual(
            tuple(experiment.model_name for experiment in tabular_state.plan.experiments),
            ("tabular_flow", "tabular_vae"),
        )
        self.assertEqual(
            tuple(experiment.model_name for experiment in image_state.plan.experiments),
            ("image_diffusion", "image_gan"),
        )

    def test_feasibility_rejects_unsupported_features_missingness_and_capabilities(self):
        checker = FeasibilityChecker()

        text_profile = DatasetInspector().inspect(
            DatasetHandle.from_rows("texty", long_text_rows(), target_column="label")
        )
        text_report = checker.check(
            text_profile,
            ObjectiveInterpreter().interpret("Generate synthetic samples."),
        )
        self.assertFalse(text_report.feasible)
        self.assertIn("Unsupported feature types detected: free_text.", text_report.reasons)

        missing_profile = DatasetInspector().inspect(
            DatasetHandle.from_rows("missing", fully_missing_rows(), target_column="label")
        )
        missing_report = checker.check(
            missing_profile,
            ObjectiveInterpreter().interpret("Generate synthetic samples."),
        )
        self.assertFalse(missing_report.feasible)
        self.assertIn("Features with no observed values cannot be modeled: x.", missing_report.reasons)

        unsupported_objective = checker.check(
            DatasetInspector().inspect(DatasetHandle.from_rows("toy", toy_rows(), target_column="label")),
            ObjectiveInterpreter().interpret("Find anomalies and outliers."),
        )
        self.assertFalse(unsupported_objective.feasible)
        self.assertIn("No planned model capability matches this dataset/objective pair.", unsupported_objective.reasons)

    def test_resource_constraints_are_recorded_and_can_reject(self):
        dataset = DatasetHandle.from_rows("toy", toy_rows(), target_column="label")
        profile = DatasetInspector().inspect(dataset)
        objective = ObjectiveInterpreter().interpret("Generate synthetic samples.")

        with self.assertRaises(ValueError):
            ExperimentBudget(max_models=0)

        budget = ExperimentBudget(max_wall_minutes=4)
        report = FeasibilityChecker().check(profile, objective, budget)

        self.assertFalse(report.feasible)
        self.assertIn("max_wall_minutes=4", report.constraints)
        self.assertIn("At least 5 wall-clock minutes are required for planned experiments.", report.reasons)

    def test_split_manifest_is_reproducible_and_immutable(self):
        records = toy_rows()
        config = SplitConfig(seed=123, stratify_column="label")
        splitter = DatasetSplitter()

        first = splitter.manifest(records, config)
        second = splitter.manifest(records, config)

        self.assertEqual(first, second)
        self.assertEqual(sorted(first.indices.all_indices), list(range(len(records))))
        with self.assertRaises(FrozenInstanceError):
            first.dataset_fingerprint = "changed"
        with self.assertRaises(FrozenInstanceError):
            first.config.seed = 999

    def test_preprocessor_rejects_validation_or_test_fit_indices(self):
        records = toy_rows()
        manifest = DatasetSplitter().manifest(records, SplitConfig(seed=123))
        holdout_index = manifest.indices.validation[0]

        with self.assertRaises(ValueError):
            TabularPreprocessor().fit_from_split(records, manifest, manifest.indices.train + (holdout_index,))

        fitted = TabularPreprocessor().fit_from_split(records, manifest)
        self.assertEqual(fitted.fitted_on, manifest.indices.train)

    def test_decisions_record_reasons_outcomes_and_constraints(self):
        state = AgentFoundation().prepare(
            DatasetHandle.from_rows("tiny", toy_rows()[:5], target_column="label"),
            "Generate synthetic samples.",
            budget=ExperimentBudget(max_models=1, max_epochs=3, max_wall_minutes=4, seed=7),
        )

        self.assertFalse(state.feasibility.feasible)
        self.assertIn("At least 10 samples are required before generative modeling is considered.", state.feasibility.reasons)
        self.assertIn("max_models=1", state.feasibility.constraints)
        self.assertIn("seed=7", state.feasibility.constraints)
        self.assertEqual(state.plan.experiments, tuple())

    def test_artifact_store_persists_json_interface(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalArtifactStore(Path(directory))
            record = store.save_json("phase 1 plan", {"ok": True}, kind="plan")

            self.assertEqual(record.kind, "plan")
            self.assertTrue(record.path.exists())
            self.assertIn('"ok": true', record.path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
