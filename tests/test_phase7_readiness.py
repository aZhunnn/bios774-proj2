import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from generative_agent.cli import _infer_modality
from generative_agent.image_data import ImagePreprocessConfig
from generative_agent.image_workflow import run_image_generation_workflow
from generative_agent.objectives import ObjectiveType, ObjectiveUseCase
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles
from generative_agent.tabular_planning import build_tabular_training_plan, tabular_dataset_fingerprint
from generative_agent.tabular_preprocessing import TabularVaeSchema
from generative_agent.workflow import run_tabular_generation_workflow


def image_npz(path: Path):
    rng = np.random.default_rng(7)
    def images(n):
        return rng.integers(0, 255, size=(n, 16, 16, 3), dtype=np.uint8)
    np.savez(
        path,
        train_images=images(24),
        train_labels=np.arange(24) % 2,
        val_images=images(8),
        val_labels=np.arange(8) % 2,
        test_images=images(8),
        test_labels=np.arange(8) % 2,
    )


class Phase7ReadinessTests(unittest.TestCase):
    def test_prior_evidence_disabled_and_rejects_incompatible_provenance(self):
        frame = pd.DataFrame({"x1": np.linspace(0, 1, 50), "x2": np.linspace(1, 2, 50)})
        schema = TabularVaeSchema(continuous=("x1", "x2"), binary=tuple(), excluded=tuple())
        raw = build_tabular_training_plan(
            train_frame=frame,
            schema=schema,
            objective_type=ObjectiveType.DATA_GENERATION,
            use_case=ObjectiveUseCase.INTERNAL_TESTING,
            budget=ExperimentBudget(max_models=1),
            prior_validation_scores={"copula": 1.0},
        )
        self.assertFalse(raw.prior_evidence_status["used"])

        incompatible = build_tabular_training_plan(
            train_frame=frame,
            schema=schema,
            objective_type=ObjectiveType.DATA_GENERATION,
            use_case=ObjectiveUseCase.INTERNAL_TESTING,
            budget=ExperimentBudget(max_models=1),
            prior_evidence={
                "dataset_fingerprint": "wrong",
                "objective_type": "data_generation",
                "evaluation_protocol": "tabular_selection_score_v1",
                "source_split": "test",
                "scores": {"copula": 1.0},
            },
            enable_prior_evidence=True,
        )
        self.assertFalse(incompatible.prior_evidence_status["used"])
        self.assertIn("mismatches", incompatible.prior_evidence_status)

        compatible = build_tabular_training_plan(
            train_frame=frame,
            schema=schema,
            objective_type=ObjectiveType.DATA_GENERATION,
            use_case=ObjectiveUseCase.INTERNAL_TESTING,
            budget=ExperimentBudget(max_models=1),
            prior_evidence={
                "dataset_fingerprint": tabular_dataset_fingerprint(frame, schema),
                "objective_type": "data_generation",
                "evaluation_protocol": "tabular_selection_score_v1",
                "source_split": "validation",
                "scores": {"copula": 1.0},
            },
            enable_prior_evidence=True,
        )
        self.assertTrue(compatible.prior_evidence_status["used"])

    def test_unified_cli_modality_inference(self):
        self.assertEqual(_infer_modality(Path("data.csv"), "auto"), "tabular")
        self.assertEqual(_infer_modality(Path("images.npz"), "auto"), "image")
        self.assertEqual(_infer_modality(Path("whatever.bin"), "image"), "image")

    def test_tabular_end_to_end_reproducible_planning_and_budget(self):
        rng = np.random.default_rng(8)
        frame = pd.DataFrame({"x1": rng.normal(size=72), "x2": rng.exponential(size=72), "quality": np.arange(72) % 3})
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "toy.csv"
            frame.to_csv(csv, index=False)
            outputs = []
            for name in ("a", "b"):
                out = Path(directory) / name
                run_tabular_generation_workflow(
                    csv_path=csv,
                    objective_text="Generate synthetic tabular data for internal testing.",
                    output_dir=out,
                    roles=ColumnRoles(target="quality"),
                    budget=ExperimentBudget(max_models=2, max_epochs=3, seed=101),
                    n_synthetic=20,
                )
                outputs.append(json.loads((out / "candidate_search.json").read_text()))

            self.assertEqual(outputs[0]["planning"], outputs[1]["planning"])
            self.assertEqual(outputs[0]["budget"]["trained_candidates"], 2)

    def test_image_end_to_end_records_planning_and_rejects_unsupported_objective(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.npz"
            output = Path(directory) / "image"
            image_npz(path)
            run_image_generation_workflow(
                npz_path=path,
                output_dir=output,
                objective_text="Generate synthetic images for internal testing.",
                budget=ExperimentBudget(max_models=1, max_epochs=1, seed=33),
                preprocess=ImagePreprocessConfig(max_train_samples=16, max_validation_samples=4, max_test_samples=4, seed=33),
                n_generated=4,
                repeated_generation_seeds=1,
            )
            planning = json.loads((output / "planning.json").read_text())
            self.assertEqual(planning["allocated_models"], ["image_vae"])
            self.assertTrue((output / "final_test_evaluation.json").exists())

            with self.assertRaisesRegex(ValueError, "Unsupported image objective"):
                run_image_generation_workflow(
                    npz_path=path,
                    output_dir=Path(directory) / "unsupported",
                    objective_text="Impute missing images.",
                    budget=ExperimentBudget(max_models=1, max_epochs=1, seed=33),
                    preprocess=ImagePreprocessConfig(max_train_samples=16, max_validation_samples=4, max_test_samples=4, seed=33),
                )


if __name__ == "__main__":
    unittest.main()
