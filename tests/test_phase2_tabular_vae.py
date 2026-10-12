import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generative_agent.evaluation import bootstrap_baseline, evaluate_synthetic
from generative_agent.generation import generate_synthetic
from generative_agent.splitting import DatasetSplitter, SplitConfig
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles, load_csv_dataset
from generative_agent.tabular_preprocessing import TabularVaePreprocessor
from generative_agent.vae import TabularVAE, VaeConfig, VaeTrainer
from generative_agent.workflow import run_tabular_vae_workflow


def phase2_frame(n=48):
    rng = np.random.default_rng(2026)
    return pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(n)],
            "x1": np.linspace(-1.0, 1.0, n),
            "x2": rng.normal(0.0, 0.5, n),
            "flag": [0, 1] * (n // 2),
            "target": ["case" if index % 3 == 0 else "control" for index in range(n)],
        }
    )


class Phase2TabularVaeTests(unittest.TestCase):
    def test_vae_forward_and_loss_are_finite(self):
        config = VaeConfig(input_dim=3, continuous_dim=2, binary_dim=1, latent_dim=2, seed=9)
        model = TabularVAE(config)
        x = torch.tensor([[0.1, -0.2, 1.0], [0.0, 0.3, 0.0]], dtype=torch.float32)

        recon, mu, logvar = model(x)
        loss = model.loss(x, recon, mu, logvar)

        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(mu.shape, (2, 2))
        self.assertTrue(torch.isfinite(loss.total))
        self.assertGreater(float(loss.kl.detach()), 0.0)

    def test_preprocessing_roles_schema_and_inverse_transform(self):
        frame = phase2_frame()
        roles = ColumnRoles(identifiers=("id",), target="target")
        manifest = DatasetSplitter().manifest(frame.to_dict(orient="records"), SplitConfig(seed=4, stratify_column="target"))

        preprocessor = TabularVaePreprocessor(roles=roles).fit(frame, manifest)
        encoded = preprocessor.transform(frame, manifest.indices.train)
        restored = preprocessor.inverse_transform(encoded[:3])

        metadata = preprocessor._metadata()
        self.assertEqual(metadata.schema.continuous, ("x1", "x2"))
        self.assertEqual(metadata.schema.binary, ("flag",))
        self.assertEqual(encoded.shape[1], 3)
        self.assertEqual(list(restored.columns), ["x1", "x2", "flag"])
        self.assertNotIn("id", metadata.schema.model_columns)
        self.assertNotIn("target", metadata.schema.model_columns)

    def test_preprocessing_rejects_missing_and_unsupported_columns(self):
        frame = phase2_frame()
        frame.loc[0, "x1"] = np.nan
        manifest = DatasetSplitter().manifest(frame.to_dict(orient="records"), SplitConfig(seed=4))

        with self.assertRaisesRegex(ValueError, "Missing values are not supported"):
            TabularVaePreprocessor(roles=ColumnRoles(identifiers=("id",), target="target")).fit(frame, manifest)

        unsupported = phase2_frame()
        unsupported["note"] = [f"long narrative note with too much free text {i}" for i in range(len(unsupported))]
        manifest = DatasetSplitter().manifest(unsupported.to_dict(orient="records"), SplitConfig(seed=4))
        with self.assertRaisesRegex(ValueError, "Unsupported column 'note'"):
            TabularVaePreprocessor(roles=ColumnRoles(identifiers=("id",), target="target")).fit(unsupported, manifest)

    def test_training_checkpoint_generation_and_reproducibility(self):
        frame = phase2_frame()
        roles = ColumnRoles(identifiers=("id",), target="target")
        manifest = DatasetSplitter().manifest(frame.to_dict(orient="records"), SplitConfig(seed=5, stratify_column="target"))
        preprocessor = TabularVaePreprocessor(roles=roles).fit(frame, manifest)
        train = preprocessor.transform(frame, manifest.indices.train)
        validation = preprocessor.transform(frame, manifest.indices.validation)

        with tempfile.TemporaryDirectory() as directory:
            config = VaeConfig(
                input_dim=train.shape[1],
                continuous_dim=2,
                binary_dim=1,
                latent_dim=2,
                hidden_dims=(8,),
                epochs=3,
                batch_size=8,
                seed=11,
                early_stopping_patience=2,
            )
            model_a = TabularVAE(config)
            result_a = VaeTrainer().train(model_a, train, validation, Path(directory) / "a.pt")
            samples_a = generate_synthetic(model_a, preprocessor, n_samples=5, seed=99).samples

            model_b = TabularVAE(config)
            result_b = VaeTrainer().train(model_b, train, validation, Path(directory) / "b.pt")
            samples_b = generate_synthetic(model_b, preprocessor, n_samples=5, seed=99).samples

            self.assertFalse(result_a.failed)
            self.assertTrue(result_a.checkpoint_path.exists())
            self.assertEqual(result_a.history, result_b.history)
            pd.testing.assert_frame_equal(samples_a, samples_b)
            self.assertEqual(samples_a["synthetic_record"].tolist(), [True] * 5)

    def test_evaluation_metrics_and_bootstrap_baseline_use_same_schema(self):
        frame = phase2_frame()
        roles = ColumnRoles(identifiers=("id",), target="target")
        manifest = DatasetSplitter().manifest(frame.to_dict(orient="records"), SplitConfig(seed=5, stratify_column="target"))
        preprocessor = TabularVaePreprocessor(roles=roles).fit(frame, manifest)
        schema = preprocessor._metadata().schema
        train_real = frame.iloc[list(manifest.indices.train)].reset_index(drop=True)
        test_real = frame.iloc[list(manifest.indices.test)].reset_index(drop=True)
        baseline = bootstrap_baseline(train_real[list(schema.model_columns)], 10, seed=3)
        baseline.insert(0, "synthetic_record", True)

        result = evaluate_synthetic(
            train_real=train_real,
            test_real=test_real,
            synthetic=baseline,
            schema=schema,
            method="training_row_bootstrap",
        )

        self.assertEqual(result.method, "training_row_bootstrap")
        self.assertIn("x1", result.feature_differences)
        self.assertIn("flag", result.feature_differences)
        self.assertGreaterEqual(result.duplicate_fraction, 0.0)
        self.assertGreaterEqual(result.unique_fraction, 0.0)
        self.assertIn("data_generation_fidelity", result.objective_metrics)
        self.assertIn("proxies", result.notes[0])

    def test_csv_loading_and_end_to_end_workflow(self):
        frame = phase2_frame()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "toy.csv"
            output_dir = Path(directory) / "artifacts"
            frame.to_csv(csv_path, index=False)

            loaded, handle = load_csv_dataset(csv_path, roles=ColumnRoles(identifiers=("id",), target="target"))
            result = run_tabular_vae_workflow(
                csv_path=csv_path,
                objective_text="Generate synthetic tabular data.",
                output_dir=output_dir,
                roles=ColumnRoles(identifiers=("id",), target="target"),
                n_synthetic=8,
            )

            self.assertEqual(len(loaded), len(frame))
            self.assertEqual(handle.target_column, "target")
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.synthetic_path.exists())
            self.assertTrue(result.checkpoint_path.exists())
            search = json.loads((output_dir / "candidate_search.json").read_text(encoding="utf-8"))
            self.assertEqual(len(search["candidates"]), 2)
            self.assertIn("validation loss", search["selection_reason"])
            synthetic = pd.read_csv(result.synthetic_path)
            self.assertEqual(len(synthetic), 8)
            self.assertEqual(set(synthetic.columns), {"synthetic_record", "x1", "x2", "flag"})

    def test_workflow_can_select_baseline_without_vae_when_budget_is_too_small(self):
        frame = phase2_frame(12)
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "toy.csv"
            output_dir = Path(directory) / "artifacts"
            frame.to_csv(csv_path, index=False)

            result = run_tabular_vae_workflow(
                csv_path=csv_path,
                objective_text="Generate synthetic tabular data.",
                output_dir=output_dir,
                roles=ColumnRoles(identifiers=("id",), target="target"),
                budget=ExperimentBudget(max_models=1, max_epochs=2, max_wall_minutes=30),
                n_synthetic=5,
            )

            self.assertIsNone(result.checkpoint_path)
            decisions = json.loads((output_dir / "agent_decisions.json").read_text(encoding="utf-8"))
            self.assertEqual(decisions["objective_decision"]["action"], "use_baseline")
            self.assertFalse((output_dir / "candidate_search.json").exists())

    def test_preprocessing_does_not_use_test_missingness_for_fit_decisions(self):
        frame = phase2_frame()
        manifest = DatasetSplitter().manifest(frame.to_dict(orient="records"), SplitConfig(seed=4))
        frame.loc[manifest.indices.test[0], "x1"] = np.nan

        preprocessor = TabularVaePreprocessor(roles=ColumnRoles(identifiers=("id",), target="target")).fit(frame, manifest)
        self.assertEqual(preprocessor._metadata().fitted_train_indices, manifest.indices.train)
        with self.assertRaisesRegex(ValueError, "Missing values are not supported during transform"):
            preprocessor.transform(frame, manifest.indices.test)

    def test_workflow_rejects_unsupported_objective(self):
        frame = phase2_frame()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "toy.csv"
            frame.to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "Unsupported Phase 2.5 objective"):
                run_tabular_vae_workflow(
                    csv_path=csv_path,
                    objective_text="Balance the minority class.",
                    output_dir=Path(directory) / "artifacts",
                    roles=ColumnRoles(identifiers=("id",), target="target"),
                )


if __name__ == "__main__":
    unittest.main()
