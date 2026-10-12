import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from generative_agent.flow import FlowConfig, FlowTrainer, RealNVPFlow, load_flow_checkpoint
from generative_agent.flow_search import propose_flow_configs, run_flow_search
from generative_agent.search import CandidateResult, SearchResult
from generative_agent.splitting import DatasetSplitter, SplitConfig
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles
from generative_agent.tabular_preprocessing import TabularVaePreprocessor
from generative_agent.workflow import run_tabular_vae_workflow
from generative_agent.vae import TabularVAE, VaeConfig


def continuous_frame(n=48):
    x = np.linspace(-2.0, 2.0, n)
    return pd.DataFrame(
        {
            "id": [f"row_{index}" for index in range(n)],
            "x1": x,
            "x2": 0.5 * x + np.sin(x),
            "target": ["case" if index % 4 == 0 else "control" for index in range(n)],
        }
    )


class Phase3AFlowTests(unittest.TestCase):
    def test_flow_invertibility_and_logdet_cancel(self):
        config = FlowConfig(input_dim=2, num_layers=4, hidden_dim=8, seed=3)
        model = RealNVPFlow(config)
        x = torch.tensor([[0.1, -0.2], [0.5, 0.8]], dtype=torch.float32)

        z, forward_logdet = model.forward(x)
        restored, inverse_logdet = model.inverse(z)

        self.assertTrue(torch.allclose(x, restored, atol=1e-5))
        self.assertTrue(torch.allclose(forward_logdet + inverse_logdet, torch.zeros(2), atol=1e-5))

    def test_flow_logdet_matches_autograd_jacobian(self):
        config = FlowConfig(input_dim=2, num_layers=2, hidden_dim=8, seed=5)
        model = RealNVPFlow(config)
        x = torch.tensor([[0.25, -0.4]], dtype=torch.float32, requires_grad=True)

        def transform(single_x):
            z, _ = model.forward(single_x[None, :])
            return z[0]

        _, logdet = model.forward(x)
        jacobian = torch.autograd.functional.jacobian(transform, x[0])
        autograd_logdet = torch.logdet(jacobian)

        self.assertTrue(torch.allclose(logdet[0], autograd_logdet, atol=1e-4))

    def test_flow_likelihood_sampling_checkpoint_and_reproducibility(self):
        frame = continuous_frame()
        manifest = DatasetSplitter().manifest(frame.to_dict(orient="records"), SplitConfig(seed=7, stratify_column="target"))
        preprocessor = TabularVaePreprocessor(roles=ColumnRoles(identifiers=("id",), target="target")).fit(frame, manifest)
        train = preprocessor.transform(frame, manifest.indices.train)
        validation = preprocessor.transform(frame, manifest.indices.validation)

        with tempfile.TemporaryDirectory() as directory:
            config = FlowConfig(input_dim=2, num_layers=2, hidden_dim=12, epochs=3, batch_size=8, seed=13)
            model_a = RealNVPFlow(config)
            result_a = FlowTrainer().train(model_a, train, validation, Path(directory) / "flow_a.pt")
            samples_a = model_a.sample(4, seed=99).detach().numpy()

            model_b = RealNVPFlow(config)
            result_b = FlowTrainer().train(model_b, train, validation, Path(directory) / "flow_b.pt")
            samples_b = model_b.sample(4, seed=99).detach().numpy()
            restored = load_flow_checkpoint(result_a.checkpoint_path)
            restored_samples = restored.sample(4, seed=99).detach().numpy()

            self.assertFalse(result_a.failed)
            self.assertTrue(np.isfinite(result_a.best_validation_nll))
            self.assertEqual(result_a.history, result_b.history)
            np.testing.assert_allclose(samples_a, samples_b, atol=1e-6)
            np.testing.assert_allclose(samples_a, restored_samples, atol=1e-6)

    def test_flow_likelihood_loss_and_gradients_are_finite(self):
        config = FlowConfig(input_dim=2, num_layers=2, hidden_dim=8, seed=17)
        model = RealNVPFlow(config)
        x = torch.tensor([[0.0, 0.1], [0.2, -0.3], [0.4, 0.5]], dtype=torch.float32)

        log_prob = model.log_prob(x)
        loss = -log_prob.mean()
        loss.backward()

        self.assertEqual(log_prob.shape, (3,))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))

    def test_flow_search_is_budget_limited_and_validation_only(self):
        frame = continuous_frame()
        manifest = DatasetSplitter().manifest(frame.to_dict(orient="records"), SplitConfig(seed=7))
        preprocessor = TabularVaePreprocessor(roles=ColumnRoles(identifiers=("id",), target="target")).fit(frame, manifest)
        train = preprocessor.transform(frame, manifest.indices.train)
        validation = preprocessor.transform(frame, manifest.indices.validation)

        with tempfile.TemporaryDirectory() as directory:
            configs = propose_flow_configs(input_dim=2, n_train=len(train), budget=ExperimentBudget(max_models=1, max_epochs=2))
            search, _ = run_flow_search(
                train_array=train,
                validation_array=validation,
                configs=configs,
                checkpoint_dir=Path(directory),
            )

            self.assertEqual(len(search.candidates), 1)
            self.assertIn("validation negative log likelihood", search.selection_reason)

    def test_end_to_end_continuous_comparison_records_flow_and_test_isolation(self):
        frame = continuous_frame()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "continuous.csv"
            output_dir = Path(directory) / "artifacts"
            frame.to_csv(csv_path, index=False)

            result = run_tabular_vae_workflow(
                csv_path=csv_path,
                objective_text="Generate synthetic tabular data.",
                output_dir=output_dir,
                roles=ColumnRoles(identifiers=("id",), target="target"),
                budget=ExperimentBudget(max_models=1, max_epochs=3),
                n_synthetic=8,
            )

            self.assertTrue(result.synthetic_path.exists())
            search = json.loads((output_dir / "candidate_search.json").read_text(encoding="utf-8"))
            decisions = json.loads((output_dir / "agent_decisions.json").read_text(encoding="utf-8"))
            comparison = json.loads((output_dir / "validation_comparison.json").read_text(encoding="utf-8"))

            self.assertIn("vae", search)
            self.assertIn("flow", search)
            self.assertIn("test was not used", decisions["selection"])
            self.assertIn("decision_rule", comparison)
            self.assertIn("baseline", comparison["scores"])
            self.assertIn("selection_score", comparison["scores"]["baseline"])

    def test_total_cross_family_budget_is_enforced(self):
        frame = continuous_frame()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "continuous.csv"
            output_dir = Path(directory) / "artifacts"
            frame.to_csv(csv_path, index=False)

            run_tabular_vae_workflow(
                csv_path=csv_path,
                objective_text="Generate synthetic tabular data.",
                output_dir=output_dir,
                roles=ColumnRoles(identifiers=("id",), target="target"),
                budget=ExperimentBudget(max_models=1, max_epochs=3),
                n_synthetic=6,
            )

            checkpoints = sorted(output_dir.glob("tabular_*_candidate_*.pt"))
            search = json.loads((output_dir / "candidate_search.json").read_text(encoding="utf-8"))
            self.assertEqual(len(checkpoints), 1)
            self.assertEqual(search["budget"]["max_trained_candidates"], 1)
            self.assertEqual(search["budget"]["trained_candidates"], 1)
            self.assertTrue(search["flow"]["skipped"])

    def test_partial_family_failure_does_not_crash_selection(self):
        frame = continuous_frame()
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "continuous.csv"
            output_dir = Path(directory) / "artifacts"
            frame.to_csv(csv_path, index=False)
            failed_config = VaeConfig(input_dim=2, continuous_dim=2, binary_dim=0, epochs=1)
            failed_model = TabularVAE(failed_config)
            failed_search = SearchResult(
                candidates=(
                    CandidateResult(
                        config=failed_config,
                        validation_loss=float("inf"),
                        epochs_ran=0,
                        wall_seconds=0.0,
                        checkpoint_path=output_dir / "failed_vae.pt",
                        failed=True,
                        failure_reason="forced failure",
                        history={"train_loss": [], "validation_loss": []},
                    ),
                ),
                selected_index=0,
                selection_reason="All VAE candidates failed; no generative candidate selected.",
            )

            with patch("generative_agent.workflow.run_vae_search", return_value=(failed_search, failed_model)):
                result = run_tabular_vae_workflow(
                    csv_path=csv_path,
                    objective_text="Generate synthetic tabular data.",
                    output_dir=output_dir,
                    roles=ColumnRoles(identifiers=("id",), target="target"),
                    budget=ExperimentBudget(max_models=2, max_epochs=3),
                    n_synthetic=6,
                )

            self.assertTrue(result.synthetic_path.exists())
            search = json.loads((output_dir / "candidate_search.json").read_text(encoding="utf-8"))
            comparison = json.loads((output_dir / "validation_comparison.json").read_text(encoding="utf-8"))
            self.assertTrue(search["vae"]["candidates"][0]["failed"])
            self.assertIn("flow", search)
            self.assertNotIn("vae", comparison["scores"])

    def test_workflow_rejects_missing_and_skips_flow_for_binary_features(self):
        missing = continuous_frame()
        missing.loc[0, "x1"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "missing.csv"
            missing.to_csv(csv_path, index=False)
            with self.assertRaisesRegex(ValueError, "Missing values are not supported"):
                run_tabular_vae_workflow(
                    csv_path=csv_path,
                    objective_text="Generate synthetic tabular data.",
                    output_dir=Path(directory) / "artifacts",
                    roles=ColumnRoles(identifiers=("id",), target="target"),
                )


if __name__ == "__main__":
    unittest.main()
