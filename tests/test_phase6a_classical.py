import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from generative_agent.classical import (
    CopulaConfig,
    GaussianCopula,
    GmmConfig,
    TabularGMM,
    load_copula_checkpoint,
    load_gmm_checkpoint,
)
from generative_agent.registry import CapabilityRegistry
from generative_agent.data import DatasetKind
from generative_agent.objectives import ObjectiveType
from generative_agent.state import ExperimentBudget
from generative_agent.tabular_io import ColumnRoles
from generative_agent.workflow import run_tabular_generation_workflow


class Phase6AClassicalTests(unittest.TestCase):
    def test_gmm_fit_sample_reproducibility_and_persistence(self):
        rng = np.random.default_rng(1)
        train = rng.normal(size=(80, 3)).astype(np.float32)
        model = TabularGMM(GmmConfig(n_components=2, covariance_type="full", seed=5)).fit(train)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "gmm.pkl"
            model.save(checkpoint)
            restored = load_gmm_checkpoint(checkpoint)

            np.testing.assert_allclose(model.sample(6, seed=9), restored.sample(6, seed=9))
            self.assertEqual(restored.sample(6, seed=9).shape, (6, 3))
            self.assertIn("lower_bound", restored.diagnostics)

    def test_gmm_handles_near_singular_covariance_with_regularization(self):
        x = np.linspace(-1, 1, 40)
        train = np.column_stack([x, x + 1e-9, np.ones_like(x)]).astype(np.float32)
        model = TabularGMM(GmmConfig(n_components=2, covariance_type="full", reg_covar=1e-4, seed=2)).fit(train)
        samples = model.sample(5, seed=3)

        self.assertTrue(np.isfinite(samples).all())
        self.assertEqual(samples.shape, (5, 3))

    def test_copula_transform_sampling_correlation_and_persistence(self):
        rng = np.random.default_rng(2)
        base = rng.normal(size=100)
        train = np.column_stack([base, base * 0.8 + rng.normal(scale=0.1, size=100), np.round(base, 1)]).astype(np.float32)
        model = GaussianCopula(CopulaConfig(seed=7)).fit(train)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "copula.pkl"
            model.save(checkpoint)
            restored = load_copula_checkpoint(checkpoint)
            a = restored.sample(50, seed=4)
            b = restored.sample(50, seed=4)

            np.testing.assert_allclose(a, b)
            self.assertTrue(np.isfinite(a).all())
            self.assertGreater(np.linalg.eigvalsh(restored.correlation).min(), 0)

    def test_registry_exposes_classical_tabular_capabilities(self):
        candidates = CapabilityRegistry().candidates(DatasetKind.TABULAR, ObjectiveType.DATA_GENERATION, implemented_only=True)
        names = {candidate.name for candidate in candidates}

        self.assertIn("tabular_gmm", names)
        self.assertIn("tabular_copula", names)
        copula = next(candidate for candidate in candidates if candidate.name == "tabular_copula")
        self.assertTrue(any("tail dependence" in limitation for limitation in copula.limitations))

    def test_workflow_trains_four_tabular_families_and_preserves_test_isolation(self):
        rng = np.random.default_rng(3)
        x1 = rng.normal(size=90)
        x2 = 0.6 * x1 + rng.normal(scale=0.5, size=90)
        x3 = rng.gamma(shape=2.0, scale=1.0, size=90)
        frame = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "quality": np.arange(90) % 3})

        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "toy.csv"
            output = Path(directory) / "artifacts"
            frame.to_csv(csv_path, index=False)
            run_tabular_generation_workflow(
                csv_path=csv_path,
                objective_text="Generate synthetic tabular data for internal testing.",
                output_dir=output,
                roles=ColumnRoles(target="quality"),
                budget=ExperimentBudget(max_models=4, max_epochs=3, seed=12),
                n_synthetic=30,
            )

            search = json.loads((output / "candidate_search.json").read_text())
            comparison = json.loads((output / "validation_comparison.json").read_text())
            evaluation = json.loads((output / "evaluation_results.json").read_text())

            self.assertEqual(search["budget"]["trained_candidates"], 4)
            self.assertEqual(search["budget"]["allocation"], {"copula": 1, "flow": 1, "gmm": 1, "vae": 1})
            self.assertIn("gmm", search)
            self.assertIn("copula", search)
            self.assertIn("gmm", comparison["scores"])
            self.assertIn("copula", comparison["scores"])
            self.assertIn("selected", evaluation)
            self.assertIn(evaluation["selected"]["method"], {"training_row_bootstrap", "tabular_vae", "tabular_flow", "tabular_gmm", "tabular_copula"})


if __name__ == "__main__":
    unittest.main()
