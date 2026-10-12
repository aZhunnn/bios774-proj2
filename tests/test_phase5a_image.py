import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from generative_agent.data import DatasetKind
from generative_agent.image_data import ImagePreprocessConfig, SplitNpzImageAdapter
from generative_agent.image_vae import ConvImageVAE, ImageVaeConfig, ImageVaeTrainer, load_image_vae_checkpoint
from generative_agent.image_workflow import run_image_generation_workflow
from generative_agent.objectives import ObjectiveType
from generative_agent.registry import CapabilityRegistry
from generative_agent.state import ExperimentBudget
from generative_agent.eligibility import method_eligibility
from generative_agent.objectives import ObjectiveUseCase


def write_image_npz(path: Path, *, channels=3, size=16, n_train=24, n_val=8, n_test=8):
    rng = np.random.default_rng(2026)
    def images(n):
        return rng.integers(0, 256, size=(n, size, size, channels), dtype=np.uint8)
    np.savez(
        path,
        train_images=images(n_train),
        train_labels=np.arange(n_train) % 3,
        val_images=images(n_val),
        val_labels=np.arange(n_val) % 3,
        test_images=images(n_test),
        test_labels=np.arange(n_test) % 3,
    )


class Phase5AImageTests(unittest.TestCase):
    def test_image_adapter_profiles_and_preserves_split_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.npz"
            write_image_npz(path)
            bundle = SplitNpzImageAdapter().load(
                path,
                config=ImagePreprocessConfig(max_train_samples=10, max_validation_samples=4, max_test_samples=3, seed=7),
            )

            self.assertEqual(bundle.profile.image_shape, (16, 16, 3))
            self.assertEqual(bundle.profile.split_counts, {"train": 10, "validation": 4, "test": 3})
            self.assertTrue(bundle.profile.labels_are_metadata)
            self.assertEqual(len(bundle.train), 10)
            self.assertEqual(len(bundle.validation), 4)
            self.assertEqual(len(bundle.test), 3)
            self.assertGreaterEqual(float(bundle.train.tensors[0].min()), 0.0)
            self.assertLessEqual(float(bundle.train.tensors[0].max()), 1.0)

    def test_image_adapter_rejects_unsupported_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            write_image_npz(path, channels=4)
            with self.assertRaisesRegex(ValueError, "1 or 3 channels"):
                SplitNpzImageAdapter().load(path)

    def test_registry_exposes_image_vae_capability(self):
        candidates = CapabilityRegistry().candidates(DatasetKind.IMAGE, ObjectiveType.DATA_GENERATION, implemented_only=True)

        self.assertTrue(any(candidate.name == "image_vae" for candidate in candidates))
        image_vae = next(candidate for candidate in candidates if candidate.name == "image_vae")
        self.assertEqual(image_vae.modalities, ("image",))
        self.assertIn("reconstruct", image_vae.operations)

    def test_image_vae_external_eligibility_does_not_claim_privacy(self):
        eligibility = method_eligibility("image_vae", ObjectiveUseCase.EXTERNAL_SHARING)

        self.assertEqual(eligibility.method, "vae")
        self.assertTrue(eligibility.eligible)
        self.assertTrue(any("no formal privacy guarantee" in item for item in eligibility.limitations))

    def test_image_vae_forward_backward_sampling_and_checkpoint(self):
        config = ImageVaeConfig(image_shape=(16, 16, 3), latent_dim=4, hidden_channels=(4, 8), epochs=1, batch_size=4)
        model = ConvImageVAE(config)
        x = torch.rand(4, 3, 16, 16)

        recon, mu, logvar = model(x)
        loss = model.loss(x, recon, mu, logvar)
        loss["total"].backward()
        samples = model.sample(3, seed=11)

        self.assertEqual(recon.shape, x.shape)
        self.assertEqual(mu.shape, (4, 4))
        self.assertTrue(torch.isfinite(loss["total"]))
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad))
        self.assertEqual(samples.shape, (3, 3, 16, 16))

    def test_image_training_checkpoint_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.npz"
            write_image_npz(path, n_train=16, n_val=4, n_test=4)
            bundle = SplitNpzImageAdapter().load(path)
            config = ImageVaeConfig(
                image_shape=(16, 16, 3),
                latent_dim=4,
                hidden_channels=(4, 8),
                epochs=1,
                batch_size=4,
                seed=9,
            )
            model = ConvImageVAE(config)
            result = ImageVaeTrainer().train(model, bundle.train, bundle.validation, Path(directory) / "image_vae.pt")
            restored = load_image_vae_checkpoint(result.checkpoint_path)

            self.assertFalse(result.failed)
            self.assertTrue(result.checkpoint_path.exists())
            torch.testing.assert_close(model.sample(2, seed=3), restored.sample(2, seed=3))

    def test_end_to_end_image_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.npz"
            output = Path(directory) / "artifacts"
            write_image_npz(path, n_train=20, n_val=6, n_test=6)
            result = run_image_generation_workflow(
                npz_path=path,
                output_dir=output,
                objective_text="Generate synthetic images for internal testing.",
                budget=ExperimentBudget(max_models=1, max_epochs=1),
                preprocess=ImagePreprocessConfig(max_train_samples=16, max_validation_samples=4, max_test_samples=4),
                latent_dim=4,
                n_generated=4,
            )

            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.checkpoint_path.exists())
            self.assertTrue(result.generated_grid_path.exists())
            self.assertTrue((output / "dataset_profile.json").exists())
            self.assertTrue((output / "evaluation_results.json").exists())


if __name__ == "__main__":
    unittest.main()
