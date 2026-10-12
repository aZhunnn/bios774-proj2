"""Lightweight DCGAN-style model for small biomedical images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ImageGanConfig:
    image_shape: tuple[int, int, int]
    latent_dim: int = 32
    generator_channels: tuple[int, int] = (32, 16)
    discriminator_channels: tuple[int, int] = (16, 32)
    generator_learning_rate: float = 2e-4
    discriminator_learning_rate: float = 2e-4
    batch_size: int = 64
    epochs: int = 5
    seed: int = 2026
    device: str = "cpu"
    stability_patience: int = 2

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ImageGanTrainingResult:
    history: dict[str, list[float]]
    checkpoint_path: Path
    wall_seconds: float
    device: str
    epochs_ran: int
    failed: bool = False
    failure_reason: str | None = None


class ImageGenerator(nn.Module):
    def __init__(self, config: ImageGanConfig) -> None:
        super().__init__()
        height, width, channels = config.image_shape
        if height != width or height % 4 != 0:
            raise ValueError("Image GAN requires square images with side divisible by 4.")
        if channels not in (1, 3):
            raise ValueError("Image GAN supports only 1-channel or 3-channel images.")
        self.config = config
        self._start_side = height // 4
        self._start_channels = config.generator_channels[0]
        self.input = nn.Linear(config.latent_dim, self._start_channels * self._start_side * self._start_side)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(config.generator_channels[0], config.generator_channels[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config.generator_channels[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(config.generator_channels[1], channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.input(z).view(z.shape[0], self._start_channels, self._start_side, self._start_side)
        return self.net(hidden)


class ImageDiscriminator(nn.Module):
    def __init__(self, config: ImageGanConfig) -> None:
        super().__init__()
        height, _width, channels = config.image_shape
        reduced = height // 4
        self.net = nn.Sequential(
            nn.Conv2d(channels, config.discriminator_channels[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(config.discriminator_channels[0], config.discriminator_channels[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(config.discriminator_channels[1]),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(config.discriminator_channels[1] * reduced * reduced, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


class ImageGAN(nn.Module):
    """Small unconditional GAN for normalized small images.

    Supported inputs are square 1-channel or 3-channel tensors with side length
    divisible by 4, normalized to [0, 1]. Labels are not used for conditioning.
    """

    def __init__(self, config: ImageGanConfig) -> None:
        super().__init__()
        _set_seed(config.seed)
        self.config = config
        self.generator = ImageGenerator(config)
        self.discriminator = ImageDiscriminator(config)

    def sample(self, n_samples: int, *, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        device = next(self.generator.parameters()).device
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim, device=device)
            return self.generator(z)


class ImageGanTrainer:
    def train(self, model: ImageGAN, train_dataset, validation_dataset, checkpoint_path: Path) -> ImageGanTrainingResult:
        del validation_dataset
        start = time.perf_counter()
        config = model.config
        _set_seed(config.seed)
        device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
        model.to(device)
        generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
        g_optimizer = torch.optim.Adam(model.generator.parameters(), lr=config.generator_learning_rate, betas=(0.5, 0.999))
        d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=config.discriminator_learning_rate, betas=(0.5, 0.999))
        loss_fn = nn.BCEWithLogitsLoss()
        history = {"generator_loss": [], "discriminator_loss": [], "sample_std": []}
        best_state: dict[str, dict[str, torch.Tensor]] | None = None
        stale_low_diversity = 0
        try:
            for epoch in range(config.epochs):
                model.train()
                g_losses: list[float] = []
                d_losses: list[float] = []
                for real, _labels in loader:
                    real = real.to(device)
                    batch = real.shape[0]
                    real_targets = torch.full((batch,), 0.9, device=device)
                    fake_targets = torch.zeros(batch, device=device)

                    d_optimizer.zero_grad()
                    real_logits = model.discriminator(real)
                    z = torch.randn(batch, config.latent_dim, device=device)
                    fake = model.generator(z).detach()
                    fake_logits = model.discriminator(fake)
                    d_loss = 0.5 * (loss_fn(real_logits, real_targets) + loss_fn(fake_logits, fake_targets))
                    _check_finite(d_loss, "GAN discriminator loss")
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), max_norm=10.0)
                    d_optimizer.step()

                    g_optimizer.zero_grad()
                    z = torch.randn(batch, config.latent_dim, device=device)
                    generated = model.generator(z)
                    generated_logits = model.discriminator(generated)
                    g_loss = loss_fn(generated_logits, torch.ones(batch, device=device))
                    _check_finite(g_loss, "GAN generator loss")
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.generator.parameters(), max_norm=10.0)
                    g_optimizer.step()

                    g_losses.append(float(g_loss.detach().cpu()))
                    d_losses.append(float(d_loss.detach().cpu()))

                sample_std = float(model.sample(min(16, len(train_dataset)), seed=config.seed + epoch).std().detach().cpu())
                history["generator_loss"].append(float(np.mean(g_losses)))
                history["discriminator_loss"].append(float(np.mean(d_losses)))
                history["sample_std"].append(sample_std)
                best_state = _checkpoint_state(model)
                if sample_std < 1e-6:
                    stale_low_diversity += 1
                else:
                    stale_low_diversity = 0
                if stale_low_diversity >= config.stability_patience:
                    break
        except (RuntimeError, FloatingPointError) as exc:
            return ImageGanTrainingResult(history, checkpoint_path, time.perf_counter() - start, str(device), len(history["generator_loss"]), True, str(exc))
        if best_state is None:
            raise RuntimeError("Image GAN training produced no checkpoint.")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generator_state": best_state["generator"], "discriminator_state": best_state["discriminator"], "config": config.to_dict(), "history": history}, checkpoint_path)
        model.generator.load_state_dict(best_state["generator"])
        model.discriminator.load_state_dict(best_state["discriminator"])
        return ImageGanTrainingResult(history, checkpoint_path, time.perf_counter() - start, str(device), len(history["generator_loss"]))


def load_image_gan_checkpoint(path: Path) -> ImageGAN:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ImageGAN(ImageGanConfig(**payload["config"]))
    model.generator.load_state_dict(payload["generator_state"])
    model.discriminator.load_state_dict(payload["discriminator_state"])
    model.eval()
    return model


def _checkpoint_state(model: ImageGAN) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "generator": {key: value.detach().cpu().clone() for key, value in model.generator.state_dict().items()},
        "discriminator": {key: value.detach().cpu().clone() for key, value in model.discriminator.state_dict().items()},
    }


def _check_finite(loss: torch.Tensor, name: str) -> None:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"NaN/Inf encountered in {name}.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
