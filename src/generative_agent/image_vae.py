"""Lightweight convolutional VAE for small biomedical images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class ImageVaeConfig:
    image_shape: tuple[int, int, int]
    latent_dim: int = 16
    hidden_channels: tuple[int, int] = (16, 32)
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 5
    kl_weight: float = 1.0
    early_stopping_patience: int = 3
    seed: int = 2026
    device: str = "cpu"
    reconstruction_assumption: str = "gaussian_unit_variance"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ImageTrainingResult:
    history: dict[str, list[float]]
    best_validation_loss: float
    epochs_ran: int
    checkpoint_path: Path
    wall_seconds: float
    device: str
    failed: bool = False
    failure_reason: str | None = None


class ConvImageVAE(nn.Module):
    def __init__(self, config: ImageVaeConfig) -> None:
        super().__init__()
        _set_seed(config.seed)
        self.config = config
        height, width, channels = config.image_shape
        if height != width or height % 4 != 0:
            raise ValueError("ConvImageVAE requires square images with side divisible by 4.")
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, config.hidden_channels[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.hidden_channels[0], config.hidden_channels[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        reduced = height // 4
        flattened = config.hidden_channels[1] * reduced * reduced
        self._encoded_shape = (config.hidden_channels[1], reduced, reduced)
        self.mu = nn.Linear(flattened, config.latent_dim)
        self.logvar = nn.Linear(flattened, config.latent_dim)
        self.decoder_input = nn.Linear(config.latent_dim, flattened)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_channels[1], config.hidden_channels[0], kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels[0], channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x).flatten(start_dim=1)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder_input(z).view(z.shape[0], *self._encoded_shape)
        return self.decoder(hidden)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar

    def loss(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.config.reconstruction_assumption != "gaussian_unit_variance":
            raise ValueError("Only gaussian_unit_variance reconstruction is implemented for image VAE.")
        residual = x - recon
        reconstruction = 0.5 * torch.sum(residual.pow(2) + math.log(2 * math.pi))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        batch = x.shape[0]
        total = (reconstruction + self.config.kl_weight * kl) / batch
        return {"total": total, "reconstruction": reconstruction / batch, "kl": kl / batch}

    def sample(self, n_samples: int, *, seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        device = next(self.parameters()).device
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.latent_dim, device=device)
            return self.decode(z)


class ImageVaeTrainer:
    def train(self, model: ConvImageVAE, train_dataset, validation_dataset, checkpoint_path: Path) -> ImageTrainingResult:
        start = time.perf_counter()
        config = model.config
        _set_seed(config.seed)
        device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
        model.to(device)
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        history = {"train_loss": [], "validation_loss": [], "validation_reconstruction": []}
        best = float("inf")
        best_state = None
        stale = 0
        try:
            for _epoch in range(config.epochs):
                model.train()
                train_losses = []
                for batch, _labels in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    recon, mu, logvar = model(batch)
                    loss = model.loss(batch, recon, mu, logvar)
                    if not torch.isfinite(loss["total"]):
                        raise FloatingPointError("NaN/Inf encountered in image VAE training loss.")
                    loss["total"].backward()
                    optimizer.step()
                    train_losses.append(float(loss["total"].detach().cpu()))
                val_loss, val_recon = _evaluate_loss(model, val_loader, device)
                history["train_loss"].append(float(np.mean(train_losses)))
                history["validation_loss"].append(val_loss)
                history["validation_reconstruction"].append(val_recon)
                if val_loss < best:
                    best = val_loss
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= config.early_stopping_patience:
                    break
        except (RuntimeError, FloatingPointError) as exc:
            return ImageTrainingResult(history, best, len(history["train_loss"]), checkpoint_path, time.perf_counter() - start, str(device), True, str(exc))
        if best_state is None:
            raise RuntimeError("Image VAE training produced no checkpoint.")
        model.load_state_dict(best_state)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "config": config.to_dict(), "history": history}, checkpoint_path)
        return ImageTrainingResult(history, best, len(history["train_loss"]), checkpoint_path, time.perf_counter() - start, str(device))


def load_image_vae_checkpoint(path: Path) -> ConvImageVAE:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = ImageVaeConfig(**payload["config"])
    model = ConvImageVAE(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _evaluate_loss(model: ConvImageVAE, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses = []
    recons = []
    with torch.no_grad():
        for batch, _labels in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss = model.loss(batch, recon, mu, logvar)
            if not torch.isfinite(loss["total"]):
                raise FloatingPointError("NaN/Inf encountered in image VAE validation loss.")
            losses.append(float(loss["total"].detach().cpu()))
            recons.append(float(loss["reconstruction"].detach().cpu()))
    return float(np.mean(losses)), float(np.mean(recons))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
