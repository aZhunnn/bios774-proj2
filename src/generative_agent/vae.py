"""PyTorch tabular VAE and training loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class VaeConfig:
    input_dim: int
    continuous_dim: int
    binary_dim: int
    latent_dim: int = 4
    hidden_dims: tuple[int, ...] = (32, 16)
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    kl_weight: float = 1.0
    early_stopping_patience: int = 8
    seed: int = 2026
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.input_dim != self.continuous_dim + self.binary_dim:
            raise ValueError("input_dim must equal continuous_dim + binary_dim.")
        if min(self.input_dim, self.latent_dim, self.batch_size, self.epochs) <= 0:
            raise ValueError("VAE dimensions, batch size, and epochs must be positive.")
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be non-negative.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VaeLoss:
    total: torch.Tensor
    reconstruction: torch.Tensor
    continuous: torch.Tensor
    binary: torch.Tensor
    kl: torch.Tensor


class TabularVAE(nn.Module):
    """VAE with Gaussian continuous outputs and Bernoulli binary outputs."""

    def __init__(self, config: VaeConfig) -> None:
        super().__init__()
        _set_seed(config.seed)
        self.config = config
        encoder_layers: list[nn.Module] = []
        previous = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        self.encoder = nn.Sequential(*encoder_layers)
        self.mu = nn.Linear(previous, config.latent_dim)
        self.logvar = nn.Linear(previous, config.latent_dim)

        decoder_layers: list[nn.Module] = []
        previous = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        decoder_layers.append(nn.Linear(previous, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> VaeLoss:
        cdim = self.config.continuous_dim
        continuous_loss = torch.tensor(0.0, device=x.device)
        binary_loss = torch.tensor(0.0, device=x.device)
        if cdim:
            # Unit-variance Gaussian observation model on standardized continuous features.
            residual = x[:, :cdim] - recon[:, :cdim]
            continuous_loss = 0.5 * torch.sum(residual.pow(2) + math.log(2 * math.pi))
        if self.config.binary_dim:
            binary_loss = F.binary_cross_entropy_with_logits(
                recon[:, cdim:],
                x[:, cdim:],
                reduction="sum",
            )
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        reconstruction = continuous_loss + binary_loss
        total = reconstruction + self.config.kl_weight * kl
        batch = x.shape[0]
        return VaeLoss(
            total=total / batch,
            reconstruction=reconstruction / batch,
            continuous=continuous_loss / batch,
            binary=binary_loss / batch,
            kl=kl / batch,
        )


@dataclass(frozen=True)
class TrainingResult:
    history: dict[str, list[float]]
    best_validation_loss: float
    epochs_ran: int
    checkpoint_path: Path
    device: str
    failed: bool = False
    failure_reason: str | None = None


class VaeTrainer:
    def train(
        self,
        model: TabularVAE,
        train_array: np.ndarray,
        validation_array: np.ndarray,
        checkpoint_path: Path,
    ) -> TrainingResult:
        config = model.config
        _set_seed(config.seed)
        device = torch.device(config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        train_loader = DataLoader(
            TensorDataset(torch.as_tensor(train_array, dtype=torch.float32)),
            batch_size=config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        validation_tensor = torch.as_tensor(validation_array, dtype=torch.float32, device=device)
        history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0

        try:
            for _epoch in range(config.epochs):
                model.train()
                train_losses: list[float] = []
                for (batch,) in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    recon, mu, logvar = model(batch)
                    loss = model.loss(batch, recon, mu, logvar)
                    if not torch.isfinite(loss.total):
                        raise FloatingPointError("NaN/Inf encountered in training loss.")
                    loss.total.backward()
                    optimizer.step()
                    train_losses.append(float(loss.total.detach().cpu()))
                model.eval()
                with torch.no_grad():
                    recon, mu, logvar = model(validation_tensor)
                    validation_loss = model.loss(validation_tensor, recon, mu, logvar).total
                if not torch.isfinite(validation_loss):
                    raise FloatingPointError("NaN/Inf encountered in validation loss.")
                validation_value = float(validation_loss.detach().cpu())
                history["train_loss"].append(float(np.mean(train_losses)))
                history["validation_loss"].append(validation_value)
                if validation_value < best_loss:
                    best_loss = validation_value
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= config.early_stopping_patience:
                    break
        except (FloatingPointError, RuntimeError) as exc:
            return TrainingResult(
                history=history,
                best_validation_loss=best_loss,
                epochs_ran=len(history["train_loss"]),
                checkpoint_path=checkpoint_path,
                device=str(device),
                failed=True,
                failure_reason=str(exc),
            )

        if best_state is None:
            raise RuntimeError("Training produced no checkpoint.")
        model.load_state_dict(best_state)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "config": config.to_dict(), "history": history}, checkpoint_path)
        return TrainingResult(
            history=history,
            best_validation_loss=best_loss,
            epochs_ran=len(history["train_loss"]),
            checkpoint_path=checkpoint_path,
            device=str(device),
        )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
