"""Continuous tabular RealNVP normalizing flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class FlowConfig:
    input_dim: int
    num_layers: int = 4
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 8
    seed: int = 2026
    device: str = "cpu"

    def __post_init__(self) -> None:
        if min(self.input_dim, self.num_layers, self.hidden_dim, self.batch_size, self.epochs) <= 0:
            raise ValueError("Flow dimensions, batch size, and epochs must be positive.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class AffineCoupling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = x * self.mask
        shift, log_scale = self.net(masked).chunk(2, dim=1)
        log_scale = torch.tanh(log_scale) * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        y = masked + (1.0 - self.mask) * (x * torch.exp(log_scale) + shift)
        return y, log_scale.sum(dim=1)

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        masked = y * self.mask
        shift, log_scale = self.net(masked).chunk(2, dim=1)
        log_scale = torch.tanh(log_scale) * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        x = masked + (1.0 - self.mask) * ((y - shift) * torch.exp(-log_scale))
        return x, -log_scale.sum(dim=1)


class RealNVPFlow(nn.Module):
    """RealNVP with affine coupling layers and standard normal base density."""

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        _set_seed(config.seed)
        self.config = config
        self.layers = nn.ModuleList(
            [
                AffineCoupling(config.input_dim, config.hidden_dim, _mask(config.input_dim, offset=i % 2))
                for i in range(config.num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            z, log_det = layer(z)
            total_log_det = total_log_det + log_det
        return z, total_log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = z
        total_log_det = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            x, log_det = layer.inverse(x)
            total_log_det = total_log_det + log_det
        return x, total_log_det

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        base = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=1)
        return base + log_det

    def sample(self, n_samples: int, *, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        device = next(self.parameters()).device
        z = torch.randn(n_samples, self.config.input_dim, device=device)
        x, _ = self.inverse(z)
        return x


@dataclass(frozen=True)
class FlowTrainingResult:
    history: dict[str, list[float]]
    best_validation_nll: float
    epochs_ran: int
    checkpoint_path: Path
    wall_seconds: float
    device: str
    failed: bool = False
    failure_reason: str | None = None


class FlowTrainer:
    def train(
        self,
        model: RealNVPFlow,
        train_array: np.ndarray,
        validation_array: np.ndarray,
        checkpoint_path: Path,
    ) -> FlowTrainingResult:
        start = time.perf_counter()
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
        history: dict[str, list[float]] = {"train_nll": [], "validation_nll": []}
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
                    loss = -model.log_prob(batch).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("NaN/Inf encountered in flow training loss.")
                    loss.backward()
                    optimizer.step()
                    train_losses.append(float(loss.detach().cpu()))
                model.eval()
                with torch.no_grad():
                    validation_loss = -model.log_prob(validation_tensor).mean()
                if not torch.isfinite(validation_loss):
                    raise FloatingPointError("NaN/Inf encountered in flow validation loss.")
                validation_value = float(validation_loss.detach().cpu())
                history["train_nll"].append(float(np.mean(train_losses)))
                history["validation_nll"].append(validation_value)
                if validation_value < best_loss:
                    best_loss = validation_value
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                if stale_epochs >= config.early_stopping_patience:
                    break
        except (FloatingPointError, RuntimeError) as exc:
            return FlowTrainingResult(
                history=history,
                best_validation_nll=best_loss,
                epochs_ran=len(history["train_nll"]),
                checkpoint_path=checkpoint_path,
                wall_seconds=time.perf_counter() - start,
                device=str(device),
                failed=True,
                failure_reason=str(exc),
            )
        if best_state is None:
            raise RuntimeError("Flow training produced no checkpoint.")
        model.load_state_dict(best_state)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state, "config": config.to_dict(), "history": history}, checkpoint_path)
        return FlowTrainingResult(
            history=history,
            best_validation_nll=best_loss,
            epochs_ran=len(history["train_nll"]),
            checkpoint_path=checkpoint_path,
            wall_seconds=time.perf_counter() - start,
            device=str(device),
        )


def load_flow_checkpoint(path: Path) -> RealNVPFlow:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = FlowConfig(**payload["config"])
    model = RealNVPFlow(config)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _mask(input_dim: int, *, offset: int) -> torch.Tensor:
    return torch.tensor([(index + offset) % 2 for index in range(input_dim)], dtype=torch.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
