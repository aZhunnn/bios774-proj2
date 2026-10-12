"""Synthetic sample generation for trained tabular VAEs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from generative_agent.tabular_preprocessing import TabularVaePreprocessor
from generative_agent.vae import TabularVAE
from generative_agent.flow import RealNVPFlow
from generative_agent.classical import GaussianCopula, TabularGMM


@dataclass(frozen=True)
class GenerationResult:
    samples: pd.DataFrame
    metadata: dict[str, object]


def generate_synthetic(
    model: TabularVAE,
    preprocessor: TabularVaePreprocessor,
    *,
    n_samples: int,
    seed: int,
    output_path: Path | None = None,
) -> GenerationResult:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.config.latent_dim, device=device)
        decoded = model.decode(z).detach().cpu().numpy()
    cdim = model.config.continuous_dim
    encoded = np.concatenate(
        [
            decoded[:, :cdim],
            (1 / (1 + np.exp(-decoded[:, cdim:])) >= 0.5).astype(np.float32),
        ],
        axis=1,
    )
    if not np.isfinite(encoded).all():
        raise ValueError("Generated encoded samples contain non-finite values.")
    samples = preprocessor.inverse_transform(encoded)
    samples.insert(0, "synthetic_record", True)
    if len(samples) != n_samples:
        raise ValueError("Generated sample count does not match request.")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(output_path, index=False)
    return GenerationResult(
        samples=samples,
        metadata={
            "n_samples": n_samples,
            "seed": seed,
            "synthetic_record_column": "synthetic_record",
            "method": "tabular_vae",
            "output_path": str(output_path) if output_path else None,
        },
    )


def generate_flow_synthetic(
    model: RealNVPFlow,
    preprocessor: TabularVaePreprocessor,
    *,
    n_samples: int,
    seed: int,
    output_path: Path | None = None,
) -> GenerationResult:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    samples_tensor = model.sample(n_samples, seed=seed)
    encoded = samples_tensor.detach().cpu().numpy()
    if not np.isfinite(encoded).all():
        raise ValueError("Generated flow samples contain non-finite values.")
    samples = preprocessor.inverse_transform(encoded)
    samples.insert(0, "synthetic_record", True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(output_path, index=False)
    return GenerationResult(
        samples=samples,
        metadata={
            "n_samples": n_samples,
            "seed": seed,
            "synthetic_record_column": "synthetic_record",
            "method": "tabular_flow",
            "output_path": str(output_path) if output_path else None,
        },
    )


def generate_classical_synthetic(
    model: TabularGMM | GaussianCopula,
    preprocessor: TabularVaePreprocessor,
    *,
    n_samples: int,
    seed: int,
    method: str,
    output_path: Path | None = None,
) -> GenerationResult:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    encoded = model.sample(n_samples, seed=seed)
    if not np.isfinite(encoded).all():
        raise ValueError(f"Generated {method} samples contain non-finite values.")
    samples = preprocessor.inverse_transform(encoded)
    samples.insert(0, "synthetic_record", True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(output_path, index=False)
    return GenerationResult(
        samples=samples,
        metadata={
            "n_samples": n_samples,
            "seed": seed,
            "synthetic_record_column": "synthetic_record",
            "method": method,
            "output_path": str(output_path) if output_path else None,
        },
    )
