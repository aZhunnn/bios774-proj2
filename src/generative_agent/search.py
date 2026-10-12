"""Budget-aware VAE candidate selection using validation data only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np

from generative_agent.state import ExperimentBudget
from generative_agent.vae import TabularVAE, TrainingResult, VaeConfig, VaeTrainer


@dataclass(frozen=True)
class CandidateResult:
    config: VaeConfig
    validation_loss: float
    epochs_ran: int
    wall_seconds: float
    checkpoint_path: Path
    failed: bool
    failure_reason: str | None
    history: dict[str, list[float]]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        payload["checkpoint_path"] = str(self.checkpoint_path)
        return payload


@dataclass(frozen=True)
class SearchResult:
    candidates: tuple[CandidateResult, ...]
    selected_index: int
    selection_reason: str

    @property
    def selected(self) -> CandidateResult:
        return self.candidates[self.selected_index]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_index": self.selected_index,
            "selection_reason": self.selection_reason,
        }


def propose_vae_configs(
    *,
    input_dim: int,
    continuous_dim: int,
    binary_dim: int,
    n_train: int,
    budget: ExperimentBudget,
) -> tuple[VaeConfig, ...]:
    count = min(max(1, budget.max_models), 3)
    base_batch = min(32, max(2, n_train))
    base_latent = min(4, max(1, input_dim))
    epochs = max(1, budget.max_epochs)
    candidates = [
        VaeConfig(
            input_dim=input_dim,
            continuous_dim=continuous_dim,
            binary_dim=binary_dim,
            latent_dim=base_latent,
            hidden_dims=(32, 16) if input_dim > 2 else (16,),
            batch_size=base_batch,
            epochs=epochs,
            seed=budget.seed,
            early_stopping_patience=min(5, max(1, epochs)),
        ),
        VaeConfig(
            input_dim=input_dim,
            continuous_dim=continuous_dim,
            binary_dim=binary_dim,
            latent_dim=max(1, min(2, input_dim)),
            hidden_dims=(16,),
            learning_rate=5e-4,
            batch_size=base_batch,
            epochs=epochs,
            seed=budget.seed + 17,
            early_stopping_patience=min(5, max(1, epochs)),
        ),
        VaeConfig(
            input_dim=input_dim,
            continuous_dim=continuous_dim,
            binary_dim=binary_dim,
            latent_dim=max(1, min(8, input_dim)),
            hidden_dims=(64, 32) if n_train >= 64 else (32,),
            learning_rate=1e-3,
            batch_size=base_batch,
            epochs=epochs,
            kl_weight=0.5,
            seed=budget.seed + 31,
            early_stopping_patience=min(5, max(1, epochs)),
        ),
    ]
    return tuple(candidates[:count])


def run_vae_search(
    *,
    train_array: np.ndarray,
    validation_array: np.ndarray,
    configs: tuple[VaeConfig, ...],
    checkpoint_dir: Path,
) -> tuple[SearchResult, TabularVAE]:
    results: list[CandidateResult] = []
    models: list[TabularVAE] = []
    for index, config in enumerate(configs):
        model = TabularVAE(config)
        checkpoint = checkpoint_dir / f"tabular_vae_candidate_{index}.pt"
        start = time.perf_counter()
        training = VaeTrainer().train(model, train_array, validation_array, checkpoint)
        elapsed = time.perf_counter() - start
        models.append(model)
        results.append(_candidate_result(config, training, elapsed))
    valid = [index for index, result in enumerate(results) if not result.failed]
    if not valid:
        return (
            SearchResult(
                candidates=tuple(results),
                selected_index=0,
                selection_reason="All VAE candidates failed; no generative candidate selected.",
            ),
            models[0],
        )
    selected = min(valid, key=lambda index: results[index].validation_loss)
    return (
        SearchResult(
            candidates=tuple(results),
            selected_index=selected,
            selection_reason="Selected lowest validation loss among successful candidates; test data was not used.",
        ),
        models[selected],
    )


def _candidate_result(config: VaeConfig, training: TrainingResult, wall_seconds: float) -> CandidateResult:
    return CandidateResult(
        config=config,
        validation_loss=training.best_validation_loss,
        epochs_ran=training.epochs_ran,
        wall_seconds=wall_seconds,
        checkpoint_path=training.checkpoint_path,
        failed=training.failed,
        failure_reason=training.failure_reason,
        history=training.history,
    )
