"""Budget-aware RealNVP search for continuous tabular data."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from generative_agent.flow import FlowConfig, FlowTrainer, RealNVPFlow
from generative_agent.state import ExperimentBudget


@dataclass(frozen=True)
class FlowCandidateResult:
    config: FlowConfig
    validation_nll: float
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
class FlowSearchResult:
    candidates: tuple[FlowCandidateResult, ...]
    selected_index: int
    selection_reason: str

    @property
    def selected(self) -> FlowCandidateResult:
        return self.candidates[self.selected_index]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_index": self.selected_index,
            "selection_reason": self.selection_reason,
        }


def propose_flow_configs(*, input_dim: int, n_train: int, budget: ExperimentBudget) -> tuple[FlowConfig, ...]:
    count = min(max(1, budget.max_models), 2)
    batch_size = min(32, max(2, n_train))
    epochs = max(1, budget.max_epochs)
    candidates = (
        FlowConfig(
            input_dim=input_dim,
            num_layers=4 if input_dim > 1 else 2,
            hidden_dim=32,
            batch_size=batch_size,
            epochs=epochs,
            seed=budget.seed + 101,
            early_stopping_patience=min(5, max(1, epochs)),
        ),
        FlowConfig(
            input_dim=input_dim,
            num_layers=6 if input_dim > 1 else 2,
            hidden_dim=48,
            learning_rate=5e-4,
            batch_size=batch_size,
            epochs=epochs,
            seed=budget.seed + 131,
            early_stopping_patience=min(5, max(1, epochs)),
        ),
    )
    return candidates[:count]


def run_flow_search(
    *,
    train_array: np.ndarray,
    validation_array: np.ndarray,
    configs: tuple[FlowConfig, ...],
    checkpoint_dir: Path,
) -> tuple[FlowSearchResult, RealNVPFlow]:
    results: list[FlowCandidateResult] = []
    models: list[RealNVPFlow] = []
    for index, config in enumerate(configs):
        model = RealNVPFlow(config)
        checkpoint = checkpoint_dir / f"tabular_flow_candidate_{index}.pt"
        training = FlowTrainer().train(model, train_array, validation_array, checkpoint)
        models.append(model)
        results.append(
            FlowCandidateResult(
                config=config,
                validation_nll=training.best_validation_nll,
                epochs_ran=training.epochs_ran,
                wall_seconds=training.wall_seconds,
                checkpoint_path=training.checkpoint_path,
                failed=training.failed,
                failure_reason=training.failure_reason,
                history=training.history,
            )
        )
    valid = [index for index, result in enumerate(results) if not result.failed]
    if not valid:
        return (
            FlowSearchResult(
                candidates=tuple(results),
                selected_index=0,
                selection_reason="All flow candidates failed; no flow candidate selected.",
            ),
            models[0],
        )
    selected = min(valid, key=lambda index: results[index].validation_nll)
    return (
        FlowSearchResult(
            candidates=tuple(results),
            selected_index=selected,
            selection_reason="Selected lowest validation negative log likelihood within flow family; test data was not used.",
        ),
        models[selected],
    )
