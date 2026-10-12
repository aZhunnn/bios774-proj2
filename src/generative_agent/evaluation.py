"""Evaluation metrics for synthetic tabular data and bootstrap baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from generative_agent.tabular_preprocessing import TabularVaeSchema


@dataclass(frozen=True)
class EvaluationResult:
    method: str
    feature_differences: dict[str, dict[str, float]]
    correlation_l1: float | None
    duplicate_fraction: float
    nearest_neighbor_mean: float
    nearest_neighbor_min: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_synthetic(
    *,
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: TabularVaeSchema,
    method: str,
) -> EvaluationResult:
    synthetic_model = synthetic[list(schema.model_columns)].copy()
    feature_diffs: dict[str, dict[str, float]] = {}
    for column in schema.continuous:
        feature_diffs[column] = {
            "test_mean": float(test_real[column].mean()),
            "synthetic_mean": float(synthetic_model[column].mean()),
            "absolute_mean_difference": float(abs(test_real[column].mean() - synthetic_model[column].mean())),
            "test_std": float(test_real[column].std(ddof=0)),
            "synthetic_std": float(synthetic_model[column].std(ddof=0)),
        }
    for column in schema.binary:
        high = schema.binary_values[column][1]
        feature_diffs[column] = {
            "test_positive_rate": float((test_real[column] == high).mean()),
            "synthetic_positive_rate": float((synthetic_model[column] == high).mean()),
            "absolute_rate_difference": float(
                abs((test_real[column] == high).mean() - (synthetic_model[column] == high).mean())
            ),
        }

    correlation_l1 = _correlation_l1(test_real[list(schema.continuous)], synthetic_model[list(schema.continuous)])
    encoded_train = _encode_for_distance(train_real, schema)
    encoded_synth = _encode_for_distance(synthetic_model, schema)
    nn = _nearest_neighbor_distances(encoded_synth, encoded_train)
    duplicates = float(synthetic_model.duplicated().mean()) if len(synthetic_model) else 0.0
    return EvaluationResult(
        method=method,
        feature_differences=feature_diffs,
        correlation_l1=correlation_l1,
        duplicate_fraction=duplicates,
        nearest_neighbor_mean=float(np.mean(nn)) if len(nn) else 0.0,
        nearest_neighbor_min=float(np.min(nn)) if len(nn) else 0.0,
        notes=(
            "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
            "Bootstrap baseline is evaluated with the same metrics for fairness.",
        ),
    )


def bootstrap_baseline(train_real: pd.DataFrame, n_samples: int, *, seed: int) -> pd.DataFrame:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    return train_real.sample(n=n_samples, replace=True, random_state=seed).reset_index(drop=True)


def _correlation_l1(test_cont: pd.DataFrame, synth_cont: pd.DataFrame) -> float | None:
    if test_cont.shape[1] < 2:
        return None
    return float(np.abs(test_cont.corr().fillna(0).to_numpy() - synth_cont.corr().fillna(0).to_numpy()).mean())


def _encode_for_distance(frame: pd.DataFrame, schema: TabularVaeSchema) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column in schema.continuous:
        values = frame[column].astype(float).to_numpy()
        std = values.std() if values.std() > 1e-12 else 1.0
        columns.append(((values - values.mean()) / std)[:, None])
    for column in schema.binary:
        low, high = schema.binary_values[column]
        columns.append(frame[column].map({low: 0.0, high: 1.0}).to_numpy(dtype=float)[:, None])
    return np.concatenate(columns, axis=1) if columns else np.empty((len(frame), 0))


def _nearest_neighbor_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.size == 0 or reference.size == 0:
        return np.zeros((query.shape[0],), dtype=float)
    distances = np.sqrt(((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1)
