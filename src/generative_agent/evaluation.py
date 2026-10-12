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
    unique_fraction: float
    nearest_neighbor_mean: float
    nearest_neighbor_min: float
    marginal_fidelity_score: float
    train_scale_normalized_marginal_fidelity: float
    dependence_preservation_score: float | None
    selection_score: float
    objective_metrics: dict[str, float]
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
        q25_real, q50_real, q75_real = test_real[column].quantile([0.25, 0.5, 0.75]).tolist()
        q25_synth, q50_synth, q75_synth = synthetic_model[column].quantile([0.25, 0.5, 0.75]).tolist()
        feature_diffs[column] = {
            "test_mean": float(test_real[column].mean()),
            "synthetic_mean": float(synthetic_model[column].mean()),
            "absolute_mean_difference": float(abs(test_real[column].mean() - synthetic_model[column].mean())),
            "test_std": float(test_real[column].std(ddof=0)),
            "synthetic_std": float(synthetic_model[column].std(ddof=0)),
            "absolute_median_difference": float(abs(q50_real - q50_synth)),
            "iqr_difference": float(abs((q75_real - q25_real) - (q75_synth - q25_synth))),
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
    encoded_train, scaling = _encode_for_distance(train_real, schema)
    encoded_synth, _ = _encode_for_distance(synthetic_model, schema, scaling=scaling)
    nn = _nearest_neighbor_distances(encoded_synth, encoded_train)
    duplicates = float(synthetic_model.duplicated().mean()) if len(synthetic_model) else 0.0
    marginal_fidelity = _marginal_fidelity_score(feature_diffs)
    normalized_marginal = _train_scale_normalized_marginal_score(train_real, test_real, synthetic_model, schema)
    dependence_score = None if correlation_l1 is None else float(1.0 / (1.0 + correlation_l1))
    selection_score = _selection_score(
        legacy_marginal=marginal_fidelity,
        normalized_marginal=normalized_marginal,
        dependence=dependence_score,
    )
    return EvaluationResult(
        method=method,
        feature_differences=feature_diffs,
        correlation_l1=correlation_l1,
        duplicate_fraction=duplicates,
        unique_fraction=1.0 - duplicates,
        nearest_neighbor_mean=float(np.mean(nn)) if len(nn) else 0.0,
        nearest_neighbor_min=float(np.min(nn)) if len(nn) else 0.0,
        marginal_fidelity_score=marginal_fidelity,
        train_scale_normalized_marginal_fidelity=normalized_marginal,
        dependence_preservation_score=dependence_score,
        selection_score=selection_score,
        objective_metrics={
            "data_generation_fidelity": marginal_fidelity,
            "train_scale_normalized_marginal_fidelity": normalized_marginal,
            "dependence_preservation": dependence_score if dependence_score is not None else 0.0,
            "selection_score": selection_score,
        },
        notes=(
            "Nearest-neighbor distances are memorization proxies, not formal privacy guarantees.",
            "Bootstrap baseline is evaluated with the same metrics for fairness.",
            "Selection score is 0.45 train-scale-normalized marginal fidelity, 0.35 dependence preservation when valid, and 0.20 legacy marginal fidelity.",
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


def _encode_for_distance(
    frame: pd.DataFrame,
    schema: TabularVaeSchema,
    *,
    scaling: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    columns: list[np.ndarray] = []
    resolved_scaling = dict(scaling or {})
    for column in schema.continuous:
        values = frame[column].astype(float).to_numpy()
        if column not in resolved_scaling:
            std = values.std() if values.std() > 1e-12 else 1.0
            resolved_scaling[column] = (float(values.mean()), float(std))
        mean, std = resolved_scaling[column]
        columns.append(((values - mean) / std)[:, None])
    for column in schema.binary:
        low, high = schema.binary_values[column]
        columns.append(frame[column].map({low: 0.0, high: 1.0}).to_numpy(dtype=float)[:, None])
    encoded = np.concatenate(columns, axis=1) if columns else np.empty((len(frame), 0))
    return encoded, resolved_scaling


def _nearest_neighbor_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.size == 0 or reference.size == 0:
        return np.zeros((query.shape[0],), dtype=float)
    distances = np.sqrt(((query[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1)


def _marginal_fidelity_score(feature_diffs: dict[str, dict[str, float]]) -> float:
    errors: list[float] = []
    for metrics in feature_diffs.values():
        for key, value in metrics.items():
            if key.startswith("absolute_") or key.endswith("_difference"):
                errors.append(abs(value))
    if not errors:
        return 0.0
    return float(1.0 / (1.0 + np.mean(errors)))


def _train_scale_normalized_marginal_score(
    train_real: pd.DataFrame,
    test_real: pd.DataFrame,
    synthetic: pd.DataFrame,
    schema: TabularVaeSchema,
) -> float:
    errors: list[float] = []
    for column in schema.continuous:
        train_std = float(train_real[column].astype(float).std(ddof=0))
        scale = train_std if train_std > 1e-12 else 1.0
        errors.append(abs(float(test_real[column].mean() - synthetic[column].mean())) / scale)
        errors.append(abs(float(test_real[column].median() - synthetic[column].median())) / scale)
        train_iqr = float(train_real[column].quantile(0.75) - train_real[column].quantile(0.25))
        iqr_scale = train_iqr if train_iqr > 1e-12 else scale
        test_iqr = float(test_real[column].quantile(0.75) - test_real[column].quantile(0.25))
        synth_iqr = float(synthetic[column].quantile(0.75) - synthetic[column].quantile(0.25))
        errors.append(abs(test_iqr - synth_iqr) / iqr_scale)
    for column in schema.binary:
        _low, high = schema.binary_values[column]
        errors.append(abs(float((test_real[column] == high).mean() - (synthetic[column] == high).mean())))
    if not errors:
        return 0.0
    return float(1.0 / (1.0 + np.mean(errors)))


def _selection_score(
    *,
    legacy_marginal: float,
    normalized_marginal: float,
    dependence: float | None,
) -> float:
    if dependence is None:
        return float(0.75 * normalized_marginal + 0.25 * legacy_marginal)
    return float(0.45 * normalized_marginal + 0.35 * dependence + 0.20 * legacy_marginal)
