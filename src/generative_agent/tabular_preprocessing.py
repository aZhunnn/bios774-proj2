"""Train-only preprocessing for Phase 2 tabular VAE inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from generative_agent.splitting import SplitManifest
from generative_agent.tabular_io import ColumnRoles


FeatureKind = Literal["continuous", "binary"]


@dataclass(frozen=True)
class TabularVaeSchema:
    continuous: tuple[str, ...]
    binary: tuple[str, ...]
    excluded: tuple[str, ...]
    binary_values: dict[str, tuple[object, object]] = field(default_factory=dict)

    @property
    def model_columns(self) -> tuple[str, ...]:
        return self.continuous + self.binary


@dataclass(frozen=True)
class TabularVaePreprocessorMetadata:
    schema: TabularVaeSchema
    continuous_mean: dict[str, float]
    continuous_std: dict[str, float]
    fitted_train_indices: tuple[int, ...]
    missing_strategy: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TabularVaePreprocessor:
    """Encodes continuous and binary columns for a VAE.

    Continuous columns are standardized using training-set mean and standard
    deviation. Binary columns are mapped to 0/1 using training-set observed
    values. Missing values are rejected in Phase 2 to avoid implicit imputation.
    """

    def __init__(self, *, roles: ColumnRoles | None = None, missing_strategy: str = "reject") -> None:
        self.roles = roles or ColumnRoles()
        self.missing_strategy = missing_strategy
        self.metadata: TabularVaePreprocessorMetadata | None = None

    def fit(self, frame: pd.DataFrame, manifest: SplitManifest) -> "TabularVaePreprocessor":
        if self.missing_strategy != "reject":
            raise ValueError("Only missing_strategy='reject' is implemented in Phase 2.")
        train = frame.iloc[list(manifest.indices.train)].copy()
        excluded = tuple(sorted(self.roles.modeling_exclusions()))
        model_columns = [column for column in frame.columns if column not in excluded]
        if not model_columns:
            raise ValueError("No modelable columns remain after applying identifier/target/conditioning roles.")

        decision_indices = list(manifest.indices.train + manifest.indices.validation)
        missing = frame.iloc[decision_indices][model_columns].isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            details = ", ".join(f"{column}={count}" for column, count in missing.items())
            raise ValueError(
                "Missing values are not supported by Phase 2 preprocessing in train/validation data: "
                f"{details}."
            )

        continuous: list[str] = []
        binary: list[str] = []
        binary_values: dict[str, tuple[object, object]] = {}
        means: dict[str, float] = {}
        stds: dict[str, float] = {}

        for column in model_columns:
            values = train[column]
            unique = tuple(sorted(values.dropna().unique().tolist()))
            if len(unique) == 2 and _is_binary_compatible(unique):
                binary.append(column)
                binary_values[column] = unique
                continue
            if pd.api.types.is_numeric_dtype(values):
                continuous.append(column)
                means[column] = float(values.mean())
                std = float(values.std(ddof=0))
                stds[column] = std if std > 1e-12 else 1.0
                continue
            raise ValueError(
                f"Unsupported column '{column}'. Phase 2 supports continuous numeric and binary features only."
            )

        self.metadata = TabularVaePreprocessorMetadata(
            schema=TabularVaeSchema(
                continuous=tuple(continuous),
                binary=tuple(binary),
                excluded=excluded,
                binary_values=binary_values,
            ),
            continuous_mean=means,
            continuous_std=stds,
            fitted_train_indices=manifest.indices.train,
            missing_strategy=self.missing_strategy,
        )
        return self

    def transform(self, frame: pd.DataFrame, indices: tuple[int, ...] | list[int]) -> np.ndarray:
        metadata = self._metadata()
        subset = frame.iloc[list(indices)]
        missing = subset[list(metadata.schema.model_columns)].isna().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            details = ", ".join(f"{column}={count}" for column, count in missing.items())
            raise ValueError(f"Missing values are not supported during transform: {details}.")
        columns: list[np.ndarray] = []
        for column in metadata.schema.continuous:
            values = subset[column].astype(float).to_numpy()
            columns.append(((values - metadata.continuous_mean[column]) / metadata.continuous_std[column])[:, None])
        for column in metadata.schema.binary:
            low, high = metadata.schema.binary_values[column]
            values = subset[column].map({low: 0.0, high: 1.0})
            if values.isna().any():
                raise ValueError(f"Binary column '{column}' contains values not observed during training.")
            columns.append(values.to_numpy(dtype=np.float32)[:, None])
        if not columns:
            raise ValueError("No encoded model columns are available.")
        return np.concatenate(columns, axis=1).astype(np.float32)

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        metadata = self._metadata()
        expected = len(metadata.schema.model_columns)
        if encoded.shape[1] != expected:
            raise ValueError(f"Expected {expected} generated columns, received {encoded.shape[1]}.")
        data: dict[str, object] = {}
        offset = 0
        for column in metadata.schema.continuous:
            values = encoded[:, offset] * metadata.continuous_std[column] + metadata.continuous_mean[column]
            data[column] = values.astype(float)
            offset += 1
        for column in metadata.schema.binary:
            low, high = metadata.schema.binary_values[column]
            data[column] = np.where(encoded[:, offset] >= 0.5, high, low)
            offset += 1
        return pd.DataFrame(data, columns=list(metadata.schema.model_columns))

    def _metadata(self) -> TabularVaePreprocessorMetadata:
        if self.metadata is None:
            raise RuntimeError("Preprocessor must be fit before use.")
        return self.metadata


def _is_binary_compatible(values: tuple[object, object]) -> bool:
    return all(isinstance(value, (bool, int, float, str, np.integer, np.floating)) for value in values)
