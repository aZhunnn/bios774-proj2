"""Training-only preprocessing contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from generative_agent.splitting import SplitManifest


class Preprocessor(Protocol):
    fitted_on: tuple[int, ...] | None

    def fit(
        self,
        records: Sequence[Mapping[str, Any]],
        train_indices: Sequence[int],
    ) -> "Preprocessor":
        ...

    def transform(
        self,
        records: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
    ) -> tuple[dict[str, Any], ...]:
        ...


@dataclass
class TabularPreprocessor:
    """Simple preprocessing contract that stores statistics from train only."""

    fitted_on: tuple[int, ...] | None = None
    numeric_means: dict[str, float] | None = None
    categorical_modes: dict[str, str] | None = None

    def fit(
        self,
        records: Sequence[Mapping[str, Any]],
        train_indices: Sequence[int],
    ) -> "TabularPreprocessor":
        if not train_indices:
            raise ValueError("Cannot fit preprocessing without training indices.")
        self.fitted_on = tuple(train_indices)
        train_rows = [records[index] for index in train_indices]
        columns = sorted({key for row in train_rows for key in row})
        self.numeric_means = {}
        self.categorical_modes = {}
        for column in columns:
            values = [_clean(row.get(column)) for row in train_rows]
            observed = [value for value in values if value is not None]
            numbers = [_as_float(value) for value in observed]
            if numbers and None not in numbers and len(numbers) == len(observed):
                self.numeric_means[column] = sum(numbers) / len(numbers)
            else:
                self.categorical_modes[column] = _mode([str(value) for value in observed])
        return self

    def fit_from_split(
        self,
        records: Sequence[Mapping[str, Any]],
        manifest: SplitManifest,
        fit_indices: Sequence[int] | None = None,
    ) -> "TabularPreprocessor":
        requested = tuple(manifest.indices.train if fit_indices is None else fit_indices)
        holdout = set(manifest.indices.validation + manifest.indices.test)
        leakage = sorted(set(requested).intersection(holdout))
        if leakage:
            raise ValueError(
                "Preprocessing fit indices include validation/test rows: "
                + ", ".join(str(index) for index in leakage)
            )
        return self.fit(records, requested)

    def transform(
        self,
        records: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
    ) -> tuple[dict[str, Any], ...]:
        if self.fitted_on is None or self.numeric_means is None or self.categorical_modes is None:
            raise RuntimeError("Preprocessor must be fit before transform.")
        transformed = []
        for index in indices:
            row = dict(records[index])
            for column, mean in self.numeric_means.items():
                if _clean(row.get(column)) is None:
                    row[column] = mean
            for column, mode in self.categorical_modes.items():
                if _clean(row.get(column)) is None:
                    row[column] = mode
            transformed.append(row)
        return tuple(transformed)


def _clean(value: Any) -> Any:
    return None if value in {None, "", "NA", "N/A", "nan", "NaN"} else value


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mode(values: list[str]) -> str:
    if not values:
        return "missing"
    return sorted(values, key=lambda value: (-values.count(value), value))[0]
