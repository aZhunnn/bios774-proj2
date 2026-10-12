"""Dataset abstractions used before model-specific training exists."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class DatasetKind(str, Enum):
    TABULAR = "tabular"
    IMAGE = "image"
    SINGLE_CELL = "single_cell"
    UNKNOWN = "unknown"


class FeatureType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FeatureSchema:
    name: str
    feature_type: FeatureType
    missing_count: int = 0
    unique_count: int | None = None


@dataclass(frozen=True)
class DatasetHandle:
    """Lightweight dataset reference plus optional in-memory records.

    The handle intentionally avoids committing the project to pandas, AnnData,
    or PyTorch Dataset yet. Adapters for those formats can populate this same
    contract in later phases.
    """

    name: str
    kind: DatasetKind
    records: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    path: Path | None = None
    target_column: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.records and self.path is None:
            raise ValueError("DatasetHandle requires records or a path.")
        if self.records:
            object.__setattr__(self, "records", tuple(dict(row) for row in self.records))

    @property
    def sample_count(self) -> int | None:
        return len(self.records) if self.records else None

    @property
    def feature_names(self) -> tuple[str, ...]:
        if not self.records:
            return tuple()
        names: list[str] = []
        for row in self.records:
            for key in row:
                if key not in names:
                    names.append(key)
        return tuple(names)

    @classmethod
    def from_rows(
        cls,
        name: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        kind: DatasetKind = DatasetKind.TABULAR,
        target_column: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DatasetHandle":
        return cls(
            name=name,
            kind=kind,
            records=tuple(rows),
            target_column=target_column,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class DatasetProfile:
    dataset_name: str
    kind: DatasetKind
    n_samples: int
    n_features: int
    feature_schema: tuple[FeatureSchema, ...]
    target_column: str | None = None
    class_balance: Mapping[str, int] = field(default_factory=dict)
    missing_values: Mapping[str, int] = field(default_factory=dict)
    role_exclusions: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_missing_values(self) -> bool:
        return any(count > 0 for count in self.missing_values.values())

    @property
    def is_labeled(self) -> bool:
        return self.target_column is not None
