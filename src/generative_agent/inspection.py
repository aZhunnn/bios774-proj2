"""Dataset inspection routines."""

from __future__ import annotations

from collections import Counter
from typing import Any

from generative_agent.data import (
    DatasetHandle,
    DatasetKind,
    DatasetProfile,
    FeatureSchema,
    FeatureType,
)


MISSING_VALUES = {"", "NA", "N/A", "nan", "NaN", None}


class DatasetInspector:
    """Infers a deterministic, compact profile from a dataset handle."""

    def inspect(self, dataset: DatasetHandle) -> DatasetProfile:
        if dataset.records:
            return self._inspect_records(dataset)
        return DatasetProfile(
            dataset_name=dataset.name,
            kind=dataset.kind,
            n_samples=0,
            n_features=0,
            feature_schema=tuple(),
            target_column=dataset.target_column,
            role_exclusions=_role_exclusions(dataset),
            notes=("Path-backed dataset inspection is deferred to format adapters.",),
        )

    def _inspect_records(self, dataset: DatasetHandle) -> DatasetProfile:
        feature_names = dataset.feature_names
        schema: list[FeatureSchema] = []
        missing_values: dict[str, int] = {}

        for name in feature_names:
            values = [row.get(name) for row in dataset.records]
            missing = sum(1 for value in values if value in MISSING_VALUES)
            observed = [value for value in values if value not in MISSING_VALUES]
            missing_values[name] = missing
            schema.append(
                FeatureSchema(
                    name=name,
                    feature_type=_infer_feature_type(observed),
                    missing_count=missing,
                    unique_count=len(set(map(str, observed))),
                )
            )

        class_balance: dict[str, int] = {}
        if dataset.target_column and dataset.target_column in feature_names:
            class_balance = dict(
                sorted(
                    Counter(str(row.get(dataset.target_column)) for row in dataset.records).items()
                )
            )

        notes = _profile_notes(dataset.kind, schema, missing_values, class_balance)
        return DatasetProfile(
            dataset_name=dataset.name,
            kind=dataset.kind,
            n_samples=len(dataset.records),
            n_features=len(feature_names),
            feature_schema=tuple(schema),
            target_column=dataset.target_column,
            class_balance=class_balance,
            missing_values=missing_values,
            role_exclusions=_role_exclusions(dataset),
            notes=notes,
        )


def _infer_feature_type(values: list[Any]) -> FeatureType:
    if not values:
        return FeatureType.UNKNOWN
    if all(isinstance(value, bool) for value in values):
        return FeatureType.BOOLEAN
    if all(_is_number(value) for value in values):
        return FeatureType.NUMERIC
    if all(isinstance(value, str) for value in values):
        unique = {str(value) for value in values}
        average_length = sum(len(value) for value in values) / len(values)
        if len(unique) <= max(20, int(len(values) * 0.2)) and average_length <= 50:
            return FeatureType.CATEGORICAL
        return FeatureType.TEXT
    if all(_is_categorical_scalar(value) for value in values):
        unique = {str(value) for value in values}
        if len(unique) <= max(20, int(len(values) * 0.2)):
            return FeatureType.CATEGORICAL
    return FeatureType.UNKNOWN


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False


def _is_categorical_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


def _profile_notes(
    kind: DatasetKind,
    schema: list[FeatureSchema],
    missing_values: dict[str, int],
    class_balance: dict[str, int],
) -> tuple[str, ...]:
    notes: list[str] = []
    if kind is DatasetKind.IMAGE:
        notes.append("Image dataset: convolutional or diffusion-family models may be relevant.")
    if kind is DatasetKind.SINGLE_CELL:
        notes.append("Single-cell dataset: count-aware preprocessing should be considered later.")
    if any(count > 0 for count in missing_values.values()):
        notes.append("Missing values detected; preprocessing must be fit on training data only.")
    if class_balance:
        counts = list(class_balance.values())
        if min(counts) / max(counts) < 0.5:
            notes.append("Class imbalance detected.")
    if not schema:
        notes.append("No feature schema could be inferred.")
    return tuple(notes)


def _role_exclusions(dataset: DatasetHandle) -> tuple[str, ...]:
    excluded = set()
    if dataset.target_column:
        excluded.add(dataset.target_column)
    excluded.update(dataset.metadata.get("identifiers", ()))
    excluded.update(dataset.metadata.get("conditioning", ()))
    return tuple(sorted(excluded))
