"""CSV loading and tabular role metadata for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from generative_agent.data import DatasetHandle, DatasetKind


@dataclass(frozen=True)
class ColumnRoles:
    identifiers: tuple[str, ...] = field(default_factory=tuple)
    target: str | None = None
    conditioning: tuple[str, ...] = field(default_factory=tuple)

    def modeling_exclusions(self) -> set[str]:
        excluded = set(self.identifiers).union(self.conditioning)
        if self.target:
            excluded.add(self.target)
        return excluded


def load_csv_dataset(
    path: str | Path,
    *,
    name: str | None = None,
    roles: ColumnRoles | None = None,
) -> tuple[pd.DataFrame, DatasetHandle]:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path, sep=None, engine="python")
    selected_roles = roles or ColumnRoles()
    missing_roles = selected_roles.modeling_exclusions().difference(frame.columns)
    if missing_roles:
        raise ValueError(f"Role columns not found in CSV: {', '.join(sorted(missing_roles))}.")
    handle = DatasetHandle.from_rows(
        name=name or csv_path.stem,
        rows=frame.to_dict(orient="records"),
        kind=DatasetKind.TABULAR,
        target_column=selected_roles.target,
        metadata={
            "source_path": str(csv_path),
            "identifiers": selected_roles.identifiers,
            "conditioning": selected_roles.conditioning,
        },
    )
    return frame, handle
