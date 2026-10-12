"""Artifact interfaces for reports, plans, and future generated outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    kind: str
    path: Path
    metadata: dict[str, Any]


class ArtifactStore(Protocol):
    def save_json(self, name: str, payload: dict[str, Any], *, kind: str) -> ArtifactRecord:
        ...


class LocalArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, payload: dict[str, Any], *, kind: str) -> ArtifactRecord:
        safe_name = name.replace("/", "_").replace(" ", "_")
        path = self.root / f"{safe_name}.json"
        path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
        return ArtifactRecord(name=name, kind=kind, path=path, metadata={"format": "json"})


def _json_ready(value: Any) -> Any:
    if dataclass_is_instance(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def dataclass_is_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__") and not isinstance(value, type)
