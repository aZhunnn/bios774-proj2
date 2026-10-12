"""Deterministic train/validation/test splitting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import random
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 2026
    stratify_column: str | None = None

    def __post_init__(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Split fractions must sum to 1.0.")
        if min(self.train_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ValueError("All split fractions must be positive.")


@dataclass(frozen=True)
class SplitIndices:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    @property
    def all_indices(self) -> tuple[int, ...]:
        return self.train + self.validation + self.test


@dataclass(frozen=True)
class SplitManifest:
    indices: SplitIndices
    config: SplitConfig
    dataset_fingerprint: str


class DatasetSplitter:
    def manifest(
        self,
        records: Sequence[Mapping[str, object]],
        config: SplitConfig,
    ) -> SplitManifest:
        return SplitManifest(
            indices=self.split(records, config),
            config=config,
            dataset_fingerprint=_fingerprint(records),
        )

    def split(
        self,
        records: Sequence[Mapping[str, object]],
        config: SplitConfig,
    ) -> SplitIndices:
        if len(records) < 3:
            raise ValueError("At least three records are required for train/validation/test splits.")

        if config.stratify_column:
            groups: dict[str, list[int]] = defaultdict(list)
            for index, row in enumerate(records):
                groups[str(row.get(config.stratify_column))].append(index)
            splits = [self._split_indices(indices, config) for _, indices in sorted(groups.items())]
            return SplitIndices(
                train=tuple(sorted(i for split in splits for i in split.train)),
                validation=tuple(sorted(i for split in splits for i in split.validation)),
                test=tuple(sorted(i for split in splits for i in split.test)),
            )

        return self._split_indices(list(range(len(records))), config)

    def _split_indices(self, indices: list[int], config: SplitConfig) -> SplitIndices:
        rng = random.Random(config.seed)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n = len(shuffled)
        train_end = max(1, round(n * config.train_fraction))
        validation_end = max(train_end + 1, train_end + round(n * config.validation_fraction))
        validation_end = min(validation_end, n - 1)
        return SplitIndices(
            train=tuple(sorted(shuffled[:train_end])),
            validation=tuple(sorted(shuffled[train_end:validation_end])),
            test=tuple(sorted(shuffled[validation_end:])),
        )


def _fingerprint(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in records:
        for key, value in sorted(row.items()):
            digest.update(str(key).encode("utf-8"))
            digest.update(b"=")
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        digest.update(b"\n")
    return digest.hexdigest()
