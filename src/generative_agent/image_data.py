"""Generic image dataset loading and preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


ImageNormalization = Literal["unit", "minus_one_one"]
ReconstructionAssumption = Literal["gaussian_unit_variance"]


@dataclass(frozen=True)
class ImagePreprocessConfig:
    normalization: ImageNormalization = "unit"
    reconstruction_assumption: ReconstructionAssumption = "gaussian_unit_variance"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_test_samples: int | None = None
    seed: int = 2026


@dataclass(frozen=True)
class ImageDatasetProfile:
    name: str
    n_samples: int
    split_counts: dict[str, int]
    image_shape: tuple[int, int, int]
    channels: int
    value_range: tuple[float, float]
    label_counts: dict[str, int]
    labels_are_metadata: bool
    normalization: ImageNormalization
    reconstruction_assumption: ReconstructionAssumption
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ImageDatasetBundle:
    name: str
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset
    profile: ImageDatasetProfile
    preprocessing: dict[str, object]


class SplitNpzImageAdapter:
    """Loads split NPZ image datasets with train/val/test arrays.

    Expected keys are generic: train_images, train_labels, val_images,
    val_labels, test_images, test_labels. Labels are retained as metadata and
    are not used as conditioning variables in Phase 5A.
    """

    REQUIRED_KEYS = (
        "train_images",
        "train_labels",
        "val_images",
        "val_labels",
        "test_images",
        "test_labels",
    )

    def load(self, path: str | Path, *, name: str | None = None, config: ImagePreprocessConfig | None = None) -> ImageDatasetBundle:
        selected = config or ImagePreprocessConfig()
        npz_path = Path(path)
        arrays = np.load(npz_path)
        missing = [key for key in self.REQUIRED_KEYS if key not in arrays]
        if missing:
            raise ValueError(f"Image NPZ is missing required keys: {', '.join(missing)}.")

        train_images, train_labels = _prepare_split(arrays["train_images"], arrays["train_labels"], "train")
        val_images, val_labels = _prepare_split(arrays["val_images"], arrays["val_labels"], "validation")
        test_images, test_labels = _prepare_split(arrays["test_images"], arrays["test_labels"], "test")
        _validate_compatible_shapes(train_images, val_images, test_images)

        train_images, train_labels = _deterministic_subset(train_images, train_labels, selected.max_train_samples, selected.seed)
        val_images, val_labels = _deterministic_subset(val_images, val_labels, selected.max_validation_samples, selected.seed + 1)
        test_images, test_labels = _deterministic_subset(test_images, test_labels, selected.max_test_samples, selected.seed + 2)

        train_tensor = _normalize(train_images, selected.normalization)
        val_tensor = _normalize(val_images, selected.normalization)
        test_tensor = _normalize(test_images, selected.normalization)
        train_dataset = TensorDataset(train_tensor, torch.as_tensor(train_labels, dtype=torch.long))
        val_dataset = TensorDataset(val_tensor, torch.as_tensor(val_labels, dtype=torch.long))
        test_dataset = TensorDataset(test_tensor, torch.as_tensor(test_labels, dtype=torch.long))

        profile = _profile(
            name=name or npz_path.stem,
            train=train_images,
            validation=val_images,
            test=test_images,
            train_labels=train_labels,
            validation_labels=val_labels,
            test_labels=test_labels,
            config=selected,
        )
        return ImageDatasetBundle(
            name=profile.name,
            train=train_dataset,
            validation=val_dataset,
            test=test_dataset,
            profile=profile,
            preprocessing={
                "normalization": selected.normalization,
                "reconstruction_assumption": selected.reconstruction_assumption,
                "labels": "metadata_only_not_conditioning",
                "seed": selected.seed,
            },
        )


def _prepare_split(images: np.ndarray, labels: np.ndarray, split: str) -> tuple[np.ndarray, np.ndarray]:
    if images.ndim == 3:
        images = images[..., None]
    if images.ndim != 4:
        raise ValueError(f"{split} images must have shape N x H x W x C or N x H x W.")
    if images.shape[-1] not in (1, 3):
        raise ValueError(f"{split} images must have 1 or 3 channels; got {images.shape[-1]}.")
    if images.shape[1] != images.shape[2]:
        raise ValueError(f"{split} images must be square for the Phase 5A image VAE.")
    if images.shape[1] % 4 != 0:
        raise ValueError(f"{split} image height/width must be divisible by 4.")
    if not np.issubdtype(images.dtype, np.number):
        raise ValueError(f"{split} images must be numeric arrays.")
    labels = np.asarray(labels).reshape(-1)
    if len(labels) != len(images):
        raise ValueError(f"{split} labels length does not match image count.")
    return images, labels


def _validate_compatible_shapes(*splits: np.ndarray) -> None:
    shape = splits[0].shape[1:]
    for split in splits[1:]:
        if split.shape[1:] != shape:
            raise ValueError("All image splits must share the same H x W x C shape.")


def _deterministic_subset(images: np.ndarray, labels: np.ndarray, limit: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or limit >= len(images):
        return images, labels
    if limit <= 0:
        raise ValueError("Subset limits must be positive.")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(images), size=limit, replace=False))
    return images[selected], labels[selected]


def _normalize(images: np.ndarray, normalization: ImageNormalization) -> torch.Tensor:
    values = images.astype(np.float32)
    if values.min() < 0:
        raise ValueError("Negative raw image values are unsupported.")
    if values.max() > 1.0:
        values = values / 255.0
    if values.min() < 0 or values.max() > 1:
        raise ValueError("Images could not be normalized to [0, 1].")
    if normalization == "minus_one_one":
        values = values * 2.0 - 1.0
    elif normalization != "unit":
        raise ValueError(f"Unsupported image normalization: {normalization}.")
    values = np.transpose(values, (0, 3, 1, 2))
    return torch.as_tensor(values, dtype=torch.float32)


def _profile(
    *,
    name: str,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    config: ImagePreprocessConfig,
) -> ImageDatasetProfile:
    all_images = np.concatenate([train.reshape(len(train), -1), validation.reshape(len(validation), -1), test.reshape(len(test), -1)])
    labels = np.concatenate([train_labels, validation_labels, test_labels])
    unique, counts = np.unique(labels, return_counts=True)
    h, w, c = train.shape[1:]
    return ImageDatasetProfile(
        name=name,
        n_samples=int(len(train) + len(validation) + len(test)),
        split_counts={"train": int(len(train)), "validation": int(len(validation)), "test": int(len(test))},
        image_shape=(int(h), int(w), int(c)),
        channels=int(c),
        value_range=(float(all_images.min()), float(all_images.max())),
        label_counts={str(label): int(count) for label, count in zip(unique, counts)},
        labels_are_metadata=True,
        normalization=config.normalization,
        reconstruction_assumption=config.reconstruction_assumption,
        notes=(
            "Labels are retained as metadata and are not used for conditioning in Phase 5A.",
            "Gaussian unit-variance reconstruction loss is used on normalized pixel intensities.",
        ),
    )
