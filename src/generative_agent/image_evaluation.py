"""Initial image generation and reconstruction evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "generative_agent_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from generative_agent.image_gan import ImageGAN
from generative_agent.image_vae import ConvImageVAE


@dataclass(frozen=True)
class ImageEvaluationResult:
    model_name: str
    selection_score: float
    pixel_mean_abs_error: float
    pixel_std_abs_error: float
    gradient_mean_abs_error: float
    gradient_std_abs_error: float
    local_patch_std_abs_error: float
    pairwise_distance_mean_abs_error: float
    feature_mmd: float
    reconstruction_mse: float | None
    pixel_mean_real: float
    pixel_mean_generated: float
    pixel_std_real: float
    pixel_std_generated: float
    duplicate_fraction: float
    nearest_neighbor_mean: float
    nearest_neighbor_min: float
    generation_seconds: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def image_metric_specification() -> dict[str, dict[str, object]]:
    return {
        "pixel_mean_abs_error": {
            "definition": "|mean(generated pixels) - mean(validation pixels)|",
            "direction": "lower is better",
            "normalization": "pixels are expected on [0, 1]",
            "sample_count_sensitivity": "global means stabilize quickly but can hide poor local structure",
            "limitations": "can favor bland images if used alone",
        },
        "pixel_std_abs_error": {
            "definition": "|std(generated pixels) - std(validation pixels)|",
            "direction": "lower is better",
            "normalization": "pixels are expected on [0, 1]",
            "sample_count_sensitivity": "sensitive to small sample batches and class mix",
            "limitations": "global variance does not prove realistic spatial organization",
        },
        "gradient_mean_abs_error": {
            "definition": "absolute error in mean finite-difference gradient magnitude",
            "direction": "lower is better",
            "normalization": "computed on [0, 1] pixels",
            "sample_count_sensitivity": "moderate; stabilizes with more images and pixels",
            "limitations": "detects edge amount, not medically meaningful structure",
        },
        "gradient_std_abs_error": {
            "definition": "absolute error in std of finite-difference gradient magnitude",
            "direction": "lower is better",
            "normalization": "computed on [0, 1] pixels",
            "sample_count_sensitivity": "moderate; noisy for tiny image samples",
            "limitations": "does not validate semantic features",
        },
        "local_patch_std_abs_error": {
            "definition": "absolute error in average 4x4 patch-level pixel standard deviation",
            "direction": "lower is better",
            "normalization": "computed on [0, 1] pixels",
            "sample_count_sensitivity": "sensitive to image size and local texture diversity",
            "limitations": "penalizes overly smooth images but does not identify realistic tissue morphology",
        },
        "pairwise_distance_mean_abs_error": {
            "definition": "absolute error in mean pairwise Euclidean distance between images",
            "direction": "lower is better",
            "normalization": "distances are divided by sqrt(number of pixels)",
            "sample_count_sensitivity": "high for small generated batches",
            "limitations": "measures diversity scale, not sample usefulness",
        },
        "feature_mmd": {
            "definition": "RBF-kernel MMD on handcrafted image summaries: channel means/stds, gradient summaries, local patch std, and quadrant means",
            "direction": "lower is better",
            "normalization": "features are standardized using validation-feature mean/std",
            "sample_count_sensitivity": "high for small batches; use repeated generation seeds",
            "limitations": "handcrafted features are not a validated pathology quality metric",
        },
        "duplicate_fraction": {
            "definition": "1 - unique rounded generated images / generated images",
            "direction": "lower is better",
            "normalization": "generated images rounded to 8-bit pixel values",
            "sample_count_sensitivity": "more reliable with larger generated batches",
            "limitations": "zero duplicates does not imply privacy",
        },
        "selection_score": {
            "definition": "1 / (1 + weighted error sum) using validation-only metrics",
            "direction": "higher is better for relative ranking",
            "normalization": "bounded in (0, 1], but not calibrated as probability or quality acceptance",
            "sample_count_sensitivity": "depends on generated sample count and random seed",
            "limitations": "heuristic ranking score; no threshold establishes clinical or release quality",
        },
    }


def evaluate_image_vae(
    *,
    model: ConvImageVAE,
    train_dataset,
    validation_dataset,
    n_generated: int,
    seed: int,
    grid_path: Path,
) -> ImageEvaluationResult:
    import time

    start = time.perf_counter()
    generated = model.sample(n_generated, seed=seed).detach().cpu()
    generation_seconds = time.perf_counter() - start
    common = evaluate_generated_images(
        model_name="image_vae",
        generated=generated,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        generation_seconds=generation_seconds,
        grid_path=grid_path,
    )
    return _with_reconstruction(common, _reconstruction_mse(model, validation_dataset))


def evaluate_image_gan(
    *,
    model: ImageGAN,
    train_dataset,
    validation_dataset,
    n_generated: int,
    seed: int,
    grid_path: Path,
) -> ImageEvaluationResult:
    import time

    start = time.perf_counter()
    generated = model.sample(n_generated, seed=seed).detach().cpu()
    generation_seconds = time.perf_counter() - start
    return evaluate_generated_images(
        model_name="image_gan",
        generated=generated,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        generation_seconds=generation_seconds,
        grid_path=grid_path,
    )


def evaluate_generated_images(
    *,
    model_name: str,
    generated: torch.Tensor,
    train_dataset,
    validation_dataset,
    generation_seconds: float,
    grid_path: Path,
) -> ImageEvaluationResult:
    real_train = _dataset_images(train_dataset)
    real_val = _dataset_images(validation_dataset)
    duplicates = _duplicate_fraction(generated)
    nn = _nearest_neighbor(generated, real_train)
    pixel_mean_generated = float(generated.mean())
    pixel_std_generated = float(generated.std())
    pixel_mean_real = float(real_val.mean())
    pixel_std_real = float(real_val.std())
    gradient_real = _gradient_stats(real_val)
    gradient_generated = _gradient_stats(generated)
    local_patch_std_abs_error = abs(_local_patch_std(real_val) - _local_patch_std(generated))
    pairwise_distance_mean_abs_error = abs(_pairwise_distance_mean(real_val) - _pairwise_distance_mean(generated))
    feature_mmd = _feature_mmd(real_val, generated)
    pixel_mean_abs_error = abs(pixel_mean_generated - pixel_mean_real)
    pixel_std_abs_error = abs(pixel_std_generated - pixel_std_real)
    gradient_mean_abs_error = abs(gradient_generated[0] - gradient_real[0])
    gradient_std_abs_error = abs(gradient_generated[1] - gradient_real[1])
    selection_score = _selection_score(
        pixel_mean_abs_error=pixel_mean_abs_error,
        pixel_std_abs_error=pixel_std_abs_error,
        gradient_mean_abs_error=gradient_mean_abs_error,
        gradient_std_abs_error=gradient_std_abs_error,
        local_patch_std_abs_error=local_patch_std_abs_error,
        pairwise_distance_mean_abs_error=pairwise_distance_mean_abs_error,
        feature_mmd=feature_mmd,
        duplicate_fraction=duplicates,
    )
    _save_grid(generated[: min(32, len(generated))], grid_path)
    return ImageEvaluationResult(
        model_name=model_name,
        selection_score=selection_score,
        pixel_mean_abs_error=pixel_mean_abs_error,
        pixel_std_abs_error=pixel_std_abs_error,
        gradient_mean_abs_error=gradient_mean_abs_error,
        gradient_std_abs_error=gradient_std_abs_error,
        local_patch_std_abs_error=local_patch_std_abs_error,
        pairwise_distance_mean_abs_error=pairwise_distance_mean_abs_error,
        feature_mmd=feature_mmd,
        reconstruction_mse=None,
        pixel_mean_real=pixel_mean_real,
        pixel_mean_generated=pixel_mean_generated,
        pixel_std_real=pixel_std_real,
        pixel_std_generated=pixel_std_generated,
        duplicate_fraction=duplicates,
        nearest_neighbor_mean=float(nn.mean()) if len(nn) else 0.0,
        nearest_neighbor_min=float(nn.min()) if len(nn) else 0.0,
        generation_seconds=generation_seconds,
        notes=(
            "Reconstruction metrics evaluate autoencoding behavior, not unconditional generation quality.",
            "Nearest-neighbor diagnostics are memorization proxies, not privacy guarantees.",
            "Selection score is validation-only and combines pixel and simple gradient summary errors with a duplicate penalty; it does not use VAE ELBO or GAN discriminator loss.",
            "The handcrafted feature MMD is not a validated pathology quality metric and is not clinically meaningful by itself.",
            "No numeric threshold in this report establishes adequate image quality for clinical use or external release.",
        ),
    )


def _with_reconstruction(result: ImageEvaluationResult, reconstruction_mse: float) -> ImageEvaluationResult:
    return ImageEvaluationResult(
        model_name=result.model_name,
        selection_score=result.selection_score,
        pixel_mean_abs_error=result.pixel_mean_abs_error,
        pixel_std_abs_error=result.pixel_std_abs_error,
        gradient_mean_abs_error=result.gradient_mean_abs_error,
        gradient_std_abs_error=result.gradient_std_abs_error,
        local_patch_std_abs_error=result.local_patch_std_abs_error,
        pairwise_distance_mean_abs_error=result.pairwise_distance_mean_abs_error,
        feature_mmd=result.feature_mmd,
        reconstruction_mse=reconstruction_mse,
        pixel_mean_real=result.pixel_mean_real,
        pixel_mean_generated=result.pixel_mean_generated,
        pixel_std_real=result.pixel_std_real,
        pixel_std_generated=result.pixel_std_generated,
        duplicate_fraction=result.duplicate_fraction,
        nearest_neighbor_mean=result.nearest_neighbor_mean,
        nearest_neighbor_min=result.nearest_neighbor_min,
        generation_seconds=result.generation_seconds,
        notes=result.notes,
    )


def _dataset_images(dataset) -> torch.Tensor:
    return torch.stack([dataset[index][0] for index in range(len(dataset))])


def _reconstruction_mse(model: ConvImageVAE, dataset) -> float:
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    values = []
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for batch, _labels in loader:
            batch = batch.to(device)
            recon, _mu, _logvar = model(batch)
            values.append(torch.mean((batch - recon) ** 2).detach().cpu())
    return float(torch.stack(values).mean())


def _duplicate_fraction(images: torch.Tensor) -> float:
    flat = (images.numpy().reshape(len(images), -1) * 255).round().astype(np.uint8)
    unique = np.unique(flat, axis=0)
    return float(1.0 - len(unique) / len(flat)) if len(flat) else 0.0


def _nearest_neighbor(query: torch.Tensor, reference: torch.Tensor) -> np.ndarray:
    q = query.numpy().reshape(len(query), -1)
    r = reference.numpy().reshape(len(reference), -1)
    distances = np.sqrt(((q[:, None, :] - r[None, :, :]) ** 2).sum(axis=2))
    return distances.min(axis=1)


def _gradient_stats(images: torch.Tensor) -> tuple[float, float]:
    horizontal = images[:, :, :, 1:] - images[:, :, :, :-1]
    vertical = images[:, :, 1:, :] - images[:, :, :-1, :]
    magnitudes = torch.cat([horizontal.abs().flatten(), vertical.abs().flatten()])
    return float(magnitudes.mean()), float(magnitudes.std())


def _local_patch_std(images: torch.Tensor, patch_size: int = 4) -> float:
    height = images.shape[-2] - images.shape[-2] % patch_size
    width = images.shape[-1] - images.shape[-1] % patch_size
    cropped = images[:, :, :height, :width]
    patches = cropped.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    return float(patches.contiguous().view(-1, patch_size * patch_size).std(dim=1).mean())


def _pairwise_distance_mean(images: torch.Tensor, max_images: int = 64) -> float:
    selected = images[:max_images].reshape(min(len(images), max_images), -1)
    if len(selected) < 2:
        return 0.0
    distances = torch.pdist(selected, p=2) / float(selected.shape[1] ** 0.5)
    return float(distances.mean())


def _image_features(images: torch.Tensor) -> torch.Tensor:
    channel_means = images.mean(dim=(2, 3))
    channel_stds = images.std(dim=(2, 3))
    horizontal = (images[:, :, :, 1:] - images[:, :, :, :-1]).abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    vertical = (images[:, :, 1:, :] - images[:, :, :-1, :]).abs().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
    patch = []
    for image in images:
        patch.append(_local_patch_std(image.unsqueeze(0)))
    patch_tensor = torch.tensor(patch, dtype=images.dtype).unsqueeze(1)
    h_mid = images.shape[-2] // 2
    w_mid = images.shape[-1] // 2
    quadrants = torch.stack(
        [
            images[:, :, :h_mid, :w_mid].mean(dim=(1, 2, 3)),
            images[:, :, :h_mid, w_mid:].mean(dim=(1, 2, 3)),
            images[:, :, h_mid:, :w_mid].mean(dim=(1, 2, 3)),
            images[:, :, h_mid:, w_mid:].mean(dim=(1, 2, 3)),
        ],
        dim=1,
    )
    return torch.cat([channel_means, channel_stds, horizontal, vertical, patch_tensor, quadrants], dim=1)


def _feature_mmd(real: torch.Tensor, generated: torch.Tensor) -> float:
    real_features = _image_features(real)
    generated_features = _image_features(generated)
    mean = real_features.mean(dim=0, keepdim=True)
    std = real_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (real_features - mean) / std
    y = (generated_features - mean) / std
    combined = torch.cat([x, y], dim=0)
    median_distance = torch.pdist(combined).median().clamp_min(1e-6)
    gamma = 1.0 / (2.0 * median_distance.pow(2))
    k_xx = torch.exp(-gamma * torch.cdist(x, x).pow(2)).mean()
    k_yy = torch.exp(-gamma * torch.cdist(y, y).pow(2)).mean()
    k_xy = torch.exp(-gamma * torch.cdist(x, y).pow(2)).mean()
    return float(torch.clamp(k_xx + k_yy - 2.0 * k_xy, min=0.0))


def _selection_score(
    *,
    pixel_mean_abs_error: float,
    pixel_std_abs_error: float,
    gradient_mean_abs_error: float,
    gradient_std_abs_error: float,
    local_patch_std_abs_error: float,
    pairwise_distance_mean_abs_error: float,
    feature_mmd: float,
    duplicate_fraction: float,
) -> float:
    error = (
        pixel_mean_abs_error
        + pixel_std_abs_error
        + 0.5 * gradient_mean_abs_error
        + 0.5 * gradient_std_abs_error
        + local_patch_std_abs_error
        + pairwise_distance_mean_abs_error
        + 0.25 * feature_mmd
        + duplicate_fraction
    )
    return float(1.0 / (1.0 + error))


def _save_grid(images: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images_np = images.numpy()
    n = len(images_np)
    cols = min(8, max(1, n))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for idx, ax in enumerate(axes_arr.flat):
        ax.axis("off")
        if idx < n:
            image = np.transpose(images_np[idx], (1, 2, 0))
            if image.shape[-1] == 1:
                ax.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(np.clip(image, 0, 1))
    fig.tight_layout(pad=0.05)
    fig.savefig(path, dpi=150)
    plt.close(fig)
