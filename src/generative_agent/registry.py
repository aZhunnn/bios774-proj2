"""Capability registry for planned generative modeling backends."""

from __future__ import annotations

from dataclasses import dataclass

from generative_agent.data import DatasetKind
from generative_agent.objectives import ObjectiveType


@dataclass(frozen=True)
class ModelCapability:
    name: str
    family: str
    training_framework: str
    dataset_kinds: tuple[DatasetKind, ...]
    objectives: tuple[ObjectiveType, ...]
    phase: int
    implemented: bool = False
    modalities: tuple[str, ...] = ("tabular",)
    feature_types: tuple[str, ...] = ("continuous", "binary")
    operations: tuple[str, ...] = ("train", "generate", "evaluate")
    computational_requirements: str = "cpu"
    limitations: tuple[str, ...] = ()


class CapabilityRegistry:
    def __init__(self, capabilities: tuple[ModelCapability, ...] | None = None) -> None:
        self._capabilities = capabilities or default_capabilities()

    def candidates(
        self,
        dataset_kind: DatasetKind,
        objective_type: ObjectiveType,
        *,
        implemented_only: bool = False,
    ) -> tuple[ModelCapability, ...]:
        matches = [
            capability
            for capability in self._capabilities
            if dataset_kind in capability.dataset_kinds
            and objective_type in capability.objectives
            and (capability.implemented or not implemented_only)
        ]
        return tuple(sorted(matches, key=lambda item: (item.phase, item.family, item.name)))

    def all(self) -> tuple[ModelCapability, ...]:
        return self._capabilities


def default_capabilities() -> tuple[ModelCapability, ...]:
    common_objectives = (
        ObjectiveType.DATA_GENERATION,
        ObjectiveType.IMPUTATION,
        ObjectiveType.CLASS_BALANCING,
        ObjectiveType.REPRESENTATION_LEARNING,
    )
    return (
        ModelCapability(
            name="tabular_vae",
            family="VAE",
            training_framework="pytorch",
            dataset_kinds=(DatasetKind.TABULAR, DatasetKind.SINGLE_CELL),
            objectives=common_objectives,
            phase=2,
            implemented=True,
            modalities=("tabular",),
            feature_types=("continuous", "binary"),
            limitations=("Gaussian continuous and Bernoulli binary observation assumptions.",),
        ),
        ModelCapability(
            name="tabular_flow",
            family="Normalizing Flow",
            training_framework="pytorch",
            dataset_kinds=(DatasetKind.TABULAR, DatasetKind.SINGLE_CELL),
            objectives=(ObjectiveType.DATA_GENERATION, ObjectiveType.UNCERTAINTY_QUANTIFICATION),
            phase=2,
            implemented=True,
            modalities=("tabular",),
            feature_types=("continuous",),
            limitations=("Continuous-only features; no missing values; likelihood is used only within flow-family selection.",),
        ),
        ModelCapability(
            name="tabular_gmm",
            family="Gaussian Mixture Model",
            training_framework="scikit-learn",
            dataset_kinds=(DatasetKind.TABULAR, DatasetKind.SINGLE_CELL),
            objectives=(ObjectiveType.DATA_GENERATION,),
            phase=6,
            implemented=True,
            modalities=("tabular",),
            feature_types=("continuous",),
            operations=("train", "generate", "evaluate", "checkpoint"),
            computational_requirements="cpu",
            limitations=("Continuous-only features; component count and covariance regularization affect stability.",),
        ),
        ModelCapability(
            name="tabular_copula",
            family="Gaussian Copula",
            training_framework="numpy scipy",
            dataset_kinds=(DatasetKind.TABULAR, DatasetKind.SINGLE_CELL),
            objectives=(ObjectiveType.DATA_GENERATION,),
            phase=6,
            implemented=True,
            modalities=("tabular",),
            feature_types=("continuous",),
            operations=("train", "generate", "evaluate", "checkpoint"),
            computational_requirements="cpu",
            limitations=("Continuous-only features; empirical marginals do not extrapolate tails and Gaussian copula does not capture arbitrary nonlinear or tail dependence.",),
        ),
        ModelCapability(
            name="image_vae",
            family="VAE",
            training_framework="pytorch",
            dataset_kinds=(DatasetKind.IMAGE,),
            objectives=(ObjectiveType.DATA_GENERATION, ObjectiveType.REPRESENTATION_LEARNING),
            phase=5,
            implemented=True,
            modalities=("image",),
            feature_types=("pixel_continuous",),
            operations=("train", "generate", "reconstruct", "evaluate"),
            computational_requirements="cpu_small_image",
            limitations=("Unconditional small-image VAE; labels are metadata only.",),
        ),
        ModelCapability(
            name="image_diffusion",
            family="Diffusion",
            training_framework="pytorch",
            dataset_kinds=(DatasetKind.IMAGE,),
            objectives=(ObjectiveType.DATA_GENERATION, ObjectiveType.CLASS_BALANCING),
            phase=3,
        ),
        ModelCapability(
            name="image_gan",
            family="GAN",
            training_framework="pytorch",
            dataset_kinds=(DatasetKind.IMAGE,),
            objectives=(ObjectiveType.DATA_GENERATION,),
            phase=5,
            implemented=True,
            modalities=("image",),
            feature_types=("pixel_continuous",),
            operations=("train", "generate", "evaluate"),
            computational_requirements="cpu_small_image",
            limitations=("Unconditional small-image GAN; adversarial losses are not cross-family quality metrics.",),
        ),
    )
