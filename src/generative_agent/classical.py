"""Classical continuous-tabular generative models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class GmmConfig:
    n_components: int = 3
    covariance_type: str = "full"
    reg_covar: float = 1e-5
    max_iter: int = 200
    seed: int = 2026

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CopulaConfig:
    correlation_regularization: float = 1e-6
    min_u: float = 1e-6
    seed: int = 2026

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ClassicalTrainingResult:
    config: dict[str, object]
    diagnostics: dict[str, object]
    checkpoint_path: Path
    wall_seconds: float
    failed: bool = False
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TabularGMM:
    def __init__(self, config: GmmConfig) -> None:
        if config.n_components <= 0:
            raise ValueError("GMM n_components must be positive.")
        self.config = config
        self.model: GaussianMixture | None = None
        self.diagnostics: dict[str, object] = {}

    def fit(self, train_array: np.ndarray) -> "TabularGMM":
        _validate_continuous_array(train_array)
        model = GaussianMixture(
            n_components=min(self.config.n_components, len(train_array)),
            covariance_type=self.config.covariance_type,
            reg_covar=self.config.reg_covar,
            max_iter=self.config.max_iter,
            random_state=self.config.seed,
        )
        model.fit(train_array)
        self.model = model
        self.diagnostics = {
            "converged": bool(model.converged_),
            "n_iter": int(model.n_iter_),
            "lower_bound": float(model.lower_bound_),
            "weights": model.weights_.tolist(),
            "means": model.means_.tolist(),
            "covariance_type": model.covariance_type,
            "reg_covar": self.config.reg_covar,
        }
        return self

    def validation_score(self, validation_array: np.ndarray) -> float:
        _validate_continuous_array(validation_array)
        return float(self._model().score(validation_array))

    def sample(self, n_samples: int, *, seed: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        model = self._model()
        old_seed = model.random_state
        model.random_state = seed
        try:
            samples, _labels = model.sample(n_samples)
        finally:
            model.random_state = old_seed
        if not np.isfinite(samples).all():
            raise ValueError("GMM generated non-finite samples.")
        return samples.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"config": self.config, "model": self._model(), "diagnostics": self.diagnostics}, handle)

    def _model(self) -> GaussianMixture:
        if self.model is None:
            raise RuntimeError("GMM must be fit before use.")
        return self.model


class GaussianCopula:
    def __init__(self, config: CopulaConfig) -> None:
        self.config = config
        self.sorted_marginals: list[np.ndarray] = []
        self.correlation: np.ndarray | None = None
        self.diagnostics: dict[str, object] = {}

    def fit(self, train_array: np.ndarray) -> "GaussianCopula":
        _validate_continuous_array(train_array)
        n, dim = train_array.shape
        latent_columns = []
        marginals = []
        for column in range(dim):
            values = train_array[:, column].astype(float)
            marginals.append(np.sort(values))
            ranks = rankdata(values, method="average")
            u = np.clip((ranks - 0.5) / n, self.config.min_u, 1.0 - self.config.min_u)
            latent_columns.append(norm.ppf(u))
        latent = np.column_stack(latent_columns)
        corr = np.corrcoef(latent, rowvar=False)
        if dim == 1:
            corr = np.ones((1, 1), dtype=float)
        corr = _nearest_positive_correlation(corr, self.config.correlation_regularization)
        self.sorted_marginals = marginals
        self.correlation = corr
        self.diagnostics = {
            "n_features": dim,
            "min_eigenvalue": float(np.linalg.eigvalsh(corr).min()),
            "correlation_regularization": self.config.correlation_regularization,
            "marginal_estimator": "empirical_quantile_linear_interpolation",
            "limitations": (
                "Empirical marginals reproduce observed support approximately and do not extrapolate tails beyond training extremes.",
                "Gaussian copula captures linear latent dependence, not arbitrary nonlinear or tail dependence.",
            ),
        }
        return self

    def validation_score(self, validation_array: np.ndarray) -> float:
        _validate_continuous_array(validation_array)
        generated_reference = self.sample(len(validation_array), seed=self.config.seed + 99)
        corr_error = np.abs(np.corrcoef(validation_array, rowvar=False) - np.corrcoef(generated_reference, rowvar=False))
        return float(-np.nan_to_num(corr_error, nan=0.0).mean())

    def sample(self, n_samples: int, *, seed: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        corr = self._correlation()
        rng = np.random.default_rng(seed)
        latent = rng.multivariate_normal(np.zeros(corr.shape[0]), corr, size=n_samples)
        u = np.clip(norm.cdf(latent), self.config.min_u, 1.0 - self.config.min_u)
        samples = np.column_stack([_inverse_empirical(values, u[:, idx]) for idx, values in enumerate(self.sorted_marginals)])
        if not np.isfinite(samples).all():
            raise ValueError("Gaussian copula generated non-finite samples.")
        return samples.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(
                {
                    "config": self.config,
                    "sorted_marginals": self.sorted_marginals,
                    "correlation": self._correlation(),
                    "diagnostics": self.diagnostics,
                },
                handle,
            )

    def _correlation(self) -> np.ndarray:
        if self.correlation is None:
            raise RuntimeError("Gaussian copula must be fit before use.")
        return self.correlation


def load_gmm_checkpoint(path: Path) -> TabularGMM:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    model = TabularGMM(payload["config"])
    model.model = payload["model"]
    model.diagnostics = payload["diagnostics"]
    return model


def load_copula_checkpoint(path: Path) -> GaussianCopula:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    model = GaussianCopula(payload["config"])
    model.sorted_marginals = payload["sorted_marginals"]
    model.correlation = payload["correlation"]
    model.diagnostics = payload["diagnostics"]
    return model


def fit_gmm(train_array: np.ndarray, config: GmmConfig, checkpoint_path: Path) -> tuple[ClassicalTrainingResult, TabularGMM]:
    start = time.perf_counter()
    model = TabularGMM(config)
    try:
        model.fit(train_array)
        model.save(checkpoint_path)
        return ClassicalTrainingResult(config.to_dict(), model.diagnostics, checkpoint_path, time.perf_counter() - start), model
    except Exception as exc:
        return ClassicalTrainingResult(config.to_dict(), {}, checkpoint_path, time.perf_counter() - start, True, str(exc)), model


def fit_copula(train_array: np.ndarray, config: CopulaConfig, checkpoint_path: Path) -> tuple[ClassicalTrainingResult, GaussianCopula]:
    start = time.perf_counter()
    model = GaussianCopula(config)
    try:
        model.fit(train_array)
        model.save(checkpoint_path)
        return ClassicalTrainingResult(config.to_dict(), model.diagnostics, checkpoint_path, time.perf_counter() - start), model
    except Exception as exc:
        return ClassicalTrainingResult(config.to_dict(), {}, checkpoint_path, time.perf_counter() - start, True, str(exc)), model


def propose_gmm_configs(*, n_train: int, budget_count: int, seed: int) -> tuple[GmmConfig, ...]:
    components = [1, 2, min(4, max(1, n_train // 50))]
    configs: list[GmmConfig] = []
    for n_components in dict.fromkeys(components):
        configs.append(GmmConfig(n_components=n_components, covariance_type="full", reg_covar=1e-5, seed=seed))
        configs.append(GmmConfig(n_components=n_components, covariance_type="diag", reg_covar=1e-5, seed=seed + 1))
    return tuple(configs[: max(0, budget_count)])


def propose_copula_configs(*, budget_count: int, seed: int) -> tuple[CopulaConfig, ...]:
    configs = (
        CopulaConfig(correlation_regularization=1e-6, seed=seed),
        CopulaConfig(correlation_regularization=1e-4, seed=seed + 1),
    )
    return configs[: max(0, budget_count)]


def _validate_continuous_array(array: np.ndarray) -> None:
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Classical tabular models require a non-empty 2D continuous array.")
    if not np.isfinite(array).all():
        raise ValueError("Classical tabular models require finite continuous values.")


def _nearest_positive_correlation(matrix: np.ndarray, regularization: float) -> np.ndarray:
    matrix = np.nan_to_num((matrix + matrix.T) / 2.0, nan=0.0)
    np.fill_diagonal(matrix, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, regularization, None)
    repaired = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.clip(np.diag(repaired), regularization, None))
    repaired = repaired / np.outer(diagonal, diagonal)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _inverse_empirical(sorted_values: np.ndarray, u: np.ndarray) -> np.ndarray:
    if len(sorted_values) == 1:
        return np.full_like(u, sorted_values[0], dtype=float)
    positions = u * (len(sorted_values) - 1)
    return np.interp(positions, np.arange(len(sorted_values)), sorted_values)
