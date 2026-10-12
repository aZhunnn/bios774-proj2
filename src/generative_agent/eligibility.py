"""Objective-aware release eligibility rules.

Eligibility is a hard gate. Performance ranking is applied only after methods
pass these deterministic rules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from generative_agent.objectives import ObjectiveUseCase


BOOTSTRAP_METHODS = {"baseline", "training_row_bootstrap", "validation_bootstrap"}
VAE_METHODS = {"vae", "tabular_vae", "validation_tabular_vae", "image_vae", "validation_image_vae"}
FLOW_METHODS = {"flow", "tabular_flow", "validation_tabular_flow"}
GAN_METHODS = {"gan", "image_gan", "validation_image_gan"}


@dataclass(frozen=True)
class MethodEligibility:
    method: str
    eligible: bool
    reasons: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EligibilitySelection:
    use_case: ObjectiveUseCase
    selected_method: str | None
    eligible_methods: tuple[str, ...]
    rejected_methods: tuple[str, ...]
    method_eligibility: tuple[MethodEligibility, ...]
    ranking_metric: str
    selection_reason: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["use_case"] = self.use_case.value
        payload["method_eligibility"] = [item.to_dict() for item in self.method_eligibility]
        return payload


def normalize_method_name(method: str) -> str:
    if method in BOOTSTRAP_METHODS:
        return "baseline"
    if method in VAE_METHODS:
        return "vae"
    if method in FLOW_METHODS:
        return "flow"
    if method in GAN_METHODS:
        return "gan"
    return method


def method_eligibility(method: str, use_case: ObjectiveUseCase) -> MethodEligibility:
    normalized = normalize_method_name(method)
    if use_case is ObjectiveUseCase.FORMAL_PRIVACY:
        return MethodEligibility(
            method=normalized,
            eligible=False,
            reasons=("No currently implemented method has a formal privacy-preserving mechanism or guarantee.",),
            limitations=("VAE/flow samples are not private merely because they differ from training records.",),
        )
    if normalized == "baseline" and use_case is ObjectiveUseCase.EXTERNAL_SHARING:
        return MethodEligibility(
            method=normalized,
            eligible=False,
            reasons=("Exact-record bootstrap resampling is not eligible for external synthetic-data sharing.",),
            limitations=("Bootstrap can duplicate training rows and has nearest-neighbor distance 0 by construction.",),
        )
    if normalized in {"vae", "flow", "gan"} and use_case is ObjectiveUseCase.EXTERNAL_SHARING:
        return MethodEligibility(
            method=normalized,
            eligible=True,
            reasons=("Model-generated samples may be considered for external sharing only without claiming privacy guarantees.",),
            limitations=("Requires disclosure that no formal privacy guarantee is implemented.",),
        )
    return MethodEligibility(
        method=normalized,
        eligible=True,
        reasons=("Method is eligible for internal testing or stress-testing under current deterministic rules.",),
        limitations=("Eligibility does not imply high fidelity or privacy protection.",),
    )


def select_eligible_method(
    scores: dict[str, dict[str, float]],
    *,
    use_case: ObjectiveUseCase | str,
    ranking_metric: str = "selection_score",
) -> EligibilitySelection:
    if isinstance(use_case, str):
        use_case = ObjectiveUseCase(use_case)
    eligibilities: list[MethodEligibility] = []
    best_method: str | None = None
    best_score = float("-inf")
    seen: set[str] = set()
    for method, metrics in scores.items():
        normalized = normalize_method_name(method)
        if normalized in seen:
            continue
        seen.add(normalized)
        eligibility = method_eligibility(normalized, use_case)
        eligibilities.append(eligibility)
        if eligibility.eligible and metrics.get(ranking_metric, float("-inf")) > best_score:
            best_method = normalized
            best_score = metrics.get(ranking_metric, float("-inf"))

    eligible = tuple(item.method for item in eligibilities if item.eligible)
    rejected = tuple(item.method for item in eligibilities if not item.eligible)
    if best_method is None:
        reason = "No method is eligible under the requested use case."
    else:
        reason = f"Selected highest {ranking_metric} among eligible methods; hard eligibility rules were applied before ranking."
    return EligibilitySelection(
        use_case=use_case,
        selected_method=best_method,
        eligible_methods=eligible,
        rejected_methods=rejected,
        method_eligibility=tuple(eligibilities),
        ranking_metric=ranking_metric,
        selection_reason=reason,
    )
