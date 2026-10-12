"""Objective interpretation for user requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping


class ObjectiveType(str, Enum):
    DATA_GENERATION = "data_generation"
    IMPUTATION = "imputation"
    ANOMALY_DETECTION = "anomaly_detection"
    CLASS_BALANCING = "class_balancing"
    REPRESENTATION_LEARNING = "representation_learning"
    UNCERTAINTY_QUANTIFICATION = "uncertainty_quantification"
    EXPLORATORY = "exploratory"


class ObjectiveUseCase(str, Enum):
    INTERNAL_TESTING = "internal_testing"
    EXTERNAL_SHARING = "external_sharing"
    FORMAL_PRIVACY = "formal_privacy"


@dataclass(frozen=True)
class ObjectiveSpec:
    raw_text: str
    objective_type: ObjectiveType
    confidence: float
    constraints: Mapping[str, str] = field(default_factory=dict)
    requested_outputs: tuple[str, ...] = field(default_factory=tuple)


class ObjectiveInterpreter:
    """Keyword-based interpreter to be replaced by an LLM planner later."""

    PRIORITY = (
        ObjectiveType.CLASS_BALANCING,
        ObjectiveType.IMPUTATION,
        ObjectiveType.ANOMALY_DETECTION,
        ObjectiveType.UNCERTAINTY_QUANTIFICATION,
        ObjectiveType.REPRESENTATION_LEARNING,
        ObjectiveType.DATA_GENERATION,
    )

    KEYWORDS = {
        ObjectiveType.IMPUTATION: ("missing", "impute", "imputation"),
        ObjectiveType.ANOMALY_DETECTION: ("anomaly", "outlier", "detect unusual"),
        ObjectiveType.CLASS_BALANCING: ("balance", "minority", "class imbalance", "augment class"),
        ObjectiveType.REPRESENTATION_LEARNING: ("embedding", "representation", "latent"),
        ObjectiveType.UNCERTAINTY_QUANTIFICATION: ("uncertainty", "credible", "posterior"),
        ObjectiveType.DATA_GENERATION: ("generate", "synthetic", "simulate", "sample"),
    }

    def interpret(self, text: str) -> ObjectiveSpec:
        normalized = text.lower().strip()
        scores: dict[ObjectiveType, int] = {}
        for objective_type, keywords in self.KEYWORDS.items():
            scores[objective_type] = sum(1 for keyword in keywords if keyword in normalized)

        specific_scores = {
            key: value for key, value in scores.items() if key is not ObjectiveType.DATA_GENERATION
        }
        if any(value > 0 for value in specific_scores.values()):
            best = max(specific_scores, key=lambda key: (specific_scores[key], -self.PRIORITY.index(key)))
        else:
            best = ObjectiveType.DATA_GENERATION if scores[ObjectiveType.DATA_GENERATION] > 0 else ObjectiveType.EXPLORATORY
        if scores[best] == 0:
            best = ObjectiveType.EXPLORATORY
            confidence = 0.35
        else:
            confidence = min(0.95, 0.55 + scores[best] * 0.15)

        outputs = ("analysis_report",)
        if best in {ObjectiveType.DATA_GENERATION, ObjectiveType.CLASS_BALANCING, ObjectiveType.IMPUTATION}:
            outputs = ("synthetic_or_completed_data", "evaluation_results", "analysis_report")

        return ObjectiveSpec(
            raw_text=text,
            objective_type=best,
            confidence=confidence,
            constraints={"use_case": _infer_use_case(normalized).value},
            requested_outputs=outputs,
        )


def _infer_use_case(normalized_text: str) -> ObjectiveUseCase:
    no_privacy_terms = (
        "without a formal privacy guarantee",
        "without formal privacy",
        "no formal privacy",
        "no privacy guarantee",
    )
    privacy_terms = ("formal privacy", "privacy guarantee", "differential privacy", "private release", "provably private")
    external_terms = ("external", "share", "sharing", "release", "public", "publish", "third party")
    internal_terms = ("internal", "testing", "stress", "pipeline", "debug", "qa")
    if any(term in normalized_text for term in no_privacy_terms):
        return ObjectiveUseCase.EXTERNAL_SHARING
    if any(term in normalized_text for term in privacy_terms):
        return ObjectiveUseCase.FORMAL_PRIVACY
    if any(term in normalized_text for term in external_terms):
        return ObjectiveUseCase.EXTERNAL_SHARING
    if any(term in normalized_text for term in internal_terms):
        return ObjectiveUseCase.INTERNAL_TESTING
    return ObjectiveUseCase.INTERNAL_TESTING
