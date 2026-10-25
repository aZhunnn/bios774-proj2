"""Optional AI-assisted narrative analysis for completed workflow artifacts.

Scientific decisions remain deterministic. This module only turns stored,
already-computed evidence into a concise narrative and records whether an
external model was actually used.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FinalAnalysisConfig:
    mode: str = "auto"  # auto, openai, anthropic, or disabled
    model: str = "gpt-4o-mini"
    max_output_tokens: int = 2000


def generate_final_analysis(
    evidence: dict[str, Any],
    config: FinalAnalysisConfig | None = None,
) -> dict[str, Any]:
    """Generate a report narrative without changing workflow decisions.

    ``auto`` uses Anthropic when ``ANTHROPIC_API_KEY`` is present and OpenAI
    when ``OPENAI_API_KEY`` is present. Missing
    credentials, an unavailable optional dependency, or an API failure falls
    back to a transparent deterministic status rather than failing the run.
    """
    selected = config or FinalAnalysisConfig()
    mode = selected.mode.lower()
    if mode not in {"auto", "openai", "anthropic", "disabled"}:
        raise ValueError("AI analysis mode must be auto, openai, anthropic, or disabled.")
    if mode == "disabled":
        return _status("disabled", "AI-assisted narrative generation was disabled by the user.")
    if mode == "anthropic" or (mode == "auto" and os.getenv("ANTHROPIC_API_KEY")):
        return _anthropic_analysis(evidence, selected)
    if mode == "openai" or (mode == "auto" and os.getenv("OPENAI_API_KEY")):
        return _openai_analysis(evidence, selected)
    return _status("fallback", "No configured AI provider credential was found; the factual deterministic report was retained.")


def _anthropic_analysis(evidence: dict[str, Any], config: FinalAnalysisConfig) -> dict[str, Any]:
    try:
        from anthropic import Anthropic

        client = Anthropic()
        response = client.messages.create(
            model=config.model,
            max_tokens=config.max_output_tokens,
            system=(
                "You are the final analysis assistant for a generative modeling experiment. "
                "Summarize only the supplied evidence. Do not invent metrics, claim privacy, "
                "clinical validity, statistical significance, or autonomous replanning. "
                "State the selected method, evidence, limitations, and permitted downstream use."
            ),
            messages=[{"role": "user", "content": json.dumps(evidence, sort_keys=True, default=str)}],
        )
        text = "\n".join(
            block.text for block in getattr(response, "content", []) if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise RuntimeError("The model returned an empty analysis.")
        return {
            "status": "generated",
            "provider": "anthropic",
            "model": config.model,
            "text": text,
            "input_scope": "stored profile, objective, planning, evaluation, and decision artifacts",
        }
    except Exception as exc:  # optional provider must never break reproducible execution
        return _status("fallback", f"Anthropic analysis was unavailable ({type(exc).__name__}); deterministic report retained.")


def _openai_analysis(evidence: dict[str, Any], config: FinalAnalysisConfig) -> dict[str, Any]:
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=config.model,
            instructions=(
                "You are the final analysis assistant for a generative modeling experiment. "
                "Summarize only the supplied evidence. Do not invent metrics, claim privacy, "
                "clinical validity, statistical significance, or autonomous replanning. "
                "State the selected method, evidence, limitations, and permitted downstream use."
            ),
            input=json.dumps(evidence, sort_keys=True, default=str),
            max_output_tokens=config.max_output_tokens,
        )
        text = getattr(response, "output_text", "").strip()
        if not text:
            raise RuntimeError("The model returned an empty analysis.")
        return {
            "status": "generated",
            "provider": "openai",
            "model": config.model,
            "text": text,
            "input_scope": "stored profile, objective, planning, evaluation, and decision artifacts",
        }
    except Exception as exc:  # optional provider must never break reproducible execution
        return _status("fallback", f"OpenAI analysis was unavailable ({type(exc).__name__}); deterministic report retained.")


def _status(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "provider": "none",
        "model": None,
        "text": None,
        "reason": reason,
        "input_scope": "none",
    }
