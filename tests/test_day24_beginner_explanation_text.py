# tests/test_day24_beginner_explanation_text.py

"""Regression tests for beginner-first explanation text integrity."""

from __future__ import annotations

from pathlib import Path

from src.agent.operational_explainer import (
    _build_system_prompt,
    _build_user_prompt,
)


def build_prediction_result() -> dict:
    """Return a small confirmed prediction context."""
    return {
        "prediction": 1,
        "probability": 0.993,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "recommended_action": (
            "설비 점검을 권장합니다."
        ),
        "evidence": [],
        "warnings": [],
        "limitations": [],
    }


def test_operational_prompts_are_not_broken_and_forbid_model_labels() -> None:
    """Prompt text must be intact and keep labels in the UI layer."""
    combined = (
        _build_system_prompt()
        + "\n"
        + _build_user_prompt(
            build_prediction_result()
        )
    )

    assert "?" * 2 not in combined
    assert "\ufffd" not in combined
    assert "Do not use category labels" in combined
    assert (
        "Do not start with brackets, labels, headings, "
        "or evidence type names."
        in combined
    )
    assert "simple comparison or example" in combined
    assert "Do not use emojis" in combined


def test_day24_explanation_sources_have_no_broken_text_markers() -> None:
    """Relevant source sections must not contain known broken markers."""
    project_root = Path(__file__).resolve().parents[1]

    operational_text = (
        project_root
        / "src/agent/operational_explainer.py"
    ).read_text(
        encoding="utf-8"
    )

    page_text = (
        project_root
        / "src/dashboard/pages/failure_prediction.py"
    ).read_text(
        encoding="utf-8"
    )

    schemas_text = (
        project_root
        / "src/api/schemas.py"
    ).read_text(
        encoding="utf-8"
    )

    schema_start = schemas_text.index(
        "class FailurePredictionExplanationRequest"
    )

    relevant_schema_text = schemas_text[
        schema_start:
    ]

    for text in (
        operational_text,
        page_text,
        relevant_schema_text,
    ):
        assert "?" * 2 not in text
        assert "\ufffd" not in text
