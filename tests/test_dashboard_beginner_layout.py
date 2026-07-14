# tests/test_dashboard_beginner_layout.py

"""Regression tests for the beginner-first failure prediction layout."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PAGE_PATH = (
    PROJECT_ROOT
    / "src/dashboard/pages/failure_prediction.py"
)


def test_failure_prediction_page_contains_beginner_sections() -> None:
    """The page must contain the full beginner-first explanation flow."""
    text = PAGE_PATH.read_text(
        encoding="utf-8"
    )

    required_labels = [
        "한눈에 보는 결론",
        "지금 할 일",
        "핵심 결과",
        "무엇부터 확인하면 되나요?",
        "왜 이런 판정이 나왔나요?",
        "숫자를 예로 들어 이해하기",
        "판단 근거는 세 종류입니다",
        "기술 상세와 모델 한계",
    ]

    for label in required_labels:
        assert label in text


def test_failure_prediction_result_uses_expected_runtime_order() -> None:
    """The result renderer must call sections in a beginner-first order."""
    text = PAGE_PATH.read_text(
        encoding="utf-8"
    )

    start = text.index(
        "def render_failure_prediction_result"
    )

    end = text.index(
        "def main()",
        start,
    )

    block = text[
        start:end
    ]

    calls = [
        "render_result_overview(",
        "render_key_metrics(",
        "render_check_steps(",
        "render_top_evidence(",
        "render_score_example(",
        "render_evidence_summary(",
        "render_operational_explanation_section(",
        "render_technical_details(",
    ]

    positions = [
        block.index(
            call
        )
        for call in calls
    ]

    assert positions == sorted(
        positions
    )


def test_failure_prediction_page_avoids_cramped_result_columns() -> None:
    """Long result labels and explanations must avoid four-column layouts."""
    text = PAGE_PATH.read_text(
        encoding="utf-8"
    )

    assert "st.columns(\n            4\n        )" not in text


def test_failure_prediction_page_has_no_broken_text_or_emoji() -> None:
    """The rebuilt page must not contain broken text markers or emoji."""
    text = PAGE_PATH.read_text(
        encoding="utf-8"
    )

    assert "?" * 2 not in text
    assert "\ufffd" not in text

    for character in text:
        code_point = ord(
            character
        )

        assert not (
            0x1F300
            <= code_point
            <= 0x1FAFF
        )
