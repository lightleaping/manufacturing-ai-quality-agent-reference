# tests/test_dashboard_agent_chat_beginner_layout.py

"""Regression tests for the beginner-first Agent chat page."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PAGE_PATH = (
    PROJECT_ROOT
    / "src/dashboard/pages/agent_chat.py"
)


def test_agent_chat_page_has_beginner_guidance() -> None:
    """The page should explain raw-sample behavior before chatting."""
    text = PAGE_PATH.read_text(
        encoding="utf-8"
    )

    required = [
        "AI 질의 응답",
        "먼저 알아둘 점",
        "이번 질문에 설비 입력값 함께 보내기",
        "질문 문장에",
        "질문 예시와 사용 방법",
        "기술 상세 보기",
    ]

    for label in required:
        assert label in text


def test_agent_chat_separates_display_and_context_history() -> None:
    """Display messages and Backend context messages must be separate."""
    text = PAGE_PATH.read_text(
        encoding="utf-8"
    )

    assert '"chat_messages"' in text
    assert '"agent_context_messages"' in text
    assert "previous_context_messages" in text
    assert "build_display_answer" in text
    assert "get_original_context_answer" in text


def test_agent_chat_hides_technical_metadata_from_primary_view() -> None:
    """Technical response data should live inside a collapsed expander."""
    text = PAGE_PATH.read_text(
        encoding="utf-8"
    )

    technical_start = text.index(
        'with st.expander(\n        "기술 상세 보기"'
    )

    assert technical_start > text.index(
        "마지막 요청 상태"
    )

    assert "Trace Summary" not in text
    assert "Recommended Action" not in text
    assert "Intent Source" not in text


def test_agent_chat_has_no_broken_text_or_emoji() -> None:
    """The rebuilt page must not contain broken markers or emoji."""
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
