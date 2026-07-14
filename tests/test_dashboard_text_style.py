# tests/test_dashboard_text_style.py

"""
Dashboard의 글 중심 UI와 Typography를 검증합니다.
"""

from __future__ import annotations

from pathlib import Path
import re

from src.dashboard.styles import (
    get_dashboard_css,
)


DASHBOARD_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "src"
    / "dashboard"
)


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "]"
)


def test_dashboard_source_does_not_contain_emoji_characters() -> None:
    """
    Dashboard 사용자 화면 코드에
    이모티콘 문자가 남아 있지 않은지 검증합니다.
    """

    for path in DASHBOARD_ROOT.rglob(
        "*.py"
    ):
        text = path.read_text(
            encoding="utf-8",
        )

        assert (
            EMOJI_PATTERN.search(
                text,
            )
            is None
        ), (
            f"이모티콘 문자가 남아 있습니다: {path}"
        )

        assert (
            "\ufe0f"
            not in text
        )

        assert (
            "\u200d"
            not in text
        )


def test_dashboard_css_uses_readable_korean_font_stack() -> None:
    """
    별도 Font 파일 없이
    운영체제의 가독성 좋은 한글 Font를 우선 사용합니다.
    """

    css = get_dashboard_css()

    assert (
        "font-family"
        in css
    )

    assert (
        "Noto Sans KR"
        in css
    )

    assert (
        "Malgun Gothic"
        in css
    )
