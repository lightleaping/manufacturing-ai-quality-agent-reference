# tests/test_dashboard_styles.py

"""
Dashboard 공통 Style Test입니다.

CSS의 모든 색상·간격을 세부적으로 고정하지 않고,
사용성에 중요한 핵심 규칙이 유지되는지 검증합니다.
"""

from unittest.mock import patch

from src.dashboard.styles import (
    apply_dashboard_styles,
    get_dashboard_css,
)


def test_dashboard_css_contains_readability_rules() -> None:
    """
    긴 문장 줄바꿈과 읽기 쉬운 행간 규칙이
    공통 CSS에 포함되는지 검증합니다.
    """

    css = get_dashboard_css()

    assert (
        "overflow-wrap: anywhere"
        in css
    )

    assert (
        "word-break: keep-all"
        in css
    )

    assert (
        "line-height"
        in css
    )


def test_dashboard_css_contains_responsive_rules() -> None:
    """
    좁은 Browser 화면을 위한
    반응형 규칙이 포함되는지 검증합니다.
    """

    css = get_dashboard_css()

    assert (
        "@media"
        in css
    )

    assert (
        "max-width: 900px"
        in css
    )

    assert (
        "max-width: 560px"
        in css
    )


def test_apply_dashboard_styles_uses_style_tag() -> None:
    """
    공통 Style 함수가 CSS를
    Streamlit Markdown으로 적용하는지 검증합니다.
    """

    with patch(
        "src.dashboard.styles.st.markdown"
    ) as mock_markdown:
        apply_dashboard_styles()

    mock_markdown.assert_called_once()

    args, kwargs = (
        mock_markdown.call_args
    )

    assert (
        "<style>"
        in args[0]
    )

    assert (
        "</style>"
        in args[0]
    )

    assert (
        kwargs[
            "unsafe_allow_html"
        ]
        is True
    )
