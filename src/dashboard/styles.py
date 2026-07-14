# src/dashboard/styles.py

"""
Streamlit Dashboard 전체에 적용하는 공통 Presentation Style입니다.

목표:
- 긴 문장이 화면 밖으로 잘리지 않도록 합니다.
- Card·Metric·Tab·Form의 시각적 계층을 통일합니다.
- 넓은 Desktop과 좁은 Browser 화면 모두 읽기 쉽게 만듭니다.
- 기존 Backend Response와 비즈니스 로직은 변경하지 않습니다.

중요:
이 파일은 CSS와 화면 표시만 담당합니다.

Prediction, Probability, Risk Level, Evidence 값은
계산하거나 수정하지 않습니다.
"""

from __future__ import annotations

import streamlit as st


DASHBOARD_CSS = """

/* ---------------------------------------------------------
   Common Typography
   --------------------------------------------------------- */

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
button,
input,
textarea,
select {
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        "Noto Sans KR",
        "Malgun Gothic",
        sans-serif !important;
}

body {
    font-size: 16px;
    letter-spacing: -0.012em;
}

h1 {
    font-weight: 760;
}

h2 {
    font-weight: 730;
}

h3,
h4,
h5 {
    font-weight: 680;
}

[data-testid="stMarkdownContainer"] p {
    font-size: 1rem;
}

[data-testid="stMetricLabel"] p {
    font-weight: 650;
}

[data-testid="stMetricValue"] {
    font-weight: 720;
}


/* ---------------------------------------------------------
   1. 전체 Page 폭과 기본 여백
   --------------------------------------------------------- */

.block-container {
    max-width: 1440px;
    padding-top: 1.6rem;
    padding-right: 2.2rem;
    padding-bottom: 4rem;
    padding-left: 2.2rem;
}


/* ---------------------------------------------------------
   2. 모든 설명 문장의 줄바꿈과 행간
   --------------------------------------------------------- */

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stCaptionContainer"] {
    overflow-wrap: anywhere;
    word-break: keep-all;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    line-height: 1.72;
}

[data-testid="stCaptionContainer"] {
    line-height: 1.58;
    opacity: 0.82;
}


/* ---------------------------------------------------------
   3. Page Title과 Section 간격
   --------------------------------------------------------- */

h1 {
    letter-spacing: -0.035em;
    line-height: 1.25;
    margin-bottom: 0.65rem;
}

h2 {
    letter-spacing: -0.025em;
    margin-top: 1.7rem;
    margin-bottom: 0.8rem;
}

h3 {
    letter-spacing: -0.018em;
    line-height: 1.38;
}

h4,
h5 {
    line-height: 1.48;
}


/* ---------------------------------------------------------
   4. Border Container를 실제 Card처럼 표시
   --------------------------------------------------------- */

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 16px;
    border-color: color-mix(
        in srgb,
        var(--text-color) 16%,
        transparent
    );
    background:
        color-mix(
            in srgb,
            var(--secondary-background-color) 72%,
            var(--background-color) 28%
        );
    box-shadow:
        0 2px 10px
        color-mix(
            in srgb,
            var(--text-color) 7%,
            transparent
        );
}

[data-testid="stVerticalBlockBorderWrapper"]
> div {
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
}


/* ---------------------------------------------------------
   5. Metric Card
   --------------------------------------------------------- */

[data-testid="stMetric"] {
    min-height: 116px;
    padding: 1rem 1.05rem;
    border-radius: 14px;
    border:
        1px solid
        color-mix(
            in srgb,
            var(--text-color) 13%,
            transparent
        );
    background:
        color-mix(
            in srgb,
            var(--secondary-background-color) 82%,
            var(--background-color) 18%
        );
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: keep-all;
    line-height: 1.35;
}

[data-testid="stMetricValue"] {
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.18;
    letter-spacing: -0.025em;
}


/* ---------------------------------------------------------
   6. Form
   --------------------------------------------------------- */

[data-testid="stForm"] {
    padding: 1.25rem 1.35rem 1.4rem;
    border-radius: 18px;
    border:
        1px solid
        color-mix(
            in srgb,
            var(--text-color) 15%,
            transparent
        );
    background:
        color-mix(
            in srgb,
            var(--secondary-background-color) 60%,
            var(--background-color) 40%
        );
}


/* ---------------------------------------------------------
   7. Button
   --------------------------------------------------------- */

.stButton > button,
[data-testid="stFormSubmitButton"] button {
    min-height: 2.9rem;
    border-radius: 11px;
    font-weight: 700;
    letter-spacing: -0.015em;
}


/* ---------------------------------------------------------
   8. Tab
   --------------------------------------------------------- */

[data-baseweb="tab-list"] {
    gap: 0.4rem;
    flex-wrap: wrap;
}

button[data-baseweb="tab"] {
    min-height: 2.75rem;
    padding-right: 1rem;
    padding-left: 1rem;
    border-radius: 10px 10px 0 0;
}

button[data-baseweb="tab"] p {
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: keep-all;
    text-align: center;
}


/* ---------------------------------------------------------
   9. Expander
   --------------------------------------------------------- */

[data-testid="stExpander"] details {
    overflow: hidden;
    border-radius: 13px;
    border-color:
        color-mix(
            in srgb,
            var(--text-color) 14%,
            transparent
        );
}

[data-testid="stExpander"] summary {
    min-height: 3rem;
    font-weight: 650;
}

[data-testid="stExpander"] summary p {
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: keep-all;
}


/* ---------------------------------------------------------
   10. Alert
   --------------------------------------------------------- */

[data-testid="stAlert"] {
    border-radius: 13px;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] li {
    overflow-wrap: anywhere;
    word-break: keep-all;
    line-height: 1.65;
}


/* ---------------------------------------------------------
   11. DataFrame
   --------------------------------------------------------- */

[data-testid="stDataFrame"] {
    width: 100%;
    overflow-x: auto;
    border-radius: 12px;
}


/* ---------------------------------------------------------
   12. Chat Message
   --------------------------------------------------------- */

[data-testid="stChatMessage"] {
    padding: 1rem 1.1rem;
    border-radius: 15px;
    border:
        1px solid
        color-mix(
            in srgb,
            var(--text-color) 12%,
            transparent
        );
}

[data-testid="stChatMessage"]
[data-testid="stMarkdownContainer"] {
    max-width: 100%;
    overflow-wrap: anywhere;
    word-break: keep-all;
}


/* ---------------------------------------------------------
   13. Progress
   --------------------------------------------------------- */

[data-testid="stProgress"] {
    margin-top: 0.6rem;
}


/* ---------------------------------------------------------
   14. Divider
   --------------------------------------------------------- */

hr {
    margin-top: 1.45rem;
    margin-bottom: 1.45rem;
    opacity: 0.55;
}


/* ---------------------------------------------------------
   15. 좁은 화면 대응
   --------------------------------------------------------- */

@media (
    max-width: 900px
) {
    .block-container {
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
    }

    [data-testid="stMetric"] {
        min-height: 104px;
    }

    [data-testid="stForm"] {
        padding-right: 0.9rem;
        padding-left: 0.9rem;
    }
}


/* ---------------------------------------------------------
   16. 매우 좁은 화면 대응
   --------------------------------------------------------- */

@media (
    max-width: 560px
) {
    .block-container {
        padding-right: 0.65rem;
        padding-left: 0.65rem;
    }

    h1 {
        font-size: 1.8rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 13px;
    }

    [data-testid="stMetric"] {
        min-height: auto;
        padding: 0.85rem;
    }
}
"""


def get_dashboard_css() -> str:
    """
    Dashboard 공통 CSS 문자열을 반환합니다.

    별도 함수로 제공하는 이유:
    Test에서 Style의 핵심 규칙을
    Streamlit 실행 없이 검증할 수 있습니다.
    """

    return DASHBOARD_CSS


def apply_dashboard_styles() -> None:
    """
    Dashboard 공통 Style을 현재 Streamlit App에 적용합니다.

    unsafe_allow_html=True는
    내부 CSS <style> 태그를 적용하기 위해서만 사용합니다.

    사용자 입력 HTML을 출력하지 않습니다.
    """

    st.markdown(
        (
            "<style>"
            f"{get_dashboard_css()}"
            "</style>"
        ),
        unsafe_allow_html=True,
    )
