"""
Day 24 Streamlit Dashboard Entry Point Test입니다.

공통 Page Config와 Navigation을 통해
기본 Dashboard Page가 정상 실행되는지 검증합니다.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import (
    AppTest,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DASHBOARD_APP_PATH = (
    PROJECT_ROOT
    / "src"
    / "dashboard"
    / "app.py"
)


def test_dashboard_app_runs_default_page() -> None:
    """
    Dashboard Entry Point가 기본 Prediction Page를
    Exception 없이 실행하는지 검증합니다.
    """

    app = AppTest.from_file(
        DASHBOARD_APP_PATH,
        default_timeout=10,
    )

    app.run(
        timeout=10,
    )

    assert not app.exception

    assert len(
        app.title
    ) >= 1

    assert (
        app.title[0].value
        == "설비 고장 위험 분석"
    )

    assert len(
        app.number_input
    ) == 5
