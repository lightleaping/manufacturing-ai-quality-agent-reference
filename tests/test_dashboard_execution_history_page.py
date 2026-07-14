"""
Day 24 Trace·Execution History Streamlit Page Test입니다.

실제 FastAPI와 SQLite를 실행하지 않고,
Mock DashboardApiClient를 사용하여 목록·상세 조회를 검증합니다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

from streamlit.testing.v1 import (
    AppTest,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

EXECUTION_HISTORY_PAGE_PATH = (
    PROJECT_ROOT
    / "src"
    / "dashboard"
    / "pages"
    / "execution_history.py"
)


def build_execution_summary() -> dict:
    """
    실행 이력 목록 테스트용 Summary를 생성합니다.
    """

    return {
        "id": 1,
        "trace_id": "trace-history-001",
        "question": (
            "이 설비 조건의 고장 위험을 예측해줘."
        ),
        "intent": "failure_prediction",
        "intent_source": "openai",
        "confidence": 0.95,
        "selected_route": "final",
        "prediction": 1,
        "probability": 0.9929,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "trace_status": "success",
        "fallback_occurred": False,
        "trace_duration_ms": 2450.2,
        "warning_count": 0,
        "error_count": 0,
        "created_at": (
            "2026-07-14T00:00:00+00:00"
        ),
    }


def build_execution_detail() -> dict:
    """
    실행 이력 상세 테스트용 Response를 생성합니다.
    """

    detail = (
        build_execution_summary()
    )

    detail.update(
        {
            "intent_reason": (
                "고장 위험 예측 요청입니다."
            ),
            "recommended_action": (
                "설비 점검을 권장합니다."
            ),
            "answer": (
                "고장 위험이 높게 예측되었습니다."
            ),
            "trace_started_at": (
                "2026-07-14T00:00:00+00:00"
            ),
            "trace_finished_at": (
                "2026-07-14T00:00:02+00:00"
            ),
            "raw_sample": {
                "Torque [Nm]": 62.0,
            },
            "evidence": [
                {
                    "evidence_type": (
                        "prediction_summary"
                    ),
                    "title": "모델 예측 요약",
                }
            ],
            "trace_events": [
                {
                    "sequence": 1,
                    "event_type": "node",
                    "event_name": (
                        "validate_question"
                    ),
                    "status": "success",
                    "started_at": (
                        "2026-07-14T00:00:00+00:00"
                    ),
                    "finished_at": (
                        "2026-07-14T00:00:00.1+00:00"
                    ),
                    "duration_ms": 100.0,
                    "metadata": {},
                }
            ],
            "warnings": [],
            "errors": [],
            "limitations": [],
        }
    )

    return detail


def test_execution_history_page_displays_empty_state() -> None:
    """
    실행 이력을 조회하기 전 안내 상태를 검증합니다.
    """

    app = AppTest.from_file(
        EXECUTION_HISTORY_PAGE_PATH,
    )

    app.run()

    assert not app.exception

    assert len(
        app.number_input
    ) == 1

    assert len(
        app.button
    ) == 1

    assert len(
        app.info
    ) >= 1


def test_execution_history_page_refreshes_summary_list() -> None:
    """
    목록 조회 성공 시 Session State와 DataFrame을 검증합니다.
    """

    expected_executions = [
        build_execution_summary()
    ]

    app = AppTest.from_file(
        EXECUTION_HISTORY_PAGE_PATH,
    )

    app.run()

    mock_context_client = (
        MagicMock()
    )

    mock_api_client = (
        MagicMock()
    )

    mock_context_client.__enter__.return_value = (
        mock_api_client
    )

    mock_api_client.get_agent_executions.return_value = (
        expected_executions
    )

    app.button[0].click()

    with patch(
        "src.dashboard.api_client."
        "DashboardApiClient",
        return_value=mock_context_client,
    ):
        app.run()

    assert not app.exception

    assert (
        app.session_state[
            "execution_history"
        ]
        == expected_executions
    )

    assert len(
        app.dataframe
    ) >= 1

    assert len(
        app.selectbox
    ) == 1

    mock_api_client.get_agent_executions.assert_called_once_with(
        limit=20,
    )


def test_execution_history_page_loads_selected_detail() -> None:
    """
    Trace ID 선택 후 상세 Response를 저장하고 표시하는지 검증합니다.
    """

    expected_detail = (
        build_execution_detail()
    )

    app = AppTest.from_file(
        EXECUTION_HISTORY_PAGE_PATH,
    )

    app.session_state[
        "execution_history"
    ] = [
        build_execution_summary()
    ]

    app.run()

    mock_context_client = (
        MagicMock()
    )

    mock_api_client = (
        MagicMock()
    )

    mock_context_client.__enter__.return_value = (
        mock_api_client
    )

    mock_api_client.get_agent_execution_detail.return_value = (
        expected_detail
    )

    app.button[1].click()

    with patch(
        "src.dashboard.api_client."
        "DashboardApiClient",
        return_value=mock_context_client,
    ):
        app.run()

    assert not app.exception

    assert (
        app.session_state[
            "selected_execution_detail"
        ]
        == expected_detail
    )

    assert len(
        app.metric
    ) == 4

    mock_api_client.get_agent_execution_detail.assert_called_once_with(
        trace_id="trace-history-001",
    )
