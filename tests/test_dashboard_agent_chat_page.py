"""
Day 24 LangGraph Agent Chat Streamlit Page Test입니다.

실제 FastAPI·OpenAI·LangGraph·PyTorch를 실행하지 않고,
Streamlit Widget과 Mock DashboardApiClient 연결을 검증합니다.
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

AGENT_CHAT_PAGE_PATH = (
    PROJECT_ROOT
    / "src"
    / "dashboard"
    / "pages"
    / "agent_chat.py"
)


def build_agent_response() -> dict:
    """
    Agent Chat Page 테스트용 고정 Response를 생성합니다.
    """

    return {
        "question": (
            "AI4I 데이터셋의 feature는 뭐야?"
        ),
        "intent": "dataset_schema_query",
        "confidence": 0.95,
        "intent_source": "openai",
        "intent_reason": (
            "데이터셋 구조 질문입니다."
        ),
        "prediction": None,
        "probability": None,
        "threshold": None,
        "risk_level": None,
        "recommended_action": None,
        "answer": (
            "AI4I 데이터셋은 6개 feature를 사용합니다."
        ),
        "evidence": [],
        "warnings": [],
        "errors": [],
        "limitations": [],
        "trace_id": "trace-chat-001",
        "trace_status": "success",
        "trace_started_at": (
            "2026-07-14T00:00:00+00:00"
        ),
        "trace_finished_at": (
            "2026-07-14T00:00:01+00:00"
        ),
        "trace_duration_ms": 1000.0,
        "fallback_occurred": False,
        "trace_events": [],
    }


def test_agent_chat_page_renders_expected_widgets() -> None:
    """
    Agent Chat Page의 입력 Widget과 Chat Input을 검증합니다.
    """

    app = AppTest.from_file(
        AGENT_CHAT_PAGE_PATH,
    )

    app.run()

    assert not app.exception

    assert len(
        app.number_input
    ) == 5

    assert len(
        app.selectbox
    ) == 1

    assert len(
        app.checkbox
    ) == 3

    assert len(
        app.chat_input
    ) == 1

    assert (
        app.checkbox[0].value
        is False
    )


def test_agent_chat_page_calls_api_without_automatic_raw_sample() -> None:
    """
    기본 상태에서는 이전 Raw Sample을 자동 포함하지 않고,
    Agent Response를 Session State에 저장하는지 검증합니다.
    """

    expected_response = (
        build_agent_response()
    )

    app = AppTest.from_file(
        AGENT_CHAT_PAGE_PATH,
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

    mock_api_client.query_langgraph_agent.return_value = (
        expected_response
    )

    app.chat_input[0].set_value(
        "AI4I 데이터셋의 feature는 뭐야?"
    )

    with patch(
        "src.dashboard.api_client."
        "DashboardApiClient",
        return_value=mock_context_client,
    ):
        app.run()

    assert not app.exception

    assert len(
        app.success
    ) == 1

    assert (
        app.session_state[
            "last_agent_result"
        ]
        == expected_response
    )

    assert len(
        app.session_state[
            "chat_messages"
        ]
    ) == 2

    called_payload = (
        mock_api_client
        .query_langgraph_agent
        .call_args
        .args[0]
    )

    assert (
        "raw_sample"
        not in called_payload
    )

    assert (
        called_payload[
            "chat_history"
        ]
        == []
    )
