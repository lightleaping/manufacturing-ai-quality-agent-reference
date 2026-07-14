"""
Day 24 Streamlit Dashboard Session State Unit Test입니다.

이 테스트는 실제 Streamlit 서버를 실행하지 않습니다.

대신 일반 dict를 사용하여 다음 동작을 검증합니다.

- Dashboard 공통 상태 초기화
- 기존 상태 보존
- Mutable 기본값 독립성
- 예측 결과와 Evidence 화면 공유 상태
- 이전 Raw Sample 자동 저장 금지
"""

from __future__ import annotations

from src.dashboard.session_state import (
    DASHBOARD_SESSION_DEFAULTS,
    initialize_dashboard_session_state,
)


def test_initialize_dashboard_session_state_adds_defaults() -> None:
    """
    빈 Session State에 Dashboard 기본 Key가 모두 추가되는지 검증합니다.
    """

    session_state: dict = {}

    initialize_dashboard_session_state(
        session_state,
    )

    assert session_state[
        "failure_prediction_result"
    ] is None

    assert session_state[
        "chat_messages"
    ] == []

    assert session_state[
        "last_agent_result"
    ] is None

    assert session_state[
        "execution_history"
    ] == []

    assert session_state[
        "selected_execution_trace_id"
    ] is None

    assert session_state[
        "selected_execution_detail"
    ] is None


def test_initialize_dashboard_session_state_preserves_existing_values() -> None:
    """
    Streamlit rerun 시 기존 사용자 상태를 덮어쓰지 않는지 검증합니다.
    """

    existing_result = {
        "prediction": 1,
        "risk_level": "HIGH",
    }

    session_state = {
        "failure_prediction_result": (
            existing_result
        ),
        "chat_messages": [
            {
                "role": "user",
                "content": "기존 질문",
            }
        ],
    }

    initialize_dashboard_session_state(
        session_state,
    )

    assert (
        session_state[
            "failure_prediction_result"
        ]
        is existing_result
    )

    assert session_state[
        "chat_messages"
    ] == [
        {
            "role": "user",
            "content": "기존 질문",
        }
    ]


def test_initialize_dashboard_session_state_creates_independent_lists() -> None:
    """
    서로 다른 사용자 Session이 같은 List 객체를 공유하지 않는지 검증합니다.
    """

    first_session: dict = {}
    second_session: dict = {}

    initialize_dashboard_session_state(
        first_session,
    )

    initialize_dashboard_session_state(
        second_session,
    )

    first_session[
        "chat_messages"
    ].append(
        {
            "role": "user",
            "content": "첫 번째 Session 질문",
        }
    )

    assert second_session[
        "chat_messages"
    ] == []

    assert (
        first_session[
            "chat_messages"
        ]
        is not second_session[
            "chat_messages"
        ]
    )


def test_failure_prediction_result_can_be_shared_with_evidence_page() -> None:
    """
    마지막 Prediction Response를 저장하여
    Evidence 화면에서 같은 Backend 결과를 재사용할 수 있는지 검증합니다.
    """

    session_state: dict = {}

    initialize_dashboard_session_state(
        session_state,
    )

    prediction_result = {
        "prediction": 1,
        "probability": 0.9929,
        "risk_level": "HIGH",
        "evidence": [
            {
                "evidence_type": (
                    "prediction_summary"
                ),
            }
        ],
    }

    session_state[
        "failure_prediction_result"
    ] = prediction_result

    assert (
        session_state[
            "failure_prediction_result"
        ]["evidence"]
        == prediction_result["evidence"]
    )


def test_session_defaults_do_not_store_agent_raw_sample() -> None:
    """
    이전 설비 입력을 Agent 요청에 자동 재사용하지 않도록
    공통 Session 기본 상태에 Agent Raw Sample을 저장하지 않는지 검증합니다.
    """

    assert (
        "agent_raw_sample"
        not in DASHBOARD_SESSION_DEFAULTS
    )

    assert (
        "last_raw_sample"
        not in DASHBOARD_SESSION_DEFAULTS
    )
