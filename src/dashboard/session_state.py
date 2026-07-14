# src/dashboard/session_state.py

"""
Streamlit Dashboard의 공통 Session State를 관리합니다.

Streamlit은 사용자가 Widget을 조작하거나 버튼을 누르면
Python Script를 위에서 아래로 다시 실행합니다.

따라서 다음과 같이 rerun 이후에도 유지해야 하는 값은
Session State에 저장합니다.

- 마지막 설비 고장 예측 Response
- 사용자와 Agent의 화면용 대화 기록
- 마지막 LangGraph Agent Response
- 최근 Agent 실행 이력
- 사용자가 선택한 Trace ID
- 선택한 Agent 실행 상세 Response

중요:
- 이 모듈은 Streamlit 화면을 직접 렌더링하지 않습니다.
- FastAPI를 직접 호출하지 않습니다.
- PyTorch 모델을 직접 실행하지 않습니다.
- LangGraph workflow를 직접 실행하지 않습니다.
- SQLite를 직접 조회하지 않습니다.
- 이전 Raw Sample을 Agent 요청에 자동 재사용하지 않습니다.

Session State의 역할:

    Streamlit rerun
        ↓
    화면 상태와 API Response 유지

Session State가 하지 않는 일:

    이전 Raw Sample
        ↓
    다음 Agent Prediction 요청에 자동 첨부
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    MutableMapping,
)
from typing import Any


# Session State 기본값은 값 자체가 아니라
# "기본값을 새로 생성하는 함수"로 관리합니다.
#
# 예:
#
#     "chat_messages": list
#
# initialize_dashboard_session_state()가 실행될 때
# list()를 호출하므로 사용자 Session마다
# 새로운 빈 List가 생성됩니다.
#
# 만약 하나의 빈 List 객체를 기본값으로 직접 공유하면
# 서로 다른 Session이 같은 Mutable 객체를
# 참조할 가능성이 있으므로 피합니다.
DASHBOARD_SESSION_DEFAULTS: dict[
    str,
    Callable[[], Any],
] = {
    # 마지막 Direct Failure Prediction API Response입니다.
    #
    # Evidence 분석 화면은 이 Response의 evidence를 재사용합니다.
    "failure_prediction_result": (
        lambda: None
    ),

    # Streamlit Chat 화면에 표시할
    # user·assistant 메시지 기록입니다.
    #
    # 이 값은 대화 문맥용이며,
    # 이전 설비 Raw Sample을 저장하지 않습니다.
    "chat_messages": list,

    # 마지막 LangGraph Agent Query API Response입니다.
    "last_agent_result": (
        lambda: None
    ),

    # GET /agent/executions에서 받은
    # 최근 Agent 실행 이력 목록입니다.
    "execution_history": list,

    # 실행 이력 화면에서 사용자가 선택한 Trace ID입니다.
    "selected_execution_trace_id": (
        lambda: None
    ),

    # GET /agent/executions/{trace_id}에서 받은
    # 선택한 실행의 상세 Response입니다.
    "selected_execution_detail": (
        lambda: None
    ),
}


def initialize_dashboard_session_state(
    session_state: MutableMapping[
        str,
        Any,
    ],
) -> None:
    """
    Dashboard 공통 Session State Key를 초기화합니다.

    Parameters
    ----------
    session_state:
        Streamlit의 st.session_state 또는
        Unit Test에서 사용하는 일반 dict입니다.

    동작 규칙
    ---------
    1. Key가 아직 없으면 기본값을 생성합니다.

    2. Key가 이미 있으면 기존 값을 유지합니다.

    3. List 같은 Mutable 기본값은
       Session마다 새로운 객체를 생성합니다.

    Streamlit 사용 예
    -----------------

        initialize_dashboard_session_state(
            st.session_state,
        )

    중요
    ----
    Streamlit은 Widget 조작 때 Script를 다시 실행합니다.

    따라서 이미 저장된 Prediction Result,
    Chat Message, Execution History를
    매 rerun마다 초기값으로 덮어쓰면 안 됩니다.

    이 함수는 없는 Key만 추가하므로
    기존 사용자 상태를 보존합니다.
    """

    for (
        key,
        default_factory,
    ) in DASHBOARD_SESSION_DEFAULTS.items():
        # 기존 값이 있으면 그대로 유지합니다.
        #
        # 단순히 아래처럼 대입하면:
        #
        #     session_state[key] = default_value
        #
        # Streamlit rerun 때마다 사용자의 상태가
        # 초기화될 수 있으므로 사용하지 않습니다.
        if key in session_state:
            continue

        # 기본값 생성 함수를 현재 Session에서 호출합니다.
        #
        # list factory:
        #     새로운 빈 List 생성
        #
        # lambda: None:
        #     None 기본값 반환
        session_state[
            key
        ] = default_factory()
