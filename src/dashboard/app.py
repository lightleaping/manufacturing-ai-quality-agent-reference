# src/dashboard/app.py

"""
Manufacturing AI Quality Agent Streamlit Dashboard Entry Point입니다.

이 파일의 역할:
- 공통 Streamlit Page Config 설정
- Dashboard Session State 초기화
- Dashboard Navigation 구성
- 화면별 Page 연결

중요:
- PyTorch 모델을 직접 로드하지 않습니다.
- LangGraph workflow를 직접 실행하지 않습니다.
- SQLite를 직접 조회하지 않습니다.
- Backend 비즈니스 로직을 다시 구현하지 않습니다.

실행:

    streamlit run src/dashboard/app.py

전체 구조:

    User
        ↓
    Streamlit Dashboard
        ↓
    DashboardApiClient
        ↓
    Existing FastAPI
        ↓
    Service / LangGraph / PyTorch / Persistence
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.dashboard.styles import apply_dashboard_styles

from src.dashboard.config import (
    load_dashboard_api_config,
)
from src.dashboard.session_state import (
    initialize_dashboard_session_state,
)


DASHBOARD_DIRECTORY = (
    Path(__file__)
    .resolve()
    .parent
)

PAGES_DIRECTORY = (
    DASHBOARD_DIRECTORY
    / "pages"
)


def main() -> None:
    """
    Dashboard 공통 설정과 Navigation을 실행합니다.
    """

    st.set_page_config(
        page_title=(
            "Manufacturing AI Quality Agent"
        ),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_dashboard_session_state(
        st.session_state,
    )

    config = (
        load_dashboard_api_config()
    )

    with st.sidebar:
        st.header(
            "Manufacturing AI Dashboard"
        )

        st.caption(
            "Presentation Layer"
        )

        st.markdown(
            "**FastAPI 연결 설정**"
        )

        st.code(
            config.base_url,
            language=None,
        )

        st.caption(
            "Timeout: "
            f"{config.timeout_seconds:.1f}초"
        )

        st.divider()

        st.caption(
            "이 화면은 AI 모델을 직접 실행하지 않습니다. "
            "입력한 내용은 FastAPI Backend로 전달되며, "
            "Backend가 AI 예측, Agent 처리, 실행 이력 저장을 담당합니다. "
            "Dashboard는 Backend가 반환한 결과를 "
            "사용자가 이해하기 쉽게 보여주는 역할을 합니다."
        )

    prediction_page = st.Page(
        PAGES_DIRECTORY
        / "failure_prediction.py",
        title="설비 고장 위험 분석",
        default=True,
    )

    evidence_page = st.Page(
        PAGES_DIRECTORY
        / "evidence_analysis.py",
        title="Evidence 분석",
    )

    agent_chat_page = st.Page(
        PAGES_DIRECTORY
        / "agent_chat.py",
        title="AI 질의 응답",
    )

    execution_history_page = st.Page(
        PAGES_DIRECTORY
        / "execution_history.py",
        title="Trace·Execution History",
    )

    navigation = st.navigation(
        {
            "Prediction": [
                prediction_page,
                evidence_page,
            ],
            "Agent": [
                agent_chat_page,
            ],
            "Observability": [
                execution_history_page,
            ],
        }
    )

    navigation.run()


main()

