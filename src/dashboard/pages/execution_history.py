# src/dashboard/pages/execution_history.py

"""
LangGraph Agent 실행 이력과 Trace를 조회하는 Streamlit Page입니다.

이 Page의 역할:
- 최근 Agent 실행 이력 목록 조회
- Trace ID 선택
- 선택한 실행 상세 조회
- Answer·Evidence·Trace Event·Warning·Error 표시
- Loading·Success·Error 상태 표시

중요:
- SQLite를 직접 조회하지 않습니다.
- SQL Query를 직접 실행하지 않습니다.
- Trace를 다시 생성하지 않습니다.
- 기존 FastAPI Persistence Endpoint를 사용합니다.

요청 흐름:

    Streamlit
        ↓
    DashboardApiClient
        ↓
    GET /agent/executions
        ↓
    Execution Summary 목록

    Trace ID 선택
        ↓
    GET /agent/executions/{trace_id}
        ↓
    Execution Detail
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.api_client import (
    DashboardApiClient,
    DashboardApiClientError,
)
from src.dashboard.session_state import (
    initialize_dashboard_session_state,
)


EXECUTION_TABLE_COLUMNS = [
    "created_at",
    "trace_id",
    "question",
    "intent",
    "intent_source",
    "selected_route",
    "prediction",
    "probability",
    "risk_level",
    "trace_status",
    "fallback_occurred",
    "trace_duration_ms",
    "warning_count",
    "error_count",
]


def build_execution_table(
    executions: list[
        dict[str, Any]
    ],
) -> pd.DataFrame:
    """
    실행 이력 Summary를 화면 표시용 DataFrame으로 정리합니다.

    Backend 값을 다시 계산하지 않고,
    목록 화면에서 필요한 필드 순서만 정리합니다.
    """

    rows: list[
        dict[str, Any]
    ] = []

    for execution in executions:
        rows.append(
            {
                column: execution.get(
                    column,
                )
                for column in (
                    EXECUTION_TABLE_COLUMNS
                )
            }
        )

    return pd.DataFrame(
        rows,
        columns=(
            EXECUTION_TABLE_COLUMNS
        ),
    )


def render_message_list(
    *,
    title: str,
    items: Any,
    message_type: str,
) -> None:
    """
    실행 상세 Response의 Warning·Error·Limitation을 표시합니다.
    """

    if not isinstance(
        items,
        list,
    ):
        return

    normalized_items = [
        str(item).strip()
        for item in items
        if str(item).strip()
    ]

    if not normalized_items:
        return

    st.markdown(
        f"**{title}**"
    )

    for item in normalized_items:
        if message_type == "error":
            st.error(
                item,
            )
        elif message_type == "warning":
            st.warning(
                item,
            )
        else:
            st.info(
                item,
            )


def render_execution_detail(
    detail: dict[str, Any],
) -> None:
    """
    선택한 Agent 실행의 상세 Response를 표시합니다.
    """

    st.divider()

    st.subheader(
        "실행 상세",
    )

    summary_columns = st.columns(
        4,
    )

    summary_columns[0].metric(
        "Intent",
        str(
            detail.get(
                "intent",
            )
            or "-"
        ),
    )

    summary_columns[1].metric(
        "Risk Level",
        str(
            detail.get(
                "risk_level",
            )
            or "-"
        ),
    )

    summary_columns[2].metric(
        "Trace Status",
        str(
            detail.get(
                "trace_status",
            )
            or "-"
        ),
    )

    duration = detail.get(
        "trace_duration_ms",
    )

    summary_columns[3].metric(
        "Duration [ms]",
        (
            f"{float(duration):.2f}"
            if duration is not None
            else "-"
        ),
    )

    st.markdown(
        "**Question**"
    )

    st.write(
        str(
            detail.get(
                "question",
                "-",
            )
        )
    )

    answer = detail.get(
        "answer",
    )

    if answer:
        st.markdown(
            "**Answer**"
        )

        st.write(
            str(
                answer,
            )
        )

    recommended_action = (
        detail.get(
            "recommended_action",
        )
    )

    if recommended_action:
        st.markdown(
            "**Recommended Action**"
        )

        st.info(
            str(
                recommended_action,
            )
        )

    with st.expander(
        "실행 메타데이터",
        expanded=False,
    ):
        metadata = {
            "id": detail.get(
                "id",
            ),
            "trace_id": detail.get(
                "trace_id",
            ),
            "created_at": detail.get(
                "created_at",
            ),
            "intent_source": detail.get(
                "intent_source",
            ),
            "confidence": detail.get(
                "confidence",
            ),
            "intent_reason": detail.get(
                "intent_reason",
            ),
            "selected_route": detail.get(
                "selected_route",
            ),
            "prediction": detail.get(
                "prediction",
            ),
            "probability": detail.get(
                "probability",
            ),
            "threshold": detail.get(
                "threshold",
            ),
            "fallback_occurred": (
                detail.get(
                    "fallback_occurred",
                    False,
                )
            ),
            "trace_started_at": (
                detail.get(
                    "trace_started_at",
                )
            ),
            "trace_finished_at": (
                detail.get(
                    "trace_finished_at",
                )
            ),
        }

        st.json(
            metadata,
        )

    raw_sample = detail.get(
        "raw_sample",
    )

    if isinstance(
        raw_sample,
        dict,
    ) and raw_sample:
        with st.expander(
            "Raw Sample",
            expanded=False,
        ):
            st.json(
                raw_sample,
            )

    evidence = detail.get(
        "evidence",
    )

    if isinstance(
        evidence,
        list,
    ) and evidence:
        st.subheader(
            "Evidence",
        )

        st.dataframe(
            pd.DataFrame(
                evidence,
            ),
            use_container_width=True,
            hide_index=True,
        )

    trace_events = detail.get(
        "trace_events",
    )

    if isinstance(
        trace_events,
        list,
    ) and trace_events:
        st.subheader(
            "Trace Events",
        )

        st.dataframe(
            pd.DataFrame(
                trace_events,
            ),
            use_container_width=True,
            hide_index=True,
        )

        for event in trace_events:
            event_name = (
                event.get(
                    "event_name",
                )
                or "Trace Event"
            )

            sequence = event.get(
                "sequence",
                "-",
            )

            with st.expander(
                f"{sequence}. {event_name}",
                expanded=False,
            ):
                st.json(
                    event,
                )

    render_message_list(
        title="Warnings",
        items=detail.get(
            "warnings",
        ),
        message_type="warning",
    )

    render_message_list(
        title="Errors",
        items=detail.get(
            "errors",
        ),
        message_type="error",
    )

    render_message_list(
        title="Limitations",
        items=detail.get(
            "limitations",
        ),
        message_type="info",
    )


def main() -> None:
    """
    Trace·Execution History Page를 렌더링합니다.
    """

    initialize_dashboard_session_state(
        st.session_state,
    )

    st.title(
        "Trace·Execution History",
    )

    st.write(
        "기존 FastAPI Persistence Endpoint를 통해 "
        "LangGraph Agent 실행 이력과 Trace를 조회합니다."
    )

    st.caption(
        "Dashboard는 SQLite를 직접 조회하지 않습니다."
    )

    limit = st.number_input(
        "최근 실행 조회 개수",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
    )

    refresh_clicked = st.button(
        "실행 이력 새로고침",
        use_container_width=True,
    )

    if refresh_clicked:
        try:
            with st.spinner(
                "최근 Agent 실행 이력을 조회하고 있습니다...",
            ):
                with DashboardApiClient() as api_client:
                    executions = (
                        api_client.get_agent_executions(
                            limit=int(
                                limit,
                            ),
                        )
                    )

        except DashboardApiClientError as exc:
            st.error(
                str(
                    exc,
                )
            )

        else:
            st.session_state[
                "execution_history"
            ] = executions

            st.session_state[
                "selected_execution_trace_id"
            ] = None

            st.session_state[
                "selected_execution_detail"
            ] = None

            st.success(
                "Agent 실행 이력 조회가 완료되었습니다."
            )

    executions = st.session_state.get(
        "execution_history",
    )

    if not isinstance(
        executions,
        list,
    ) or not executions:
        st.info(
            "'실행 이력 새로고침' 버튼을 눌러 "
            "최근 Agent 실행을 조회해주세요."
        )
        return

    st.subheader(
        "최근 실행 목록",
    )

    execution_table = (
        build_execution_table(
            executions,
        )
    )

    st.dataframe(
        execution_table,
        use_container_width=True,
        hide_index=True,
    )

    trace_ids = [
        str(
            execution.get(
                "trace_id",
            )
        )
        for execution in executions
        if execution.get(
            "trace_id",
        )
    ]

    if not trace_ids:
        st.warning(
            "상세 조회에 사용할 Trace ID가 없습니다."
        )
        return

    selected_trace_id = st.selectbox(
        "Trace ID 선택",
        options=trace_ids,
    )

    st.session_state[
        "selected_execution_trace_id"
    ] = selected_trace_id

    detail_clicked = st.button(
        "선택 실행 상세 조회",
        use_container_width=True,
    )

    if detail_clicked:
        try:
            with st.spinner(
                "선택한 Agent 실행 상세를 조회하고 있습니다...",
            ):
                with DashboardApiClient() as api_client:
                    detail = (
                        api_client.get_agent_execution_detail(
                            trace_id=(
                                selected_trace_id
                            ),
                        )
                    )

        except (
            DashboardApiClientError,
            ValueError,
        ) as exc:
            st.error(
                str(
                    exc,
                )
            )

        else:
            st.session_state[
                "selected_execution_detail"
            ] = detail

            st.success(
                "Agent 실행 상세 조회가 완료되었습니다."
            )

    selected_detail = (
        st.session_state.get(
            "selected_execution_detail",
        )
    )

    if isinstance(
        selected_detail,
        dict,
    ):
        render_execution_detail(
            selected_detail,
        )


main()
