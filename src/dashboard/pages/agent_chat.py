# src/dashboard/pages/agent_chat.py

"""
Beginner-first LangGraph Agent chat page.

Important:
- The Dashboard calls the existing FastAPI API.
- It does not run LangGraph directly.
- It does not reuse a previous raw_sample automatically.
- Display chat history and Backend context history are separated.
"""

from __future__ import annotations

import json
import math
from typing import Any

import streamlit as st

from src.dashboard.api_client import (
    DashboardApiClient,
    DashboardApiClientError,
)
from src.dashboard.session_state import (
    initialize_dashboard_session_state,
)
from src.dashboard.ui_helpers import (
    build_langgraph_agent_payload,
    build_raw_sample_payload,
)


INTENT_LABELS = {
    "failure_prediction": "설비 고장 위험 예측",
    "dataset_schema_query": "데이터셋 구조 설명",
    "unknown": "지원 범위 밖 질문",
}

TRACE_STATUS_LABELS = {
    "success": "처리 완료",
    "fallback": "추가 정보 필요",
    "failure": "처리 실패",
}


def format_optional_value(
    value: Any,
    *,
    digits: int = 4,
) -> str:
    """Format an optional finite numeric value."""
    if value is None:
        return "-"

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )

    if not math.isfinite(
        numeric_value
    ):
        return "-"

    return (
        f"{numeric_value:.{digits}f}"
    )


def format_probability(
    value: Any,
) -> str:
    """Format a probability-like ratio as a percentage."""
    if value is None:
        return "-"

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )

    if not math.isfinite(
        numeric_value
    ):
        return "-"

    return (
        f"{numeric_value * 100:.2f}%"
    )


def get_intent_label(
    intent: Any,
) -> str:
    """Return a beginner-friendly intent label."""
    normalized = str(
        intent
        or "unknown"
    ).strip()

    return INTENT_LABELS.get(
        normalized,
        normalized,
    )


def get_trace_status_label(
    status: Any,
) -> str:
    """Return a beginner-friendly processing status."""
    normalized = str(
        status
        or "unknown"
    ).strip().lower()

    return TRACE_STATUS_LABELS.get(
        normalized,
        "상태 확인 필요",
    )


def contains_raw_sample_error(
    result: dict[str, Any],
) -> bool:
    """Detect the confirmed Backend missing-raw-sample fallback."""
    texts: list[
        str
    ] = []

    for key in (
        "answer",
        "recommended_action",
    ):
        value = result.get(
            key
        )

        if value:
            texts.append(
                str(
                    value
                )
            )

    errors = result.get(
        "errors"
    )

    if isinstance(
        errors,
        list,
    ):
        texts.extend(
            str(
                error
            )
            for error in errors
        )

    combined = " ".join(
        texts
    ).lower()

    return (
        "raw_sample"
        in combined
        or "raw sample"
        in combined
    )


def build_display_answer(
    result: dict[str, Any],
) -> str:
    """
    Build a concise display answer without changing Backend results.

    The original Backend answer is still stored separately
    for the next request's context history.
    """
    if contains_raw_sample_error(
        result
    ):
        return (
            "이번 질문에는 설비 입력값이 함께 전달되지 않아 "
            "고장 위험을 계산하지 못했습니다.\n\n"
            "고장 위험을 예측하려면 위의 "
            "**'이번 질문에 설비 입력값 함께 보내기'**를 선택한 뒤, "
            "현재 설비 값을 확인하고 질문을 다시 보내세요.\n\n"
            "대화 기록은 질문의 문맥만 전달합니다. "
            "이전 질문에서 사용한 설비 값은 안전을 위해 자동으로 다시 사용하지 않습니다."
        )

    intent = str(
        result.get(
            "intent"
        )
        or ""
    ).strip()

    if intent == "unknown":
        return (
            "이 질문은 현재 Agent가 지원하는 범위에서 답하기 어렵습니다.\n\n"
            "현재는 다음 두 종류의 질문을 지원합니다.\n\n"
            "- 설비 입력값을 함께 보낸 고장 위험 예측\n"
            "- AI4I 데이터셋의 입력 항목, 예측 대상, 구조 설명\n\n"
            "고장 위험 질문이라면 설비 입력값 포함 옵션을 선택하고, "
            "데이터셋 질문이라면 예를 들어 "
            "'AI4I 데이터셋은 어떤 입력값을 사용해?'처럼 질문해보세요."
        )

    answer = str(
        result.get(
            "answer",
            "",
        )
    ).strip()

    if answer:
        return answer

    return (
        "Agent 응답에 화면에 표시할 설명이 없습니다. "
        "아래 기술 상세에서 Backend 응답을 확인하세요."
    )


def get_original_context_answer(
    result: dict[str, Any],
) -> str:
    """
    Return the original Backend answer for context history.

    Display copy and Backend context copy are intentionally separated.
    """
    answer = str(
        result.get(
            "answer",
            "",
        )
    ).strip()

    if answer:
        return answer

    return build_display_answer(
        result
    )


def render_page_intro() -> None:
    """Explain the supported tasks and the raw-sample rule."""
    st.title(
        "AI 질의 응답"
    )

    st.write(
        "자연어로 질문하면 기존 LangGraph Agent API가 "
        "질문 유형을 판단하고 지원 가능한 작업을 처리합니다."
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "#### 먼저 알아둘 점"
        )

        st.write(
            "대화 기록은 이전 질문의 뜻을 이해하는 데 사용됩니다. "
            "하지만 이전 질문에서 사용한 설비 입력값은 다음 고장 예측에 자동으로 재사용하지 않습니다."
        )

        st.write(
            "고장 위험을 예측하려면 질문할 때마다 "
            "**'이번 질문에 설비 입력값 함께 보내기'**를 선택해야 합니다."
        )

        st.caption(
            "질문 문장에 'Torque 62, Tool wear 220'이라고 적는 것만으로는 "
            "구조화된 설비 입력값이 자동 생성되지 않습니다."
        )


def render_question_guide() -> None:
    """Show clear examples by supported task."""
    with st.expander(
        "질문 예시와 사용 방법",
        expanded=False,
    ):
        st.markdown(
            "#### 데이터셋을 물어볼 때"
        )

        st.write(
            "설비 입력값을 함께 보낼 필요가 없습니다."
        )

        st.code(
            "AI4I 데이터셋은 어떤 입력값을 사용해?\n"
            "예측 대상은 무엇이야?\n"
            "제품 유형 L, M, H는 무슨 뜻이야?",
            language=None,
        )

        st.markdown(
            "#### 고장 위험을 물어볼 때"
        )

        st.write(
            "먼저 '이번 질문에 설비 입력값 함께 보내기'를 선택하고 "
            "현재 값을 확인한 뒤 질문합니다."
        )

        st.code(
            "현재 입력값으로 고장 위험을 예측해줘.\n"
            "이번 설비 상태에서 무엇을 먼저 점검해야 해?",
            language=None,
        )

        st.markdown(
            "#### 후속 질문을 할 때"
        )

        st.write(
            "Agent는 이전 대화의 문장을 참고할 수 있습니다. "
            "다만 새 고장 예측에는 현재 요청의 설비 입력값이 다시 필요합니다."
        )


def render_request_options() -> tuple[
    bool,
    dict[str, Any],
]:
    """Render current-request-only raw sample and evidence options."""
    with st.expander(
        "이번 질문에 사용할 설비 입력값",
        expanded=True,
    ):
        include_raw_sample = (
            st.checkbox(
                "이번 질문에 설비 입력값 함께 보내기",
                value=False,
                help=(
                    "고장 위험을 예측하는 현재 질문에서만 선택하세요. "
                    "이 값은 다음 질문에 자동으로 재사용되지 않습니다."
                ),
            )
        )

        if include_raw_sample:
            st.write(
                "아래 값은 이번 질문에만 함께 전달됩니다."
            )

        else:
            st.write(
                "현재는 설비 입력값을 보내지 않습니다. "
                "데이터셋 구조 질문은 그대로 할 수 있지만, "
                "고장 위험 예측은 입력값 부족으로 처리되지 않습니다."
            )

        input_columns = st.columns(
            2
        )

        with input_columns[0]:
            air_temperature = (
                st.number_input(
                    "공기 온도 (K)",
                    value=303.0,
                    step=0.1,
                    disabled=(
                        not include_raw_sample
                    ),
                    help=(
                        "설비 주변 공기의 절대 온도입니다. "
                        "303 K는 약 29.9°C입니다."
                    ),
                )
            )

            process_temperature = (
                st.number_input(
                    "공정 온도 (K)",
                    value=312.5,
                    step=0.1,
                    disabled=(
                        not include_raw_sample
                    ),
                    help=(
                        "공정에서 측정한 절대 온도입니다. "
                        "312.5 K는 약 39.4°C입니다."
                    ),
                )
            )

            rotational_speed = (
                st.number_input(
                    "회전 속도 (rpm)",
                    value=1380.0,
                    step=1.0,
                    disabled=(
                        not include_raw_sample
                    ),
                    help=(
                        "1분 동안의 회전 횟수입니다."
                    ),
                )
            )

        with input_columns[1]:
            torque = st.number_input(
                "토크 (Nm)",
                value=62.0,
                step=0.1,
                disabled=(
                    not include_raw_sample
                ),
                help=(
                    "회전축에 걸리는 힘의 크기입니다."
                ),
            )

            tool_wear = (
                st.number_input(
                    "공구 마모 시간 (분)",
                    value=220.0,
                    step=1.0,
                    disabled=(
                        not include_raw_sample
                    ),
                    help=(
                        "공구 사용과 관련된 누적 시간 입력값입니다."
                    ),
                )
            )

            machine_type = (
                st.selectbox(
                    "제품 유형",
                    options=[
                        "L",
                        "M",
                        "H",
                    ],
                    index=0,
                    disabled=(
                        not include_raw_sample
                    ),
                    help=(
                        "AI4I 학습 데이터에서 사용한 제품 유형 코드입니다."
                    ),
                )
            )

        with st.expander(
            "판단 근거 상세 옵션",
            expanded=False,
        ):
            st.write(
                "처음 보는 사용자는 기본 설정을 그대로 사용해도 됩니다."
            )

            option_columns = st.columns(
                2
            )

            with option_columns[0]:
                include_shap = (
                    st.checkbox(
                        "이번 입력값의 영향 분석 포함",
                        value=True,
                        help=(
                            "각 입력값이 이번 AI 판단을 "
                            "높이거나 낮춘 방향을 함께 요청합니다."
                        ),
                    )
                )

            with option_columns[1]:
                include_global_importance = (
                    st.checkbox(
                        "전체 데이터 중요도 포함",
                        value=True,
                        help=(
                            "전체 참고 데이터에서 AI가 중요하게 본 "
                            "입력값 정보를 함께 요청합니다."
                        ),
                    )
                )

    values = {
        "air_temperature": (
            air_temperature
        ),
        "process_temperature": (
            process_temperature
        ),
        "rotational_speed": (
            rotational_speed
        ),
        "torque": torque,
        "tool_wear": (
            tool_wear
        ),
        "machine_type": (
            machine_type
        ),
        "include_shap": (
            include_shap
        ),
        "include_global_importance": (
            include_global_importance
        ),
    }

    return (
        include_raw_sample,
        values,
    )


def render_chat_messages(
    messages: Any,
) -> None:
    """Render only user-facing chat content."""
    if not isinstance(
        messages,
        list,
    ):
        return

    for message in messages:
        if not isinstance(
            message,
            dict,
        ):
            continue

        role = str(
            message.get(
                "role",
                "assistant",
            )
        )

        if role not in {
            "user",
            "assistant",
        }:
            role = "assistant"

        content = str(
            message.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        with st.chat_message(
            role
        ):
            st.markdown(
                content
            )


def render_prediction_summary(
    result: dict[str, Any],
) -> None:
    """Render prediction values only when the Agent returned them."""
    prediction = result.get(
        "prediction"
    )

    probability = result.get(
        "probability"
    )

    risk_level = result.get(
        "risk_level"
    )

    if (
        prediction is None
        and probability is None
        and risk_level is None
    ):
        return

    with st.container(
        border=True
    ):
        st.markdown(
            "#### 예측 결과"
        )

        first_row = st.columns(
            2
        )

        first_row[0].metric(
            "판정",
            (
                "고장 위험 있음"
                if prediction == 1
                else (
                    "고장 위험 낮음"
                    if prediction == 0
                    else "확인 필요"
                )
            ),
        )

        first_row[1].metric(
            "위험 단계",
            str(
                risk_level
                or "UNKNOWN"
            ),
        )

        second_row = st.columns(
            2
        )

        second_row[0].metric(
            "AI 모델 위험 점수",
            format_probability(
                probability
            ),
        )

        second_row[1].metric(
            "위험 판정 기준선",
            format_probability(
                result.get(
                    "threshold"
                )
            ),
        )

        st.caption(
            "AI 모델 위험 점수는 실제 고장 발생을 확정하는 값이 아닙니다."
        )


def render_string_list(
    *,
    title: str,
    items: Any,
) -> None:
    """Render a compact string list inside technical details."""
    if not isinstance(
        items,
        list,
    ):
        return

    normalized = [
        str(
            item
        ).strip()
        for item in items
        if str(
            item
        ).strip()
    ]

    if not normalized:
        return

    st.markdown(
        f"**{title}**"
    )

    for item in normalized:
        st.write(
            f"- {item}"
        )


def render_agent_result(
    result: dict[str, Any],
) -> None:
    """
    Render a compact user summary and keep metadata in one technical expander.
    """
    st.divider()

    with st.container(
        border=True
    ):
        st.markdown(
            "### 마지막 요청 상태"
        )

        status = get_trace_status_label(
            result.get(
                "trace_status"
            )
        )

        intent_label = get_intent_label(
            result.get(
                "intent"
            )
        )

        st.write(
            f"**처리 상태:** {status}"
        )

        st.write(
            f"**처리한 질문 유형:** {intent_label}"
        )

        if contains_raw_sample_error(
            result
        ):
            st.caption(
                "고장 위험 예측 질문으로 분류됐지만, "
                "이번 요청에 설비 입력값이 포함되지 않아 예측은 실행되지 않았습니다."
            )

        elif (
            result.get(
                "trace_status"
            )
            == "success"
        ):
            st.caption(
                "Agent가 요청을 정상적으로 처리했습니다."
            )

    render_prediction_summary(
        result
    )

    with st.expander(
        "기술 상세 보기",
        expanded=False,
    ):
        st.caption(
            "아래 내용은 개발자와 기술 검토자를 위한 정보입니다. "
            "일반 사용자는 위 대화 내용만 확인해도 됩니다."
        )

        st.markdown(
            "#### 질문 분류 정보"
        )

        detail_rows = [
            (
                "분류 결과",
                result.get(
                    "intent"
                )
                or "-",
            ),
            (
                "분류 신뢰도",
                format_optional_value(
                    result.get(
                        "confidence"
                    )
                ),
            ),
            (
                "분류 출처",
                result.get(
                    "intent_source"
                )
                or "-",
            ),
            (
                "처리 상태",
                result.get(
                    "trace_status"
                )
                or "-",
            ),
            (
                "처리 시간",
                (
                    format_optional_value(
                        result.get(
                            "trace_duration_ms"
                        ),
                        digits=2,
                    )
                    + " ms"
                ),
            ),
        ]

        for (
            label,
            value,
        ) in detail_rows:
            st.write(
                f"**{label}:** {value}"
            )

        intent_reason = result.get(
            "intent_reason"
        )

        if intent_reason:
            st.markdown(
                "#### 분류 이유"
            )

            st.write(
                str(
                    intent_reason
                )
            )

        recommended_action = (
            result.get(
                "recommended_action"
            )
        )

        if recommended_action:
            st.markdown(
                "#### Backend 권장 조치"
            )

            st.write(
                str(
                    recommended_action
                )
            )

        trace_data = {
            "trace_id": result.get(
                "trace_id"
            ),
            "trace_status": result.get(
                "trace_status"
            ),
            "trace_started_at": result.get(
                "trace_started_at"
            ),
            "trace_finished_at": result.get(
                "trace_finished_at"
            ),
            "trace_duration_ms": result.get(
                "trace_duration_ms"
            ),
            "fallback_occurred": result.get(
                "fallback_occurred",
                False,
            ),
        }

        st.markdown(
            "#### 실행 추적 정보"
        )

        st.code(
            json.dumps(
                trace_data,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            language="json",
        )

        evidence = result.get(
            "evidence"
        )

        if isinstance(
            evidence,
            list,
        ) and evidence:
            st.markdown(
                "#### 판단 근거 원본"
            )

            st.code(
                json.dumps(
                    evidence,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                language="json",
            )

        trace_events = result.get(
            "trace_events"
        )

        if isinstance(
            trace_events,
            list,
        ) and trace_events:
            st.markdown(
                "#### 실행 단계 기록"
            )

            st.code(
                json.dumps(
                    trace_events,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                language="json",
            )

        render_string_list(
            title="경고",
            items=result.get(
                "warnings"
            ),
        )

        render_string_list(
            title="오류",
            items=result.get(
                "errors"
            ),
        )

        render_string_list(
            title="한계",
            items=result.get(
                "limitations"
            ),
        )


def ensure_chat_context_state() -> None:
    """Initialize Backend-only context history without reusing raw samples."""
    if (
        "agent_context_messages"
        not in st.session_state
    ):
        st.session_state[
            "agent_context_messages"
        ] = []


def main() -> None:
    """Render the LangGraph Agent chat page."""
    initialize_dashboard_session_state(
        st.session_state
    )

    ensure_chat_context_state()

    render_page_intro()

    render_question_guide()

    (
        include_raw_sample,
        values,
    ) = render_request_options()

    st.divider()

    st.subheader(
        "대화"
    )

    st.caption(
        "아래에는 질문과 쉬운 답변만 표시합니다. "
        "분류 정보와 실행 기록은 맨 아래 기술 상세에서 확인할 수 있습니다."
    )

    chat_history_container = (
        st.container()
    )

    question = st.chat_input(
        "질문을 입력하세요."
    )

    if question:
        question_text = (
            question.strip()
        )

        previous_context_messages = list(
            st.session_state.get(
                "agent_context_messages",
                [],
            )
        )

        raw_sample = None

        if include_raw_sample:
            raw_sample = (
                build_raw_sample_payload(
                    air_temperature=(
                        values[
                            "air_temperature"
                        ]
                    ),
                    process_temperature=(
                        values[
                            "process_temperature"
                        ]
                    ),
                    rotational_speed=(
                        values[
                            "rotational_speed"
                        ]
                    ),
                    torque=(
                        values[
                            "torque"
                        ]
                    ),
                    tool_wear=(
                        values[
                            "tool_wear"
                        ]
                    ),
                    machine_type=(
                        values[
                            "machine_type"
                        ]
                    ),
                )
            )

        try:
            payload = (
                build_langgraph_agent_payload(
                    question=question_text,
                    chat_history=(
                        previous_context_messages
                    ),
                    raw_sample=raw_sample,
                    include_shap=(
                        values[
                            "include_shap"
                        ]
                    ),
                    include_global_importance=(
                        values[
                            "include_global_importance"
                        ]
                    ),
                )
            )

            st.session_state[
                "chat_messages"
            ].append(
                {
                    "role": "user",
                    "content": (
                        question_text
                    ),
                }
            )

            with st.spinner(
                "Agent가 질문을 처리하고 있습니다..."
            ):
                with (
                    DashboardApiClient()
                    as api_client
                ):
                    result = (
                        api_client
                        .query_langgraph_agent(
                            payload
                        )
                    )

        except (
            DashboardApiClientError,
            ValueError,
        ) as exc:
            display_error = (
                "요청을 처리하지 못했습니다. "
                "FastAPI 연결 상태와 입력 내용을 확인하세요."
            )

            st.session_state[
                "chat_messages"
            ].append(
                {
                    "role": "assistant",
                    "content": (
                        display_error
                    ),
                }
            )

            st.error(
                display_error
            )

            with st.expander(
                "개발자용 오류 정보",
                expanded=False,
            ):
                st.write(
                    str(
                        exc
                    )
                )

        else:
            display_answer = (
                build_display_answer(
                    result
                )
            )

            original_answer = (
                get_original_context_answer(
                    result
                )
            )

            st.session_state[
                "chat_messages"
            ].append(
                {
                    "role": "assistant",
                    "content": (
                        display_answer
                    ),
                }
            )

            st.session_state[
                "agent_context_messages"
            ].extend(
                [
                    {
                        "role": "user",
                        "content": (
                            question_text
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": (
                            original_answer
                        ),
                    },
                ]
            )

            st.session_state[
                "last_agent_result"
            ] = result

            st.success(
                "Agent 요청이 완료되었습니다."
            )

    with chat_history_container:
        render_chat_messages(
            st.session_state.get(
                "chat_messages"
            )
        )

    last_agent_result = (
        st.session_state.get(
            "last_agent_result"
        )
    )

    if isinstance(
        last_agent_result,
        dict,
    ):
        render_agent_result(
            last_agent_result
        )


main()
