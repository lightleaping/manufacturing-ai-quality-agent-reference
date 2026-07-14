# src/dashboard/ui_helpers.py

"""
Streamlit Dashboard에서 사용할 순수 UI Helper 함수입니다.

이 모듈은 Streamlit Widget과 FastAPI Client 사이에서
데이터 구조를 정리하는 역할을 담당합니다.

주요 역할:
- 설비 입력값을 FastAPI Request Payload로 변환
- Direct Failure Prediction Payload 생성
- Chat History 정규화
- LangGraph Agent Request Payload 생성
- Evidence를 evidence_type 기준으로 분류

중요:
- Streamlit 화면을 직접 렌더링하지 않습니다.
- FastAPI를 직접 호출하지 않습니다.
- PyTorch 모델을 직접 실행하지 않습니다.
- LangGraph workflow를 직접 실행하지 않습니다.
- SQLite를 직접 조회하지 않습니다.
- Prediction, Risk Level, Evidence를 다시 계산하지 않습니다.

요청 흐름:

    Streamlit Widget
        ↓
    UI Helper
        ↓
    DashboardApiClient
        ↓
    Existing FastAPI
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


# FastAPI ChatMessageRequest가 허용하는 role입니다.
#
# 사용자 입력으로 system·developer role을 전달하지 않도록
# Dashboard에서도 동일한 계약을 유지합니다.
SUPPORTED_CHAT_ROLES = {
    "user",
    "assistant",
}

# FastAPI ChatMessageRequest의 content 최대 길이입니다.
MAX_CHAT_MESSAGE_LENGTH = 1000

# FastAPI LangGraphAgentQueryRequest가 허용하는
# chat_history 최대 메시지 개수입니다.
MAX_CHAT_HISTORY_MESSAGES = 6


def build_raw_sample_payload(
    *,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    machine_type: str,
) -> dict[str, Any]:
    """
    Dashboard 설비 입력값을 FastAPI Raw Sample JSON으로 변환합니다.

    Parameters
    ----------
    air_temperature:
        공기 온도입니다.

    process_temperature:
        공정 온도입니다.

    rotational_speed:
        회전 속도입니다.

    torque:
        토크입니다.

    tool_wear:
        공구 마모 시간입니다.

    machine_type:
        Dashboard Python 코드에서 사용하는 설비 Type 변수입니다.

    Returns
    -------
    dict[str, Any]
        FastAPI Request에 포함할 Raw Sample JSON입니다.

    중요
    ----
    Python UI 변수명은 machine_type이지만,
    실제 FastAPI JSON 필드명은 type입니다.

    변환:

        Dashboard Python

        machine_type

                ↓

        FastAPI JSON

        "type"
    """

    normalized_machine_type = (
        str(
            machine_type,
        )
        .strip()
        .upper()
    )

    return {
        "air_temperature": air_temperature,
        "process_temperature": process_temperature,
        "rotational_speed": rotational_speed,
        "torque": torque,
        "tool_wear": tool_wear,
        "type": normalized_machine_type,
    }


def build_failure_prediction_payload(
    *,
    air_temperature: float,
    process_temperature: float,
    rotational_speed: float,
    torque: float,
    tool_wear: float,
    machine_type: str,
    include_shap: bool = True,
    include_global_importance: bool = True,
) -> dict[str, Any]:
    """
    Direct Failure Prediction API Request Payload를 생성합니다.

    Endpoint:

        POST /agent/failure-prediction

    이 함수는 설비 입력값과 Evidence 옵션을 하나의
    JSON Payload 구조로 결합합니다.

    Prediction이나 Risk Level은 계산하지 않습니다.

    해당 값은 기존 FastAPI Backend가 계산하고,
    Dashboard는 Response를 표시만 합니다.
    """

    payload = build_raw_sample_payload(
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
        machine_type=machine_type,
    )

    payload.update(
        {
            "include_shap": bool(
                include_shap,
            ),
            "include_global_importance": bool(
                include_global_importance,
            ),
        }
    )

    return payload


def normalize_chat_history(
    messages: Iterable[
        Mapping[str, Any]
    ]
    | None,
) -> list[dict[str, str]]:
    """
    Session State의 대화 기록을 FastAPI 계약에 맞게 정규화합니다.

    정규화 규칙:
    1. role은 user·assistant만 유지합니다.
    2. content는 문자열만 사용합니다.
    3. content 앞뒤 공백을 제거합니다.
    4. 공백 제거 후 빈 메시지는 제외합니다.
    5. content는 최대 1000자로 제한합니다.
    6. 최종적으로 최근 메시지 6개만 유지합니다.

    Chat History의 목적:
        현재 질문의 대화 문맥 이해

    Chat History가 하지 않는 일:
        이전 Raw Sample 자동 재사용

    따라서 이 함수는 대화 내용만 정규화하며,
    설비 입력값을 추출하거나 Prediction 입력으로 변환하지 않습니다.
    """

    if messages is None:
        return []

    normalized_messages: list[
        dict[str, str]
    ] = []

    for message in messages:
        role = message.get(
            "role",
        )

        content = message.get(
            "content",
        )

        if role not in SUPPORTED_CHAT_ROLES:
            continue

        if not isinstance(
            content,
            str,
        ):
            continue

        normalized_content = (
            content.strip()
        )

        if not normalized_content:
            continue

        normalized_messages.append(
            {
                "role": role,
                "content": (
                    normalized_content[
                        :MAX_CHAT_MESSAGE_LENGTH
                    ]
                ),
            }
        )

    return normalized_messages[
        -MAX_CHAT_HISTORY_MESSAGES:
    ]


def build_langgraph_agent_payload(
    *,
    question: str,
    chat_history: Iterable[
        Mapping[str, Any]
    ]
    | None,
    raw_sample: Mapping[
        str,
        Any,
    ]
    | None,
    include_shap: bool = True,
    include_global_importance: bool = True,
) -> dict[str, Any]:
    """
    LangGraph Agent Query API Request Payload를 생성합니다.

    Endpoint:

        POST /agent/langgraph-query

    중요한 정책:
    - chat_history는 질문 문맥 이해용입니다.
    - raw_sample은 현재 Prediction 요청의 설비 입력값입니다.
    - 이전 요청의 raw_sample은 자동으로 재사용하지 않습니다.
    - raw_sample이 None이면 Request Payload에도 포함하지 않습니다.
    """

    normalized_question = (
        str(
            question,
        )
        .strip()
    )

    if not normalized_question:
        raise ValueError(
            "question은 비어 있을 수 없습니다."
        )

    payload: dict[str, Any] = {
        "question": normalized_question,
        "chat_history": (
            normalize_chat_history(
                chat_history,
            )
        ),
        "include_shap": bool(
            include_shap,
        ),
        "include_global_importance": bool(
            include_global_importance,
        ),
    }

    # Raw Sample은 현재 요청에서 명시적으로 전달된 경우에만
    # LangGraph Agent Payload에 포함합니다.
    #
    # Session State에 과거 설비값이 존재하더라도
    # 이 함수가 자동으로 가져오거나 재사용하지 않습니다.
    if raw_sample is not None:
        payload["raw_sample"] = dict(
            raw_sample,
        )

    return payload


def group_evidence_by_type(
    evidence_items: Iterable[
        Mapping[str, Any]
    ]
    | None,
) -> dict[
    str,
    list[dict[str, Any]],
]:
    """
    FastAPI Evidence 목록을 evidence_type 기준으로 분류합니다.

    예:

        {
            "prediction_summary": [
                {...}
            ],
            "rule_based": [
                {...}
            ],
            "shap_local": [
                {...}
            ],
            "global_importance": [
                {...}
            ]
        }

    알려지지 않은 Evidence Type도 제거하지 않습니다.

    이유:
    향후 Backend에 새로운 Evidence Type이 추가되더라도
    Dashboard에서 데이터를 손실하지 않고 표시할 수 있어야 합니다.
    """

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    if evidence_items is None:
        return grouped

    for evidence in evidence_items:
        evidence_type = evidence.get(
            "evidence_type",
        )

        if not isinstance(
            evidence_type,
            str,
        ):
            normalized_type = (
                "unknown"
            )
        else:
            normalized_type = (
                evidence_type.strip()
                or "unknown"
            )

        grouped.setdefault(
            normalized_type,
            [],
        ).append(
            dict(
                evidence,
            )
        )

    return grouped
