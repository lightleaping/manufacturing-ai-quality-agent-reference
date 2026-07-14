"""
Day 24 Streamlit Dashboard UI Helper Unit Test입니다.

이 테스트는 Streamlit 화면을 직접 실행하지 않습니다.

대신 다음과 같이 UI와 분리 가능한 순수 데이터 변환 로직을 검증합니다.

- 설비 입력값을 FastAPI Request Payload로 변환
- LangGraph Agent Request Payload 생성
- Chat History 정규화
- 현재 요청의 Raw Sample 선택적 포함
- Evidence Type별 분류

중요:
- 실제 FastAPI 서버를 실행하지 않습니다.
- 실제 OpenAI API를 호출하지 않습니다.
- 실제 LangGraph를 실행하지 않습니다.
- 실제 PyTorch 모델을 실행하지 않습니다.
- 실제 SQLite를 조회하지 않습니다.
"""

from __future__ import annotations

import pytest

from src.dashboard.ui_helpers import (
    build_failure_prediction_payload,
    build_langgraph_agent_payload,
    build_raw_sample_payload,
    group_evidence_by_type,
    normalize_chat_history,
)


def test_build_raw_sample_payload_uses_api_type_alias() -> None:
    """
    Dashboard 설비 입력값이 FastAPI JSON 계약에 맞게 변환되는지 검증합니다.

    Python UI 변수명은 machine_type을 사용할 수 있지만,
    실제 API JSON 필드명은 type이어야 합니다.
    """

    payload = build_raw_sample_payload(
        air_temperature=303.0,
        process_temperature=312.5,
        rotational_speed=1380.0,
        torque=62.0,
        tool_wear=220.0,
        machine_type="L",
    )

    assert payload == {
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
    }

    assert "machine_type" not in payload


def test_build_failure_prediction_payload_adds_evidence_options() -> None:
    """
    Direct Failure Prediction 요청에
    SHAP과 Global Importance 옵션이 포함되는지 검증합니다.
    """

    payload = build_failure_prediction_payload(
        air_temperature=303.0,
        process_temperature=312.5,
        rotational_speed=1380.0,
        torque=62.0,
        tool_wear=220.0,
        machine_type="L",
        include_shap=False,
        include_global_importance=True,
    )

    assert payload["type"] == "L"
    assert payload["include_shap"] is False
    assert payload["include_global_importance"] is True


def test_normalize_chat_history_keeps_only_supported_messages() -> None:
    """
    user·assistant 메시지만 유지하고,
    공백 메시지와 지원하지 않는 role을 제외하는지 검증합니다.
    """

    messages = [
        {
            "role": "system",
            "content": "사용자 입력으로 허용되지 않는 메시지",
        },
        {
            "role": "user",
            "content": "  첫 번째 질문  ",
        },
        {
            "role": "assistant",
            "content": "  첫 번째 답변  ",
        },
        {
            "role": "user",
            "content": "   ",
        },
    ]

    normalized = normalize_chat_history(
        messages,
    )

    assert normalized == [
        {
            "role": "user",
            "content": "첫 번째 질문",
        },
        {
            "role": "assistant",
            "content": "첫 번째 답변",
        },
    ]


def test_normalize_chat_history_keeps_latest_six_messages() -> None:
    """
    FastAPI 계약에 맞게 최근 메시지 6개만 유지하는지 검증합니다.
    """

    messages = [
        {
            "role": "user",
            "content": f"message-{index}",
        }
        for index in range(8)
    ]

    normalized = normalize_chat_history(
        messages,
    )

    assert len(normalized) == 6
    assert normalized[0]["content"] == "message-2"
    assert normalized[-1]["content"] == "message-7"


def test_normalize_chat_history_limits_content_to_1000_characters() -> None:
    """
    FastAPI ChatMessageRequest의 최대 길이인
    1000자를 넘지 않도록 정규화하는지 검증합니다.
    """

    messages = [
        {
            "role": "assistant",
            "content": "A" * 1200,
        }
    ]

    normalized = normalize_chat_history(
        messages,
    )

    assert len(
        normalized[0]["content"]
    ) == 1000


def test_build_langgraph_agent_payload_omits_raw_sample_when_not_provided() -> None:
    """
    이전 설비 입력값을 자동 재사용하지 않도록,
    현재 요청에 Raw Sample이 없으면 Payload에서도 제외하는지 검증합니다.
    """

    payload = build_langgraph_agent_payload(
        question="  AI4I 데이터셋의 feature는 뭐야?  ",
        chat_history=[],
        raw_sample=None,
        include_shap=True,
        include_global_importance=True,
    )

    assert payload["question"] == (
        "AI4I 데이터셋의 feature는 뭐야?"
    )

    assert payload["chat_history"] == []
    assert "raw_sample" not in payload


def test_build_langgraph_agent_payload_includes_explicit_raw_sample() -> None:
    """
    사용자가 현재 요청에서 Raw Sample을 명시한 경우에만
    Agent Request에 포함되는지 검증합니다.
    """

    raw_sample = build_raw_sample_payload(
        air_temperature=303.0,
        process_temperature=312.5,
        rotational_speed=1380.0,
        torque=62.0,
        tool_wear=220.0,
        machine_type="L",
    )

    payload = build_langgraph_agent_payload(
        question="이 설비 조건의 고장 위험을 예측해줘.",
        chat_history=[],
        raw_sample=raw_sample,
        include_shap=True,
        include_global_importance=False,
    )

    assert payload["raw_sample"] == raw_sample
    assert payload["include_shap"] is True
    assert (
        payload["include_global_importance"]
        is False
    )


def test_build_langgraph_agent_payload_rejects_blank_question() -> None:
    """
    공백만 있는 질문이 FastAPI까지 전달되지 않도록 검증합니다.
    """

    with pytest.raises(
        ValueError,
        match="question",
    ):
        build_langgraph_agent_payload(
            question="   ",
            chat_history=[],
            raw_sample=None,
        )


def test_group_evidence_by_type_preserves_unknown_types() -> None:
    """
    알려진 Evidence뿐 아니라 향후 추가되는 유형도
    손실 없이 Type별로 분류하는지 검증합니다.
    """

    evidence_items = [
        {
            "evidence_type": "prediction_summary",
            "title": "모델 예측 요약",
        },
        {
            "evidence_type": "shap_local",
            "title": "Torque SHAP",
        },
        {
            "evidence_type": "future_evidence",
            "title": "향후 Evidence",
        },
    ]

    grouped = group_evidence_by_type(
        evidence_items,
    )

    assert len(
        grouped["prediction_summary"]
    ) == 1

    assert len(
        grouped["shap_local"]
    ) == 1

    assert len(
        grouped["future_evidence"]
    ) == 1
