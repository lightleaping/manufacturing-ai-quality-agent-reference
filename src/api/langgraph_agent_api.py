from typing import Any

from fastapi import APIRouter

from src.agent.failure_agent_graph import run_failure_agent_graph
from src.api.schemas import (
    LangGraphAgentQueryRequest,
    LangGraphAgentQueryResponse,
)

# APIRouter는 FastAPI endpoint들을 묶는 작은 라우터입니다.
#
# main.py에서 app.include_router(router)를 호출하면,
# 이 파일에 정의된 endpoint가 전체 FastAPI app에 등록됩니다.
router = APIRouter()


def _raw_sample_to_dict(request: LangGraphAgentQueryRequest) -> dict[str, Any] | None:
    """
    Pydantic raw_sample 객체를 일반 dict로 바꾸는 helper 함수입니다.

    왜 필요한가?
    - API request body는 Pydantic BaseModel로 검증됩니다.
    - 그런데 Day 13 LangGraph AgentState는 dict 기반으로 값을 들고 다닙니다.
    - 따라서 endpoint 경계에서 Pydantic 객체를 dict로 바꿔주는 것이 안전합니다.

    raw_sample이 없는 경우:
    - dataset_schema_query나 unknown 질문일 수 있습니다.
    - failure_prediction 질문이라도 입력값이 빠진 상황일 수 있습니다.
    - 이때는 None을 그대로 반환해서 LangGraph workflow가 판단하게 합니다.
    """

    if request.raw_sample is None:
        return None

    # Pydantic v2에서는 model_dump()를 사용합니다.
    # dict로 바꿔야 LangGraph AgentState에 넣기 쉽습니다.
    return request.raw_sample.model_dump()


def _state_to_response(
    *,
    question: str,
    state: dict[str, Any],
) -> LangGraphAgentQueryResponse:
    """
    LangGraph AgentState를 API response schema로 변환합니다.

    왜 별도 함수로 분리하는가?
    - endpoint 함수 안에 변환 로직이 길게 들어가면 읽기 어렵습니다.
    - 테스트할 때도 'endpoint 역할'과 'state 변환 역할'을 구분하기 어렵습니다.
    - 신입 포트폴리오 기준으로도 이런 변환 함수를 분리하면 구조 설명이 쉬워집니다.

    핵심:
    - LangGraph 내부 state는 많은 중간값을 가질 수 있습니다.
    - API response에는 사용자에게 필요한 값만 정리해서 반환합니다.
    """

    return LangGraphAgentQueryResponse(
        question=question,
        intent=state.get("intent", "unknown"),
        confidence=state.get("confidence"),
        intent_source=state.get("intent_source"),
        intent_reason=state.get("intent_reason"),
        prediction=state.get("prediction"),
        probability=state.get("probability"),
        threshold=state.get("threshold"),
        risk_level=state.get("risk_level"),
        recommended_action=state.get("recommended_action"),
        answer=state.get(
            "answer",
            "요청을 처리했지만 생성된 답변이 없습니다.",
        ),
        evidence=state.get("evidence", []),
        warnings=state.get("warnings", []),
        errors=state.get("errors", []),
        limitations=state.get("limitations", []),
    )


@router.post(
    "/agent/langgraph-query",
    response_model=LangGraphAgentQueryResponse,
)
def query_langgraph_agent(
    request: LangGraphAgentQueryRequest,
) -> LangGraphAgentQueryResponse:
    """
    자연어 질문을 LangGraph Agent workflow로 전달하는 endpoint입니다.

    기존 /agent/failure-prediction과 다른 점:
    - 기존 endpoint는 raw sensor 값을 바로 받아 prediction service를 호출했습니다.
    - 이 endpoint는 question을 먼저 받고, LangGraph가 intent를 판단합니다.

    처리 흐름:
    1. request에서 question을 받습니다.
    2. raw_sample이 있으면 dict로 변환합니다.
    3. run_failure_agent_graph()를 호출합니다.
    4. LangGraph AgentState 결과를 API response schema로 변환합니다.

    중요한 설계 원칙:
    - 이 endpoint는 OpenAI API를 직접 호출하지 않습니다.
    - 이 endpoint는 모델 artifact를 직접 로드하지 않습니다.
    - 실제 intent 분류와 prediction 판단은 LangGraph workflow와 service layer가 담당합니다.
    """

    raw_sample = _raw_sample_to_dict(request)

    # Day 13에서 만든 LangGraph runner를 호출합니다.
    #
    # 여기서 endpoint가 직접 intent를 분류하지 않는 이유:
    # - intent 분류는 LangGraph workflow의 책임입니다.
    # - API는 입력을 받아 workflow에 넘기는 얇은 계층으로 두는 것이 좋습니다.
    #
    # include_shap / include_global_importance는 prediction service에서 사용할 수 있도록
    # LangGraph workflow에 함께 전달합니다.
    state = run_failure_agent_graph(
        question=request.question,
        raw_sample=raw_sample,
        include_shap=request.include_shap,
        include_global_importance=request.include_global_importance,
    )

    return _state_to_response(
        question=request.question,
        state=state,
    )