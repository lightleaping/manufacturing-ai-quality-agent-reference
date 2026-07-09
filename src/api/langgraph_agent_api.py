from typing import Any

from fastapi import APIRouter

from src.agent.failure_agent_graph import (
    run_failure_agent_graph,
)
from src.agent.state import (
    AgentState,
    ChatMessage,
)
from src.api.schemas import (
    LangGraphAgentQueryRequest,
    LangGraphAgentQueryResponse,
)


# APIRouter는 FastAPI endpoint들을 묶는 작은 라우터입니다.
#
# main.py에서 app.include_router(router)를 호출하면,
# 이 파일에 정의된 endpoint가 전체 FastAPI app에 등록됩니다.
router = APIRouter()


def _raw_sample_to_dict(
    request: LangGraphAgentQueryRequest,
) -> dict[str, Any] | None:
    """
    Pydantic raw_sample 객체를 일반 dict로 바꾸는 helper 함수입니다.

    왜 필요한가?
    -------------
    - API request body는 Pydantic BaseModel로 검증됩니다.
    - 그런데 Day 13 LangGraph AgentState는 dict 기반으로 값을 들고 다닙니다.
    - 따라서 endpoint 경계에서 Pydantic 객체를 dict로 바꿔주는 것이 안전합니다.

    raw_sample이 없는 경우
    ----------------------
    - dataset_schema_query나 unknown 질문일 수 있습니다.
    - failure_prediction 질문이라도 입력값이 빠진 상황일 수 있습니다.
    - 이때는 None을 그대로 반환해서 LangGraph workflow가 판단하게 합니다.
    """

    if request.raw_sample is None:
        return None

    # Pydantic v2에서는 model_dump()를 사용합니다.
    #
    # dict로 바꿔야 LangGraph AgentState에 넣기 쉽습니다.
    return request.raw_sample.model_dump()


def _chat_history_to_dicts(
    request: LangGraphAgentQueryRequest,
) -> list[ChatMessage]:
    """
    Pydantic chat_history 객체들을
    LangGraph AgentState가 사용할 일반 dict 목록으로 변환합니다.

    입력 구조
    --------
    FastAPI와 Pydantic이 검증한 객체 목록입니다.

        list[ChatMessageRequest]

    예:

        [
            ChatMessageRequest(
                role="user",
                content="이 설비 조건이면 고장 위험이 높아?",
            ),
            ChatMessageRequest(
                role="assistant",
                content="고장 위험이 높게 예측되었습니다.",
            ),
        ]

    출력 구조
    --------
    LangGraph AgentState가 사용할 일반 dict 목록입니다.

        list[ChatMessage]

    예:

        [
            {
                "role": "user",
                "content": "이 설비 조건이면 고장 위험이 높아?",
            },
            {
                "role": "assistant",
                "content": "고장 위험이 높게 예측되었습니다.",
            },
        ]

    왜 endpoint 경계에서 변환하는가?
    ------------------------------
    FastAPI request 계층에서는
    Pydantic BaseModel을 사용하여 외부 입력을 검증합니다.

    LangGraph 내부에서는
    TypedDict 기반의 일반 Python dict를 사용합니다.

    따라서 API 계층과 Agent 계층의 경계에서
    Pydantic 객체를 일반 dict로 변환합니다.

    각 계층의 책임
    --------------
    Pydantic schema:
        외부 HTTP 입력 검증

    LangGraph AgentState:
        node 사이에서 상태 전달

    intent classifier:
        이전 대화 문맥을 이용한 intent 분류

    중요
    ----
    chat_history는 현재 질문의 문맥을 이해하기 위한 데이터입니다.

    이전 대화에 probability나 설비 값이 적혀 있어도
    해당 텍스트를 PyTorch 모델의 새로운 입력으로 사용하지 않습니다.

    실제 prediction은 계속 raw_sample을 사용합니다.
    """

    # request.chat_history는
    # LangGraphAgentQueryRequest에서
    # default_factory=list로 정의되어 있습니다.
    #
    # 따라서 클라이언트가 chat_history를 생략하면:
    #
    #     request.chat_history == []
    #
    # 가 됩니다.
    #
    # 빈 list를 순회하면 결과도 빈 list가 되므로
    # 별도의 None 검사가 필요하지 않습니다.
    return [
        {
            # ChatMessageRequest schema에서
            # role은 "user" 또는 "assistant"로 이미 검증되었습니다.
            "role": message.role,

            # content도 Pydantic schema에서
            # 문자열 길이 검증을 통과한 값입니다.
            "content": message.content,
        }
        for message in request.chat_history
    ]


def _state_to_response(
    *,
    question: str,
    state: AgentState,
) -> LangGraphAgentQueryResponse:
    """
    LangGraph AgentState를 API response schema로 변환합니다.

    왜 별도 함수로 분리하는가?
    ------------------------
    - endpoint 함수 안에 변환 로직이 길게 들어가면 읽기 어렵습니다.
    - 테스트할 때도 'endpoint 역할'과 'state 변환 역할'을 구분하기 어렵습니다.
    - 신입 포트폴리오 기준으로도 이런 변환 함수를 분리하면 구조 설명이 쉬워집니다.

    핵심
    ----
    - LangGraph 내부 state는 많은 중간값을 가질 수 있습니다.
    - API response에는 사용자에게 필요한 값만 정리해서 반환합니다.

    Day 16 확장
    -----------
    기존 Day 15까지는 최종 처리 결과를 중심으로 반환했습니다.

    예:

        intent

        prediction

        probability

        risk_level

        answer

        warnings

        errors

    Day 16에서는 내부 구조화 trace도 API response에 연결합니다.

    추가 항목:

        trace_id

        trace_status

        trace_started_at

        trace_finished_at

        trace_duration_ms

        fallback_occurred

        trace_events

    trace_events 변환
    -----------------
    AgentState 내부의 trace_events는
    list[TraceEvent] 형태의 일반 dict 목록입니다.

    LangGraphAgentQueryResponse에서는:

        list[TraceEventResponse]

    를 사용합니다.

    여기서 각 dict를 직접 TraceEventResponse 객체로
    하나씩 변환하지 않아도 됩니다.

    Pydantic은 LangGraphAgentQueryResponse를 생성할 때
    trace_events 안의 각 dict를 읽고
    TraceEventResponse schema에 맞는지 자동 검증합니다.

    예:

        trace_events=[
            {
                "sequence": 1,
                "event_type": "node",
                "event_name": "validate_question",
                ...
            }
        ]

                │

                ▼

        Pydantic 자동 검증

                │

                ▼

        list[TraceEventResponse]

    따라서 API 계층에 불필요한 중복 변환 코드를 추가하지 않습니다.
    """

    return LangGraphAgentQueryResponse(
        # 현재 사용자가 API request에 전달한 질문입니다.
        question=question,

        # ----------------------------------------------------
        # Intent classification 결과
        # ----------------------------------------------------

        # intent가 없는 예외적 상태에서는
        # 안전한 기본값 "unknown"을 사용합니다.
        intent=state.get(
            "intent",
            "unknown",
        ),

        confidence=state.get(
            "confidence"
        ),

        intent_source=state.get(
            "intent_source"
        ),

        intent_reason=state.get(
            "intent_reason"
        ),

        # ----------------------------------------------------
        # Failure prediction 결과
        # ----------------------------------------------------
        #
        # dataset_schema_query 또는 unknown intent에서는
        # prediction을 수행하지 않을 수 있으므로
        # 값이 없으면 None이 전달됩니다.

        prediction=state.get(
            "prediction"
        ),

        probability=state.get(
            "probability"
        ),

        threshold=state.get(
            "threshold"
        ),

        risk_level=state.get(
            "risk_level"
        ),

        recommended_action=state.get(
            "recommended_action"
        ),

        # ----------------------------------------------------
        # Agent answer
        # ----------------------------------------------------

        answer=state.get(
            "answer",
            "요청을 처리했지만 생성된 답변이 없습니다.",
        ),

        # ----------------------------------------------------
        # Evidence / warning / error / limitation
        # ----------------------------------------------------

        evidence=state.get(
            "evidence",
            [],
        ),

        warnings=state.get(
            "warnings",
            [],
        ),

        errors=state.get(
            "errors",
            [],
        ),

        limitations=state.get(
            "limitations",
            [],
        ),

        # ----------------------------------------------------
        # Day 16 - 전체 trace 요약
        # ----------------------------------------------------

        # 요청 하나를 구분하는 UUID 기반 고유 ID입니다.
        #
        # 예:
        #
        # "908759dd97bd4a3eb7494b68f76f871c"
        #
        # create_initial_agent_state()에서
        # 요청마다 새로운 값을 생성합니다.
        trace_id=state.get(
            "trace_id"
        ),

        # 전체 LangGraph workflow의 최종 상태입니다.
        #
        # 일반적인 최종 API 응답에서는:
        #
        # "success"
        #
        # "fallback"
        #
        # "error"
        #
        # 중 하나입니다.
        #
        # "running"은 workflow 실행 중 상태이므로
        # finalize_trace() 이후의 정상 응답에서는
        # 일반적으로 남아 있지 않습니다.
        trace_status=state.get(
            "trace_status"
        ),

        # 전체 trace를 시작한 UTC 시각입니다.
        #
        # ISO 8601 문자열 예:
        #
        # "2026-07-10T01:12:30.120000+00:00"
        trace_started_at=state.get(
            "trace_started_at"
        ),

        # 전체 LangGraph workflow가 종료된 UTC 시각입니다.
        trace_finished_at=state.get(
            "trace_finished_at"
        ),

        # 전체 workflow 실행 시간입니다.
        #
        # 단위:
        #
        # millisecond
        trace_duration_ms=state.get(
            "trace_duration_ms"
        ),

        # 실제 LangGraph fallback route 또는
        # fallback answer node가 실행됐는지 나타냅니다.
        #
        # intent_source == "fallback"과는
        # 서로 다른 개념입니다.
        fallback_occurred=state.get(
            "fallback_occurred",
            False,
        ),

        # node와 route 실행 기록입니다.
        #
        # Pydantic이 각 dict를
        # TraceEventResponse schema로 자동 검증합니다.
        trace_events=state.get(
            "trace_events",
            [],
        ),
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

    기존 /agent/failure-prediction과 다른 점
    ----------------------------------------
    기존 endpoint:
        정형화된 raw sensor 값을 바로 받아
        prediction service를 호출합니다.

    LangGraph endpoint:
        현재 question과 선택적 chat_history를 받아
        LangGraph가 intent를 판단합니다.

    처리 흐름
    --------
    1. request에서 현재 question을 받습니다.

    2. raw_sample이 있으면 일반 dict로 변환합니다.

    3. chat_history의 Pydantic 객체들을
       AgentState용 일반 dict 목록으로 변환합니다.

    4. run_failure_agent_graph()에
       question, chat_history, raw_sample을 전달합니다.

    5. LangGraph AgentState 결과를
       API response schema로 변환합니다.

    Day 15 multi-turn 흐름
    ----------------------
        이전 chat_history
        +
        현재 question

                │

                ▼

        FastAPI request 검증

                │

                ▼

        run_failure_agent_graph()

                │

                ▼

        AgentState

                │

                ▼

        intent classifier

                │

                ▼

        LangGraph routing


    Day 16 trace 흐름
    -----------------
        FastAPI request

                │

                ▼

        run_failure_agent_graph()

                │

                ▼

        trace_id 생성

                │

                ▼

        node 실행

                │

                ▼

        node trace event 추가

                │

                ▼

        route 선택

                │

                ▼

        route trace event 추가

                │

                ▼

        finalize_trace()

                │

                ▼

        AgentState

        trace_id

        trace_status

        trace_duration_ms

        trace_events

                │

                ▼

        _state_to_response()

                │

                ▼

        LangGraphAgentQueryResponse

                │

                ▼

        Swagger JSON response


    중요한 설계 원칙
    ----------------
    - 이 endpoint는 OpenAI API를 직접 호출하지 않습니다.

    - 이 endpoint는 모델 artifact를 직접 로드하지 않습니다.

    - 실제 intent 분류는
      LangGraph workflow와 intent classifier가 담당합니다.

    - 실제 prediction은
      Day 12 prediction service가 담당합니다.

    - chat_history는 질문 문맥 이해용입니다.

    - raw_sample은 실제 PyTorch prediction 입력용입니다.

    - chat_history에 이전 설비 값이나 probability가 있어도
      새로운 prediction 입력으로 자동 사용하지 않습니다.

    - endpoint는 trace 시간을 직접 측정하지 않습니다.

    - endpoint는 trace event를 직접 생성하지 않습니다.

    - trace 생성은 LangGraph Agent 계층의 책임입니다.

    - endpoint는 완성된 AgentState trace를
      API response schema에 연결하는 역할만 합니다.
    """

    # 선택적으로 전달된 raw_sample을
    # Pydantic 객체에서 일반 dict로 변환합니다.
    raw_sample = _raw_sample_to_dict(
        request
    )

    # 이전 대화 기록을
    # Pydantic ChatMessageRequest 객체 목록에서
    # LangGraph AgentState용 일반 dict 목록으로 변환합니다.
    #
    # chat_history가 요청에 없다면
    # request.chat_history는 빈 list이므로
    # 결과도 빈 list가 됩니다.
    chat_history = (
        _chat_history_to_dicts(
            request
        )
    )

    # Day 13에서 만든 LangGraph runner를 호출합니다.
    #
    # 여기서 endpoint가 직접 intent를 분류하지 않는 이유:
    # - intent 분류는 LangGraph workflow의 책임입니다.
    # - API는 입력을 받아 workflow에 넘기는
    #   얇은 계층으로 두는 것이 좋습니다.
    #
    # 각 입력의 역할:
    #
    # question:
    #   현재 intent를 분류할 사용자 질문
    #
    # chat_history:
    #   현재 질문의 문맥을 이해하기 위한 이전 대화
    #
    # raw_sample:
    #   실제 고장 prediction에 사용할 설비 입력값
    #
    # include_shap / include_global_importance:
    #   prediction service에서 생성할 evidence 옵션
    state = run_failure_agent_graph(
        question=request.question,
        raw_sample=raw_sample,
        include_shap=(
            request.include_shap
        ),
        include_global_importance=(
            request.include_global_importance
        ),

        # Day 15:
        # 검증 및 변환이 끝난 이전 대화 기록을
        # LangGraph runner에 전달합니다.
        chat_history=chat_history,
    )

    # LangGraph 내부 AgentState를
    # FastAPI response schema로 변환합니다.
    #
    # Day 16부터는 최종 prediction과 answer뿐 아니라
    # 구조화 trace 정보도 함께 반환합니다.
    return _state_to_response(
        question=request.question,
        state=state,
    )