import logging

from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
)

from src.agent.failure_agent_graph import (
    run_failure_agent_graph,
)
from src.agent.state import (
    AgentState,
    ChatMessage,
    append_warning,
)
from src.api.schemas import (
    AgentExecutionDetailResponse,
    AgentExecutionSummaryResponse,
    LangGraphAgentQueryRequest,
    LangGraphAgentQueryResponse,
)
from src.persistence.execution_history import (
    get_execution_by_trace_id,
    insert_execution,
    list_recent_executions,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

# __name__은 현재 Python module 이름입니다.
#
# 현재 module:
#
# src.api.langgraph_agent_api
#
# logger를 사용하면 Persistence 저장 실패를
# 사용자 질문·raw_sample 전체를 출력하지 않고
# 서버 로그에 기록할 수 있습니다.
#
# 보안상 다음 값은 로그에 출력하지 않습니다.
#
#     OpenAI API key
#
#     환경 변수 전체
#
#     사용자 question 원문
#
#     raw_sample 전체
#
#     chat_history 전체
#
# 저장 실패 로그에는 trace_id만 사용합니다.
logger = logging.getLogger(
    __name__
)

# APIRouter는 FastAPI endpoint들을 묶는 작은 라우터입니다.
#
# main.py에서 app.include_router(router)를 호출하면,
# 이 파일에 정의된 endpoint가 전체 FastAPI app에 등록됩니다.
router = APIRouter()

# ---------------------------------------------------------------------
# Day 19 - Agent 실행 이력 안전 저장
# ---------------------------------------------------------------------

def _save_execution_history_safely(
    *,
    state: AgentState,
) -> None:
    """
    final AgentState를 SQLite에 저장합니다.

    저장 성공:

        별도 반환값 없이 정상 종료

    저장 실패:

        서버 로그 기록

        AgentState warnings에 안내 추가

        예외를 endpoint 밖으로 다시 전달하지 않음


    왜 저장 실패 예외를 다시 발생시키지 않는가?
    ----------------------------------------
    Agent 실행 성공과
    실행 이력 저장 성공은 서로 다른 결과입니다.

    예:

        OpenAI intent 분류 성공

        PyTorch prediction 성공

        Evidence 생성 성공

        Agent answer 생성 성공

        SQLite 파일 권한 오류

    이 상황에서 DB 저장 실패 때문에
    이미 성공한 고장 예측 결과까지
    HTTP 500으로 바꾸는 것은
    Day 19 초기 정책에 맞지 않습니다.

    따라서:

        Agent 결과

        -> 사용자에게 반환

        Persistence 실패

        -> warning과 서버 로그로 기록

    정책을 사용합니다.


    왜 Exception을 넓게 처리하는가?
    ------------------------------
    일반적으로는 가능한 구체적인 예외를
    처리하는 것이 좋습니다.

    하지만 이 helper는 FastAPI와 Persistence 사이의
    최종 격리 경계입니다.

    Persistence에서 다음 문제가 발생할 수 있습니다.

        sqlite3.Error

        파일 경로·권한 오류

        JSON 직렬화 오류

        필수 저장 필드 오류

    Day 19 정책은 저장 계층의 오류가
    Agent 응답을 실패시키지 않도록 하는 것입니다.

    따라서 이 작은 Persistence 호출 범위에서만
    Exception을 처리합니다.

    오류를 완전히 숨기지는 않고
    logger.exception()으로 traceback을 기록합니다.
    """

    try:
        # final AgentState를 SQLite에 저장합니다.
        #
        # 기본 DB:
        #
        # data/runtime/
        # agent_execution_history.db
        insert_execution(
            state=state,
        )

    except Exception:
        # exception 정보와 traceback을
        # 서버 로그에 기록합니다.
        #
        # 사용자 질문·설비 입력 전체는
        # 개인정보·민감정보 위험 때문에
        # 로그 메시지에 직접 넣지 않습니다.
        logger.exception(
            (
                "Failed to save Agent "
                "execution history. "
                "trace_id=%s"
            ),
            state.get(
                "trace_id"
            ),
        )

        # Persistence 실패는 경고입니다.
        #
        # Agent workflow가 만든 prediction과 answer는
        # 그대로 유지합니다.
        append_warning(
            state,
            (
                "Agent 실행 결과는 정상적으로 생성됐지만, "
                "실행 이력을 SQLite에 저장하지 못했습니다."
            ),
        )

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

        # --------------------------------------------------------
    # Day 19 - Agent 실행 이력 SQLite 저장
    # --------------------------------------------------------
    #
    # LangGraph workflow가 완전히 끝난 final AgentState를
    # Persistence 계층에 전달합니다.
    #
    # 이 시점에는 일반적으로 다음 값이 완성되어 있습니다.
    #
    #     trace_id
    #
    #     intent
    #
    #     prediction
    #
    #     probability
    #
    #     evidence
    #
    #     answer
    #
    #     trace_status
    #
    #     trace_duration_ms
    #
    #     trace_events
    #
    #
    # 저장 책임을 LangGraph node 안에 넣지 않는 이유:
    #
    # LangGraph:
    #
    #     Agent workflow 실행
    #
    # Persistence:
    #
    #     실행 결과 저장·조회
    #
    # FastAPI:
    #
    #     두 계층 연결
    #
    # 역할을 분리하기 위해서입니다.
    #
    #
    # 저장 실패 정책:
    #
    # SQLite 저장 실패
    #
    #     -> 서버 로그 기록
    #
    #     -> warning 추가
    #
    #     -> Agent response 유지
    _save_execution_history_safely(
        state=state,
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

# ============================================================
# Day 19 - Agent 실행 이력 조회 Endpoints
# ============================================================


@router.get(
    "/agent/executions",
    response_model=list[
        AgentExecutionSummaryResponse
    ],
)
def get_recent_agent_executions(
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description=(
            "Maximum number of recent "
            "Agent executions to return."
        ),
    ),
) -> list[AgentExecutionSummaryResponse]:
    """
    최근 Agent 실행 이력을
    최신 실행 순서로 조회합니다.

    Endpoint:

        GET /agent/executions

    Query parameter:

        limit

    기본값:

        20

    허용 범위:

        1 ~ 100

    예:

        GET /agent/executions

        -> 최근 최대 20건

        GET /agent/executions?limit=5

        -> 최근 최대 5건


    반환 데이터
    -----------
    목록 조회는 여러 실행을 반환하므로
    핵심 요약 데이터만 포함합니다.

    포함:

        trace_id

        question

        intent

        prediction

        probability

        risk_level

        trace_status

        fallback_occurred

        trace_duration_ms

        warning_count

        error_count

        created_at

    제외:

        raw_sample

        evidence

        trace_events

        warnings 전체

        errors 전체

    전체 상세 데이터는:

        GET /agent/executions/{trace_id}

    endpoint에서 조회합니다.


    limit 검증
    ----------
    FastAPI와 Pydantic이 Query 조건을 먼저 검증합니다.

    ge=1:

        1 이상

    le=100:

        100 이하

    예:

        limit=0

        -> HTTP 422

        limit=101

        -> HTTP 422
    """

    executions = list_recent_executions(
        limit=limit,
    )

    # Persistence 계층은 list[dict]를 반환합니다.
    #
    # FastAPI는 response_model을 사용해
    # 각 dict를 AgentExecutionSummaryResponse에 맞게
    # 자동 검증하고 JSON으로 변환합니다.
    return executions

# ============================================================
# Day 19 - 상세 조회 Endpoints
# ============================================================

@router.get(
    "/agent/executions/{trace_id}",
    response_model=AgentExecutionDetailResponse,
)
def get_agent_execution_detail(
    trace_id: str,
) -> AgentExecutionDetailResponse:
    """
    trace_id를 사용하여
    Agent 실행 이력 한 건을 상세 조회합니다.

    조회 성공:

        HTTP 200

    실행 이력 없음:

        HTTP 404
    """

    # Persistence 계층에서
    # trace_id 기준 실행 이력을 조회합니다.
    execution = get_execution_by_trace_id(
        trace_id=trace_id,
    )

    # Persistence 계층은
    # 실행 이력이 없을 때 None을 반환합니다.
    #
    # HTTP 상태 코드는 API 계층의 책임이므로
    # 여기에서 404로 변환합니다.
    if execution is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Agent execution history was not found. "
                f"trace_id={trace_id}"
            ),
        )

    # 조회 결과 dict는
    # response_model을 통해
    # AgentExecutionDetailResponse 구조로
    # 검증된 뒤 JSON으로 반환됩니다.
    return execution