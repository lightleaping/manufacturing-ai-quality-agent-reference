# src/api/schemas.py

from typing import Any, Literal

from pydantic import BaseModel, Field


class FailurePredictionRequest(BaseModel):
    """
    POST /agent/failure-prediction 요청 body를 정의하는 Pydantic schema입니다.

    Pydantic schema는 API 입력값을 검증하는 역할을 합니다.

    예를 들어 사용자가 Swagger UI 또는 HTTP client로 아래 JSON을 보내면,

    {
      "air_temperature": 303.0,
      "process_temperature": 312.5,
      "rotational_speed": 1380.0,
      "torque": 62.0,
      "tool_wear": 220.0,
      "type": "L",
      "include_shap": true,
      "include_global_importance": true
    }

    FastAPI는 이 JSON을 FailurePredictionRequest 객체로 변환합니다.

    주의:
    - Python에서 type이라는 이름은 내장 함수 type과 겹칠 수 있습니다.
    - 그래서 내부 변수명은 machine_type으로 두고,
      API JSON에서는 alias="type"을 사용해 type이라는 이름으로 받습니다.
    """

    air_temperature: float = Field(
        ...,
        description="Air temperature [K]",
        examples=[303.0],
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature [K]",
        examples=[312.5],
    )

    rotational_speed: float = Field(
        ...,
        description="Rotational speed [rpm]",
        examples=[1380.0],
    )

    torque: float = Field(
        ...,
        description="Torque [Nm]",
        examples=[62.0],
    )

    tool_wear: float = Field(
        ...,
        description="Tool wear [min]",
        examples=[220.0],
    )

    machine_type: str = Field(
        ...,
        alias="type",
        description="AI4I product type. Usually one of L, M, H.",
        examples=["L"],
    )

    include_shap: bool = Field(
        default=True,
        description="Whether to include SHAP local explanation evidence.",
    )

    include_global_importance: bool = Field(
        default=True,
        description="Whether to include global permutation importance evidence.",
    )

    def to_raw_sample(self) -> dict[str, Any]:
        """
        API request field 이름을 Day 5 inference 함수가 기대하는 raw_sample 형식으로 변환합니다.

        Day 5 추론 흐름에서는 AI4I 원본 feature 이름을 그대로 사용했습니다.

        즉 API에서는 사용하기 쉬운 snake_case 이름을 받고,
        내부 inference 함수에는 기존 프로젝트 feature 이름으로 넘깁니다.

        API request:
            air_temperature

        내부 raw_sample:
            Air temperature [K]
        """

        return {
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear,
            "Type": self.machine_type,
        }


class AgentEvidenceResponse(BaseModel):
    """
    API 응답에 포함될 evidence 한 개의 구조입니다.

    Day 9에서 만든 AgentEvidence dataclass를
    JSON으로 반환하기 좋게 Pydantic schema로 표현합니다.

    evidence_type 예:
    - prediction_summary
    - rule_based
    - shap_local
    - global_importance

    source 예:
    - model_prediction
    - rule_engine
    - shap
    - permutation_importance
    """

    evidence_id: str
    evidence_type: str
    source: str
    title: str
    summary: str

    feature: str | None = None
    value: Any | None = None
    direction: str | None = None
    contribution: float | None = None
    importance: float | None = None
    severity: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class FailurePredictionResponse(BaseModel):
    """
    POST /agent/failure-prediction 응답 schema입니다.

    이 response는 단순 모델 결과만 반환하지 않습니다.

    포함 항목:
    - prediction: threshold 기준 최종 0/1 판단
    - probability: 모델이 예측한 고장 확률
    - threshold: 운영 판단 기준
    - risk_level: 설명용 위험 등급
    - recommended_action: 권장 조치
    - evidence: prediction_summary, rule_based, shap_local, global_importance
    - answer: Agent가 사용자에게 보여줄 자연어 답변
    - warnings: 해석상 주의사항
    - limitations: 현재 시스템 한계
    """

    prediction: int
    probability: float
    threshold: float
    risk_level: str
    recommended_action: str

    evidence: list[AgentEvidenceResponse]
    answer: str

    warnings: list[str] = Field(
        default_factory=list
    )

    limitations: list[str] = Field(
        default_factory=list
    )


# ============================================================
# Day 14 - LangGraph Agent API Schemas
# ============================================================
#
# Day 10~12의 /agent/failure-prediction endpoint는
# "정형화된 설비 입력값"을 바로 받아서 prediction service를 호출했습니다.
#
# Day 14의 /agent/langgraph-query endpoint는
# "자연어 질문(question)"을 먼저 받고,
# 선택적으로 raw_sample을 함께 받습니다.
#
# 즉, Day 14 endpoint의 중심은 raw_sample이 아니라 question입니다.
# raw_sample은 failure_prediction intent일 때만 필요합니다.


class LangGraphRawSampleRequest(BaseModel):
    """
    LangGraph Agent가 사용할 수 있는 설비 입력값 schema입니다.

    기존 /agent/failure-prediction endpoint의 입력값과 거의 같지만,
    여기서는 LangGraph request 안에 중첩되어 들어갑니다.

    예:
    {
        "question": "이 설비 조건이면 고장 위험이 높아?",
        "raw_sample": {
            "air_temperature": 303.0,
            "process_temperature": 312.5,
            "rotational_speed": 1380.0,
            "torque": 62.0,
            "tool_wear": 220.0,
            "type": "L"
        }
    }
    """

    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type: str


# ============================================================
# Day 15 - Chat History / Multi-turn Request Schema
# ============================================================
#
# Day 14까지는 현재 question 하나만 LangGraph Agent에 전달했습니다.
#
# Day 15에서는 아래와 같이
# 이전 user 질문과 이전 assistant 답변도 함께 전달할 수 있습니다.
#
# 예:
#
# [
#     {
#         "role": "user",
#         "content": "이 설비 조건이면 고장 위험이 높아?"
#     },
#     {
#         "role": "assistant",
#         "content": "현재 입력 조건에서는 고장 위험이 높게 예측되었습니다."
#     }
# ]
#
# chat_history의 목적:
#   현재 질문에 포함된
#   "그건", "그중", "방금 결과" 같은 표현의
#   이전 대화 문맥을 이해하는 것
#
# chat_history가 하지 않는 일:
#   이전 대화에 적힌 설비 값이나 probability를
#   새로운 PyTorch 모델 prediction 입력으로 자동 사용하는 것
#
# 실제 고장 prediction은 계속 raw_sample을 사용합니다.


class ChatMessageRequest(BaseModel):
    """
    /agent/langgraph-query 요청의
    chat_history에 포함되는 메시지 한 개의 schema입니다.

    예:

        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?"
        }

        {
            "role": "assistant",
            "content": "현재 입력 조건에서는 고장 위험이 높게 예측되었습니다."
        }

    role
    ----
    메시지를 작성한 주체입니다.

    현재 Day 15에서는 아래 두 값만 허용합니다.

    user:
        사용자가 입력한 이전 질문

    assistant:
        LangGraph Agent가 반환한 이전 답변

    왜 system role을 허용하지 않는가?
    --------------------------------
    API 사용자가 chat_history 안에 임의의 system message를 넣으면
    내부 system instruction과 일반 대화 데이터의 경계가 흐려질 수 있습니다.

    내부 system instruction은 개발자가 관리합니다.

    외부 API의 chat_history는
    이전 user/assistant 대화 문맥만 전달합니다.

    content
    -------
    실제 질문 또는 답변 문자열입니다.

    최소 길이:
        1자

    최대 길이:
        1000자

    한 메시지의 길이를 제한하면
    지나치게 긴 대화가 OpenAI prompt에 무제한 포함되는 것을 줄일 수 있습니다.
    """

    # Literal을 사용하면
    # "user", "assistant" 외의 문자열은
    # Pydantic request 검증 단계에서 거부됩니다.
    #
    # 예:
    #
    # 정상:
    #     "user"
    #     "assistant"
    #
    # 검증 실패:
    #     "system"
    #     "developer"
    #     "uesr"
    role: Literal[
        "user",
        "assistant",
    ] = Field(
        ...,
        description=(
            "Message author. "
            "Only user and assistant are allowed."
        ),
        examples=["user"],
    )

    # min_length=1:
    #   완전히 빈 문자열 ""을 허용하지 않습니다.
    #
    # max_length=1000:
    #   한 메시지가 지나치게 길어지는 것을 제한합니다.
    #
    # 주의:
    # min_length=1은 ""은 막지만
    # 공백만 있는 "   "까지 자동으로 제거하지는 않습니다.
    #
    # classifier 내부의 _normalize_chat_history()에서도
    # strip() 후 빈 content를 제외하므로
    # 실제 prompt 생성 단계에서 한 번 더 방어합니다.
    content: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Previous user or assistant message. "
            "Maximum length is 1000 characters."
        ),
        examples=[
            "이 설비 조건이면 고장 위험이 높아?"
        ],
    )


class LangGraphAgentQueryRequest(BaseModel):
    """
    /agent/langgraph-query endpoint의 request schema입니다.

    question:
        사용자의 현재 자연어 질문입니다.

        LangGraph workflow는
        현재 question과 필요한 이전 chat_history를 참고하여
        intent를 분류합니다.

    chat_history:
        현재 질문 이전의 user/assistant 대화 기록입니다.

        선택 입력입니다.

        전달하지 않으면 새로운 빈 list를 사용하므로
        기존 Day 14 single-turn 요청도 그대로 동작합니다.

        최대 최근 메시지 6개를 허용합니다.

        chat_history는 질문 문맥 이해용이며,
        PyTorch 모델 prediction 입력이 아닙니다.

    raw_sample:
        선택 입력값입니다.

        dataset_schema_query나 unknown 질문에는 필요하지 않습니다.

        failure_prediction 질문일 때
        실제 모델 probability를 계산하려면 필요합니다.

    include_shap:
        prediction을 수행할 때
        SHAP local explanation을 포함할지 여부입니다.

        Day 12 service layer가 이 옵션을 사용합니다.

    include_global_importance:
        prediction을 수행할 때
        global importance evidence를 포함할지 여부입니다.
    """

    # 현재 intent를 분류할 사용자 질문입니다.
    question: str

    # 이전 대화 기록입니다.
    #
    # default_factory=list:
    #   chat_history를 요청에서 생략하면
    #   요청 객체를 만들 때마다 새로운 빈 list를 생성합니다.
    #
    # 다음처럼 직접 빈 list를 기본값으로 작성하지 않습니다.
    #
    #     chat_history: list[ChatMessageRequest] = []
    #
    # 일반 Python 함수에서는 mutable 기본 객체를
    # 여러 호출이 공유할 수 있으므로 피하는 것이 좋습니다.
    #
    # Pydantic은 mutable 기본값을 방어적으로 처리하지만,
    # default_factory=list를 사용하면
    # "요청 객체마다 새로운 list를 생성한다"는 의도가
    # 코드에 명확하게 드러납니다.
    #
    # max_length=6:
    #   대화 기록 전체를 무제한 받지 않고
    #   최대 메시지 6개까지만 허용합니다.
    #
    # 일반적으로:
    #
    # user
    # -> assistant
    #
    # 가 한 번의 대화라고 보면,
    # 메시지 6개는 최근 약 3회의 대화에 해당합니다.
    chat_history: list[
        ChatMessageRequest
    ] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "Recent conversation history for multi-turn "
            "intent classification. "
            "Maximum 6 user/assistant messages."
        ),
    )

    # 실제 고장 prediction에 사용할 설비 입력값입니다.
    #
    # chat_history에 이전 설비 값이 적혀 있더라도
    # raw_sample을 자동으로 대체하지 않습니다.
    raw_sample: (
        LangGraphRawSampleRequest
        |
        None
    ) = None

    # SHAP local explanation 사용 여부입니다.
    include_shap: bool = True

    # global permutation importance evidence 사용 여부입니다.
    include_global_importance: bool = True


# ============================================================
# Day 16 - LangGraph Trace / Observability Response Schema
# ============================================================
#
# Day 15까지 LangGraph API response에서는
# 최종 intent, prediction, answer, warning, error를 확인할 수 있었습니다.
#
# 하지만 최종 결과만으로는 아래 정보를 확인하기 어려웠습니다.
#
# 어떤 node가 실행됐는가?
#
# node가 어떤 순서로 실행됐는가?
#
# node 실행에 얼마나 시간이 걸렸는가?
#
# 어떤 route가 선택됐는가?
#
# fallback 경로가 사용됐는가?
#
# 전체 workflow가 success, fallback, error 중
# 어떤 상태로 끝났는가?
#
# Day 16에서는 AgentState의 구조화 trace를
# FastAPI JSON response로 반환하기 위한 schema를 추가합니다.


class TraceEventResponse(BaseModel):
    """
    LangGraph 실행 과정에서 발생한
    node 또는 route trace event 한 개의 API 응답 schema입니다.

    AgentState 내부에서는
    src/agent/state.py의 TraceEvent TypedDict를 사용합니다.

    FastAPI 응답에서는
    TraceEventResponse Pydantic model을 사용합니다.

    왜 내부 TypedDict와 API Pydantic model을 구분하는가?
    ---------------------------------------------------
    TypedDict:
        Python 내부 코드의 dict 구조를
        정적 타입 수준에서 표현합니다.

    Pydantic BaseModel:
        FastAPI response를 검증하고,
        JSON schema와 Swagger 문서를 생성합니다.

    즉:

        Agent 내부

        TraceEvent TypedDict

                │

                ▼

        FastAPI 응답

        TraceEventResponse


    응답 예
    -------
    {
        "sequence": 3,
        "event_type": "node",
        "event_name": "classify_intent",
        "status": "success",
        "started_at": "2026-07-10T01:12:30.120000+00:00",
        "finished_at": "2026-07-10T01:12:30.932000+00:00",
        "duration_ms": 812.0,
        "metadata": {
            "intent": "failure_prediction",
            "intent_source": "openai",
            "confidence": 0.95
        }
    }
    """

    # 하나의 trace 안에서 event가 실행된 순서입니다.
    #
    # 첫 번째 event:
    #
    # sequence = 1
    #
    # 두 번째 event:
    #
    # sequence = 2
    #
    # ge=1:
    #   sequence는 1 이상이어야 합니다.
    sequence: int = Field(
        ...,
        ge=1,
        description=(
            "Execution order of the trace event. "
            "Starts from 1."
        ),
        examples=[1],
    )

    # trace event 종류입니다.
    #
    # node:
    #   실제 LangGraph node 실행
    #
    # route:
    #   conditional routing 결과
    event_type: Literal[
        "node",
        "route",
    ] = Field(
        ...,
        description=(
            "Trace event type. "
            "Either node execution or route selection."
        ),
        examples=["node"],
    )

    # 실제 실행된 node 또는 route 이름입니다.
    #
    # 예:
    #
    # validate_question
    #
    # classify_intent
    #
    # route_after_classification
    #
    # call_failure_prediction
    event_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Executed LangGraph node or route name."
        ),
        examples=["classify_intent"],
    )

    # 개별 trace event 처리 상태입니다.
    #
    # success:
    #   정상 실행
    #
    # warning:
    #   실행은 완료됐지만 warning 추가
    #
    # error:
    #   현재 node 실행에서 error 추가 또는 예외 발생
    #
    # fallback:
    #   fallback route 또는 fallback node 실행
    status: Literal[
        "success",
        "warning",
        "error",
        "fallback",
    ] = Field(
        ...,
        description=(
            "Execution status of this trace event."
        ),
        examples=["success"],
    )

    # 해당 node 또는 route 실행을 시작한 UTC 시각입니다.
    #
    # 현재 내부 AgentState에서는
    # ISO 8601 문자열로 저장합니다.
    #
    # 예:
    #
    # 2026-07-10T01:12:30.120000+00:00
    started_at: str = Field(
        ...,
        description=(
            "UTC start time in ISO 8601 format."
        ),
        examples=[
            "2026-07-10T01:12:30.120000+00:00"
        ],
    )

    # 해당 node 또는 route 실행이 끝난 UTC 시각입니다.
    finished_at: str = Field(
        ...,
        description=(
            "UTC finish time in ISO 8601 format."
        ),
        examples=[
            "2026-07-10T01:12:30.932000+00:00"
        ],
    )

    # node 또는 route 실행 시간입니다.
    #
    # 단위:
    #
    # millisecond
    #
    # ge=0:
    #   실행 시간은 음수가 될 수 없습니다.
    duration_ms: float = Field(
        ...,
        ge=0.0,
        description=(
            "Execution duration in milliseconds."
        ),
        examples=[812.0],
    )

    # node 또는 route별 추가 요약 정보입니다.
    #
    # intent node 예:
    #
    # {
    #     "intent": "failure_prediction",
    #     "intent_source": "openai",
    #     "confidence": 0.95
    # }
    #
    # route 예:
    #
    # {
    #     "selected_route": "failure_prediction"
    # }
    #
    # prediction 예:
    #
    # {
    #     "prediction_succeeded": true,
    #     "prediction": 1,
    #     "risk_level": "HIGH"
    # }
    #
    # metadata는 event마다 구조가 다르므로
    # dict[str, Any]로 표현합니다.
    metadata: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
        description=(
            "Structured metadata for the node or route."
        ),
    )


class LangGraphAgentQueryResponse(BaseModel):
    """
    /agent/langgraph-query endpoint의 response schema입니다.

    이 response는 LangGraph AgentState에서 나온 값을
    API 사용자가 보기 좋은 JSON 형태로 정리한 것입니다.

    prediction 관련 필드는 optional입니다.
    이유:
        dataset_schema_query나 unknown intent에서는
        prediction 자체를 수행하지 않기 때문입니다.

    Day 16 확장
    -----------
    최종 Agent 결과뿐 아니라
    내부 실행 과정을 확인할 수 있도록
    구조화 trace 정보를 추가합니다.

    추가 항목:

        trace_id

        trace_status

        trace_started_at

        trace_finished_at

        trace_duration_ms

        fallback_occurred

        trace_events


    기존 API 호환성
    ---------------
    Day 16 trace 필드에는 기본값을 둡니다.

    따라서 기존 테스트나 다른 Python 코드에서
    trace field 없이 response model을 직접 생성하더라도
    기존 방식이 즉시 깨지지 않습니다.

    실제 /agent/langgraph-query endpoint에서는
    다음 단계에서 AgentState의 trace 값을 연결하여
    실제 trace 정보를 반환합니다.
    """

    question: str
    intent: str

    confidence: float | None = None

    intent_source: str | None = None

    intent_reason: str | None = None

    prediction: int | None = None

    probability: float | None = None

    threshold: float | None = None

    risk_level: str | None = None

    recommended_action: str | None = None

    answer: str

    # Pydantic은 mutable 기본값을 방어적으로 처리하지만,
    # default_factory=list를 사용하면
    # response 객체마다 새로운 list를 만든다는 의도가 명확합니다.
    #
    # 위 AgentEvidenceResponse의 metadata에서도
    # 같은 방식으로 Field(default_factory=dict)를 사용하고 있습니다.
    evidence: list[
        dict[str, Any]
    ] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    errors: list[str] = Field(
        default_factory=list
    )

    limitations: list[str] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Day 16 - 전체 trace 요약
    # --------------------------------------------------------

    # 요청 하나의 전체 LangGraph 실행을 구분하는 ID입니다.
    #
    # 예:
    #
    # "908759dd97bd4a3eb7494b68f76f871c"
    #
    # create_initial_agent_state()에서
    # uuid4().hex를 사용해 생성합니다.
    #
    # None 기본값을 사용하는 이유:
    #   기존 response model 생성 코드와의
    #   하위 호환성을 유지하기 위해서입니다.
    #
    # 실제 LangGraph endpoint에서는
    # 다음 단계에서 AgentState 값을 전달합니다.
    trace_id: str | None = Field(
        default=None,
        description=(
            "Unique identifier for one LangGraph execution."
        ),
        examples=[
            "908759dd97bd4a3eb7494b68f76f871c"
        ],
    )

    # 요청 하나의 전체 workflow 처리 상태입니다.
    #
    # running:
    #   아직 workflow 실행 중
    #
    # success:
    #   정상 경로로 완료
    #
    # fallback:
    #   fallback 경로로 사용자 응답 완료
    #
    # error:
    #   정상 결과를 만들지 못한 오류 상태
    #
    # 실제 API response는 graph.invoke()와
    # finalize_trace() 이후 반환되므로
    # 일반적으로 success, fallback, error 중 하나입니다.
    trace_status: (
        Literal[
            "running",
            "success",
            "fallback",
            "error",
        ]
        |
        None
    ) = Field(
        default=None,
        description=(
            "Final status of the entire LangGraph execution."
        ),
        examples=["success"],
    )

    # 전체 LangGraph trace를 시작한 UTC 시각입니다.
    #
    # ISO 8601 문자열 예:
    #
    # "2026-07-10T01:12:30.120000+00:00"
    trace_started_at: str | None = Field(
        default=None,
        description=(
            "UTC start time of the entire trace "
            "in ISO 8601 format."
        ),
        examples=[
            "2026-07-10T01:12:30.120000+00:00"
        ],
    )

    # 전체 LangGraph trace가 종료된 UTC 시각입니다.
    #
    # graph 실행이 끝난 뒤
    # finalize_trace()에서 저장합니다.
    trace_finished_at: str | None = Field(
        default=None,
        description=(
            "UTC finish time of the entire trace "
            "in ISO 8601 format."
        ),
        examples=[
            "2026-07-10T01:12:30.979000+00:00"
        ],
    )

    # 전체 LangGraph workflow 실행 시간입니다.
    #
    # 단위:
    #
    # millisecond
    #
    # ge=0:
    #   실행 시간은 음수가 될 수 없습니다.
    trace_duration_ms: (
        float
        |
        None
    ) = Field(
        default=None,
        ge=0.0,
        description=(
            "Total LangGraph execution duration "
            "in milliseconds."
        ),
        examples=[859.34],
    )

    # 실제 LangGraph fallback 경로가
    # 한 번이라도 사용됐는지 나타냅니다.
    #
    # 주의:
    #
    # intent_source == "fallback"
    #
    # 과
    #
    # fallback_occurred == True
    #
    # 는 서로 다른 의미입니다.
    #
    # intent_source == "fallback":
    #   OpenAI intent 분류 실패 후
    #   rule-based classifier를 사용했다는 의미
    #
    # fallback_occurred == True:
    #   LangGraph가 실제 fallback route 또는
    #   fallback answer node를 실행했다는 의미
    fallback_occurred: bool = Field(
        default=False,
        description=(
            "Whether the LangGraph workflow "
            "used an actual fallback route."
        ),
    )

    # 요청 하나에서 실행된
    # 모든 node와 route trace event입니다.
    #
    # 실행 순서 예:
    #
    # 1.
    # validate_question
    #
    # 2.
    # route_after_validation
    #
    # 3.
    # classify_intent
    #
    # 4.
    # route_after_classification
    #
    # 5.
    # call_failure_prediction
    #
    # 각 event는 TraceEventResponse schema로
    # FastAPI와 Swagger에서 검증·문서화됩니다.
    trace_events: list[
        TraceEventResponse
    ] = Field(
        default_factory=list,
        description=(
            "Ordered LangGraph node and route trace events."
        ),
    )