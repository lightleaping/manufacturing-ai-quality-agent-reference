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
    metadata: dict[str, Any] = Field(default_factory=dict)


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

    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


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
    role: Literal["user", "assistant"] = Field(
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
    chat_history: list[ChatMessageRequest] = Field(
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
    raw_sample: LangGraphRawSampleRequest | None = None

    # SHAP local explanation 사용 여부입니다.
    include_shap: bool = True

    # global permutation importance evidence 사용 여부입니다.
    include_global_importance: bool = True


class LangGraphAgentQueryResponse(BaseModel):
    """
    /agent/langgraph-query endpoint의 response schema입니다.

    이 response는 LangGraph AgentState에서 나온 값을
    API 사용자가 보기 좋은 JSON 형태로 정리한 것입니다.

    prediction 관련 필드는 optional입니다.
    이유:
        dataset_schema_query나 unknown intent에서는
        prediction 자체를 수행하지 않기 때문입니다.
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
    evidence: list[dict[str, Any]] = Field(
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