"""
Day 13 - LangGraph AgentState 정의

이 파일의 역할
----------------
LangGraph workflow 안에서 각 node가 공유할 상태 구조를 정의합니다.

LangGraph에서는 하나의 질문을 처리할 때 여러 node가 순서대로 실행됩니다.

예시 흐름:
    validate_question
    -> classify_intent
    -> prepare_input
    -> call_failure_prediction_service
    -> build_final_answer

각 node는 따로 실행되지만, 같은 상태(state)를 주고받습니다.

즉, AgentState는 LangGraph workflow가 들고 다니는 "상태 상자"입니다.

기존 manufacturing-mcp-agent와 비교
-----------------------------------
기존 manufacturing-mcp-agent도 내부적으로 question, intent, tool_name, answer, evidence 등을
처리했지만, 규칙 기반 흐름이 상대적으로 단순했습니다.

이번 Day 13에서는 OpenAI intent classifier, fallback, prediction service,
evidence, warnings, errors를 하나의 workflow state에 담아 관리합니다.

Day 15 확장
-----------
Day 15에서는 이전 사용자 질문과 Agent 답변을 저장할 수 있도록
chat_history 구조를 더 명확하게 정의합니다.

기존에는 chat_history를 list[dict[str, str]] 타입으로 표현했지만,
이 타입만으로는 dict에 어떤 key가 반드시 필요한지 알기 어렵습니다.

Day 15에서는 ChatMessage TypedDict를 추가하여
각 대화 메시지가 반드시 아래 두 값을 가지도록 타입을 명확하게 표현합니다.

    role:
        메시지를 작성한 주체
        "user" 또는 "assistant"

    content:
        실제 질문 또는 답변 내용

chat_history는 모델 prediction 입력이 아닙니다.

chat_history:
    현재 질문의 문맥을 이해하고 intent를 분류하기 위한 대화 기록

raw_sample:
    PyTorch MLP가 실제 고장 probability를 계산할 때 사용하는 설비 입력값

두 데이터의 책임을 구분합니다.

Day 16 확장
-----------
Day 16에서는 LangGraph Agent의 실행 과정을 추적할 수 있도록
내부 구조화 trace 데이터를 AgentState에 추가합니다.

기존에는 최종 intent, prediction, warning, error는 확인할 수 있었지만,
다음 정보는 하나의 실행 기록으로 남지 않았습니다.

    어떤 node가 실행됐는지

    node가 어떤 순서로 실행됐는지

    node 실행을 언제 시작하고 종료했는지

    node 실행에 몇 ms가 걸렸는지

    conditional edge가 어떤 route를 선택했는지

    fallback 경로가 실행됐는지

    전체 workflow가 성공, fallback, error 중
    어떤 상태로 종료됐는지

Day 16에서는 각 요청에 고유한 trace_id를 생성하고,
전체 trace 요약과 개별 trace event를 AgentState에 저장합니다.

전체 trace 요약:

    trace_id

    trace_status

    trace_started_at

    trace_finished_at

    trace_duration_ms

    fallback_occurred

개별 실행 기록:

    trace_events

trace_events에는 이후 각 node와 route 실행 결과가
순서대로 추가될 예정입니다.

중요:
이 파일은 trace 데이터의 "구조"와 "초기값"을 정의합니다.

실제로 node 실행 시간을 측정하고,
trace event를 추가하고,
최종 trace 상태를 결정하는 로직은
다음 단계의 src/agent/trace.py에서 분리하여 구현합니다.

중요한 점
---------
1. 처음부터 모든 값이 존재하지 않습니다.
2. 처음에는 question 정도만 있습니다.
3. node를 지나면서 intent, prediction, evidence, answer 등이 하나씩 채워집니다.
4. 그래서 TypedDict(total=False)를 사용합니다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, NotRequired, Required, TypedDict
from uuid import uuid4


# risk_level은 모델 prediction 결과를 사람이 이해하기 쉽게 바꾼 위험 등급입니다.
#
# LOW:
#   현재 입력 기준 고장 위험이 낮은 상태
#
# MEDIUM:
#   주의가 필요한 중간 위험 상태
#
# HIGH:
#   고장 위험이 높다고 판단되는 상태
#
# UNKNOWN:
#   아직 예측을 수행하지 않았거나, 오류로 인해 위험도를 판단할 수 없는 상태
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


# AgentIntent는 Day 13 LangGraph에서 우선 지원할 intent입니다.
#
# failure_prediction:
#   설비 입력값을 바탕으로 고장 위험 예측을 수행하는 intent
#
# dataset_schema_query:
#   AI4I 데이터셋의 feature, target, 컬럼 의미를 설명하는 intent
#
# unknown:
#   현재 지원하지 않거나, 입력이 부족해서 처리하기 어려운 intent
AgentIntent = Literal[
    "failure_prediction",
    "dataset_schema_query",
    "unknown",
]


# ChatRole은 대화 메시지를 작성한 주체를 나타냅니다.
#
# user:
#   사용자가 입력한 질문 또는 후속 질문
#
# assistant:
#   이전 LangGraph Agent가 생성한 답변
#
# 왜 일반 str이 아니라 Literal을 사용하는가?
# -----------------------------------------
# 다음처럼 일반 str을 사용하면:
#
#     role: str
#
# "user", "assistant" 외에도
# 오타나 지원하지 않는 문자열이 타입상 허용됩니다.
#
# 예:
#     "uesr"
#     "system"
#     "developer"
#
# 현재 Day 15 API에서는 이전 사용자 질문과
# 이전 Agent 답변만 대화 이력으로 받습니다.
#
# 따라서 Literal을 사용하여 허용할 값을
# "user"와 "assistant"로 명확하게 제한합니다.
#
# 특히 외부 API 사용자가 임의로 "system" role을 전달하면,
# 대화 기록과 내부 system instruction의 경계가 흐려질 수 있습니다.
#
# 내부 system instruction은 개발자가 관리하고,
# API의 chat_history는 문맥 이해용 user/assistant 데이터로만 사용합니다.
ChatRole = Literal["user", "assistant"]


class ChatMessage(TypedDict):
    """
    chat_history에 저장할 대화 메시지 한 개의 구조입니다.

    예:
        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?"
        }

        {
            "role": "assistant",
            "content": "현재 입력 조건에서는 고장 위험이 높게 예측되었습니다."
        }

    왜 dict[str, str] 대신 TypedDict를 사용하는가?
    ---------------------------------------------
    기존 타입:

        dict[str, str]

    이 타입은 key와 value가 문자열이라는 사실만 표현합니다.

    따라서 아래처럼 필요한 key가 빠져도
    타입 구조가 명확하지 않습니다.

        {
            "speaker": "user",
            "message": "고장 위험이 높아?"
        }

    Day 15에서는 ChatMessage TypedDict를 사용하여
    각 메시지가 반드시 role과 content를 가진다는 사실을
    타입 수준에서 명확하게 표현합니다.

    ChatMessage는 total=True가 기본값입니다.

    따라서 role과 content는 모두 필수 key입니다.
    """

    # 메시지를 작성한 주체입니다.
    #
    # 현재는 "user" 또는 "assistant"만 허용합니다.
    role: ChatRole

    # 실제 질문 또는 답변 내용입니다.
    content: str


# TraceEventType은 trace 기록 한 개가
# 어떤 종류의 실행을 나타내는지 구분합니다.
#
# node:
#   LangGraph node 함수 실행 기록
#
# 예:
#   validate_question
#   classify_intent
#   call_failure_prediction
#   build_final_answer
#
# route:
#   conditional edge가 다음 실행 경로를
#   선택한 결과를 기록
#
# 예:
#   route_after_validation
#   route_after_classification
#   route_after_prediction
#
# node와 route를 구분하는 이유
# ----------------------------
# node는 실제 검증, intent 분류, prediction,
# 답변 생성 등의 작업을 수행합니다.
#
# route는 현재 state를 확인한 뒤
# 다음에 어느 node로 이동할지 결정합니다.
#
# 두 실행의 책임이 다르므로 event_type으로 구분합니다.
TraceEventType = Literal[
    "node",
    "route",
]


# TraceEventStatus는 개별 node 또는 route 실행 결과를 나타냅니다.
#
# success:
#   해당 실행이 정상적으로 완료됨
#
# warning:
#   핵심 실행은 완료됐지만 경고가 추가됨
#
# 예:
#   OpenAI intent 분류 실패 후
#   rule-based classifier로 정상 처리
#
# error:
#   해당 실행 중 workflow에 영향을 주는 문제가 발생함
#
# 예:
#   raw_sample이 없어 prediction을 수행하지 못함
#
# fallback:
#   fallback 경로 또는 fallback 답변을 사용함
#
# 주의:
# intent_source == "fallback"과
# LangGraph fallback 경로는 서로 다른 개념입니다.
#
# intent_source == "fallback":
#   OpenAI classifier가 실패하여
#   rule-based classifier를 사용했다는 의미
#
# TraceEventStatus == "fallback":
#   LangGraph가 실제 fallback 경로를
#   선택하거나 fallback 답변을 사용했다는 의미
TraceEventStatus = Literal[
    "success",
    "warning",
    "error",
    "fallback",
]


# TraceStatus는 요청 하나의 전체 LangGraph 실행 상태입니다.
#
# running:
#   AgentState를 만들었고
#   workflow가 아직 실행 중인 상태
#
# success:
#   요청이 정상 경로로 완료된 상태
#
# fallback:
#   요청 자체는 응답했지만
#   fallback 답변 경로를 사용한 상태
#
# error:
#   처리 과정에서 오류가 발생하여
#   정상 결과를 만들지 못한 상태
#
# 개별 TraceEventStatus와의 차이
# -----------------------------
# TraceEventStatus:
#   node 또는 route 하나의 실행 결과
#
# TraceStatus:
#   요청 하나의 전체 workflow 결과
TraceStatus = Literal[
    "running",
    "success",
    "fallback",
    "error",
]


class TraceEvent(TypedDict):
    """
    LangGraph 실행 과정에서 발생한
    node 또는 route 기록 한 개의 구조입니다.

    예 - intent 분류 node:

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

    예 - conditional routing:

        {
            "sequence": 4,
            "event_type": "route",
            "event_name": "route_after_classification",
            "status": "success",
            "started_at": "2026-07-10T01:12:30.933000+00:00",
            "finished_at": "2026-07-10T01:12:30.933020+00:00",
            "duration_ms": 0.02,
            "metadata": {
                "intent": "failure_prediction",
                "selected_route": "failure_prediction"
            }
        }

    왜 TypedDict를 사용하는가?
    -------------------------
    다음처럼 일반 dict만 사용하면:

        dict[str, Any]

    어떤 key가 반드시 필요한지
    코드만 보고 알기 어렵습니다.

    TraceEvent TypedDict를 사용하면
    trace event가 어떤 필드를 가져야 하는지
    타입 수준에서 명확하게 표현할 수 있습니다.

    TraceEvent는 total=True가 기본값입니다.

    따라서 아래 필드는 모두 필수입니다.

        sequence

        event_type

        event_name

        status

        started_at

        finished_at

        duration_ms

        metadata
    """

    # 하나의 trace 안에서 event가 발생한 순서입니다.
    #
    # 첫 번째 event:
    #   sequence = 1
    #
    # 두 번째 event:
    #   sequence = 2
    #
    # list 자체도 순서를 유지하지만,
    # sequence 값을 별도로 저장하면
    # 이후 JSONL, DB, Dashboard에서
    # 실행 순서를 명확하게 정렬할 수 있습니다.
    sequence: int

    # 이 event가 LangGraph node 실행인지,
    # conditional route 실행인지 구분합니다.
    event_type: TraceEventType

    # 실제 실행된 node 또는 route의 이름입니다.
    #
    # 예:
    # "validate_question"
    # "classify_intent"
    # "route_after_classification"
    # "call_failure_prediction"
    event_name: str

    # 개별 event의 실행 결과입니다.
    #
    # success
    # warning
    # error
    # fallback
    status: TraceEventStatus

    # 해당 event 실행을 시작한 UTC 시각입니다.
    #
    # ISO 8601 문자열 형식을 사용합니다.
    #
    # 예:
    # "2026-07-10T01:12:30.120000+00:00"
    started_at: str

    # 해당 event 실행이 끝난 UTC 시각입니다.
    #
    # ISO 8601 문자열 형식을 사용합니다.
    finished_at: str

    # 해당 event 실행에 걸린 시간입니다.
    #
    # 단위:
    #   millisecond
    #
    # 예:
    #   12.5
    #
    # 의미:
    #   12.5 ms
    duration_ms: float

    # event 종류에 따라 추가로 확인할 값을 저장합니다.
    #
    # intent 분류 node 예:
    #
    # {
    #   "intent": "failure_prediction",
    #   "intent_source": "openai",
    #   "confidence": 0.95
    # }
    #
    # routing 예:
    #
    # {
    #   "selected_route": "failure_prediction"
    # }
    #
    # prediction node 예:
    #
    # {
    #   "prediction_succeeded": True,
    #   "prediction": 1,
    #   "risk_level": "HIGH",
    #   "evidence_count": 8
    # }
    #
    # metadata에 모든 원본 데이터를 넣지는 않습니다.
    #
    # 전체 chat_history,
    # 전체 raw_sample,
    # API key,
    # 환경 변수,
    # OpenAI 원본 응답 전문 등은
    # 민감 정보 노출과 trace 크기 증가를 막기 위해
    # 구조화 trace에 그대로 저장하지 않습니다.
    metadata: dict[str, Any]


class AgentState(TypedDict, total=False):
    """
    LangGraph workflow에서 node들이 공유하는 상태입니다.

    왜 total=False를 사용하는가?
    ---------------------------
    AgentState의 모든 필드가 처음부터 존재하는 것이 아니기 때문입니다.

    예를 들어 사용자가 질문을 막 입력한 직후에는 보통 아래 정도만 있습니다.

        {
            "question": "Torque 62이면 고장 위험이 높아?"
        }

    이후 classify_intent node를 지나면 intent가 추가됩니다.

        {
            "question": "...",
            "intent": "failure_prediction",
            "confidence": 0.91
        }

    prediction node를 지나면 prediction, probability, evidence 등이 추가됩니다.

    즉, node가 실행될수록 state가 점점 채워지는 구조입니다.

    Required와 NotRequired
    ----------------------
    Required:
        이 state를 만들 때 반드시 있어야 하는 필드입니다.

    NotRequired:
        workflow 중간에 생길 수도 있고, 없을 수도 있는 필드입니다.

    현재는 question만 Required로 둡니다.

    Day 16 trace 필드도 왜 NotRequired인가?
    -------------------------------------
    create_initial_agent_state()를 사용하면
    trace_id와 trace 초기값을 항상 생성합니다.

    하지만 기존 테스트와 일부 node 테스트에서는
    AgentState를 helper 함수 없이 직접 만들 수도 있습니다.

    예:

        state: AgentState = {
            "question": "고장 위험을 예측해줘."
        }

    기존 수동 state 생성 방식과의 호환성을 유지하기 위해
    trace 필드는 NotRequired로 둡니다.

    실제 public runner에서는
    create_initial_agent_state()를 사용하므로
    trace 초기값이 생성될 예정입니다.
    """

    # 사용자의 원본 질문입니다.
    #
    # LangGraph workflow의 시작점에서 반드시 필요합니다.
    question: Required[str]

    # 대화 기록입니다.
    #
    # Day 13에서는 필수는 아니지만,
    # 이후 multi-turn Agent로 확장할 때 사용할 수 있습니다.
    #
    # Day 15에서는 ChatMessage TypedDict를 사용하여
    # 각 메시지가 role과 content를 가지도록 타입을 구체화합니다.
    #
    # 예:
    # [
    #   {"role": "user", "content": "이 설비 위험해?"},
    #   {"role": "assistant", "content": "고장 위험이 높습니다."}
    # ]
    #
    # chat_history의 목적:
    #   이전 대화를 참고하여
    #   "그건 왜 그래?"
    #   "그중 중요한 것은 뭐야?"
    #   같은 후속 질문의 문맥을 이해하는 것
    #
    # chat_history가 하지 않는 일:
    #   이전 대화에 적힌 설비 값을
    #   자동으로 PyTorch 모델 입력으로 사용하는 것
    #
    # 실제 prediction은 계속 raw_sample을 사용합니다.
    chat_history: NotRequired[list[ChatMessage]]

    # OpenAI 또는 rule-based classifier가 분류한 intent입니다.
    #
    # 예:
    # "failure_prediction"
    # "dataset_schema_query"
    # "unknown"
    intent: NotRequired[str]

    # intent 분류 신뢰도입니다.
    #
    # OpenAI classifier가 반환한 confidence이거나,
    # rule-based fallback에서 임의로 지정한 confidence입니다.
    #
    # 0.0 ~ 1.0 범위를 사용합니다.
    confidence: NotRequired[float]

    # intent를 그렇게 판단한 이유입니다.
    #
    # 면접/디버깅/trace 관점에서 중요합니다.
    intent_reason: NotRequired[str]

    # intent 분류 방식입니다.
    #
    # 예:
    # "openai"
    # "rule_based"
    # "fallback"
    # "validation"
    intent_source: NotRequired[str]

    # intent 분류 과정의 원본 응답입니다.
    #
    # OpenAI 응답 원문을 디버깅 목적으로 보관할 수 있습니다.
    # 운영 응답에 그대로 노출할 필요는 없습니다.
    intent_raw_response: NotRequired[str | None]

    # 설비 고장 예측에 사용할 raw sample입니다.
    #
    # Day 12 API request에서 받던 값을 LangGraph state에 담는 용도입니다.
    #
    # 예:
    # {
    #   "Air temperature [K]": 303.0,
    #   "Process temperature [K]": 312.5,
    #   "Rotational speed [rpm]": 1380.0,
    #   "Torque [Nm]": 62.0,
    #   "Tool wear [min]": 220.0,
    #   "Type": "L"
    # }
    raw_sample: NotRequired[dict[str, Any]]

    # 모델의 최종 예측값입니다.
    #
    # 0:
    #   정상으로 예측
    #
    # 1:
    #   고장 위험으로 예측
    #
    # None:
    #   아직 예측하지 않았거나 예측 실패
    prediction: NotRequired[int | None]

    # 모델이 계산한 고장 probability입니다.
    #
    # 주의:
    #   probability는 모델이 추정한 고장 위험 확률이고,
    #   prediction은 threshold를 기준으로 0/1로 변환한 최종 판단입니다.
    probability: NotRequired[float | None]

    # prediction을 결정할 때 사용한 threshold입니다.
    #
    # 예:
    # probability >= 0.7이면 prediction = 1
    threshold: NotRequired[float | None]

    # 사람이 이해하기 쉬운 위험 등급입니다.
    #
    # probability와 threshold를 바탕으로 LOW/MEDIUM/HIGH 등으로 표현합니다.
    risk_level: NotRequired[RiskLevel]

    # 사용자에게 권장할 조치입니다.
    #
    # 예:
    # "고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다."
    recommended_action: NotRequired[str]

    # Agent evidence 목록입니다.
    #
    # Day 9에서 만든 AgentEvidence 구조를 dict로 변환해서 담을 수 있습니다.
    #
    # evidence 예:
    # [
    #   {
    #     "evidence_type": "prediction_summary",
    #     "source": "model_prediction",
    #     "summary": "모델은 고장 probability를 0.9930으로 예측했습니다."
    #   },
    #   {
    #     "evidence_type": "shap_local",
    #     "source": "shap",
    #     "feature": "Torque [Nm]",
    #     "contribution": 5.1592
    #   }
    # ]
    evidence: NotRequired[list[dict[str, Any]]]

    # 최종 사용자 답변입니다.
    #
    # generate_answer node 또는 service layer에서 생성됩니다.
    answer: NotRequired[str]

    # 경고 메시지 목록입니다.
    #
    # warnings는 prediction 자체는 가능하지만,
    # 부가 기능 일부가 실패했을 때 사용합니다.
    #
    # 예:
    # - SHAP artifact가 없어 shap_local evidence를 생략했습니다.
    # - global importance artifact가 없어 global_importance evidence를 생략했습니다.
    warnings: NotRequired[list[str]]

    # 오류 메시지 목록입니다.
    #
    # errors는 workflow 수행에 영향을 주는 문제를 기록합니다.
    #
    # 예:
    # - question이 비어 있습니다.
    # - raw_sample이 없어 failure_prediction을 수행할 수 없습니다.
    # - model artifact 로딩에 실패했습니다.
    errors: NotRequired[list[str]]

    # 현재 Agent가 가진 한계 설명입니다.
    #
    # Day 12 API response에서 limitations를 사용했듯이,
    # LangGraph state에서도 사용자에게 안내할 한계를 담을 수 있습니다.
    limitations: NotRequired[list[str]]

    # trace_id는 요청 하나의 전체 실행을 식별하는 고유 ID입니다.
    #
    # Day 13에서는 향후 확장을 위해 필드만 준비했습니다.
    #
    # Day 16에서는 create_initial_agent_state()가 실행될 때
    # uuid4().hex를 사용하여 실제 값을 생성합니다.
    #
    # 예:
    # "908759dd97bd4a3eb7494b68f76f871c"
    #
    # 같은 요청 안에서 발생하는 모든 node와 route event는
    # 같은 trace_id 아래에 묶입니다.
    #
    # 이후 확장:
    # - 구조화 로그
    # - 실행 이력 DB
    # - LangSmith
    # - OpenTelemetry
    # - Streamlit trace Dashboard
    trace_id: NotRequired[str]

    # 요청 하나의 전체 workflow 처리 상태입니다.
    #
    # 초기값:
    #   "running"
    #
    # workflow 완료 후에는
    # 다음 단계의 trace 종료 함수가 상태를 변경할 예정입니다.
    #
    # 정상 완료:
    #   "success"
    #
    # fallback 답변 사용:
    #   "fallback"
    #
    # 처리 실패:
    #   "error"
    trace_status: NotRequired[TraceStatus]

    # 전체 trace를 시작한 UTC 시각입니다.
    #
    # create_initial_agent_state()를 호출할 때 생성합니다.
    #
    # ISO 8601 예:
    # "2026-07-10T01:12:30.120000+00:00"
    #
    # 왜 UTC를 사용하는가?
    # --------------------
    # 서버가 다른 지역이나 다른 time zone에서 실행돼도
    # 같은 시간 기준으로 로그를 비교하기 쉽기 때문입니다.
    trace_started_at: NotRequired[str]

    # 전체 trace가 종료된 UTC 시각입니다.
    #
    # 초기 state에서는 아직 workflow가 끝나지 않았으므로
    # None으로 초기화합니다.
    #
    # workflow가 끝나면
    # 다음 단계의 trace 종료 함수가 실제 시각을 저장합니다.
    trace_finished_at: NotRequired[str | None]

    # 전체 LangGraph workflow 실행 시간입니다.
    #
    # 단위:
    #   millisecond
    #
    # 초기 state에서는 아직 workflow가 끝나지 않았으므로
    # None으로 초기화합니다.
    #
    # 예:
    #   859.34
    #
    # 의미:
    #   전체 workflow 실행에 859.34 ms가 걸림
    trace_duration_ms: NotRequired[float | None]

    # LangGraph가 실제 fallback 경로를 사용했는지 나타냅니다.
    #
    # 초기값:
    #   False
    #
    # fallback route 또는 fallback answer가 실행되면
    # 다음 단계의 trace 로직에서 True로 변경합니다.
    #
    # 주의:
    # intent_source == "fallback"과는 다른 값입니다.
    #
    # intent_source == "fallback":
    #   OpenAI intent 분류 실패 후
    #   rule-based classifier를 사용했다는 의미
    #
    # fallback_occurred == True:
    #   LangGraph가 실제 fallback 처리 경로를
    #   실행했다는 의미
    fallback_occurred: NotRequired[bool]

    # 요청 하나에서 발생한 node와 route 실행 기록입니다.
    #
    # 초기값:
    #   []
    #
    # 이후 실행 예:
    #
    # [
    #   {
    #     "sequence": 1,
    #     "event_type": "node",
    #     "event_name": "validate_question",
    #     "status": "success",
    #     ...
    #   },
    #   {
    #     "sequence": 2,
    #     "event_type": "route",
    #     "event_name": "route_after_validation",
    #     "status": "success",
    #     ...
    #   }
    # ]
    #
    # node와 route가 실행될 때마다
    # 다음 단계의 src/agent/trace.py가
    # TraceEvent를 생성하여 이 list에 추가할 예정입니다.
    trace_events: NotRequired[list[TraceEvent]]

    # 가장 최근 route 판단 결과입니다.
    #
    # Day 16에서는 routing 함수가 선택한 경로를
    # 먼저 AgentState에 저장합니다.
    #
    # 예:
    #
    # "classify"
    #
    # "failure_prediction"
    #
    # "dataset_schema"
    #
    # "fallback"
    #
    # "final"
    #
    # 이후 실제 LangGraph conditional edge는
    # route 함수를 다시 실행하지 않고
    # 이 값을 읽어 다음 node를 선택합니다.
    #
    # 이 값은 내부 workflow 제어용이며
    # 사용자 API response에 반드시 노출할 필요는 없습니다.
    selected_route: NotRequired[str]

    include_shap: NotRequired[bool]
    include_global_importance: NotRequired[bool]


def create_initial_agent_state(
    *,
    question: str,
    raw_sample: dict[str, Any] | None = None,

    # list는 값을 추가하거나 삭제할 수 있는 mutable 객체입니다.
    #
    # 다음처럼 함수 매개변수의 기본값을 직접 []로 작성하면:
    #
    #     chat_history: list[ChatMessage] = []
    #
    # 함수가 호출될 때마다 새로운 빈 list가 생성되는 것이 아니라,
    # 함수가 정의될 때 생성된 하나의 기본 list 객체가
    # 이후 여러 함수 호출에서 계속 재사용될 수 있습니다.
    #
    # 따라서 함수 내부에서 append(), extend() 등으로
    # 기본 list의 내용을 수정하면,
    # 이전 함수 호출에서 추가한 대화 기록이
    # 다음 함수 호출에도 남을 수 있습니다.
    #
    # Agent 요청마다 대화 기록은 서로 독립적이어야 합니다.
    #
    # 이전 요청의 chat_history가 새로운 요청에 섞이면,
    # 서로 관련 없는 사용자의 질문이나 Agent 답변이
    # 다음 intent 분류의 문맥으로 잘못 사용될 수 있습니다.
    #
    # 이를 방지하기 위해 수정 가능한 빈 list인 []를
    # 함수 기본값으로 직접 사용하지 않고,
    # 기본값을 None으로 둡니다.
    #
    # 이후 함수가 실제로 실행될 때
    # 새로운 빈 list를 생성합니다.
    chat_history: list[ChatMessage] | None = None,
) -> AgentState:
    """
    LangGraph 실행을 시작하기 위한 초기 AgentState를 만듭니다.

    왜 helper 함수가 필요한가?
    -------------------------
    매번 dict를 직접 만들면 기본값이 빠지거나 key 이름을 잘못 쓸 수 있습니다.

    helper 함수를 사용하면 초기 state의 기본 구조를 일관되게 만들 수 있습니다.

    Parameters
    ----------
    question:
        사용자의 자연어 질문입니다.

    raw_sample:
        failure_prediction intent일 때 사용할 설비 입력값입니다.
        없을 수도 있으므로 None을 허용합니다.

    chat_history:
        이전 대화 기록입니다.

        Day 13에서는 선택값으로 준비했습니다.

        Day 15에서는 값이 전달되지 않아도
        초기 state에 새로운 빈 list를 넣습니다.

        따라서 이후 node는 chat_history key가 있는지
        매번 검사하지 않고 대화 기록을 읽을 수 있습니다.

    Returns
    -------
    AgentState
        LangGraph workflow에 넣을 초기 상태입니다.

    Day 16 trace 초기화
    ------------------
    이 함수는 일반 Agent 값뿐 아니라
    요청 하나의 trace 초기 상태도 함께 생성합니다.

    생성되는 값:

        trace_id:
            요청을 구분하는 고유 ID

        trace_status:
            아직 workflow 실행 중이므로 "running"

        trace_started_at:
            초기 state를 만든 UTC 시각

        trace_finished_at:
            아직 끝나지 않았으므로 None

        trace_duration_ms:
            아직 전체 시간이 계산되지 않았으므로 None

        fallback_occurred:
            아직 fallback이 발생하지 않았으므로 False

        trace_events:
            아직 node나 route가 실행되지 않았으므로 빈 list
    """

    # uuid4()는 무작위 기반 UUID 객체를 생성합니다.
    #
    # 예:
    #
    # UUID(
    #     "908759dd-97bd-4a3e-b749-4b68f76f871c"
    # )
    #
    # .hex를 사용하면 하이픈이 없는
    # 32자리 16진수 문자열을 얻습니다.
    #
    # 예:
    #
    # "908759dd97bd4a3eb7494b68f76f871c"
    #
    # 요청마다 새로운 값이 생성되므로
    # 여러 Agent 실행을 서로 구분할 수 있습니다.
    trace_id = uuid4().hex

    # datetime.now(timezone.utc)는
    # 현재 UTC 시각을 timezone 정보와 함께 만듭니다.
    #
    # 예:
    #
    # datetime(
    #     2026,
    #     7,
    #     10,
    #     1,
    #     12,
    #     30,
    #     tzinfo=timezone.utc,
    # )
    #
    # .isoformat()은 datetime 객체를
    # 로그와 JSON에 저장하기 쉬운 문자열로 변환합니다.
    #
    # 예:
    #
    # "2026-07-10T01:12:30.120000+00:00"
    #
    # Day 16에서는 trace 시각을 UTC 기반
    # ISO 8601 문자열로 통일합니다.
    trace_started_at = datetime.now(timezone.utc).isoformat()

    state: AgentState = {
        # question은 Required 필드이므로 반드시 넣습니다.
        "question": question,

        # chat_history가 전달되지 않았다면:
        #
        #     chat_history is None
        #
        # 아래 표현은 새로운 빈 list를 만듭니다.
        #
        #     list([])
        #
        # chat_history가 전달되었다면:
        #
        #     list(chat_history)
        #
        # 를 실행하여 새로운 최상위 list 객체를 만듭니다.
        #
        # 따라서 외부에서 전달한 원본 chat_history list와
        # AgentState 내부의 chat_history list가
        # 같은 최상위 list 객체를 직접 공유하지 않습니다.
        #
        # 예:
        #
        # original_history = [
        #     {
        #         "role": "user",
        #         "content": "고장 위험이 높아?"
        #     }
        # ]
        #
        # state = create_initial_agent_state(
        #     question="그건 왜 그래?",
        #     chat_history=original_history,
        # )
        #
        # state["chat_history"].append(...)
        #
        # 위 코드에서 state 내부 list에 새 메시지를 추가해도
        # original_history의 최상위 list에는
        # 같은 메시지가 자동으로 추가되지 않습니다.
        #
        # 주의:
        # list(...)는 얕은 복사이므로
        # list 안의 개별 dict까지 깊게 복사하는 것은 아닙니다.
        #
        # 현재 구조에서는 메시지 dict 자체를 수정하지 않고,
        # 새로운 메시지를 list에 추가하는 방식으로 사용할 예정이므로
        # 최상위 list를 분리하는 것으로 충분합니다.
        "chat_history": list(chat_history or []),

        # warnings와 errors는 여러 node에서 append할 가능성이 높습니다.
        # 처음부터 빈 list로 만들어두면 이후 코드에서 setdefault를 반복하지 않아도 됩니다.
        "warnings": [],
        "errors": [],

        # limitations도 최종 답변에 붙일 수 있으므로 기본값을 빈 list로 둡니다.
        "limitations": [],

        # Day 16:
        # 요청 하나를 구분하는 고유 trace ID입니다.
        #
        # create_initial_agent_state()를 호출할 때마다
        # 새로운 UUID 기반 문자열이 생성됩니다.
        "trace_id": trace_id,

        # 초기 state를 만든 직후에는
        # LangGraph workflow가 아직 끝나지 않았으므로
        # 전체 trace 상태를 "running"으로 설정합니다.
        "trace_status": "running",

        # 요청 trace를 시작한 UTC 시각입니다.
        "trace_started_at": trace_started_at,

        # 아직 workflow가 끝나지 않았으므로
        # 종료 시각은 None입니다.
        "trace_finished_at": None,

        # 아직 전체 workflow 실행 시간이 계산되지 않았으므로
        # duration도 None입니다.
        "trace_duration_ms": None,

        # 초기 state에서는 아직 fallback 경로를
        # 실행하지 않았으므로 False입니다.
        "fallback_occurred": False,

        # 아직 LangGraph node나 route가 실행되지 않았으므로
        # trace event 목록은 빈 list로 시작합니다.
        #
        # 다음 단계의 src/agent/trace.py가
        # 각 실행 기록을 이 list에 추가할 예정입니다.
        "trace_events": [],
    }

    # raw_sample이 있는 경우에만 state에 넣습니다.
    #
    # 왜 None으로 넣지 않고 아예 생략하는가?
    # -------------------------------------
    # total=False TypedDict에서는 없는 값과 None인 값을 구분할 수 있습니다.
    #
    # - key 없음: 아직 준비되지 않음
    # - key는 있는데 None: 명시적으로 값이 없음
    #
    # 지금은 raw_sample이 제공된 경우에만 추가합니다.
    if raw_sample is not None:
        state["raw_sample"] = raw_sample

    # Day 13에서는 chat_history가 제공된 경우에만
    # 아래와 같이 state에 추가했습니다.
    #
    #     if chat_history is not None:
    #         state["chat_history"] = chat_history
    #
    # Day 15에서는 chat_history가 없어도
    # 항상 새로운 빈 list를 초기 state에 넣습니다.
    #
    # 따라서 기존 조건문은 제거하고,
    # 위 state 생성 부분에서 chat_history를 초기화합니다.

    return state


def append_warning(state: AgentState, message: str) -> AgentState:
    """
    AgentState에 warning 메시지를 추가합니다.

    warnings와 errors의 차이
    ------------------------
    warning:
        핵심 기능은 수행됐지만, 부가 기능 일부가 실패한 경우

        예:
        - prediction은 성공했지만 SHAP 계산 실패
        - prediction은 성공했지만 global importance 로딩 실패

    error:
        workflow 수행에 직접 문제가 되는 경우

        예:
        - 질문이 비어 있음
        - raw_sample이 없음
        - 모델 artifact 로딩 실패
    """

    # state에 warnings key가 없을 수도 있으므로 setdefault를 사용합니다.
    #
    # setdefault("warnings", [])의 의미:
    #   warnings가 이미 있으면 기존 list를 사용
    #   warnings가 없으면 빈 list를 새로 만들고 state에 넣음
    state.setdefault("warnings", [])

    # 이제 warnings list가 보장되므로 append할 수 있습니다.
    state["warnings"].append(message)

    return state


def append_error(state: AgentState, message: str) -> AgentState:
    """
    AgentState에 error 메시지를 추가합니다.

    LangGraph node에서 예외가 발생했거나,
    다음 단계로 진행할 수 없는 조건을 발견했을 때 사용합니다.
    """

    # state에 errors key가 없을 수도 있으므로 setdefault를 사용합니다.
    state.setdefault("errors", [])

    # 에러 메시지를 누적합니다.
    state["errors"].append(message)

    return state


def has_errors(state: AgentState) -> bool:
    """
    현재 AgentState에 error가 있는지 확인합니다.

    LangGraph conditional edge에서 사용할 수 있습니다.

    예:
        if has_errors(state):
            return "generate_fallback_answer"
        return "continue"
    """

    return len(state.get("errors", [])) > 0