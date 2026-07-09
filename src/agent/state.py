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

중요한 점
---------
1. 처음부터 모든 값이 존재하지 않습니다.
2. 처음에는 question 정도만 있습니다.
3. node를 지나면서 intent, prediction, evidence, answer 등이 하나씩 채워집니다.
4. 그래서 TypedDict(total=False)를 사용합니다.
"""

from __future__ import annotations

from typing import Any, Literal, NotRequired, Required, TypedDict


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
    # 예:
    # [
    #   {"role": "user", "content": "이 설비 위험해?"},
    #   {"role": "assistant", "content": "고장 위험이 높습니다."}
    # ]
    chat_history: NotRequired[list[dict[str, str]]]

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

    # trace_id는 요청 하나를 추적하기 위한 ID입니다.
    #
    # 지금 당장 필수는 아니지만,
    # 나중에 로그, LangSmith, OpenTelemetry 등과 연결할 때 유용합니다.
    trace_id: NotRequired[str]


def create_initial_agent_state(
    *,
    question: str,
    raw_sample: dict[str, Any] | None = None,
    chat_history: list[dict[str, str]] | None = None,
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
        Day 13에서는 선택값입니다.

    Returns
    -------
    AgentState
        LangGraph workflow에 넣을 초기 상태입니다.
    """

    state: AgentState = {
        # question은 Required 필드이므로 반드시 넣습니다.
        "question": question,

        # warnings와 errors는 여러 node에서 append할 가능성이 높습니다.
        # 처음부터 빈 list로 만들어두면 이후 코드에서 setdefault를 반복하지 않아도 됩니다.
        "warnings": [],
        "errors": [],

        # limitations도 최종 답변에 붙일 수 있으므로 기본값을 빈 list로 둡니다.
        "limitations": [],
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

    # chat_history도 제공된 경우에만 추가합니다.
    if chat_history is not None:
        state["chat_history"] = chat_history

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


def has_raw_sample(state: AgentState) -> bool:
    """
    failure_prediction을 수행할 raw_sample이 있는지 확인합니다.

    intent가 failure_prediction이어도 raw_sample이 없으면
    실제 모델 예측을 수행할 수 없습니다.
    """

    raw_sample = state.get("raw_sample")

    return isinstance(raw_sample, dict) and len(raw_sample) > 0