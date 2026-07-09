"""
Day 13 - LangGraph 기반 Failure Agent workflow

이 파일의 역할
----------------
사용자의 자연어 질문을 LangGraph workflow로 처리합니다.

전체 흐름
---------
사용자 질문 + 선택적 raw_sample
-> validate_question_node
-> classify_intent_node
-> intent에 따라 분기
   - failure_prediction     -> call_failure_prediction_node
   - dataset_schema_query    -> build_dataset_schema_answer_node
   - unknown                 -> build_fallback_answer_node
-> 최종 answer 반환

기존 manufacturing-mcp-agent와의 차이
--------------------------------------
기존 manufacturing-mcp-agent:
    question
    -> rule-based intent routing
    -> tool 호출
    -> answer/evidence 반환

이번 Day 13 reference project:
    question
    -> OpenAI gpt-4o-mini intent classifier
    -> JSON 검증
    -> 실패 시 rule-based fallback
    -> LangGraph AgentState 기반 workflow
    -> Day 12 failure_agent_service 재사용

중요한 설계 원칙
----------------
1. LLM은 최종 고장 예측을 하지 않습니다.
2. LLM은 질문을 intent로 분류하는 역할만 합니다.
3. 실제 prediction은 Day 12의 failure_agent_service가 담당합니다.
4. raw_sample이 없으면 억지로 예측하지 않고 fallback answer를 반환합니다.
5. workflow 중 오류는 errors에 누적하고, 부가 기능 실패는 warnings에 누적합니다.
"""

from __future__ import annotations

from typing import Any, Mapping

from langgraph.graph import END, START, StateGraph

from src.agent.intent_classifier import classify_intent
from src.agent.state import (
    AgentState,
    ChatMessage,
    append_error,
    append_warning,
    create_initial_agent_state,
    has_errors,
    has_raw_sample,
)


def validate_question_node(state: AgentState) -> AgentState:
    """
    사용자 질문이 비어 있지 않은지 검증하는 node입니다.

    LangGraph node란?
    ----------------
    LangGraph에서 node는 workflow의 한 처리 단계입니다.

    이 node는 state를 입력으로 받고,
    수정된 state를 반환합니다.

    여기서는 question이 비어 있으면 errors에 메시지를 추가합니다.
    """

    question = state.get("question", "")

    if not isinstance(question, str) or not question.strip():
        append_error(
            state,
            "question이 비어 있어 Agent workflow를 실행할 수 없습니다.",
        )
        return state

    # 앞뒤 공백을 제거한 질문으로 정리합니다.
    state["question"] = question.strip()

    return state


def classify_intent_node(state: AgentState) -> AgentState:
    """
    사용자 질문을 intent로 분류하는 node입니다.

    처리 흐름
    --------
    1. state["question"]을 읽습니다.
    2. state["chat_history"]를 읽습니다.
    3. intent_classifier.classify_intent()를 호출합니다.
    4. 결과를 AgentState에 저장합니다.

    classify_intent() 내부에서는 다음 구조가 동작합니다.

        최근 chat_history
        +
        현재 question
        -> OpenAI gpt-4o-mini intent classification
        -> JSON 검증
        -> 실패 시 rule-based fallback

    Day 15 확장
    -----------
    기존 Day 13에서는 현재 question만 intent classifier에 전달했습니다.

        question
        -> classify_intent()
        -> intent

    Day 15에서는 이전 대화 기록도 함께 전달합니다.

        chat_history
        +
        question
        -> classify_intent()
        -> intent

    예:

        이전 user:
            "이 설비 조건이면 고장 위험이 높아?"

        이전 assistant:
            "현재 입력 조건에서는 고장 위험이 높게 예측되었습니다."

        현재 user:
            "그건 왜 그래?"

    현재 질문만 보면 "그건"이 무엇을 의미하는지 알기 어렵습니다.

    이전 대화 문맥을 함께 전달하면
    현재 질문이 이전 고장 예측과 관련된 후속 질문이라는 점을
    intent classifier가 참고할 수 있습니다.

    주의
    ----
    LLM은 고장 예측을 직접 하지 않습니다.

    LLM은 현재 질문이 어떤 workflow에 해당하는지
    intent만 분류합니다.

    chat_history는 질문의 문맥을 이해하기 위한 데이터입니다.

    실제 고장 probability와 prediction은
    계속 state["raw_sample"]을 사용하며,
    Day 12 failure prediction service가 계산합니다.
    """

    # 현재 intent를 분류할 사용자 질문을 읽습니다.
    question = state.get("question", "")

    # AgentState에 저장된 이전 대화 기록을 읽습니다.
    #
    # Day 15부터 create_initial_agent_state()는
    # chat_history가 전달되지 않은 single-turn 요청에서도
    # 새로운 빈 list를 state에 넣습니다.
    #
    # 따라서 일반적인 workflow에서는
    # state["chat_history"]가 존재합니다.
    #
    # 하지만 LangGraph node는 테스트나 다른 Python 코드에서
    # 부분적인 state를 직접 전달받을 수도 있습니다.
    #
    # 이런 경우에도 KeyError가 발생하지 않도록
    # get("chat_history", [])를 사용해 방어적으로 읽습니다.
    chat_history = state.get(
        "chat_history",
        [],
    )

    # 현재 질문과 이전 대화 기록을
    # intent classifier에 함께 전달합니다.
    #
    # question:
    #   지금 intent를 분류해야 하는 현재 사용자 질문
    #
    # chat_history:
    #   "그건", "그중", "방금 결과" 같은
    #   후속 표현의 문맥을 이해하기 위한 이전 대화
    #
    # chat_history는 PyTorch 모델 입력이 아닙니다.
    #
    # 실제 prediction은 이후
    # call_failure_prediction_node()에서
    # state["raw_sample"]을 사용해 수행합니다.
    result = classify_intent(
        question,
        chat_history=chat_history,
    )

    # intent classifier가 반환한 결과를
    # 이후 LangGraph node와 router가 사용할 수 있도록
    # AgentState에 저장합니다.
    state["intent"] = result.intent
    state["confidence"] = result.confidence
    state["intent_reason"] = result.reason
    state["intent_source"] = result.source
    state["intent_raw_response"] = result.raw_response

    # OpenAI 호출 실패, JSON 검증 실패 등으로
    # rule-based fallback이 사용된 경우
    # Agent 전체를 중단하지 않고 warning으로 기록합니다.
    #
    # intent 자체는 fallback으로 얻었으므로
    # workflow를 계속 실행할 수 있습니다.
    if result.error is not None:
        append_warning(
            state,
            f"intent 분류 과정에서 fallback 또는 오류가 발생했습니다: {result.error}",
        )

    return state


def call_failure_prediction_node(state: AgentState) -> AgentState:
    """
    failure_prediction intent일 때 Day 12 prediction service를 호출하는 node입니다.

    raw_sample이 없으면 prediction을 수행하지 않습니다.
    이유:
    - 고장 예측 모델은 설비 입력값이 있어야 동작할 수 있습니다.
    - 자연어 질문만으로 probability를 만들어내면 안 됩니다.
    """

    raw_sample = state.get("raw_sample")

    # raw_sample이 None이거나 빈 dict이면
    # 모델 예측에 필요한 설비 입력값이 없는 상태입니다.
    #
    # 자연어 질문만 보고 모델 probability를 임의로 만들면 안 되므로
    # prediction service를 호출하지 않습니다.
    #
    # 이 상황은 failure_prediction workflow를 완료할 수 없는 상태이므로
    # warnings가 아니라 errors에 기록합니다.
    #
    # 이후 route_after_prediction()은 errors가 있는 것을 확인하고
    # build_fallback_answer_node로 이동합니다.
    if not raw_sample:
        state["prediction"] = None
        state["probability"] = None
        state["threshold"] = None
        state["risk_level"] = "UNKNOWN"

        state["recommended_action"] = (
            "현재 요청에는 raw_sample이 없어 고장 예측을 수행할 수 없습니다. "
            "이전 대화의 설비 조건이나 raw_sample은 자동으로 재사용하지 않으므로, "
            "현재 예측에 사용할 설비 입력값을 요청에 함께 보내주세요."
        )

        state["answer"] = (
            "현재 요청에는 raw_sample이 없어 고장 위험 예측을 수행할 수 없습니다.\n\n"
            "chat_history는 현재 질문의 문맥을 이해하는 용도로만 사용하며, "
            "이전 대화의 설비 조건이나 raw_sample은 자동으로 재사용하지 않습니다.\n\n"
            "고장 위험을 다시 예측하려면 "
            "air_temperature, process_temperature, rotational_speed, "
            "torque, tool_wear, type 값을 포함한 "
            "새 raw_sample을 요청에 함께 제공해주세요."
        )

        state.setdefault("errors", []).append(
            "failure_prediction intent이지만 현재 요청에 raw_sample이 없어 "
            "prediction을 수행할 수 없습니다. "
            "chat_history는 질문 문맥 이해용이며, "
            "이전 대화의 설비 조건이나 raw_sample은 자동으로 재사용하지 않습니다."
        )

        return state

    try:
        # Day 12 prediction service를 실행합니다.
        #
        # include_shap과 include_global_importance는
        # Day 14 FastAPI request에서 전달된 옵션입니다.
        prediction_result = _run_failure_prediction_service(
            raw_sample=raw_sample,
            include_shap=state.get("include_shap", True),
            include_global_importance=state.get(
                "include_global_importance",
                True,
            ),
        )

    except Exception as exc:
        # prediction service에서 예외가 발생해도
        # LangGraph workflow 자체를 즉시 종료하지 않습니다.
        #
        # 오류를 AgentState에 기록하면
        # route_after_prediction()이 errors를 확인한 뒤
        # fallback answer node로 이동할 수 있습니다.
        state.setdefault("errors", []).append(
            f"failure prediction service 실행 중 오류가 발생했습니다: {exc}"
        )

        return state

    state["prediction"] = prediction_result.get("prediction")
    state["probability"] = prediction_result.get("probability")
    state["threshold"] = prediction_result.get("threshold")
    state["risk_level"] = prediction_result.get("risk_level")
    state["recommended_action"] = prediction_result.get("recommended_action")
    state["answer"] = prediction_result.get("answer", "")
    state["evidence"] = prediction_result.get("evidence", [])
    state["warnings"] = prediction_result.get("warnings", [])
    state["errors"] = prediction_result.get("errors", [])
    state["limitations"] = prediction_result.get("limitations", [])

    return state


def build_dataset_schema_answer_node(state: AgentState) -> AgentState:
    """
    AI4I 데이터셋 schema 질문에 답하는 node입니다.

    현재 Day 13에서는 dataset_schema_query intent를
    실제 DB나 문서 검색으로 처리하지 않고,
    프로젝트에서 사용 중인 AI4I feature 정보를 정적으로 답변합니다.

    이후 확장 방향
    --------------
    - docs/ 문서 검색
    - RAG 연결
    - dataset metadata artifact 로딩
    """

    state["answer"] = (
        "현재 프로젝트는 AI4I 2020 Predictive Maintenance Dataset을 사용합니다.\n\n"
        "모델 입력 feature는 다음 6개입니다.\n"
        "- Air temperature [K]\n"
        "- Process temperature [K]\n"
        "- Rotational speed [rpm]\n"
        "- Torque [Nm]\n"
        "- Tool wear [min]\n"
        "- Type\n\n"
        "target은 Machine failure입니다.\n"
        "UDI와 Product ID는 식별자이므로 학습 feature에서 제외합니다.\n"
        "Type은 현재 L/M/H를 숫자로 mapping해 사용하며, 이후 one-hot encoding으로 개선할 수 있습니다."
    )

    state["evidence"] = [
        {
            "evidence_id": "dataset_schema_001",
            "evidence_type": "dataset_schema",
            "source": "project_schema",
            "title": "AI4I 데이터셋 schema",
            "summary": "현재 모델은 AI4I feature 6개를 사용하고, Machine failure를 target으로 사용합니다.",
            "feature": None,
            "value": None,
            "direction": None,
            "contribution": None,
            "importance": None,
            "severity": "LOW",
            "metadata": {
                "features": [
                    "Air temperature [K]",
                    "Process temperature [K]",
                    "Rotational speed [rpm]",
                    "Torque [Nm]",
                    "Tool wear [min]",
                    "Type",
                ],
                "target": "Machine failure",
                "excluded_columns": ["UDI", "Product ID"],
            },
        }
    ]

    return state


def build_fallback_answer_node(state: AgentState) -> AgentState:
    """
    unknown intent 또는 오류 상황에서 fallback answer를 만드는 node입니다.

    fallback이 필요한 경우
    ---------------------
    1. question이 비어 있음
    2. intent가 unknown임
    3. failure_prediction intent인데 raw_sample이 없음
    4. prediction service 호출 실패
    """

    errors = state.get("errors", [])
    intent = state.get("intent", "unknown")

    if errors:
        state["answer"] = (
            "요청을 처리하는 중 문제가 발생했습니다.\n\n"
            "확인된 문제:\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\n\n"
            "고장 위험을 다시 예측하려면 "
            "현재 예측에 사용할 새 raw_sample을 요청에 함께 제공해주세요."
        )
        return state

    if intent == "unknown":
        state["answer"] = (
            "현재 질문은 이 Agent가 지원하는 작업으로 분류되지 않았습니다.\n\n"
            "현재 지원하는 질문 유형은 다음과 같습니다.\n"
            "- 설비 입력값 기반 고장 위험 예측\n"
            "- AI4I 데이터셋 feature / target / schema 설명\n\n"
            "예시 질문:\n"
            "'Torque 62, Tool wear 220이면 고장 위험이 높아?'"
        )
        return state

    state["answer"] = (
        "현재 요청을 처리할 수 없습니다. "
        "질문 유형과 입력값을 다시 확인해주세요."
    )

    return state


def build_final_answer_node(state: AgentState) -> AgentState:
    """
    최종 answer를 확인하는 node입니다.

    Day 12 failure_agent_service가 이미 answer를 만들어주는 경우에는
    그 answer를 그대로 사용합니다.

    만약 answer가 비어 있다면 최소 fallback answer를 생성합니다.
    """

    answer = state.get("answer")

    if isinstance(answer, str) and answer.strip():
        return state

    risk_level = state.get("risk_level", "UNKNOWN")
    probability = state.get("probability")
    recommended_action = state.get("recommended_action")

    state["answer"] = (
        f"모델 예측 결과 risk_level={risk_level}입니다. "
        f"probability={probability}. "
        f"{recommended_action or '추가 확인이 필요합니다.'}"
    )

    return state


def route_after_validation(state: AgentState) -> str:
    """
    validate_question_node 이후 어느 node로 갈지 결정합니다.

    errors가 있으면 더 진행하지 않고 fallback answer로 이동합니다.
    errors가 없으면 intent 분류 단계로 이동합니다.
    """

    if has_errors(state):
        return "fallback"

    return "classify"


def route_after_classification(state: AgentState) -> str:
    """
    classify_intent_node 이후 intent에 따라 workflow 경로를 결정합니다.

    LangGraph conditional edge에서 사용하는 router 함수입니다.
    """

    if has_errors(state):
        return "fallback"

    intent = state.get("intent", "unknown")

    if intent == "failure_prediction":
        return "failure_prediction"

    if intent == "dataset_schema_query":
        return "dataset_schema"

    return "fallback"


def route_after_prediction(state: AgentState) -> str:
    """
    prediction service 호출 이후 다음 경로를 결정합니다.

    prediction service에서 hard failure가 발생하면 fallback answer로 이동합니다.
    성공하면 final answer node로 이동합니다.
    """

    if has_errors(state):
        return "fallback"

    return "final"


def build_failure_agent_graph():
    """
    LangGraph workflow를 생성하고 compile합니다.

    LangGraph 기본 구성
    -------------------
    1. StateGraph(AgentState)로 graph builder 생성
    2. add_node()로 node 등록
    3. add_edge() 또는 add_conditional_edges()로 흐름 연결
    4. compile()로 실행 가능한 graph 생성

    Returns
    -------
    CompiledStateGraph
        invoke()로 실행할 수 있는 LangGraph workflow입니다.
    """

    graph_builder = StateGraph(AgentState)

    # node 등록
    graph_builder.add_node("validate_question", validate_question_node)
    graph_builder.add_node("classify_intent", classify_intent_node)
    graph_builder.add_node("call_failure_prediction", call_failure_prediction_node)
    graph_builder.add_node("build_dataset_schema_answer", build_dataset_schema_answer_node)
    graph_builder.add_node("build_fallback_answer", build_fallback_answer_node)
    graph_builder.add_node("build_final_answer", build_final_answer_node)

    # 시작점 연결
    graph_builder.add_edge(START, "validate_question")

    # question 검증 결과에 따라 분기
    graph_builder.add_conditional_edges(
        "validate_question",
        route_after_validation,
        {
            "classify": "classify_intent",
            "fallback": "build_fallback_answer",
        },
    )

    # intent 분류 결과에 따라 분기
    graph_builder.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "failure_prediction": "call_failure_prediction",
            "dataset_schema": "build_dataset_schema_answer",
            "fallback": "build_fallback_answer",
        },
    )

    # prediction 결과에 따라 분기
    graph_builder.add_conditional_edges(
        "call_failure_prediction",
        route_after_prediction,
        {
            "final": "build_final_answer",
            "fallback": "build_fallback_answer",
        },
    )

    # 종료 edge
    graph_builder.add_edge("build_dataset_schema_answer", END)
    graph_builder.add_edge("build_fallback_answer", END)
    graph_builder.add_edge("build_final_answer", END)

    return graph_builder.compile()


def run_failure_agent_graph(
    question: str,
    raw_sample: dict[str, Any] | None = None,
    include_shap: bool = True,
    include_global_importance: bool = True,
    *,
    # 이전 사용자 질문과 Agent 답변입니다.
    #
    # Day 15에서는 현재 question의 문맥을 이해할 수 있도록
    # chat_history를 LangGraph 초기 state에 저장합니다.
    #
    # 기본값을 []로 직접 작성하지 않는 이유
    # --------------------------------------
    # list는 append(), extend() 등으로
    # 내부 값을 변경할 수 있는 mutable 객체입니다.
    #
    # 다음처럼 작성하면:
    #
    #     chat_history: list[ChatMessage] = []
    #
    # 함수가 호출될 때마다 새로운 빈 list를 만드는 것이 아니라,
    # 함수가 정의될 때 만들어진 같은 기본 list 객체가
    # 여러 호출에서 재사용될 수 있습니다.
    #
    # 그러면 이전 Agent 요청에서 추가한 대화가
    # 다음 Agent 요청에 잘못 남을 가능성이 있습니다.
    #
    # 따라서 기본값은 None으로 두고,
    # create_initial_agent_state()에서
    # 요청마다 새로운 list를 생성합니다.
    chat_history: list[ChatMessage] | None = None,
) -> AgentState:
    """
    LangGraph workflow를 실행하는 공개 runner 함수입니다.

    FastAPI endpoint는 AgentState 내부 구조를 직접 만들지 않고
    question, 선택적 raw_sample, 선택적 chat_history를 전달합니다.

    runner가 API 입력을 AgentState로 변환하고
    compiled LangGraph workflow를 실행합니다.

    Day 15 확장
    -----------
    기존 Day 14 흐름:

        question
        +
        raw_sample
        -> AgentState
        -> LangGraph workflow

    Day 15 흐름:

        question
        +
        chat_history
        +
        raw_sample
        -> AgentState
        -> LangGraph workflow

    각 입력의 책임
    --------------
    question:
        현재 intent를 분류할 사용자 질문

    chat_history:
        현재 질문의 문맥을 이해하기 위한
        이전 user/assistant 대화 기록

    raw_sample:
        PyTorch MLP가 실제 고장 probability를 계산할 때 사용하는
        설비 입력값

    chat_history에 이전 probability나 설비 값이 적혀 있어도
    해당 텍스트를 새로운 모델 입력으로 자동 사용하지 않습니다.

    실제 prediction은 계속 raw_sample을 사용합니다.
    """

    # FastAPI request 또는 다른 Python 코드에서 전달한 값을
    # LangGraph node들이 공유할 초기 AgentState로 변환합니다.
    initial_state = create_initial_agent_state(
        question=question,
        raw_sample=raw_sample,

        # 이전 대화 기록을 초기 state에 저장합니다.
        #
        # chat_history가 None이면
        # create_initial_agent_state() 내부에서
        # 새로운 빈 list를 생성합니다.
        chat_history=chat_history,
    )

    # SHAP local evidence를 포함할지 저장합니다.
    #
    # 이후 call_failure_prediction_node()에서
    # Day 12 prediction service에 전달합니다.
    initial_state["include_shap"] = include_shap

    # global permutation importance evidence를
    # 포함할지 저장합니다.
    initial_state["include_global_importance"] = (
        include_global_importance
    )

    # LangGraph workflow를 생성하고 compile합니다.
    graph = build_failure_agent_graph()

    # 초기 AgentState를 graph에 전달하여
    # validate -> classify -> routing -> answer 흐름을 실행합니다.
    final_state = graph.invoke(initial_state)

    # 모든 node 실행이 끝난 최종 AgentState를 반환합니다.
    return final_state
    

def _run_failure_prediction_service(
    raw_sample: dict[str, Any],
    include_shap: bool = True,
    include_global_importance: bool = True,
) -> dict[str, Any]:
    """
    Day 12 failure_agent_service를 호출하는 내부 helper입니다.

    왜 helper로 분리하는가?
    ---------------------
    call_failure_prediction_node 안에 API request 변환과 service 호출 코드가 길게 들어가면
    LangGraph node의 책임이 흐려집니다.

    node는 workflow orchestration을 담당하고,
    이 helper는 Day 12 service 호출 방식을 담당합니다.

    raw_sample key 지원
    -------------------
    아래 두 형태를 모두 허용합니다.

    1. API 스타일 key
        {
            "air_temperature": 303.0,
            "process_temperature": 312.5,
            "rotational_speed": 1380.0,
            "torque": 62.0,
            "tool_wear": 220.0,
            "type": "L"
        }

    2. AI4I 원본 feature 스타일 key
        {
            "Air temperature [K]": 303.0,
            "Process temperature [K]": 312.5,
            "Rotational speed [rpm]": 1380.0,
            "Torque [Nm]": 62.0,
            "Tool wear [min]": 220.0,
            "Type": "L"
        }
    """

    from src.api.failure_agent_service import run_failure_prediction_agent
    from src.api.schemas import FailurePredictionRequest

    request = FailurePredictionRequest(
        air_temperature=_get_sample_value(
            raw_sample,
            "air_temperature",
            "Air temperature [K]",
        ),
        process_temperature=_get_sample_value(
            raw_sample,
            "process_temperature",
            "Process temperature [K]",
        ),
        rotational_speed=_get_sample_value(
            raw_sample,
            "rotational_speed",
            "Rotational speed [rpm]",
        ),
        torque=_get_sample_value(
            raw_sample,
            "torque",
            "Torque [Nm]",
        ),
        tool_wear=_get_sample_value(
            raw_sample,
            "tool_wear",
            "Tool wear [min]",
        ),
        type=_get_sample_value(
            raw_sample,
            "type",
            "Type",
        ),
        
        include_shap=include_shap,
        include_global_importance=include_global_importance,
    )

    response = run_failure_prediction_agent(request=request)

    return _response_to_dict(response)


def _get_sample_value(
    raw_sample: Mapping[str, Any],
    api_key: str,
    feature_key: str,
) -> Any:
    """
    raw_sample에서 값을 꺼내는 helper입니다.

    api_key가 있으면 api_key 값을 우선 사용하고,
    없으면 AI4I feature_key 값을 사용합니다.

    둘 다 없으면 KeyError를 발생시켜
    call_failure_prediction_node에서 error로 처리되게 합니다.
    """

    if api_key in raw_sample:
        return raw_sample[api_key]

    if feature_key in raw_sample:
        return raw_sample[feature_key]

    raise KeyError(f"raw_sample에 필요한 값이 없습니다: {api_key} 또는 {feature_key}")


def _response_to_dict(response: Any) -> dict[str, Any]:
    """
    service 응답을 dict로 변환합니다.

    FastAPI/Pydantic 응답 객체일 수도 있고,
    이미 dict일 수도 있으므로 방어적으로 처리합니다.
    """

    if isinstance(response, dict):
        return response

    # Pydantic v2
    if hasattr(response, "model_dump"):
        return response.model_dump()

    # Pydantic v1
    if hasattr(response, "dict"):
        return response.dict()

    raise TypeError(f"지원하지 않는 response 타입입니다: {type(response).__name__}")