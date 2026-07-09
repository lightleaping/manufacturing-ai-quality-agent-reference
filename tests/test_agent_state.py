"""
Day 13 - AgentState 테스트

이 테스트 파일의 역할
---------------------
src/agent/state.py에서 정의한 AgentState helper 함수들이
의도대로 동작하는지 검증합니다.

왜 AgentState를 테스트하는가?
----------------------------
LangGraph workflow에서는 여러 node가 하나의 state를 주고받습니다.

예:
    validate_question
    -> classify_intent
    -> call_prediction_service
    -> build_answer

이때 state 구조가 불안정하면 이후 node에서 KeyError가 발생하거나,
warnings/errors가 제대로 누적되지 않을 수 있습니다.

따라서 LangGraph workflow를 만들기 전에
state 생성, warning 추가, error 추가, raw_sample 확인 같은
기본 동작을 먼저 테스트합니다.
"""

from src.agent.state import (
    append_error,
    append_warning,
    create_initial_agent_state,
    has_errors,
    has_raw_sample,
)


def test_create_initial_agent_state_has_required_defaults():
    """
    초기 AgentState에는 question, warnings, errors, limitations가 있어야 합니다.

    question:
        사용자의 원본 질문이므로 필수입니다.

    warnings:
        prediction은 성공했지만 SHAP 등 부가 기능이 실패했을 때 누적합니다.

    errors:
        workflow 진행에 문제가 되는 오류를 누적합니다.

    limitations:
        현재 Agent의 한계나 주의사항을 사용자에게 안내할 때 사용합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    assert state["question"] == "이 설비 조건이면 고장 위험이 높아?"
    assert state["warnings"] == []
    assert state["errors"] == []
    assert state["limitations"] == []


def test_create_initial_agent_state_includes_raw_sample_when_provided():
    """
    raw_sample을 전달하면 초기 state에 raw_sample이 포함되어야 합니다.

    raw_sample은 failure_prediction intent일 때
    실제 모델 예측에 사용할 설비 입력값입니다.
    """

    raw_sample = {
        "Air temperature [K]": 303.0,
        "Process temperature [K]": 312.5,
        "Rotational speed [rpm]": 1380.0,
        "Torque [Nm]": 62.0,
        "Tool wear [min]": 220.0,
        "Type": "L",
    }

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?",
        raw_sample=raw_sample,
    )

    assert state["question"] == "이 설비 조건이면 고장 위험이 높아?"
    assert state["raw_sample"] == raw_sample


def test_create_initial_agent_state_omits_raw_sample_when_not_provided():
    """
    raw_sample을 전달하지 않으면 state에 raw_sample key를 만들지 않습니다.

    왜 None으로 넣지 않는가?
    -----------------------
    total=False TypedDict에서는
    key가 없는 상태와 key가 있지만 값이 None인 상태를 구분할 수 있습니다.

    여기서는 raw_sample이 제공되지 않았다는 의미를 명확히 하기 위해
    아예 key를 생략합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    assert "raw_sample" not in state


def test_create_initial_agent_state_includes_chat_history_when_provided():
    """
    chat_history를 전달하면 초기 state에 chat_history가 포함되어야 합니다.

    Day 13에서는 필수는 아니지만,
    이후 multi-turn Agent로 확장할 때 사용할 수 있습니다.
    """

    chat_history = [
        {"role": "user", "content": "이 설비 위험해?"},
        {"role": "assistant", "content": "입력값을 알려주세요."},
    ]

    state = create_initial_agent_state(
        question="Torque 62면 어때?",
        chat_history=chat_history,
    )

    assert state["chat_history"] == chat_history


def test_append_warning_adds_warning_to_existing_list():
    """
    append_warning()은 기존 warnings list에 메시지를 추가해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    append_warning(
        state,
        "SHAP artifact가 없어 shap_local evidence를 생략했습니다.",
    )

    assert state["warnings"] == [
        "SHAP artifact가 없어 shap_local evidence를 생략했습니다."
    ]


def test_append_warning_creates_warning_list_if_missing():
    """
    state에 warnings key가 없어도 append_warning()은 안전하게 동작해야 합니다.

    LangGraph node 중간에서 state가 부분적으로 만들어졌을 수도 있으므로
    setdefault()로 방어하는 구조를 테스트합니다.
    """

    state = {
        "question": "이 설비 조건이면 고장 위험이 높아?"
    }

    append_warning(
        state,
        "global importance artifact가 없어 생략했습니다.",
    )

    assert state["warnings"] == [
        "global importance artifact가 없어 생략했습니다."
    ]


def test_append_error_adds_error_to_existing_list():
    """
    append_error()는 기존 errors list에 오류 메시지를 추가해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    append_error(
        state,
        "raw_sample이 없어 failure_prediction을 수행할 수 없습니다.",
    )

    assert state["errors"] == [
        "raw_sample이 없어 failure_prediction을 수행할 수 없습니다."
    ]


def test_append_error_creates_error_list_if_missing():
    """
    state에 errors key가 없어도 append_error()는 안전하게 동작해야 합니다.
    """

    state = {
        "question": "이 설비 조건이면 고장 위험이 높아?"
    }

    append_error(
        state,
        "question이 비어 있습니다.",
    )

    assert state["errors"] == [
        "question이 비어 있습니다."
    ]


def test_has_errors_returns_false_when_no_errors():
    """
    errors가 비어 있으면 has_errors()는 False를 반환해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    assert has_errors(state) is False


def test_has_errors_returns_true_when_errors_exist():
    """
    errors에 하나라도 메시지가 있으면 has_errors()는 True를 반환해야 합니다.

    이 함수는 나중에 LangGraph conditional edge에서 사용할 수 있습니다.

    예:
        errors가 있으면 fallback answer node로 이동
        errors가 없으면 prediction node로 이동
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    append_error(
        state,
        "raw_sample이 없어 failure_prediction을 수행할 수 없습니다.",
    )

    assert has_errors(state) is True


def test_has_raw_sample_returns_true_when_valid_raw_sample_exists():
    """
    raw_sample이 dict이고 비어 있지 않으면 True를 반환해야 합니다.
    """

    raw_sample = {
        "Air temperature [K]": 303.0,
        "Process temperature [K]": 312.5,
        "Rotational speed [rpm]": 1380.0,
        "Torque [Nm]": 62.0,
        "Tool wear [min]": 220.0,
        "Type": "L",
    }

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?",
        raw_sample=raw_sample,
    )

    assert has_raw_sample(state) is True


def test_has_raw_sample_returns_false_when_raw_sample_missing():
    """
    raw_sample key가 없으면 False를 반환해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    assert has_raw_sample(state) is False


def test_has_raw_sample_returns_false_when_raw_sample_is_empty_dict():
    """
    raw_sample이 빈 dict이면 실제 예측에 사용할 입력값이 없으므로 False를 반환해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?",
        raw_sample={},
    )

    assert has_raw_sample(state) is False