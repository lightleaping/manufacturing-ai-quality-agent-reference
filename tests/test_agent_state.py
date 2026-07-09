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

Day 15 확장
-----------
Day 15에서는 chat_history를 실제 multi-turn Agent에 연결합니다.

따라서 아래 동작도 추가로 검증합니다.

1. chat_history를 전달하지 않아도
   초기 state에 새로운 빈 list가 생성되는가?

2. Agent 실행마다 서로 독립된 chat_history list를 사용하는가?

3. 외부에서 전달한 chat_history list와
   AgentState 내부 list가 같은 최상위 객체를 공유하지 않는가?

Agent 요청마다 대화 이력은 서로 독립적이어야 합니다.

이전 요청의 대화가 다음 요청에 잘못 섞이면
현재 질문의 문맥이나 intent를 잘못 판단할 수 있으므로,
chat_history의 초기화와 복사 동작을 테스트합니다.
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
    초기 AgentState에는
    question, chat_history, warnings, errors, limitations가 있어야 합니다.

    question:
        사용자의 원본 질문이므로 필수입니다.

    chat_history:
        이전 대화가 없는 single-turn 요청에서도
        이후 node가 안전하게 사용할 수 있도록
        새로운 빈 list로 초기화합니다.

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

    # Day 15부터 chat_history를 전달하지 않아도
    # 초기 AgentState에 새로운 빈 list를 넣습니다.
    #
    # 따라서 이후 node는 아래와 같이
    # chat_history key의 존재 여부를 매번 검사하지 않고
    # 바로 대화 기록을 읽을 수 있습니다.
    #
    #     state["chat_history"]
    assert state["chat_history"] == []

    assert state["warnings"] == []
    assert state["errors"] == []
    assert state["limitations"] == []


def test_create_initial_agent_state_creates_independent_chat_history_for_each_call():
    """
    create_initial_agent_state()를 여러 번 호출하면
    각 AgentState는 서로 독립된 chat_history list를 가져야 합니다.

    왜 이 테스트가 필요한가?
    -------------------------
    다음처럼 수정 가능한 빈 list를
    함수 매개변수의 기본값으로 직접 사용하면:

        def create_initial_agent_state(
            chat_history=[],
        ):
            ...

    함수가 호출될 때마다 새로운 빈 list가 만들어지는 것이 아니라,
    함수가 정의될 때 생성된 같은 기본 list 객체가
    여러 함수 호출에서 재사용될 수 있습니다.

    그러면 첫 번째 Agent 요청에서 추가한 대화가
    두 번째 Agent 요청에도 남을 수 있습니다.

    예:
        첫 번째 요청:
            "이 설비 조건이면 고장 위험이 높아?"

        두 번째 요청:
            "AI4I 데이터셋 feature는 뭐야?"

    두 요청은 서로 독립적이어야 하므로,
    첫 번째 요청의 대화가
    두 번째 요청의 chat_history에 들어가면 안 됩니다.

    현재 create_initial_agent_state()는
    chat_history의 기본값을 None으로 두고,
    함수가 실행될 때마다 새로운 빈 list를 생성합니다.
    """

    # 첫 번째 Agent 요청의 초기 state를 만듭니다.
    first_state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    # 두 번째 Agent 요청의 초기 state를 만듭니다.
    second_state = create_initial_agent_state(
        question="AI4I 데이터셋 feature는 뭐야?"
    )

    # 첫 번째 state의 chat_history에만
    # 새로운 대화 메시지를 추가합니다.
    first_state["chat_history"].append(
        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?",
        }
    )

    # 첫 번째 AgentState에는 메시지가 추가되어야 합니다.
    assert first_state["chat_history"] == [
        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?",
        }
    ]

    # 두 번째 AgentState의 chat_history는
    # 첫 번째 state의 변경에 영향을 받지 않고
    # 계속 빈 list여야 합니다.
    assert second_state["chat_history"] == []

    # 값이 처음에는 둘 다 빈 list였더라도,
    # 실제 메모리에서는 서로 다른 list 객체여야 합니다.
    #
    # ==:
    #   두 객체 안의 값이 같은지 비교
    #
    # is:
    #   두 변수가 실제로 같은 객체를 가리키는지 비교
    #
    # 여기서는 서로 다른 Agent 실행이므로
    # 같은 list 객체를 공유하면 안 됩니다.
    assert (
        first_state["chat_history"]
        is not second_state["chat_history"]
    )


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

    Day 13에서는 이후 multi-turn Agent 확장을 위해
    선택 필드로 구조를 준비했습니다.

    Day 15에서는 실제 이전 대화 문맥을
    intent classifier까지 전달하기 위해 사용합니다.
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


def test_create_initial_agent_state_copies_provided_chat_history_list():
    """
    외부에서 전달한 chat_history와
    AgentState 내부 chat_history는
    같은 최상위 list 객체를 공유하지 않아야 합니다.

    현재 create_initial_agent_state()에서는:

        list(chat_history or [])

    를 사용합니다.

    chat_history가 전달되었다면:

        list(chat_history)

    가 실행되어 새로운 최상위 list 객체를 만듭니다.

    따라서 AgentState 내부 history에
    새로운 메시지를 append해도,
    외부에서 전달한 원본 list에는
    그 메시지가 자동으로 추가되지 않아야 합니다.

    주의:
    -----
    list(...)는 얕은 복사입니다.

    즉, 최상위 list는 새로 만들지만
    list 안에 들어 있는 개별 메시지 dict까지
    모두 새 객체로 깊게 복사하는 것은 아닙니다.

    현재 Day 15에서는 기존 메시지 dict를 직접 수정하기보다,
    새로운 메시지 dict를 list에 추가하는 방식으로 사용할 예정이므로
    최상위 list를 분리하는 것으로 충분합니다.
    """

    # FastAPI request 또는 외부 코드에서 전달했다고 가정하는
    # 원본 대화 기록입니다.
    original_chat_history = [
        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?",
        }
    ]

    state = create_initial_agent_state(
        question="그건 왜 그래?",
        chat_history=original_chat_history,
    )

    # 내부 값은 원본 대화 기록과 같아야 합니다.
    assert state["chat_history"] == original_chat_history

    # 값은 같지만,
    # 최상위 list 객체는 서로 달라야 합니다.
    assert state["chat_history"] is not original_chat_history

    # AgentState 내부 history에만
    # 새로운 assistant 메시지를 추가합니다.
    state["chat_history"].append(
        {
            "role": "assistant",
            "content": "고장 위험 예측 결과를 설명합니다.",
        }
    )

    # AgentState 내부에는 기존 user 메시지와
    # 새 assistant 메시지가 모두 있어야 합니다.
    assert len(state["chat_history"]) == 2

    # 외부에서 전달한 원본 list는
    # AgentState 내부 list의 append에 영향을 받지 않아야 합니다.
    assert original_chat_history == [
        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?",
        }
    ]

    assert len(original_chat_history) == 1


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