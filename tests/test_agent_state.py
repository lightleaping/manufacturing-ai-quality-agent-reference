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

Day 16 확장
-----------
Day 16에서는 LangGraph Agent 실행 과정을 추적할 수 있도록
AgentState에 구조화 trace 초기값을 추가합니다.

따라서 아래 동작도 추가로 검증합니다.

1. 초기 state에 trace 관련 기본 필드가 생성되는가?

2. trace_id가 요청마다 새롭게 생성되는가?

3. trace_id가 UUID 기반 32자리 16진수 문자열인가?

4. trace_started_at이 UTC 기반 ISO 8601 문자열인가?

5. Agent 실행마다 서로 독립된 trace_events list를 사용하는가?

현재 단계에서는 trace 데이터의 구조와 초기값만 테스트합니다.

실제 node 실행 시간 측정,
node event 추가,
route event 추가,
trace 종료 상태 계산은
다음 단계의 src/agent/trace.py에서 구현하고 별도로 테스트합니다.
"""

from datetime import datetime, timezone

from src.agent.state import (
    append_error,
    append_warning,
    create_initial_agent_state,
    has_errors,
)


def test_create_initial_agent_state_has_required_defaults():
    """
    초기 AgentState에는
    question, chat_history, warnings, errors, limitations가 있어야 합니다.

    Day 16부터는 아래 trace 초기값도 있어야 합니다.

    trace_id:
        요청 하나를 구분하는 고유 ID

    trace_status:
        workflow가 아직 종료되지 않았으므로 "running"

    trace_started_at:
        trace가 시작된 UTC 시각

    trace_finished_at:
        아직 workflow가 종료되지 않았으므로 None

    trace_duration_ms:
        아직 전체 실행 시간이 계산되지 않았으므로 None

    fallback_occurred:
        아직 fallback 경로를 실행하지 않았으므로 False

    trace_events:
        아직 node나 route가 실행되지 않았으므로 빈 list

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

    # Day 16:
    # trace_id는 요청마다 생성되는 문자열이어야 합니다.
    #
    # 구체적인 길이와 UUID 형식은
    # 별도의 테스트에서 자세히 검증합니다.
    assert isinstance(state["trace_id"], str)

    # 초기 state를 만든 직후에는
    # LangGraph workflow가 아직 실행 중이므로
    # trace_status는 "running"이어야 합니다.
    assert state["trace_status"] == "running"

    # trace 시작 시각은 문자열로 생성되어야 합니다.
    #
    # ISO 8601 형식과 UTC 여부는
    # 별도의 테스트에서 자세히 검증합니다.
    assert isinstance(state["trace_started_at"], str)

    # 아직 workflow가 끝나지 않았으므로
    # 종료 시각은 None이어야 합니다.
    assert state["trace_finished_at"] is None

    # 아직 전체 workflow 실행 시간이
    # 계산되지 않았으므로 None이어야 합니다.
    assert state["trace_duration_ms"] is None

    # 초기 state에서는 아직 fallback 경로를
    # 실행하지 않았으므로 False여야 합니다.
    assert state["fallback_occurred"] is False

    # 아직 node와 route가 실행되지 않았으므로
    # trace event 목록은 빈 list여야 합니다.
    assert state["trace_events"] == []


def test_create_initial_agent_state_creates_unique_trace_id_for_each_call():
    """
    create_initial_agent_state()를 호출할 때마다
    서로 다른 trace_id가 생성되어야 합니다.

    왜 요청마다 새로운 trace_id가 필요한가?
    ---------------------------------------
    trace_id는 Agent 요청 하나의 전체 실행을 구분하는 ID입니다.

    예:

        첫 번째 요청:
            "이 설비의 고장 위험을 예측해줘."

        두 번째 요청:
            "AI4I 데이터셋 feature를 알려줘."

    두 요청은 서로 다른 LangGraph 실행입니다.

    따라서 같은 trace_id를 공유하면
    이후 로그, 실행 이력 DB, Dashboard에서
    서로 다른 요청의 node 실행 기록이
    하나의 요청처럼 잘못 섞일 수 있습니다.

    create_initial_agent_state()는 호출할 때마다:

        uuid4().hex

    를 실행하여 새로운 trace_id를 생성합니다.
    """

    first_state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    second_state = create_initial_agent_state(
        question="AI4I 데이터셋 feature는 뭐야?"
    )

    # 서로 다른 Agent 요청이므로
    # trace_id도 서로 달라야 합니다.
    assert first_state["trace_id"] != second_state["trace_id"]


def test_create_initial_agent_state_creates_32_character_hex_trace_id():
    """
    trace_id는 uuid4().hex로 만든
    32자리 16진수 문자열이어야 합니다.

    uuid4() 예:

        UUID(
            "908759dd-97bd-4a3e-b749-4b68f76f871c"
        )

    uuid4().hex 예:

        "908759dd97bd4a3eb7494b68f76f871c"

    .hex를 사용하므로:

    1. 하이픈이 없습니다.

    2. 문자열 길이는 32입니다.

    3. 0~9, a~f 범위의
       16진수 문자열로 변환할 수 있습니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    trace_id = state["trace_id"]

    # uuid4().hex 결과는 문자열입니다.
    assert isinstance(trace_id, str)

    # 하이픈을 제거한 UUID hex 문자열은
    # 길이가 32여야 합니다.
    assert len(trace_id) == 32

    # int(trace_id, 16)은 문자열을
    # 16진수 정수로 변환합니다.
    #
    # trace_id 안에 16진수로 사용할 수 없는 문자가 있다면
    # ValueError가 발생하여 테스트가 실패합니다.
    #
    # 반환값 자체가 중요한 것이 아니라,
    # 정상적인 16진수 문자열인지 검증하는 것이 목적입니다.
    int(trace_id, 16)


def test_create_initial_agent_state_creates_utc_iso_trace_started_at():
    """
    trace_started_at은
    UTC 기반 ISO 8601 문자열이어야 합니다.

    예:

        "2026-07-10T01:12:30.120000+00:00"

    왜 UTC를 사용하는가?
    --------------------
    서버가 서울, 미국, 유럽 등
    서로 다른 time zone에서 실행되더라도
    같은 시간 기준으로 trace를 비교하기 쉽기 때문입니다.

    ISO 8601 문자열을 사용하는 이유
    --------------------------------
    datetime 객체는 그대로 JSON 응답에 넣기 어려울 수 있습니다.

    .isoformat()을 사용하면
    로그, JSON, DB에 저장하기 쉬운
    표준 형태의 문자열을 만들 수 있습니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    trace_started_at = state["trace_started_at"]

    # datetime.fromisoformat()은
    # ISO 8601 형식 문자열을 datetime 객체로 변환합니다.
    #
    # 문자열 형식이 올바르지 않다면
    # ValueError가 발생하여 테스트가 실패합니다.
    parsed_started_at = datetime.fromisoformat(
        trace_started_at
    )

    # timezone 정보가 없는 datetime은
    # tzinfo가 None입니다.
    #
    # Day 16에서는 UTC timezone 정보를
    # 포함해야 하므로 None이면 안 됩니다.
    assert parsed_started_at.tzinfo is not None

    # utcoffset()은 현재 datetime이
    # UTC와 얼마나 차이 나는지 반환합니다.
    #
    # UTC라면 차이가 0이어야 합니다.
    assert (
        parsed_started_at.utcoffset()
        == timezone.utc.utcoffset(parsed_started_at)
    )


def test_create_initial_agent_state_creates_independent_trace_events_for_each_call():
    """
    create_initial_agent_state()를 여러 번 호출하면
    각 AgentState는 서로 독립된 trace_events list를 가져야 합니다.

    왜 이 테스트가 필요한가?
    -------------------------
    trace_events에는 앞으로 다음 실행 기록이 누적됩니다.

        validate_question

        route_after_validation

        classify_intent

        route_after_classification

        call_failure_prediction

        build_final_answer

    서로 다른 Agent 요청이
    같은 trace_events list를 공유하면,
    첫 번째 요청의 node 기록이
    두 번째 요청 trace에 섞일 수 있습니다.

    따라서 요청마다 새로운 빈 list를 생성해야 합니다.
    """

    first_state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    second_state = create_initial_agent_state(
        question="AI4I 데이터셋 feature는 뭐야?"
    )

    # 아직 node와 route를 실행하지 않았으므로
    # 두 trace event 목록의 값은 모두 빈 list입니다.
    assert first_state["trace_events"] == []
    assert second_state["trace_events"] == []

    # 첫 번째 요청의 trace_events에만
    # 테스트용 trace event를 추가합니다.
    first_state["trace_events"].append(
        {
            "sequence": 1,
            "event_type": "node",
            "event_name": "validate_question",
            "status": "success",
            "started_at": (
                "2026-07-10T01:12:30.120000+00:00"
            ),
            "finished_at": (
                "2026-07-10T01:12:30.121000+00:00"
            ),
            "duration_ms": 1.0,
            "metadata": {},
        }
    )

    # 첫 번째 AgentState에는
    # trace event가 한 개 추가되어야 합니다.
    assert len(first_state["trace_events"]) == 1

    # 두 번째 AgentState는
    # 첫 번째 state의 변경에 영향을 받지 않아야 합니다.
    assert second_state["trace_events"] == []

    # 두 list의 값이 처음에는 모두 빈 list였더라도,
    # 실제 메모리에서는 서로 다른 객체여야 합니다.
    assert (
        first_state["trace_events"]
        is not second_state["trace_events"]
    )


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
    state에 errors key가 없어도 append_error()은 안전하게 동작해야 합니다.
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
