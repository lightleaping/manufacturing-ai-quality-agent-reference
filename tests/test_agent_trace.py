"""
Day 16 - LangGraph 내부 구조화 Trace helper 테스트

이 테스트 파일의 역할
---------------------
src/agent/trace.py에서 구현한
구조화 trace helper 함수들이
의도대로 동작하는지 검증합니다.

Day 15까지는 Agent의 최종 결과를 중심으로 테스트했습니다.

예:

    intent

    confidence

    prediction

    probability

    risk_level

    warnings

    errors

Day 16에서는 최종 결과뿐 아니라
Agent 내부 실행 과정도 추적합니다.

예:

    어떤 node가 실행됐는가?

    어떤 순서로 실행됐는가?

    node 실행에 몇 ms가 걸렸는가?

    어떤 route가 선택됐는가?

    warning 또는 error가 추가됐는가?

    fallback 경로가 사용됐는가?

    전체 workflow가
    success, fallback, error 중
    어떤 상태로 끝났는가?


테스트 범위
-----------
1. UTC ISO 8601 시각 생성

2. millisecond 실행 시간 계산

3. trace 기본값 보완

4. 기존 trace 값 유지

5. 새로운 AgentState로 trace 문맥 전달

6. trace event 추가

7. event sequence 자동 증가

8. metadata 복사

9. fallback 발생 기록

10. 정상 node 실행 기록

11. warning node 실행 기록

12. error node 실행 기록

13. fallback node 실행 기록

14. node 예외 기록 후 원래 예외 재발생

15. 새로운 dict를 반환하는 node의 trace 유지

16. 정상 route 선택 기록

17. fallback route 선택 기록

18. route 예외 기록 후 원래 예외 재발생

19. 전체 trace 최종 상태 결정

20. 전체 trace 종료 처리


중요
----
이 파일은 trace helper의 단위 테스트입니다.

아직 실제 LangGraph compiled workflow에
trace wrapper를 연결하지 않습니다.

실제 node 실행 순서와 route event가
최종 AgentState에 남는지는
다음 단계에서 failure_agent_graph.py를 수정한 뒤
통합 테스트로 별도 검증합니다.
"""

from datetime import datetime, timezone
from time import perf_counter

import pytest

from src.agent import trace as agent_trace
from src.agent.state import (
    AgentState,
    append_error,
    append_warning,
    create_initial_agent_state,
)
from src.agent.trace import (
    append_trace_event,
    calculate_duration_ms,
    carry_trace_context,
    determine_final_trace_status,
    ensure_trace_state,
    finalize_trace,
    mark_fallback_occurred,
    run_traced_node,
    run_traced_route,
    utc_now_iso,
)


def test_utc_now_iso_returns_utc_iso_8601_string():
    """
    utc_now_iso()는
    UTC timezone 정보가 포함된
    ISO 8601 문자열을 반환해야 합니다.

    반환 예:

        "2026-07-10T01:12:30.120000+00:00"

    검증 내용
    ---------
    1. 반환값이 str인가?

    2. datetime.fromisoformat()으로
       다시 datetime 객체로 변환할 수 있는가?

    3. timezone 정보가 존재하는가?

    4. UTC offset이 0인가?
    """

    result = utc_now_iso()

    # JSON과 로그에 저장하기 쉬운
    # 문자열 형식이어야 합니다.
    assert isinstance(result, str)

    # 올바른 ISO 8601 문자열이 아니라면
    # ValueError가 발생하여 테스트가 실패합니다.
    parsed_result = datetime.fromisoformat(result)

    # timezone 정보가 없는 datetime은
    # tzinfo가 None입니다.
    #
    # Day 16 trace는 UTC timezone 정보를
    # 반드시 포함해야 합니다.
    assert parsed_result.tzinfo is not None

    # UTC는 UTC와의 시간 차이가 0입니다.
    assert (
        parsed_result.utcoffset()
        ==
        timezone.utc.utcoffset(parsed_result)
    )


def test_calculate_duration_ms_converts_seconds_to_milliseconds(
    monkeypatch,
):
    """
    calculate_duration_ms()는
    perf_counter() 차이를 millisecond로 변환해야 합니다.

    테스트 예:

        시작:
            100.000000초

        종료:
            100.012345초

        경과 시간:
            0.012345초

        millisecond:
            12.345 ms


    왜 실제 시간을 기다리지 않는가?
    -----------------------------
    time.sleep()을 사용하면
    운영체제와 실행 환경에 따라
    실제 대기 시간이 조금씩 달라질 수 있습니다.

    단위 테스트에서는 perf_counter()를
    고정된 fake 값으로 교체하여
    계산 결과를 정확하게 검증합니다.
    """

    # calculate_duration_ms() 내부에서 호출할
    # perf_counter() 값을 고정합니다.
    monkeypatch.setattr(
        agent_trace,
        "perf_counter",
        lambda: 100.012345,
    )

    result = calculate_duration_ms(
        started_perf_counter=100.0,
    )

    # 0.012345초 × 1,000
    #
    # =
    #
    # 12.345 ms
    assert result == 12.345


def test_ensure_trace_state_adds_missing_trace_defaults():
    """
    trace field가 하나도 없는 부분 AgentState에도
    ensure_trace_state()가 기본값을 추가해야 합니다.

    기존 node 단위 테스트에서는
    다음처럼 state를 직접 만들 수도 있습니다.

        {
            "question": "고장 위험을 예측해줘."
        }

    이런 state에서도 trace helper가
    KeyError 없이 동작해야 합니다.
    """

    state: AgentState = {
        "question": "고장 위험을 예측해줘."
    }

    result = ensure_trace_state(state)

    assert isinstance(
        result["trace_id"],
        str,
    )

    assert len(
        result["trace_id"]
    ) == 32

    # 초기 workflow 상태입니다.
    assert (
        result["trace_status"]
        ==
        "running"
    )

    assert isinstance(
        result["trace_started_at"],
        str,
    )

    assert (
        result["trace_finished_at"]
        is None
    )

    assert (
        result["trace_duration_ms"]
        is None
    )

    assert (
        result["fallback_occurred"]
        is False
    )

    assert (
        result["trace_events"]
        ==
        []
    )


def test_ensure_trace_state_preserves_existing_trace_values():
    """
    trace 값이 이미 존재한다면
    ensure_trace_state()가 기존 값을
    덮어쓰면 안 됩니다.

    setdefault()는 key가 없을 때만
    기본값을 추가합니다.
    """

    existing_trace_events = [
        {
            "sequence": 1,
            "event_type": "node",
            "event_name": "validate_question",
            "status": "success",
            "started_at": (
                "2026-07-10T01:00:00+00:00"
            ),
            "finished_at": (
                "2026-07-10T01:00:00.001000+00:00"
            ),
            "duration_ms": 1.0,
            "metadata": {},
        }
    ]

    state: AgentState = {
        "question": "고장 위험을 예측해줘.",
        "trace_id": "existing-trace-id",
        "trace_status": "success",
        "trace_started_at": (
            "2026-07-10T01:00:00+00:00"
        ),
        "trace_finished_at": (
            "2026-07-10T01:00:01+00:00"
        ),
        "trace_duration_ms": 1000.0,
        "fallback_occurred": True,
        "trace_events": existing_trace_events,
    }

    result = ensure_trace_state(state)

    assert (
        result["trace_id"]
        ==
        "existing-trace-id"
    )

    assert (
        result["trace_status"]
        ==
        "success"
    )

    assert (
        result["trace_started_at"]
        ==
        "2026-07-10T01:00:00+00:00"
    )

    assert (
        result["trace_finished_at"]
        ==
        "2026-07-10T01:00:01+00:00"
    )

    assert (
        result["trace_duration_ms"]
        ==
        1000.0
    )

    assert (
        result["fallback_occurred"]
        is True
    )

    # 기존 list 객체도 그대로 유지해야 합니다.
    assert (
        result["trace_events"]
        is existing_trace_events
    )


def test_carry_trace_context_copies_missing_trace_fields():
    """
    node가 새로운 AgentState dict를 반환하더라도
    기존 trace 문맥이 유지되어야 합니다.

    node 실행 전:

        source_state

    node 실행 후:

        target_state

    target_state에 trace field가 없다면
    source_state의 trace 값을 이어받습니다.
    """

    source_state = create_initial_agent_state(
        question="고장 위험을 예측해줘."
    )

    target_state: AgentState = {
        "question": "고장 위험을 예측해줘.",
        "intent": "failure_prediction",
    }

    result = carry_trace_context(
        source_state=source_state,
        target_state=target_state,
    )

    assert (
        result["trace_id"]
        ==
        source_state["trace_id"]
    )

    assert (
        result["trace_status"]
        ==
        source_state["trace_status"]
    )

    assert (
        result["trace_started_at"]
        ==
        source_state["trace_started_at"]
    )

    # trace event list도 동일한 실행 문맥을
    # 이어가야 하므로 같은 list 객체를 사용합니다.
    assert (
        result["trace_events"]
        is source_state["trace_events"]
    )


def test_append_trace_event_adds_sequence_and_copies_metadata():
    """
    append_trace_event()는
    event를 순서대로 추가해야 합니다.

    첫 번째 event:

        sequence = 1

    두 번째 event:

        sequence = 2

    또한 외부 metadata dict와
    trace 내부 metadata dict는
    같은 객체를 공유하지 않아야 합니다.
    """

    state = create_initial_agent_state(
        question="고장 위험을 예측해줘."
    )

    original_metadata = {
        "intent": "failure_prediction",
    }

    append_trace_event(
        state,
        event_type="node",
        event_name="classify_intent",
        status="success",
        started_at=(
            "2026-07-10T01:00:00+00:00"
        ),
        finished_at=(
            "2026-07-10T01:00:00.010000+00:00"
        ),
        duration_ms=10.0,
        metadata=original_metadata,
    )

    append_trace_event(
        state,
        event_type="route",
        event_name="route_after_classification",
        status="success",
        started_at=(
            "2026-07-10T01:00:00.011000+00:00"
        ),
        finished_at=(
            "2026-07-10T01:00:00.012000+00:00"
        ),
        duration_ms=1.0,
        metadata={
            "selected_route": (
                "failure_prediction"
            ),
        },
    )

    assert (
        len(state["trace_events"])
        ==
        2
    )

    first_event = (
        state["trace_events"][0]
    )

    second_event = (
        state["trace_events"][1]
    )

    assert (
        first_event["sequence"]
        ==
        1
    )

    assert (
        second_event["sequence"]
        ==
        2
    )

    assert (
        first_event["event_name"]
        ==
        "classify_intent"
    )

    assert (
        second_event["event_type"]
        ==
        "route"
    )

    # append_trace_event()는
    # dict(metadata)를 사용하여
    # 새로운 최상위 metadata dict를 만듭니다.
    assert (
        first_event["metadata"]
        is not original_metadata
    )

    # 외부 원본 metadata를 수정합니다.
    original_metadata["intent"] = (
        "unknown"
    )

    # trace event 내부 metadata는
    # 기존 값을 유지해야 합니다.
    assert (
        first_event["metadata"]["intent"]
        ==
        "failure_prediction"
    )


def test_mark_fallback_occurred_sets_fallback_flag():
    """
    mark_fallback_occurred()는
    fallback_occurred를 True로 변경해야 합니다.
    """

    state = create_initial_agent_state(
        question="오늘 점심 메뉴 추천해줘."
    )

    assert (
        state["fallback_occurred"]
        is False
    )

    result = mark_fallback_occurred(
        state
    )

    assert (
        result["fallback_occurred"]
        is True
    )


def test_run_traced_node_records_success_and_metadata():
    """
    정상 node 실행은
    success trace event로 기록되어야 합니다.

    metadata_builder가 반환한 값도
    event metadata에 포함되어야 합니다.
    """

    def fake_node(
        state: AgentState,
    ) -> AgentState:
        state["intent"] = (
            "failure_prediction"
        )

        state["confidence"] = 0.95

        state["intent_source"] = (
            "openai"
        )

        return state

    def build_metadata(
        state: AgentState,
    ) -> dict[str, object]:
        return {
            "intent": state.get(
                "intent"
            ),
            "confidence": state.get(
                "confidence"
            ),
            "intent_source": state.get(
                "intent_source"
            ),
        }

    state = create_initial_agent_state(
        question="고장 위험을 예측해줘."
    )

    result = run_traced_node(
        state,
        node_name="classify_intent",
        node_function=fake_node,
        metadata_builder=build_metadata,
    )

    assert (
        result["intent"]
        ==
        "failure_prediction"
    )

    assert (
        len(result["trace_events"])
        ==
        1
    )

    event = result["trace_events"][0]

    assert (
        event["event_type"]
        ==
        "node"
    )

    assert (
        event["event_name"]
        ==
        "classify_intent"
    )

    assert (
        event["status"]
        ==
        "success"
    )

    assert (
        event["duration_ms"]
        >=
        0.0
    )

    assert (
        event["metadata"]["warnings_added"]
        ==
        0
    )

    assert (
        event["metadata"]["errors_added"]
        ==
        0
    )

    assert (
        event["metadata"]["intent"]
        ==
        "failure_prediction"
    )

    assert (
        event["metadata"]["confidence"]
        ==
        0.95
    )

    assert (
        event["metadata"]["intent_source"]
        ==
        "openai"
    )


def test_run_traced_node_records_warning_when_warning_is_added():
    """
    node 실행 중 warning이 새로 추가되면
    event status는 warning이어야 합니다.
    """

    def fake_warning_node(
        state: AgentState,
    ) -> AgentState:
        append_warning(
            state,
            (
                "OpenAI 분류 실패 후 "
                "rule-based fallback을 사용했습니다."
            ),
        )

        return state

    state = create_initial_agent_state(
        question="고장 위험을 예측해줘."
    )

    result = run_traced_node(
        state,
        node_name="classify_intent",
        node_function=fake_warning_node,
    )

    event = result["trace_events"][0]

    assert (
        event["status"]
        ==
        "warning"
    )

    assert (
        event["metadata"]["warnings_added"]
        ==
        1
    )

    assert (
        event["metadata"]["errors_added"]
        ==
        0
    )


def test_run_traced_node_records_error_when_error_is_added():
    """
    node가 예외를 발생시키지 않았더라도
    errors list에 새 오류를 추가했다면
    event status는 error여야 합니다.

    예:

        raw_sample 없음

        ↓

        errors에 안내 추가

        ↓

        fallback route로 이동
    """

    def fake_error_node(
        state: AgentState,
    ) -> AgentState:
        append_error(
            state,
            (
                "raw_sample이 없어 "
                "prediction을 수행할 수 없습니다."
            ),
        )

        return state

    state = create_initial_agent_state(
        question="고장 위험을 예측해줘."
    )

    result = run_traced_node(
        state,
        node_name="call_failure_prediction",
        node_function=fake_error_node,
    )

    event = result["trace_events"][0]

    assert (
        event["status"]
        ==
        "error"
    )

    assert (
        event["metadata"]["warnings_added"]
        ==
        0
    )

    assert (
        event["metadata"]["errors_added"]
        ==
        1
    )


def test_run_traced_node_records_fallback_node():
    """
    is_fallback_node=True이면:

    1. event status는 fallback

    2. fallback_occurred는 True

    여야 합니다.
    """

    def fake_fallback_node(
        state: AgentState,
    ) -> AgentState:
        state["answer"] = (
            "현재 요청을 처리할 수 없습니다."
        )

        return state

    state = create_initial_agent_state(
        question="오늘 점심 메뉴 추천해줘."
    )

    result = run_traced_node(
        state,
        node_name="build_fallback_answer",
        node_function=fake_fallback_node,
        is_fallback_node=True,
    )

    event = result["trace_events"][0]

    assert (
        event["status"]
        ==
        "fallback"
    )

    assert (
        result["fallback_occurred"]
        is True
    )


def test_run_traced_node_records_exception_and_reraises():
    """
    node 함수가 예상하지 못한 예외를 발생시키면:

    1. error trace event 기록

    2. exception_type 기록

    3. 원래 예외 다시 발생

    해야 합니다.


    왜 pytest.raises를 사용하는가?
    -----------------------------
    예외가 발생하는 것이
    이 테스트의 예상 동작이기 때문입니다.
    """

    def failing_node(
        state: AgentState,
    ) -> AgentState:
        raise ValueError(
            "테스트용 node 오류"
        )

    state = create_initial_agent_state(
        question="고장 위험을 예측해줘."
    )

    with pytest.raises(
        ValueError,
        match="테스트용 node 오류",
    ):
        run_traced_node(
            state,
            node_name="failing_node",
            node_function=failing_node,
        )

    # 예외는 다시 발생했지만,
    # 발생 전에 trace event는 기록되어야 합니다.
    assert (
        len(state["trace_events"])
        ==
        1
    )

    event = state["trace_events"][0]

    assert (
        event["status"]
        ==
        "error"
    )

    assert (
        event["metadata"]["exception_type"]
        ==
        "ValueError"
    )


def test_run_traced_node_preserves_trace_when_node_returns_new_state():
    """
    node가 기존 state를 직접 수정하지 않고
    새로운 dict를 반환해도
    trace 문맥이 유지되어야 합니다.

    기존 state:

        trace_id 존재

        trace_events 존재

    새로운 node 반환값:

        question

        intent

    run_traced_node():

        기존 trace 문맥 전달

        현재 node event 추가
    """

    def new_state_node(
        state: AgentState,
    ) -> AgentState:
        return {
            "question": state["question"],
            "intent": "dataset_schema_query",
        }

    state = create_initial_agent_state(
        question="AI4I feature를 알려줘."
    )

    original_trace_id = (
        state["trace_id"]
    )

    result = run_traced_node(
        state,
        node_name="classify_intent",
        node_function=new_state_node,
    )

    assert (
        result["trace_id"]
        ==
        original_trace_id
    )

    assert (
        result["intent"]
        ==
        "dataset_schema_query"
    )

    assert (
        len(result["trace_events"])
        ==
        1
    )

    assert (
        result["trace_events"][0]["event_name"]
        ==
        "classify_intent"
    )


def test_run_traced_route_records_selected_route_and_metadata():
    """
    정상 route 함수는:

    1. 선택한 route 문자열 반환

    2. route trace event 추가

    3. selected_route metadata 저장

    4. 추가 metadata 저장

    해야 합니다.


    중요
    ----
    이 테스트는 run_traced_route() helper 자체를
    직접 호출하여 검증합니다.

    실제 LangGraph 연결에서는
    이 helper를 conditional edge에
    바로 등록하지 않습니다.

    상태를 반환하는 route 판단 node 안에서 실행한 뒤,
    선택 결과를 별도 conditional router가 읽도록
    연결할 예정입니다.
    """

    def fake_route(
        state: AgentState,
    ) -> str:
        return "failure_prediction"

    def build_route_metadata(
        state: AgentState,
        selected_route: str,
    ) -> dict[str, object]:
        return {
            "intent": state.get(
                "intent"
            ),
            "route_confirmed": (
                selected_route
                ==
                "failure_prediction"
            ),
        }

    state = create_initial_agent_state(
        question="고장 위험을 예측해줘."
    )

    state["intent"] = (
        "failure_prediction"
    )

    selected_route = run_traced_route(
        state,
        route_name="route_after_classification",
        route_function=fake_route,
        metadata_builder=build_route_metadata,
    )

    assert (
        selected_route
        ==
        "failure_prediction"
    )

    assert (
        len(state["trace_events"])
        ==
        1
    )

    event = state["trace_events"][0]

    assert (
        event["event_type"]
        ==
        "route"
    )

    assert (
        event["event_name"]
        ==
        "route_after_classification"
    )

    assert (
        event["status"]
        ==
        "success"
    )

    assert (
        event["metadata"]["selected_route"]
        ==
        "failure_prediction"
    )

    assert (
        event["metadata"]["intent"]
        ==
        "failure_prediction"
    )

    assert (
        event["metadata"]["route_confirmed"]
        is True
    )


def test_run_traced_route_marks_fallback_route():
    """
    selected_route가 "fallback"이면:

    1. route event status는 fallback

    2. fallback_occurred는 True

    여야 합니다.
    """

    def fake_fallback_route(
        state: AgentState,
    ) -> str:
        return "fallback"

    state = create_initial_agent_state(
        question="오늘 점심 메뉴 추천해줘."
    )

    selected_route = run_traced_route(
        state,
        route_name="route_after_classification",
        route_function=fake_fallback_route,
    )

    assert (
        selected_route
        ==
        "fallback"
    )

    assert (
        state["fallback_occurred"]
        is True
    )

    event = state["trace_events"][0]

    assert (
        event["status"]
        ==
        "fallback"
    )

    assert (
        event["metadata"]["selected_route"]
        ==
        "fallback"
    )


def test_run_traced_route_records_exception_and_reraises():
    """
    route 함수가 예외를 발생시키면:

    1. error route event 기록

    2. exception_type 기록

    3. 원래 예외 재발생

    해야 합니다.
    """

    def failing_route(
        state: AgentState,
    ) -> str:
        raise RuntimeError(
            "테스트용 route 오류"
        )

    state = create_initial_agent_state(
        question="고장 위험을 예측해줘."
    )

    with pytest.raises(
        RuntimeError,
        match="테스트용 route 오류",
    ):
        run_traced_route(
            state,
            route_name="failing_route",
            route_function=failing_route,
        )

    assert (
        len(state["trace_events"])
        ==
        1
    )

    event = state["trace_events"][0]

    assert (
        event["event_type"]
        ==
        "route"
    )

    assert (
        event["status"]
        ==
        "error"
    )

    assert (
        event["metadata"]["exception_type"]
        ==
        "RuntimeError"
    )


def test_determine_final_trace_status_returns_expected_status():
    """
    전체 trace 상태 결정 우선순위를 검증합니다.

    정상:

        errors 없음

        fallback 없음

        → success


    오류:

        errors 존재

        fallback 없음

        → error


    fallback:

        fallback_occurred = True

        → fallback


    fallback과 errors가 모두 존재:

        fallback 우선

        → fallback
    """

    success_state = (
        create_initial_agent_state(
            question="AI4I feature를 알려줘."
        )
    )

    assert (
        determine_final_trace_status(
            success_state
        )
        ==
        "success"
    )

    error_state = (
        create_initial_agent_state(
            question="고장 위험을 예측해줘."
        )
    )

    append_error(
        error_state,
        "prediction service 오류",
    )

    assert (
        determine_final_trace_status(
            error_state
        )
        ==
        "error"
    )

    fallback_state = (
        create_initial_agent_state(
            question="오늘 점심 메뉴 추천해줘."
        )
    )

    mark_fallback_occurred(
        fallback_state
    )

    assert (
        determine_final_trace_status(
            fallback_state
        )
        ==
        "fallback"
    )

    fallback_with_error_state = (
        create_initial_agent_state(
            question="고장 위험을 다시 예측해줘."
        )
    )

    append_error(
        fallback_with_error_state,
        "raw_sample이 없습니다.",
    )

    mark_fallback_occurred(
        fallback_with_error_state
    )

    # fallback 답변을 정상 반환한 요청은
    # 시스템 전체 장애와 구분합니다.
    assert (
        determine_final_trace_status(
            fallback_with_error_state
        )
        ==
        "fallback"
    )


def test_finalize_trace_sets_finished_at_duration_and_status():
    """
    finalize_trace()는
    전체 trace 종료 정보를 저장해야 합니다.

    검증 내용
    ---------
    trace_finished_at:

        UTC ISO 8601 문자열

    trace_duration_ms:

        0 이상의 float

    trace_status:

        최종 state 기준 상태
    """

    state = create_initial_agent_state(
        question="AI4I feature를 알려줘."
    )

    # 실제 public runner에서는
    # graph.invoke() 전에 이 값을 저장할 예정입니다.
    started_perf_counter = (
        perf_counter()
    )

    result = finalize_trace(
        state,
        started_perf_counter=(
            started_perf_counter
        ),
    )

    assert isinstance(
        result["trace_finished_at"],
        str,
    )

    parsed_finished_at = (
        datetime.fromisoformat(
            result["trace_finished_at"]
        )
    )

    assert (
        parsed_finished_at.tzinfo
        is not None
    )

    assert (
        parsed_finished_at.utcoffset()
        ==
        timezone.utc.utcoffset(
            parsed_finished_at
        )
    )

    assert isinstance(
        result["trace_duration_ms"],
        float,
    )

    assert (
        result["trace_duration_ms"]
        >=
        0.0
    )

    assert (
        result["trace_status"]
        ==
        "success"
    )