"""
Day 16 - LangGraph 내부 구조화 Trace helper

이 파일의 역할
----------------
LangGraph Agent가 실행될 때 발생하는
node 실행과 route 선택 과정을
구조화된 trace event로 기록합니다.

Day 15까지는 최종 AgentState를 통해
다음 결과를 확인할 수 있었습니다.

    intent

    confidence

    intent_source

    prediction

    probability

    risk_level

    warnings

    errors

하지만 최종 결과만으로는
다음 실행 과정을 자세히 확인하기 어려웠습니다.

    어떤 node가 실행됐는가?

    node는 어떤 순서로 실행됐는가?

    node 실행을 언제 시작했는가?

    node 실행을 언제 종료했는가?

    node 실행에 몇 ms가 걸렸는가?

    conditional edge는 어떤 route를 선택했는가?

    fallback 경로가 실행됐는가?

    warning 또는 error가
    어느 node 실행 중 추가됐는가?

Day 16에서는 이러한 실행 정보를
AgentState의 trace_events에 구조화하여 저장합니다.


state.py와 trace.py의 역할 차이
------------------------------
src/agent/state.py:

    trace 데이터가 어떤 구조를 가지는지 정의

    예:
        TraceEvent

        TraceStatus

        trace_id

        trace_events


src/agent/trace.py:

    실제 실행 시간을 측정

    TraceEvent 생성

    trace_events에 event 추가

    fallback 발생 기록

    전체 trace 종료 처리


failure_agent_graph.py:

    어느 node를 trace할지 결정

    어느 route를 trace할지 결정

    node 실행 결과에서
    어떤 metadata를 저장할지 결정


왜 단순 print()를 사용하지 않는가?
---------------------------------
다음과 같은 print 로그는
개발 중 빠르게 확인하기에는 편리합니다.

    print("classify_intent started")

    print("classify_intent completed")

하지만 print 문자열만 사용하면
다음 작업이 어려워집니다.

    실행 순서 정렬

    duration 계산

    특정 trace_id 검색

    JSON 응답 반환

    DB 저장

    Streamlit Dashboard 표시

    LangSmith 또는 OpenTelemetry 연결

따라서 Day 16에서는 다음처럼
정해진 key를 가진 구조화 데이터를 사용합니다.

예:

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


중요한 설계 원칙
----------------
1. trace 기능과 business logic을 분리합니다.

2. 기존 node 내부에
   시간 측정 코드를 반복해서 작성하지 않습니다.

3. node 실행은 run_traced_node()가 감쌉니다.

4. route 실행은 run_traced_route()가 감쌉니다.

5. 예외를 숨기지 않습니다.

6. 예외가 발생하면 error trace event를 남긴 뒤
   원래 예외를 다시 발생시킵니다.

7. trace에는 필요한 요약 정보만 저장합니다.

8. 전체 chat_history, 전체 raw_sample,
   API key, 환경 변수 등은
   trace metadata에 자동 저장하지 않습니다.
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from src.agent.state import (
    AgentState,
    TraceEventStatus,
    TraceEventType,
    TraceStatus,
)


# NodeFunction은 LangGraph node 함수의 타입을 표현합니다.
#
# Callable[[AgentState], AgentState]의 의미
# ----------------------------------------
# Callable:
#   호출할 수 있는 함수 또는 객체
#
# [AgentState]:
#   함수가 AgentState 하나를 입력으로 받음
#
# AgentState:
#   함수 실행 후 AgentState를 반환함
#
# 예:
#
# def classify_intent_node(
#     state: AgentState,
# ) -> AgentState:
#     ...
#
# 위 함수는 NodeFunction 타입과 맞습니다.
NodeFunction = Callable[
    [AgentState],
    AgentState,
]


# NodeMetadataBuilder는
# node 실행이 끝난 뒤
# trace metadata를 만드는 함수의 타입입니다.
#
# 입력:
#   node 실행 후 AgentState
#
# 출력:
#   trace에 저장할 metadata dict
#
# 예:
#
# def build_intent_metadata(
#     state: AgentState,
# ) -> dict[str, Any]:
#     return {
#         "intent": state.get("intent"),
#         "intent_source": state.get("intent_source"),
#         "confidence": state.get("confidence"),
#     }
NodeMetadataBuilder = Callable[
    [AgentState],
    dict[str, Any],
]


# RouteFunction은
# LangGraph conditional edge에서 사용하는
# routing 함수의 타입을 표현합니다.
#
# 입력:
#   현재 AgentState
#
# 출력:
#   다음 경로 이름
#
# 예:
#
# def route_after_classification(
#     state: AgentState,
# ) -> str:
#     return "failure_prediction"
RouteFunction = Callable[
    [AgentState],
    str,
]


# RouteMetadataBuilder는
# route 선택이 끝난 뒤
# 추가 metadata를 만드는 함수의 타입입니다.
#
# 입력:
#   현재 AgentState
#
#   route 함수가 반환한 selected_route
#
# 출력:
#   trace metadata dict
RouteMetadataBuilder = Callable[
    [AgentState, str],
    dict[str, Any],
]


# trace 관련 field 이름을 하나의 tuple로 관리합니다.
#
# 이 값들은 node 함수가 새로운 AgentState dict를
# 만들어 반환했을 때도 유지해야 하는 trace 문맥입니다.
#
# 현재 프로젝트의 node는 대부분
# 전달받은 state를 수정한 뒤 다시 반환하지만,
# 미래에 새로운 dict를 반환하는 node가 추가될 수도 있습니다.
#
# 따라서 run_traced_node()는
# node 실행 전 state의 trace 문맥을
# node 실행 후 state에도 안전하게 이어줍니다.
TRACE_CONTEXT_FIELDS = (
    "trace_id",
    "trace_status",
    "trace_started_at",
    "trace_finished_at",
    "trace_duration_ms",
    "fallback_occurred",
    "trace_events",
)


def utc_now_iso() -> str:
    """
    현재 UTC 시각을 ISO 8601 문자열로 반환합니다.

    반환 예:

        "2026-07-10T01:12:30.120000+00:00"

    왜 UTC를 사용하는가?
    --------------------
    서버가 서로 다른 지역에서 실행되더라도
    같은 시간 기준으로 로그를 비교하기 쉽기 때문입니다.

    예:

        서울 서버

        미국 서버

        클라우드 서버

    각 서버의 로컬 시간을 그대로 사용하면
    같은 사건도 서로 다른 시각처럼 보일 수 있습니다.

    UTC를 공통 기준으로 사용하면
    여러 실행 기록을 비교하기 쉽습니다.

    왜 문자열로 반환하는가?
    -----------------------
    datetime 객체보다 ISO 8601 문자열이
    JSON 응답, 로그, DB 저장에 사용하기 쉽기 때문입니다.
    """

    return datetime.now(
        timezone.utc
    ).isoformat()


def calculate_duration_ms(
    started_perf_counter: float,
) -> float:
    """
    perf_counter() 시작값을 사용하여
    현재까지 경과한 시간을 millisecond 단위로 계산합니다.

    Parameters
    ----------
    started_perf_counter:
        실행 시작 시점에 호출한
        perf_counter() 반환값입니다.

    Returns
    -------
    float
        경과 시간입니다.

        단위:
            millisecond

        예:
            12.537

        의미:
            약 12.537 ms가 걸렸음


    왜 datetime 차이만 사용하지 않는가?
    ----------------------------------
    datetime은 다음 질문에 적합합니다.

        언제 시작했는가?

        언제 끝났는가?

    perf_counter()는 다음 질문에 적합합니다.

        실제 실행에 얼마나 걸렸는가?

    perf_counter()는
    실행 시간 측정을 목적으로 제공되는
    고해상도 단조 증가 시계입니다.

    단조 증가란?
    ------------
    측정 중 시스템 시각이 조정되더라도
    시간값이 갑자기 과거로 이동하지 않도록
    경과 시간 측정에 적합한 특성을 가진다는 의미입니다.


    계산 과정
    ---------
    perf_counter()는 초 단위 값을 반환합니다.

    예:

        시작:
            100.500

        종료:
            100.512537

    차이:

        100.512537
        -
        100.500

        =
        0.012537초

    millisecond 변환:

        0.012537
        ×
        1000

        =
        12.537 ms
    """

    elapsed_seconds = (
        perf_counter()
        -
        started_perf_counter
    )

    elapsed_ms = (
        elapsed_seconds
        *
        1000.0
    )

    # 너무 긴 소수점은
    # 로그와 API 응답을 읽기 어렵게 만들 수 있습니다.
    #
    # Day 16에서는 millisecond 값을
    # 소수점 셋째 자리까지 저장합니다.
    return round(
        elapsed_ms,
        3,
    )


def ensure_trace_state(
    state: AgentState,
) -> AgentState:
    """
    AgentState에 필요한 trace 기본값이 있는지 확인합니다.

    값이 없으면 안전한 기본값을 추가합니다.

    왜 이 함수가 필요한가?
    ---------------------
    public runner에서는 일반적으로:

        create_initial_agent_state()

    를 사용하므로 trace 기본값이 이미 존재합니다.

    하지만 기존 단위 테스트에서는
    다음처럼 AgentState를 직접 만들 수도 있습니다.

        state: AgentState = {
            "question": "고장 위험을 예측해줘."
        }

    이런 state에는 다음 값이 없을 수 있습니다.

        trace_id

        trace_status

        trace_started_at

        trace_events

    trace helper가 없는 key 때문에
    KeyError를 발생시키지 않도록
    방어적으로 기본값을 보완합니다.


    setdefault()의 의미
    ------------------
    예:

        state.setdefault(
            "trace_events",
            [],
        )

    trace_events가 이미 있다면:

        기존 값을 유지

    trace_events가 없다면:

        새로운 빈 list를 추가
    """

    # trace_id가 없다면
    # 새로운 UUID 기반 고유 ID를 생성합니다.
    state.setdefault(
        "trace_id",
        uuid4().hex,
    )

    # 아직 전체 workflow가 실행 중이라고 가정합니다.
    state.setdefault(
        "trace_status",
        "running",
    )

    # 시작 시각이 없다면
    # 현재 UTC 시각을 trace 시작 시각으로 사용합니다.
    state.setdefault(
        "trace_started_at",
        utc_now_iso(),
    )

    # 아직 종료되지 않았으므로 None입니다.
    state.setdefault(
        "trace_finished_at",
        None,
    )

    # 아직 전체 실행 시간이 계산되지 않았으므로
    # None을 기본값으로 사용합니다.
    state.setdefault(
        "trace_duration_ms",
        None,
    )

    # 아직 fallback이 발생하지 않았다고 가정합니다.
    state.setdefault(
        "fallback_occurred",
        False,
    )

    # node와 route event를 누적할 list입니다.
    state.setdefault(
        "trace_events",
        [],
    )

    return state


def carry_trace_context(
    *,
    source_state: AgentState,
    target_state: AgentState,
) -> AgentState:
    """
    source_state의 trace 문맥을
    target_state에 이어서 전달합니다.

    Parameters
    ----------
    source_state:
        node 실행 전 AgentState

    target_state:
        node 실행 후 반환된 AgentState

    Returns
    -------
    AgentState
        trace 문맥이 유지된 target_state


    왜 필요한가?
    ------------
    현재 프로젝트 node는 대부분
    전달받은 state를 수정한 뒤
    같은 state를 반환합니다.

    예:

        def node(state):
            state["intent"] = "failure_prediction"

            return state

    이 경우 trace 정보도 그대로 유지됩니다.

    하지만 미래에 다음처럼
    새로운 dict를 반환하는 node가 생길 수도 있습니다.

        def node(state):
            return {
                "question": state["question"],
                "intent": "failure_prediction",
            }

    이 경우 기존 trace_events,
    trace_id 등이 빠질 수 있습니다.

    carry_trace_context()는
    target_state에 trace field가 없을 때만
    source_state 값을 전달합니다.


    왜 기존 값을 덮어쓰지 않는가?
    ---------------------------
    target_state가 이미 새로운 trace 값을 가지고 있다면
    그 값을 유지해야 하기 때문입니다.

    따라서:

        if field not in target_state:

    조건에서만 복사합니다.
    """

    for field_name in TRACE_CONTEXT_FIELDS:

        if (
            field_name
            not in target_state
            and field_name in source_state
        ):
            target_state[field_name] = (
                source_state[field_name]
            )

    return target_state


def append_trace_event(
    state: AgentState,
    *,
    event_type: TraceEventType,
    event_name: str,
    status: TraceEventStatus,
    started_at: str,
    finished_at: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    """
    AgentState의 trace_events에
    구조화된 event 한 개를 추가합니다.

    Parameters
    ----------
    state:
        trace event를 저장할 AgentState

    event_type:
        event 종류

        "node"
        또는
        "route"

    event_name:
        실제 node 또는 route 이름

        예:
            "classify_intent"

            "route_after_classification"

    status:
        event 실행 결과

        "success"

        "warning"

        "error"

        "fallback"

    started_at:
        실행 시작 UTC 시각

    finished_at:
        실행 종료 UTC 시각

    duration_ms:
        실행 시간

        단위:
            millisecond

    metadata:
        event와 관련된 추가 요약 정보

        예:

        {
            "intent": "failure_prediction",
            "intent_source": "openai",
            "confidence": 0.95
        }


    sequence 생성 방식
    ------------------
    현재 trace event 개수:

        0개

    새 event sequence:

        1


    현재 trace event 개수:

        3개

    새 event sequence:

        4

    따라서:

        len(trace_events) + 1

    을 사용합니다.
    """

    # trace 기본값이 없는 부분 state에서도
    # 안전하게 사용할 수 있도록 보완합니다.
    ensure_trace_state(state)

    trace_events = state["trace_events"]

    # trace event는 1부터 순서를 시작합니다.
    sequence = (
        len(trace_events)
        +
        1
    )

    # metadata가 전달되지 않았다면
    # 새로운 빈 dict를 사용합니다.
    #
    # 함수 기본값을 {}로 직접 사용하지 않는 이유는
    # mutable default 문제를 피하기 위해서입니다.
    event_metadata = dict(
        metadata
        or
        {}
    )

    trace_event = {
        "sequence": sequence,
        "event_type": event_type,
        "event_name": event_name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "metadata": event_metadata,
    }

    trace_events.append(
        trace_event
    )

    return state


def mark_fallback_occurred(
    state: AgentState,
) -> AgentState:
    """
    LangGraph가 실제 fallback 경로를 사용했음을 기록합니다.

    fallback_occurred를 True로 변경합니다.

    왜 intent_source와 분리하는가?
    -----------------------------
    다음 두 상황은 서로 다릅니다.


    상황 1:
    OpenAI intent 분류 실패

        ↓

    rule-based classifier 사용

        ↓

    failure_prediction 분류 성공

        ↓

    모델 prediction 성공


    이 경우:

        intent_source
        =
        "fallback"

    하지만 LangGraph는
    fallback answer node를 실행하지 않았습니다.

    따라서:

        fallback_occurred
        =
        False


    상황 2:
    OpenAI intent 분류 성공

        ↓

    intent
    =
    "failure_prediction"

        ↓

    raw_sample 없음

        ↓

    fallback answer node 실행


    이 경우:

        intent_source
        =
        "openai"

    하지만 LangGraph는
    실제 fallback 경로를 실행했습니다.

    따라서:

        fallback_occurred
        =
        True


    즉:

        intent_source == "fallback"

    과:

        fallback_occurred is True

    는 서로 다른 의미입니다.
    """

    ensure_trace_state(state)

    state["fallback_occurred"] = True

    return state


def run_traced_node(
    state: AgentState,
    *,
    node_name: str,
    node_function: NodeFunction,
    metadata_builder: NodeMetadataBuilder | None = None,
    is_fallback_node: bool = False,
) -> AgentState:
    """
    LangGraph node를 실행하고
    실행 결과를 trace event로 기록합니다.

    처리 순서
    ---------
    1. trace 기본값 확인

    2. node 실행 전 warning 개수 저장

    3. node 실행 전 error 개수 저장

    4. 시작 UTC 시각 기록

    5. perf_counter() 시작값 기록

    6. 실제 node 함수 실행

    7. 종료 UTC 시각 기록

    8. duration_ms 계산

    9. node 실행 후 새 warning 개수 계산

    10. node 실행 후 새 error 개수 계산

    11. event status 결정

    12. metadata 생성

    13. trace_events에 event 추가

    14. node 실행 후 AgentState 반환


    Parameters
    ----------
    state:
        node 실행 전 AgentState

    node_name:
        trace에 기록할 node 이름

        예:
            "validate_question"

            "classify_intent"

            "call_failure_prediction"

    node_function:
        실제 실행할 LangGraph node 함수

    metadata_builder:
        node 실행 후 state를 읽어
        추가 trace metadata를 만드는 함수

        전달하지 않아도 됩니다.

    is_fallback_node:
        현재 node가 fallback 처리 node인지 여부

        True라면:

            fallback_occurred = True

            event status = "fallback"


    예외 처리 원칙
    -------------
    실제 node 함수가 예상하지 못한 예외를 발생시키면:

    1. error trace event 기록

    2. 예외 종류를 metadata에 저장

    3. 원래 예외를 다시 raise

    합니다.

    왜 예외를 숨기지 않는가?
    -----------------------
    trace 기능 때문에
    실제 business logic 오류가
    정상 처리된 것처럼 보이면 안 되기 때문입니다.

    따라서 trace는 오류를 기록하지만,
    오류 자체를 임의로 정상 결과로 바꾸지 않습니다.
    """

    ensure_trace_state(state)

    # node 실행 전에 존재하던
    # warning과 error 개수를 기억합니다.
    #
    # node 실행 후 개수와 비교하면
    # 현재 node가 새 warning 또는 error를
    # 추가했는지 알 수 있습니다.
    warning_count_before = len(
        state.get(
            "warnings",
            [],
        )
    )

    error_count_before = len(
        state.get(
            "errors",
            [],
        )
    )

    # 사람이 읽을 수 있는
    # 실제 시작 UTC 시각입니다.
    started_at = utc_now_iso()

    # duration 계산에 사용할
    # 고해상도 시작 시간입니다.
    started_perf_counter = perf_counter()

    try:
        # 실제 LangGraph node를 실행합니다.
        result_state = node_function(
            state
        )

    except Exception as exc:
        # 예상하지 못한 예외가 발생해도
        # node 종료 시각과 실행 시간은 기록합니다.
        finished_at = utc_now_iso()

        duration_ms = calculate_duration_ms(
            started_perf_counter
        )

        # 예외 메시지 전체에는
        # 민감한 입력값이나 내부 경로가 포함될 수 있습니다.
        #
        # Day 16 기본 trace에는
        # 예외 클래스 이름만 저장합니다.
        #
        # 예:
        #
        # ValueError
        #
        # FileNotFoundError
        #
        # RuntimeError
        exception_metadata = {
            "exception_type": (
                type(exc).__name__
            ),
        }

        append_trace_event(
            state,
            event_type="node",
            event_name=node_name,
            status="error",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            metadata=exception_metadata,
        )

        # trace 기록 후
        # 원래 예외를 다시 발생시킵니다.
        #
        # raise만 사용하면
        # 현재 except가 받은
        # 원래 예외와 traceback을 유지합니다.
        raise

    # node 함수가 새로운 dict를 반환해도
    # 기존 trace 문맥이 사라지지 않도록 이어줍니다.
    carry_trace_context(
        source_state=state,
        target_state=result_state,
    )

    ensure_trace_state(
        result_state
    )

    finished_at = utc_now_iso()

    duration_ms = calculate_duration_ms(
        started_perf_counter
    )

    warning_count_after = len(
        result_state.get(
            "warnings",
            [],
        )
    )

    error_count_after = len(
        result_state.get(
            "errors",
            [],
        )
    )

    # 현재 node 실행 중
    # 새로 추가된 warning 개수입니다.
    warnings_added = max(
        0,
        (
            warning_count_after
            -
            warning_count_before
        ),
    )

    # 현재 node 실행 중
    # 새로 추가된 error 개수입니다.
    errors_added = max(
        0,
        (
            error_count_after
            -
            error_count_before
        ),
    )

    # 개별 node event의 상태를 결정합니다.
    #
    # 우선순위:
    #
    # 1. fallback node
    #
    # 2. 새 error 발생
    #
    # 3. 새 warning 발생
    #
    # 4. 정상 성공
    if is_fallback_node:
        mark_fallback_occurred(
            result_state
        )

        event_status: TraceEventStatus = (
            "fallback"
        )

    elif errors_added > 0:
        event_status = "error"

    elif warnings_added > 0:
        event_status = "warning"

    else:
        event_status = "success"

    # 모든 node trace에 공통으로 저장하는 metadata입니다.
    #
    # warnings_added:
    #   이 node 실행 중 새로 추가된 warning 수
    #
    # errors_added:
    #   이 node 실행 중 새로 추가된 error 수
    event_metadata: dict[str, Any] = {
        "warnings_added": warnings_added,
        "errors_added": errors_added,
    }

    # node별 추가 metadata builder가 있다면
    # node 실행 후 state를 사용하여
    # metadata를 생성합니다.
    #
    # 예:
    #
    # classify_intent:
    #
    # {
    #   "intent": "failure_prediction",
    #   "intent_source": "openai",
    #   "confidence": 0.95
    # }
    if metadata_builder is not None:
        additional_metadata = (
            metadata_builder(
                result_state
            )
        )

        event_metadata.update(
            additional_metadata
        )

    append_trace_event(
        result_state,
        event_type="node",
        event_name=node_name,
        status=event_status,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        metadata=event_metadata,
    )

    return result_state


def run_traced_route(
    state: AgentState,
    *,
    route_name: str,
    route_function: RouteFunction,
    metadata_builder: RouteMetadataBuilder | None = None,
) -> str:
    """
    LangGraph conditional route 함수를 실행하고
    선택 결과를 trace event로 기록합니다.

    Parameters
    ----------
    state:
        routing에 사용할 현재 AgentState

    route_name:
        trace에 기록할 route 함수 이름

        예:
            "route_after_validation"

            "route_after_classification"

            "route_after_prediction"

    route_function:
        실제 routing 함수

    metadata_builder:
        route 실행 후
        추가 metadata를 만드는 함수

        전달하지 않아도 됩니다.

    Returns
    -------
    str
        route_function이 선택한 다음 경로 이름

        예:
            "classify"

            "failure_prediction"

            "dataset_schema"

            "fallback"

            "final"


    route trace 기본 metadata
    -------------------------
    모든 route event에는
    selected_route를 자동 저장합니다.

    예:

        {
            "selected_route": "failure_prediction"
        }


    fallback route 처리
    -------------------
    selected_route가 "fallback"이면:

        fallback_occurred = True

        event status = "fallback"

    으로 기록합니다.
    """

    ensure_trace_state(state)

    started_at = utc_now_iso()

    started_perf_counter = perf_counter()

    try:
        # 실제 conditional routing 함수를 실행합니다.
        selected_route = route_function(
            state
        )

    except Exception as exc:
        finished_at = utc_now_iso()

        duration_ms = calculate_duration_ms(
            started_perf_counter
        )

        append_trace_event(
            state,
            event_type="route",
            event_name=route_name,
            status="error",
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            metadata={
                "exception_type": (
                    type(exc).__name__
                ),
            },
        )

        # routing 오류도 숨기지 않고
        # 원래 예외를 다시 발생시킵니다.
        raise

    finished_at = utc_now_iso()

    duration_ms = calculate_duration_ms(
        started_perf_counter
    )

    # route가 fallback을 선택했다면
    # 전체 trace에 fallback 발생을 기록합니다.
    if selected_route == "fallback":
        mark_fallback_occurred(
            state
        )

        route_status: TraceEventStatus = (
            "fallback"
        )

    else:
        route_status = "success"

    # 모든 route event에는
    # 선택한 다음 경로를 기본 metadata로 저장합니다.
    route_metadata: dict[str, Any] = {
        "selected_route": selected_route,
    }

    # route별 추가 metadata가 있다면 병합합니다.
    #
    # 예:
    #
    # {
    #   "intent": "failure_prediction",
    #   "selected_route": "failure_prediction"
    # }
    if metadata_builder is not None:
        additional_metadata = (
            metadata_builder(
                state,
                selected_route,
            )
        )

        route_metadata.update(
            additional_metadata
        )

    append_trace_event(
        state,
        event_type="route",
        event_name=route_name,
        status=route_status,
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        metadata=route_metadata,
    )

    return selected_route


def determine_final_trace_status(
    state: AgentState,
) -> TraceStatus:
    """
    최종 AgentState를 바탕으로
    요청 하나의 전체 trace 상태를 결정합니다.

    우선순위
    --------
    1. fallback_occurred가 True

        → "fallback"

    2. errors가 하나 이상 존재

        → "error"

    3. 그 외

        → "success"


    왜 fallback을 error보다 먼저 확인하는가?
    --------------------------------------
    현재 프로젝트에서는
    raw_sample이 없는 failure_prediction 요청처럼
    error 메시지를 기록한 뒤
    fallback 답변을 정상 반환하는 경로가 있습니다.

    예:

        errors:

        [
            "raw_sample이 없어
             prediction을 수행할 수 없습니다."
        ]

        fallback_occurred:

        True

    이 요청은 아무 응답도 반환하지 못한
    시스템 장애와는 다릅니다.

    Agent가 문제를 감지하고
    사용자에게 안전한 fallback 답변을 반환했습니다.

    따라서 전체 상태를:

        "error"

    보다는:

        "fallback"

    으로 구분합니다.
    """

    ensure_trace_state(state)

    if state.get(
        "fallback_occurred",
        False,
    ):
        return "fallback"

    if len(
        state.get(
            "errors",
            [],
        )
    ) > 0:
        return "error"

    return "success"


def finalize_trace(
    state: AgentState,
    *,
    started_perf_counter: float,
) -> AgentState:
    """
    요청 하나의 전체 trace를 종료합니다.

    Parameters
    ----------
    state:
        LangGraph 실행이 끝난 최종 AgentState

    started_perf_counter:
        전체 workflow 실행을 시작하기 전에
        runner가 저장한 perf_counter() 값

    Returns
    -------
    AgentState
        전체 trace 종료 정보가 추가된 state


    저장하는 값
    -----------
    trace_finished_at:

        전체 workflow 종료 UTC 시각

    trace_duration_ms:

        전체 workflow 실행 시간

    trace_status:

        "success"

        "fallback"

        또는

        "error"


    왜 전체 workflow timer는
    state 안에 직접 저장하지 않는가?
    --------------------------------
    perf_counter() 반환값은
    프로세스 내부의 경과 시간 측정용 값입니다.

    외부 API 응답이나 DB에 저장할
    의미 있는 절대 시각이 아닙니다.

    따라서 runner의 지역 변수로 보관하고,
    최종 duration_ms 계산에만 사용합니다.

    AgentState에는 사용자가 이해할 수 있는:

        trace_started_at

        trace_finished_at

        trace_duration_ms

    만 저장합니다.
    """

    ensure_trace_state(state)

    state["trace_finished_at"] = (
        utc_now_iso()
    )

    state["trace_duration_ms"] = (
        calculate_duration_ms(
            started_perf_counter
        )
    )

    state["trace_status"] = (
        determine_final_trace_status(
            state
        )
    )

    return state