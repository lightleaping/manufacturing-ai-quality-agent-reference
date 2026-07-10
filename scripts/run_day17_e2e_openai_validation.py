"""
Day 17 - Real OpenAI E2E Validation

이 스크립트는 monkeypatch나 mock을 사용하지 않고
실제 OpenAI API와 기존 LangGraph Agent 실행 경로를 검증합니다.

현재 첫 번째 구현 범위
-------------------
Scenario 1:

    질문:
        AI4I 데이터셋의 feature와 target은 뭐야?

    실제 실행 경로:
        question

        -> 실제 OpenAI gpt-4o-mini intent classification

        -> OpenAI JSON 응답 parsing

        -> intent payload validation

        -> LangGraph conditional routing

        -> dataset schema answer 생성

        -> trace 종료

        -> 최종 AgentState 반환

    기대 결과:
        intent == "dataset_schema_query"

        intent_source == "openai"

        trace_status == "success"

        fallback_occurred is False


중요
----
이 파일은 기본 pytest에 자동으로 포함하지 않습니다.

실제 OpenAI API는 다음 외부 조건에 영향을 받기 때문입니다.

- 네트워크 연결
- OPENAI_API_KEY
- API 사용 비용
- OpenAI 서비스 상태
- 실제 응답 시간

따라서 사용자가 아래 명령을 명시적으로 실행할 때만 동작합니다.

    python -m scripts.run_day17_e2e_openai_validation \
        --scenario schema
"""

from __future__ import annotations

import argparse
import os
import sys
from numbers import Real
from typing import Any

from dotenv import load_dotenv

from src.agent.failure_agent_graph import (
    run_failure_agent_graph,
)
from src.agent.state import (
    AgentState,
    ChatMessage,
)

from fastapi.testclient import TestClient

from src.api.main import app

# 콘솔에서 시나리오와 결과 영역을 구분하기 위한 길이입니다.
DIVIDER_LENGTH = 100

# raw_sample 누락으로 인해
# failure prediction을 완료하지 못하고
# 안전한 fallback answer로 이동하는 경로입니다.
#
# 중요한 점:
#
# OpenAI intent 분류는 성공합니다.
#
# 즉:
#
#     intent_source == "openai"
#
# 이지만,
#
# 실제 prediction에는 현재 raw_sample이 없으므로
# LangGraph fallback 경로가 실행됩니다.
#
# 따라서:
#
#     fallback_occurred is True
#
# 입니다.
#
# intent classifier fallback과
# LangGraph workflow fallback은
# 서로 다른 개념입니다.
EXPECTED_MISSING_SAMPLE_TRACE_EVENT_NAMES = [
    "validate_question",
    "route_after_validation",
    "classify_intent",
    "route_after_classification",
    "call_failure_prediction",
    "route_after_prediction",
    "build_fallback_answer",
]

# Day 16에서 확인한
# dataset schema 정상 경로의 trace event 순서입니다.
#
# Day 17에서는 최종 intent만 보는 것이 아니라,
# 실제 LangGraph 내부에서 기대한 node와 route가
# 올바른 순서로 실행되었는지도 확인합니다.
EXPECTED_SCHEMA_TRACE_EVENT_NAMES = [
    "validate_question",
    "route_after_validation",
    "classify_intent",
    "route_after_classification",
    "build_dataset_schema_answer",
]

# 실제 failure prediction 정상 경로에서
# 실행되어야 하는 TraceEvent 이름의 순서입니다.
#
# 처리 흐름:
#
# validate_question
#
# -> route_after_validation
#
# -> classify_intent
#
# -> route_after_classification
#
# -> call_failure_prediction
#
# -> route_after_prediction
#
# -> build_final_answer
#
# Day 17에서는 최종 prediction 값만 확인하지 않고,
# 실제 LangGraph가 기대한 node와 route를
# 올바른 순서로 실행했는지도 검증합니다.
EXPECTED_PREDICTION_TRACE_EVENT_NAMES = [
    "validate_question",
    "route_after_validation",
    "classify_intent",
    "route_after_classification",
    "call_failure_prediction",
    "route_after_prediction",
    "build_final_answer",
]

# 지원하지 않는 질문이 들어왔을 때의
# unknown fallback 경로입니다.
#
# 처리 흐름:
#
# validate_question
#
# -> route_after_validation
#
# -> classify_intent
#
# -> route_after_classification
#
# -> build_fallback_answer
#
# 이 경로에서는 failure prediction node를 실행하지 않습니다.
#
# 즉, 다음 node는 실행되면 안 됩니다.
#
#     call_failure_prediction
#
#     route_after_prediction
#
# 사용자가 점심 메뉴 추천처럼
# 현재 Agent가 지원하지 않는 질문을 하면
# 안전하게 fallback answer를 반환합니다.
EXPECTED_UNKNOWN_TRACE_EVENT_NAMES = [
    "validate_question",
    "route_after_validation",
    "classify_intent",
    "route_after_classification",
    "build_fallback_answer",
]

# FastAPI endpoint를 통해 failure prediction을 실행해도
# 내부 LangGraph 경로는 기존 prediction 정상 경로와 같습니다.
#
# FastAPI는 LangGraph node를 새로 추가하지 않습니다.
#
# FastAPI의 역할:
#
# HTTP JSON
#
# -> Pydantic request validation
#
# -> run_failure_agent_graph()
#
# -> AgentState
#
# -> Pydantic response validation
#
# -> HTTP JSON
#
# 따라서 response 안의 trace event 순서는
# 직접 Agent runner를 호출한 Scenario 2와 같아야 합니다.
EXPECTED_API_PREDICTION_TRACE_EVENT_NAMES = (
    EXPECTED_PREDICTION_TRACE_EVENT_NAMES
)

def print_divider(
    character: str = "=",
) -> None:
    """
    콘솔에 구분선을 출력합니다.

    Parameters
    ----------
    character:
        구분선에 사용할 문자입니다.

        기본값은 "="입니다.

    예:
        ======================================================================
    """

    print(
        character
        *
        DIVIDER_LENGTH
    )


def print_title(
    title: str,
) -> None:
    """
    시나리오 제목을 눈에 잘 보이도록 출력합니다.
    """

    print()
    print_divider()
    print(title)
    print_divider()


def format_value(
    value: Any,
) -> str:
    """
    콘솔에 출력할 값을 읽기 쉬운 문자열로 변환합니다.

    bool은 Python 기본 출력인 True, False 대신
    JSON과 비슷한 true, false로 표시합니다.

    None은 null로 표시합니다.
    """

    if value is None:
        return "null"

    if isinstance(
        value,
        bool,
    ):
        return (
            "true"
            if value
            else "false"
        )

    return str(
        value
    )


def add_check_result(
    *,
    condition: bool,
    check_name: str,
    failures: list[str],
    expected: Any | None = None,
    actual: Any | None = None,
) -> None:
    """
    검증 조건 하나의 PASS 또는 FAIL 결과를 기록합니다.

    Parameters
    ----------
    condition:
        검증 조건의 최종 결과입니다.

        True:
            검증 성공

        False:
            검증 실패

    check_name:
        어떤 조건을 검사했는지 설명하는 이름입니다.

    failures:
        실패한 검증 메시지를 모아 둘 list입니다.

        함수 밖에서 만든 list를 전달하고,
        실패할 때 append()하여 결과를 누적합니다.

    expected:
        기대값입니다.

        필요하지 않은 검증에서는 생략할 수 있습니다.

    actual:
        실제 실행 결과입니다.

        필요하지 않은 검증에서는 생략할 수 있습니다.
    """

    if condition:
        print(
            f"[PASS] {check_name}"
        )
        return

    failure_message = (
        f"[FAIL] {check_name}"
    )

    print(
        failure_message
    )

    if (
        expected is not None
        or
        actual is not None
    ):
        print(
            "       expected : "
            f"{format_value(expected)}"
        )
        print(
            "       actual   : "
            f"{format_value(actual)}"
        )

    failures.append(
        failure_message
    )


def validate_equal(
    *,
    check_name: str,
    actual: Any,
    expected: Any,
    failures: list[str],
) -> None:
    """
    실제값과 기대값이 같은지 검사합니다.

    예:
        actual:
            "dataset_schema_query"

        expected:
            "dataset_schema_query"

        결과:
            PASS
    """

    add_check_result(
        condition=(
            actual
            ==
            expected
        ),
        check_name=check_name,
        expected=expected,
        actual=actual,
        failures=failures,
    )


def validate_non_empty_string(
    *,
    check_name: str,
    actual: Any,
    failures: list[str],
) -> None:
    """
    값이 비어 있지 않은 문자열인지 검사합니다.

    다음은 실패합니다.

        None

        ""

        "   "

        123

    다음은 성공합니다.

        "openai"

        "사용자가 데이터셋 schema를 질문했습니다."
    """

    is_valid = (
        isinstance(
            actual,
            str,
        )
        and
        bool(
            actual.strip()
        )
    )

    add_check_result(
        condition=is_valid,
        check_name=check_name,
        expected=(
            "non-empty string"
        ),
        actual=actual,
        failures=failures,
    )


def validate_confidence(
    *,
    confidence: Any,
    failures: list[str],
) -> None:
    """
    OpenAI intent confidence가 정상 범위인지 검사합니다.

    정상 조건:

        숫자

        0.0 이상

        1.0 이하

    bool을 별도로 제외하는 이유
    --------------------------
    Python에서 bool은 int의 하위 타입입니다.

    따라서 아래 결과는 True입니다.

        isinstance(True, int)

    하지만 confidence=True는
    의미상 올바른 confidence가 아닙니다.

    그래서 숫자인지 검사할 때
    bool은 명시적으로 제외합니다.
    """

    is_valid_number = (
        isinstance(
            confidence,
            Real,
        )
        and
        not isinstance(
            confidence,
            bool,
        )
    )

    is_valid_range = (
        is_valid_number
        and
        0.0
        <=
        float(
            confidence
        )
        <=
        1.0
    )

    add_check_result(
        condition=is_valid_range,
        check_name=(
            "confidence is between "
            "0.0 and 1.0"
        ),
        expected=(
            "number in [0.0, 1.0]"
        ),
        actual=confidence,
        failures=failures,
    )

def validate_unit_interval_number(
    *,
    check_name: str,
    actual: Any,
    failures: list[str],
) -> None:
    """
    값이 0.0 이상 1.0 이하의
    정상적인 숫자인지 검사합니다.

    unit interval
    -------------
    수학에서 다음 범위를 의미합니다.

        [0.0, 1.0]

    대괄호 []는 양 끝값도 포함한다는 뜻입니다.

    따라서 다음 값은 모두 허용됩니다.

        0.0

        0.25

        0.7

        1.0


    이번 E2E에서 사용하는 곳
    -----------------------
    probability:

        모델이 계산한 고장 확률

    threshold:

        prediction 0과 1을 구분하는 기준


    bool을 제외하는 이유
    -------------------
    Python에서 bool은 int를 상속합니다.

    따라서:

        isinstance(True, int)

    결과가 True입니다.

    하지만 probability=True 같은 값은
    의미상 정상적인 모델 확률이 아닙니다.

    그래서 bool은 명시적으로 제외합니다.
    """

    is_valid_number = (
        isinstance(
            actual,
            Real,
        )
        and
        not isinstance(
            actual,
            bool,
        )
    )

    is_valid_range = (
        is_valid_number
        and
        0.0
        <=
        float(
            actual
        )
        <=
        1.0
    )

    add_check_result(
        condition=is_valid_range,
        check_name=check_name,
        expected=(
            "number in [0.0, 1.0]"
        ),
        actual=actual,
        failures=failures,
    )

def validate_binary_prediction(
    *,
    prediction: Any,
    failures: list[str],
) -> None:
    """
    prediction이 정상적인 이진 분류 결과인지 검사합니다.

    현재 고장 예측 모델의 출력:

        0:
            정상 또는 고장 위험 기준 미만

        1:
            고장 위험 기준 이상


    bool을 제외하는 이유
    -------------------
    Python에서는:

        True == 1

        False == 0

    이 성립합니다.

    하지만 prediction=True는
    모델의 명시적인 정수 예측값으로 보기 어렵습니다.

    따라서 정확히 int이고,
    bool은 아니며,
    0 또는 1인지 확인합니다.
    """

    is_valid = (
        isinstance(
            prediction,
            int,
        )
        and
        not isinstance(
            prediction,
            bool,
        )
        and
        prediction
        in
        {
            0,
            1,
        }
    )

    add_check_result(
        condition=is_valid,
        check_name=(
            "prediction is 0 or 1"
        ),
        expected="0 or 1",
        actual=prediction,
        failures=failures,
    )

def validate_prediction_consistency(
    *,
    prediction: Any,
    probability: Any,
    threshold: Any,
    failures: list[str],
) -> None:
    """
    prediction이 probability와 threshold의
    비교 결과와 일치하는지 검사합니다.

    현재 이진 분류 정책:

        probability >= threshold

            -> prediction = 1


        probability < threshold

            -> prediction = 0


    예:
        probability:
            0.993

        threshold:
            0.7

        기대 prediction:
            1


    이 검증이 필요한 이유
    -------------------
    prediction, probability, threshold가
    각각 정상 범위여도 서로 모순될 수 있습니다.

    잘못된 예:

        probability:
            0.9

        threshold:
            0.7

        prediction:
            0

    각 필드만 따로 검사하면 놓칠 수 있으므로
    세 값의 관계도 검증합니다.
    """

    prediction_is_valid = (
        isinstance(
            prediction,
            int,
        )
        and
        not isinstance(
            prediction,
            bool,
        )
        and
        prediction
        in
        {
            0,
            1,
        }
    )

    probability_is_valid = (
        isinstance(
            probability,
            Real,
        )
        and
        not isinstance(
            probability,
            bool,
        )
    )

    threshold_is_valid = (
        isinstance(
            threshold,
            Real,
        )
        and
        not isinstance(
            threshold,
            bool,
        )
    )

    input_values_are_valid = (
        prediction_is_valid
        and
        probability_is_valid
        and
        threshold_is_valid
    )

    if input_values_are_valid:
        expected_prediction = int(
            float(
                probability
            )
            >=
            float(
                threshold
            )
        )
    else:
        expected_prediction = (
            "valid prediction, "
            "probability, and threshold"
        )

    add_check_result(
        condition=(
            input_values_are_valid
            and
            prediction
            ==
            expected_prediction
        ),
        check_name=(
            "prediction matches "
            "probability and threshold"
        ),
        expected=expected_prediction,
        actual=prediction,
        failures=failures,
    )

def get_trace_event_names(
    state: AgentState,
) -> list[str]:
    """
    최종 AgentState의 trace event에서
    event_name만 순서대로 추출합니다.

    입력 예:
        [
            {
                "sequence": 1,
                "event_name": "validate_question",
                ...
            },
            {
                "sequence": 2,
                "event_name": "route_after_validation",
                ...
            },
        ]

    반환:
        [
            "validate_question",
            "route_after_validation",
        ]
    """

    trace_events = state.get(
        "trace_events",
        [],
    )

    if not isinstance(
        trace_events,
        list,
    ):
        return []

    event_names: list[str] = []

    for event in trace_events:
        if not isinstance(
            event,
            dict,
        ):
            continue

        event_name = event.get(
            "event_name"
        )

        if isinstance(
            event_name,
            str,
        ):
            event_names.append(
                event_name
            )

    return event_names


def validate_trace_event_sequences(
    *,
    state: AgentState,
    failures: list[str],
) -> None:
    """
    TraceEvent의 sequence가
    1부터 순서대로 증가하는지 검사합니다.

    정상 예:
        1, 2, 3, 4, 5

    비정상 예:
        1, 2, 4, 5

        0, 1, 2, 3

        1, 1, 2, 3
    """

    trace_events = state.get(
        "trace_events",
        [],
    )

    if not isinstance(
        trace_events,
        list,
    ):
        add_check_result(
            condition=False,
            check_name=(
                "trace_events is a list"
            ),
            expected="list",
            actual=type(
                trace_events
            ).__name__,
            failures=failures,
        )
        return

    actual_sequences = [
        event.get(
            "sequence"
        )
        for event in trace_events
        if isinstance(
            event,
            dict,
        )
    ]

    expected_sequences = list(
        range(
            1,
            len(
                trace_events
            )
            +
            1,
        )
    )

    add_check_result(
        condition=(
            actual_sequences
            ==
            expected_sequences
        ),
        check_name=(
            "trace event sequence "
            "is continuous"
        ),
        expected=expected_sequences,
        actual=actual_sequences,
        failures=failures,
    )


def print_agent_result(
    state: AgentState,
) -> None:
    """
    E2E 실행 후 핵심 Agent 결과를 출력합니다.

    API key나 OpenAI raw response 전체는 출력하지 않습니다.
    """

    print()
    print(
        "[AGENT RESULT]"
    )

    result_fields = [
        (
            "intent",
            state.get(
                "intent"
            ),
        ),
        (
            "intent_source",
            state.get(
                "intent_source"
            ),
        ),
        (
            "confidence",
            state.get(
                "confidence"
            ),
        ),
        (
            "intent_reason",
            state.get(
                "intent_reason"
            ),
        ),

        # Day 17 Scenario 2:
        # 실제 PyTorch prediction 결과를
        # 콘솔에서 함께 확인합니다.
        (
            "prediction",
            state.get(
                "prediction"
            ),
        ),
        (
            "probability",
            state.get(
                "probability"
            ),
        ),
        (
            "threshold",
            state.get(
                "threshold"
            ),
        ),
        (
            "risk_level",
            state.get(
                "risk_level"
            ),
        ),
        (
            "recommended_action",
            state.get(
                "recommended_action"
            ),
        ),

        # 가장 마지막 route node가 저장한
        # 현재 selected_route입니다.
        (
            "selected_route",
            state.get(
                "selected_route"
            ),
        ),
        (
            "trace_status",
            state.get(
                "trace_status"
            ),
        ),
        (
            "fallback_occurred",
            state.get(
                "fallback_occurred"
            ),
        ),
        (
            "trace_duration_ms",
            state.get(
                "trace_duration_ms"
            ),
        ),
        (
            "warning_count",
            len(
                state.get(
                    "warnings",
                    [],
                )
            ),
        ),
        (
            "error_count",
            len(
                state.get(
                    "errors",
                    [],
                )
            ),
        ),
    ]

    for (
        field_name,
        value,
    ) in result_fields:
        print(
            f"{field_name:<20}: "
            f"{format_value(value)}"
        )


def print_trace_event_names(
    state: AgentState,
) -> None:
    """
    실제 실행된 trace event 이름을 순서대로 출력합니다.
    """

    event_names = (
        get_trace_event_names(
            state
        )
    )

    print()
    print(
        "[TRACE EVENT ORDER]"
    )

    if not event_names:
        print(
            "No trace events"
        )
        return

    for (
        index,
        event_name,
    ) in enumerate(
        event_names,
        start=1,
    ):
        print(
            f"{index}. "
            f"{event_name}"
        )


def validate_schema_state(
    state: AgentState,
) -> list[str]:
    """
    Dataset schema 실제 OpenAI E2E 결과를 검증합니다.

    반환
    ----
    list[str]

    빈 list:
        모든 검증 성공

    값이 있는 list:
        하나 이상의 검증 실패
    """

    failures: list[str] = []

    print()
    print(
        "[VALIDATION]"
    )

    validate_equal(
        check_name=(
            "intent == "
            "dataset_schema_query"
        ),
        actual=state.get(
            "intent"
        ),
        expected=(
            "dataset_schema_query"
        ),
        failures=failures,
    )

    validate_equal(
        check_name=(
            "intent_source == openai"
        ),
        actual=state.get(
            "intent_source"
        ),
        expected="openai",
        failures=failures,
    )

    validate_confidence(
        confidence=state.get(
            "confidence"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "intent_reason is not empty"
        ),
        actual=state.get(
            "intent_reason"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "answer is not empty"
        ),
        actual=state.get(
            "answer"
        ),
        failures=failures,
    )

    validate_equal(
        check_name=(
            "trace_status == success"
        ),
        actual=state.get(
            "trace_status"
        ),
        expected="success",
        failures=failures,
    )

    validate_equal(
        check_name=(
            "fallback_occurred "
            "== false"
        ),
        actual=state.get(
            "fallback_occurred"
        ),
        expected=False,
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "trace_id is not empty"
        ),
        actual=state.get(
            "trace_id"
        ),
        failures=failures,
    )

    trace_duration_ms = (
        state.get(
            "trace_duration_ms"
        )
    )

    is_valid_duration = (
        isinstance(
            trace_duration_ms,
            Real,
        )
        and
        not isinstance(
            trace_duration_ms,
            bool,
        )
        and
        float(
            trace_duration_ms
        )
        >=
        0.0
    )

    add_check_result(
        condition=is_valid_duration,
        check_name=(
            "trace_duration_ms "
            "is zero or greater"
        ),
        expected=(
            "number >= 0.0"
        ),
        actual=trace_duration_ms,
        failures=failures,
    )

    error_list = state.get(
        "errors",
        [],
    )

    add_check_result(
        condition=(
            isinstance(
                error_list,
                list,
            )
            and
            len(
                error_list
            )
            ==
            0
        ),
        check_name=(
            "error_count == 0"
        ),
        expected=0,
        actual=(
            len(
                error_list
            )
            if isinstance(
                error_list,
                list,
            )
            else
            "errors is not a list"
        ),
        failures=failures,
    )

    actual_event_names = (
        get_trace_event_names(
            state
        )
    )

    add_check_result(
        condition=(
            actual_event_names
            ==
            EXPECTED_SCHEMA_TRACE_EVENT_NAMES
        ),
        check_name=(
            "trace event order "
            "matches schema route"
        ),
        expected=(
            EXPECTED_SCHEMA_TRACE_EVENT_NAMES
        ),
        actual=actual_event_names,
        failures=failures,
    )

    validate_trace_event_sequences(
        state=state,
        failures=failures,
    )

    return failures

def validate_prediction_consistency(
    *,
    prediction: Any,
    probability: Any,
    threshold: Any,
    failures: list[str],
) -> None:
    """
    prediction이 probability와 threshold의
    비교 결과와 일치하는지 검사합니다.

    현재 이진 분류 정책:

        probability >= threshold

            -> prediction = 1


        probability < threshold

            -> prediction = 0


    예:
        probability:
            0.993

        threshold:
            0.7

        기대 prediction:
            1


    이 검증이 필요한 이유
    -------------------
    prediction, probability, threshold가
    각각 정상 범위여도 서로 모순될 수 있습니다.

    잘못된 예:

        probability:
            0.9

        threshold:
            0.7

        prediction:
            0

    각 필드만 따로 검사하면 놓칠 수 있으므로
    세 값의 관계도 검증합니다.
    """

    prediction_is_valid = (
        isinstance(
            prediction,
            int,
        )
        and
        not isinstance(
            prediction,
            bool,
        )
        and
        prediction
        in
        {
            0,
            1,
        }
    )

    probability_is_valid = (
        isinstance(
            probability,
            Real,
        )
        and
        not isinstance(
            probability,
            bool,
        )
    )

    threshold_is_valid = (
        isinstance(
            threshold,
            Real,
        )
        and
        not isinstance(
            threshold,
            bool,
        )
    )

    input_values_are_valid = (
        prediction_is_valid
        and
        probability_is_valid
        and
        threshold_is_valid
    )

    if input_values_are_valid:
        expected_prediction = int(
            float(
                probability
            )
            >=
            float(
                threshold
            )
        )
    else:
        expected_prediction = (
            "valid prediction, "
            "probability, and threshold"
        )

    add_check_result(
        condition=(
            input_values_are_valid
            and
            prediction
            ==
            expected_prediction
        ),
        check_name=(
            "prediction matches "
            "probability and threshold"
        ),
        expected=expected_prediction,
        actual=prediction,
        failures=failures,
    )

def validate_message_list_contains(
    *,
    check_name: str,
    messages: Any,
    expected_text: str,
    failures: list[str],
) -> None:
    """
    메시지 목록 안에 특정 문자열이 포함되어 있는지 검사합니다.

    이번 Scenario 4에서는
    errors 목록에 raw_sample 관련 오류가
    실제로 기록되었는지 확인합니다.


    예
    --
    messages:

        [
            "failure_prediction intent이지만 "
            "현재 요청에 raw_sample이 없습니다."
        ]


    expected_text:

        "raw_sample"


    결과:

        PASS


    str(message)를 사용하는 이유
    ----------------------------
    현재 errors는 문자열 목록이지만,
    이후 구조화 error 객체로 변경될 수도 있습니다.

    각 값을 str로 변환하면
    문자열과 dict 형태 모두
    검색 가능한 문자열로 비교할 수 있습니다.
    """

    if not isinstance(
        messages,
        list,
    ):
        add_check_result(
            condition=False,
            check_name=check_name,
            expected=(
                f"list containing "
                f"{expected_text!r}"
            ),
            actual=(
                type(
                    messages
                ).__name__
            ),
            failures=failures,
        )
        return

    normalized_expected_text = (
        expected_text
        .strip()
        .lower()
    )

    contains_expected_text = any(
        normalized_expected_text
        in
        str(
            message
        ).lower()
        for message in messages
    )

    add_check_result(
        condition=(
            contains_expected_text
        ),
        check_name=check_name,
        expected=(
            f"message containing "
            f"{expected_text!r}"
        ),
        actual=messages,
        failures=failures,
    )

def run_schema_scenario(
    return_state: bool = False,
) -> bool | AgentState:
    """
    Scenario 1을 실제로 실행합니다.

    Returns
    -------
    bool

    True:
        모든 E2E 검증 성공

    False:
        하나 이상의 E2E 검증 실패
    """

    print_title(
        "[SCENARIO] "
        "Dataset schema real OpenAI E2E"
    )

    question = (
        "AI4I 데이터셋의 "
        "feature와 target은 뭐야?"
    )

    print()
    print(
        "[REQUEST]"
    )
    print(
        f"question             : "
        f"{question}"
    )
    print(
        "raw_sample_provided  : "
        "false"
    )
    print(
        "chat_history_count   : "
        "0"
    )

    try:
        # 기존 프로젝트의 공식 LangGraph runner를 호출합니다.
        #
        # 이 함수 내부에서:
        #
        # 1. AgentState 생성
        #
        # 2. LangGraph compile
        #
        # 3. 실제 OpenAI intent classification
        #
        # 4. intent JSON validation
        #
        # 5. LangGraph conditional routing
        #
        # 6. dataset schema answer 생성
        #
        # 7. trace 종료
        #
        # 8. 최종 AgentState 반환
        #
        # 순서가 실행됩니다.
        #
        # Day 17 스크립트에서 OpenAI SDK를
        # 직접 호출하지 않는 이유는
        # 기존 Agent 전체 경로를 검증하기 위해서입니다.
        state = run_failure_agent_graph(
            question=question,
            raw_sample=None,

            # Dataset schema 질문에서는
            # prediction service를 실행하지 않으므로
            # SHAP과 global importance는 사용되지 않습니다.
            #
            # 그래도 이 시나리오의 목적을
            # 더 명확하게 나타내기 위해
            # False를 명시합니다.
            include_shap=False,
            include_global_importance=False,

            # 이번 첫 번째 시나리오는
            # single-turn 질문입니다.
            chat_history=None,
        )

    except Exception as exc:
        print()
        print(
            "[ERROR] "
            "Scenario execution raised "
            "an exception"
        )
        print(
            "exception_type      : "
            f"{type(exc).__name__}"
        )
        print(
            "exception_message   : "
            f"{exc}"
        )

        return False

    print_agent_result(
        state
    )

    print_trace_event_names(
        state
    )

    failures = (
        validate_schema_state(
            state
        )
    )

    print()
    print(
        "[SCENARIO RESULT]"
    )

    if failures:
        print(
            "result              : "
            "FAILED"
        )
        print(
            "failure_count       : "
            f"{len(failures)}"
        )

        return False

    print(
        "result              : "
        "SUCCESS"
    )
    print(
        "failure_count       : "
        "0"
    )

    # 기존 Day 17 실행에서는 return_state 기본값이 False이므로
    # 성공 여부인 True를 그대로 반환합니다.
    #
    # Day 18 Benchmark에서 return_state=True로 호출하면,
    # 이미 실행과 검증이 끝난 최종 AgentState를 반환합니다.
    #
    # 이렇게 하면 Day 18이 OpenAI를 다시 호출하지 않고도
    # intent, route, trace 등의 구조화 결과를 수집할 수 있습니다.
    if return_state:
        return state

    return True

def validate_prediction_state(
    state: AgentState,
) -> list[str]:
    """
    실제 OpenAI와 실제 PyTorch failure prediction을 거친
    최종 AgentState가 기대 조건을 만족하는지 검사합니다.

    반환값
    ------
    빈 list:
        모든 검증 성공

    값이 있는 list:
        하나 이상의 검증 실패
    """

    failures: list[str] = []

    print()
    print(
        "[VALIDATION]"
    )

    # ---------------------------------------------------------
    # 1. 실제 OpenAI intent 결과 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "intent == "
            "failure_prediction"
        ),
        actual=state.get(
            "intent"
        ),
        expected=(
            "failure_prediction"
        ),
        failures=failures,
    )

    validate_equal(
        check_name=(
            "intent_source == openai"
        ),
        actual=state.get(
            "intent_source"
        ),
        expected="openai",
        failures=failures,
    )

    validate_confidence(
        confidence=state.get(
            "confidence"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "intent_reason is not empty"
        ),
        actual=state.get(
            "intent_reason"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 2. 실제 PyTorch prediction 결과 검증
    # ---------------------------------------------------------

    prediction = state.get(
        "prediction"
    )

    probability = state.get(
        "probability"
    )

    threshold = state.get(
        "threshold"
    )

    validate_binary_prediction(
        prediction=prediction,
        failures=failures,
    )

    validate_unit_interval_number(
        check_name=(
            "probability is between "
            "0.0 and 1.0"
        ),
        actual=probability,
        failures=failures,
    )

    validate_unit_interval_number(
        check_name=(
            "threshold is between "
            "0.0 and 1.0"
        ),
        actual=threshold,
        failures=failures,
    )

    validate_prediction_consistency(
        prediction=prediction,
        probability=probability,
        threshold=threshold,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 3. 위험도와 사용자 응답 검증
    # ---------------------------------------------------------

    risk_level = state.get(
        "risk_level"
    )

    allowed_risk_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    add_check_result(
        condition=(
            risk_level
            in
            allowed_risk_levels
        ),
        check_name=(
            "risk_level is "
            "LOW, MEDIUM, or HIGH"
        ),
        expected=sorted(
            allowed_risk_levels
        ),
        actual=risk_level,
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "recommended_action "
            "is not empty"
        ),
        actual=state.get(
            "recommended_action"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "answer is not empty"
        ),
        actual=state.get(
            "answer"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 4. Trace 최종 상태 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "trace_status == success"
        ),
        actual=state.get(
            "trace_status"
        ),
        expected="success",
        failures=failures,
    )

    validate_equal(
        check_name=(
            "fallback_occurred "
            "== false"
        ),
        actual=state.get(
            "fallback_occurred"
        ),
        expected=False,
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "trace_id is not empty"
        ),
        actual=state.get(
            "trace_id"
        ),
        failures=failures,
    )

    trace_duration_ms = (
        state.get(
            "trace_duration_ms"
        )
    )

    is_valid_duration = (
        isinstance(
            trace_duration_ms,
            Real,
        )
        and
        not isinstance(
            trace_duration_ms,
            bool,
        )
        and
        float(
            trace_duration_ms
        )
        >=
        0.0
    )

    add_check_result(
        condition=is_valid_duration,
        check_name=(
            "trace_duration_ms "
            "is zero or greater"
        ),
        expected=(
            "number >= 0.0"
        ),
        actual=trace_duration_ms,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 5. error 검증
    # ---------------------------------------------------------

    errors = state.get(
        "errors",
        [],
    )

    error_count = (
        len(
            errors
        )
        if isinstance(
            errors,
            list,
        )
        else
        "errors is not a list"
    )

    add_check_result(
        condition=(
            isinstance(
                errors,
                list,
            )
            and
            len(
                errors
            )
            ==
            0
        ),
        check_name=(
            "error_count == 0"
        ),
        expected=0,
        actual=error_count,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 6. 실제 LangGraph 실행 순서 검증
    # ---------------------------------------------------------

    actual_event_names = (
        get_trace_event_names(
            state
        )
    )

    add_check_result(
        condition=(
            actual_event_names
            ==
            EXPECTED_PREDICTION_TRACE_EVENT_NAMES
        ),
        check_name=(
            "trace event order "
            "matches prediction route"
        ),
        expected=(
            EXPECTED_PREDICTION_TRACE_EVENT_NAMES
        ),
        actual=actual_event_names,
        failures=failures,
    )

    validate_trace_event_sequences(
        state=state,
        failures=failures,
    )

    return failures

def validate_multi_turn_state(
    *,
    state: AgentState,
    expected_question: str,
    expected_chat_history: list[ChatMessage],
) -> list[str]:
    """
    실제 multi-turn OpenAI E2E 결과를 검증합니다.

    이번 시나리오에서는:

        이전 user 질문

        +

        이전 assistant 답변

        +

        현재 후속 질문

    이 LangGraph Agent에 전달됩니다.

    검증 범위
    ---------
    1. 현재 question 유지

    2. chat_history 유지

    3. 실제 OpenAI intent 분류

    4. dataset schema route

    5. prediction 미실행

    6. 정상 Trace 종료

    7. Trace event 순서

    8. error 없음


    반환값
    ------
    빈 list:

        모든 검증 성공


    값이 있는 list:

        하나 이상의 검증 실패
    """

    failures: list[str] = []

    print()
    print(
        "[VALIDATION]"
    )

    # ---------------------------------------------------------
    # 1. 현재 question 검증
    # ---------------------------------------------------------
    #
    # multi-turn 요청에서도
    # 최종 분류 대상은 현재 question입니다.
    #
    # 이전 history가 현재 question을
    # 덮어쓰면 안 됩니다.
    validate_equal(
        check_name=(
            "current question "
            "is preserved"
        ),
        actual=state.get(
            "question"
        ),
        expected=expected_question,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 2. chat_history 구조 검증
    # ---------------------------------------------------------

    actual_chat_history = (
        state.get(
            "chat_history"
        )
    )

    # chat_history가 list인지 확인합니다.
    add_check_result(
        condition=isinstance(
            actual_chat_history,
            list,
        ),
        check_name=(
            "chat_history is a list"
        ),
        expected="list",
        actual=(
            type(
                actual_chat_history
            ).__name__
        ),
        failures=failures,
    )

    # 이전 user 메시지 1개와
    # assistant 메시지 1개,
    # 총 2개가 유지되는지 확인합니다.
    actual_history_count = (
        len(
            actual_chat_history
        )
        if isinstance(
            actual_chat_history,
            list,
        )
        else
        None
    )

    validate_equal(
        check_name=(
            "chat_history_count == 2"
        ),
        actual=actual_history_count,
        expected=2,
        failures=failures,
    )

    # 전달한 history 내용과
    # 최종 AgentState에 남아 있는 history가
    # 같은지 확인합니다.
    #
    # ==:
    #   두 객체의 내부 값이 같은지 비교합니다.
    #
    # 여기서는 같은 객체인지 확인하는 is가 아니라,
    # role과 content 값이 같은지 확인하는 ==를 사용합니다.
    add_check_result(
        condition=(
            actual_chat_history
            ==
            expected_chat_history
        ),
        check_name=(
            "chat_history content "
            "is preserved"
        ),
        expected=(
            expected_chat_history
        ),
        actual=(
            actual_chat_history
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 3. 실제 OpenAI intent 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "intent == "
            "dataset_schema_query"
        ),
        actual=state.get(
            "intent"
        ),
        expected=(
            "dataset_schema_query"
        ),
        failures=failures,
    )

    # 실제 OpenAI 결과가 사용되어야 합니다.
    #
    # OpenAI 호출 실패 후
    # rule-based fallback이 사용되었다면
    # 이 검증은 실패합니다.
    validate_equal(
        check_name=(
            "intent_source == openai"
        ),
        actual=state.get(
            "intent_source"
        ),
        expected="openai",
        failures=failures,
    )

    validate_confidence(
        confidence=state.get(
            "confidence"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "intent_reason is not empty"
        ),
        actual=state.get(
            "intent_reason"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 4. LangGraph route 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "selected_route == "
            "dataset_schema"
        ),
        actual=state.get(
            "selected_route"
        ),
        expected=(
            "dataset_schema"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 5. Prediction 미실행 검증
    # ---------------------------------------------------------
    #
    # 이번 요청은 데이터셋 schema 질문입니다.
    #
    # 따라서 실제 PyTorch prediction을
    # 수행하면 안 됩니다.
    #
    # state.get("prediction")은
    # key가 없으면 None을 반환합니다.
    validate_equal(
        check_name=(
            "prediction is null"
        ),
        actual=state.get(
            "prediction"
        ),
        expected=None,
        failures=failures,
    )

    # dataset schema answer가
    # 실제로 생성되었는지 확인합니다.
    validate_non_empty_string(
        check_name=(
            "answer is not empty"
        ),
        actual=state.get(
            "answer"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 6. Trace 최종 결과 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "trace_status == success"
        ),
        actual=state.get(
            "trace_status"
        ),
        expected="success",
        failures=failures,
    )

    validate_equal(
        check_name=(
            "fallback_occurred "
            "== false"
        ),
        actual=state.get(
            "fallback_occurred"
        ),
        expected=False,
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "trace_id is not empty"
        ),
        actual=state.get(
            "trace_id"
        ),
        failures=failures,
    )

    trace_duration_ms = (
        state.get(
            "trace_duration_ms"
        )
    )

    is_valid_duration = (
        isinstance(
            trace_duration_ms,
            Real,
        )
        and
        not isinstance(
            trace_duration_ms,
            bool,
        )
        and
        float(
            trace_duration_ms
        )
        >=
        0.0
    )

    add_check_result(
        condition=is_valid_duration,
        check_name=(
            "trace_duration_ms "
            "is zero or greater"
        ),
        expected=(
            "number >= 0.0"
        ),
        actual=trace_duration_ms,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 7. error 검증
    # ---------------------------------------------------------

    errors = state.get(
        "errors",
        [],
    )

    error_count = (
        len(
            errors
        )
        if isinstance(
            errors,
            list,
        )
        else
        "errors is not a list"
    )

    add_check_result(
        condition=(
            isinstance(
                errors,
                list,
            )
            and
            len(
                errors
            )
            ==
            0
        ),
        check_name=(
            "error_count == 0"
        ),
        expected=0,
        actual=error_count,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 8. Trace 실행 순서 검증
    # ---------------------------------------------------------
    #
    # multi-turn 여부와 관계없이
    # 최종 intent가 dataset_schema_query이면
    # 실행 node와 route 순서는
    # Scenario 1의 schema 경로와 같습니다.
    actual_event_names = (
        get_trace_event_names(
            state
        )
    )

    add_check_result(
        condition=(
            actual_event_names
            ==
            EXPECTED_SCHEMA_TRACE_EVENT_NAMES
        ),
        check_name=(
            "trace event order "
            "matches multi-turn "
            "schema route"
        ),
        expected=(
            EXPECTED_SCHEMA_TRACE_EVENT_NAMES
        ),
        actual=actual_event_names,
        failures=failures,
    )

    validate_trace_event_sequences(
        state=state,
        failures=failures,
    )

    return failures

def validate_missing_sample_state(
    *,
    state: AgentState,
    expected_question: str,
    expected_chat_history: list[ChatMessage],
) -> list[str]:
    """
    실제 OpenAI intent 분류는 성공했지만
    현재 raw_sample이 없어서
    prediction을 수행하지 않는 E2E 결과를 검증합니다.


    핵심 정책
    ---------
    chat_history:

        현재 질문의 문맥 이해용


    raw_sample:

        실제 PyTorch prediction 입력용


    이전 대화에 설비 조건이 있어도
    현재 raw_sample로 자동 복원하거나
    자동 재사용하지 않습니다.


    반환값
    ------
    빈 list:

        모든 검증 성공


    값이 있는 list:

        하나 이상의 검증 실패
    """

    failures: list[str] = []

    print()
    print(
        "[VALIDATION]"
    )

    # ---------------------------------------------------------
    # 1. 현재 question 유지 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "current question "
            "is preserved"
        ),
        actual=state.get(
            "question"
        ),
        expected=expected_question,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 2. 이전 chat_history 유지 검증
    # ---------------------------------------------------------

    actual_chat_history = (
        state.get(
            "chat_history"
        )
    )

    add_check_result(
        condition=isinstance(
            actual_chat_history,
            list,
        ),
        check_name=(
            "chat_history is a list"
        ),
        expected="list",
        actual=(
            type(
                actual_chat_history
            ).__name__
        ),
        failures=failures,
    )

    actual_history_count = (
        len(
            actual_chat_history
        )
        if isinstance(
            actual_chat_history,
            list,
        )
        else
        None
    )

    validate_equal(
        check_name=(
            "chat_history_count == 2"
        ),
        actual=actual_history_count,
        expected=2,
        failures=failures,
    )

    add_check_result(
        condition=(
            actual_chat_history
            ==
            expected_chat_history
        ),
        check_name=(
            "chat_history content "
            "is preserved"
        ),
        expected=(
            expected_chat_history
        ),
        actual=(
            actual_chat_history
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 3. 실제 OpenAI intent 검증
    # ---------------------------------------------------------
    #
    # 이전 문맥과 현재 질문을 보고
    # failure_prediction으로 분류해야 합니다.

    validate_equal(
        check_name=(
            "intent == "
            "failure_prediction"
        ),
        actual=state.get(
            "intent"
        ),
        expected=(
            "failure_prediction"
        ),
        failures=failures,
    )

    # OpenAI intent 호출 자체는
    # 성공해야 합니다.
    validate_equal(
        check_name=(
            "intent_source == openai"
        ),
        actual=state.get(
            "intent_source"
        ),
        expected="openai",
        failures=failures,
    )

    validate_confidence(
        confidence=state.get(
            "confidence"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "intent_reason is not empty"
        ),
        actual=state.get(
            "intent_reason"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 4. 현재 raw_sample 미사용 검증
    # ---------------------------------------------------------
    #
    # 현재 요청에 raw_sample을 전달하지 않았으므로
    # AgentState에도 None이어야 합니다.
    #
    # 이전 chat_history의 설비 조건을
    # 자동 변환해서 넣었다면
    # 이 검증이 실패합니다.

    validate_equal(
        check_name=(
            "raw_sample is null"
        ),
        actual=state.get(
            "raw_sample"
        ),
        expected=None,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 5. Prediction 미수행 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "prediction is null"
        ),
        actual=state.get(
            "prediction"
        ),
        expected=None,
        failures=failures,
    )

    validate_equal(
        check_name=(
            "probability is null"
        ),
        actual=state.get(
            "probability"
        ),
        expected=None,
        failures=failures,
    )

    # prediction을 수행하지 못했으므로
    # 위험도를 확정하지 않습니다.
    validate_equal(
        check_name=(
            "risk_level == UNKNOWN"
        ),
        actual=state.get(
            "risk_level"
        ),
        expected="UNKNOWN",
        failures=failures,
    )

    # ---------------------------------------------------------
    # 6. 실제 fallback route 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "selected_route == fallback"
        ),
        actual=state.get(
            "selected_route"
        ),
        expected="fallback",
        failures=failures,
    )

    # 정상 success가 아니라
    # 안전한 fallback으로 종료되어야 합니다.
    validate_equal(
        check_name=(
            "trace_status == fallback"
        ),
        actual=state.get(
            "trace_status"
        ),
        expected="fallback",
        failures=failures,
    )

    validate_equal(
        check_name=(
            "fallback_occurred "
            "== true"
        ),
        actual=state.get(
            "fallback_occurred"
        ),
        expected=True,
        failures=failures,
    )

    # 사용자에게 raw_sample이 필요하다는
    # 권장 조치가 생성되어야 합니다.
    validate_non_empty_string(
        check_name=(
            "recommended_action "
            "is not empty"
        ),
        actual=state.get(
            "recommended_action"
        ),
        failures=failures,
    )

    # 안전한 fallback answer가
    # 생성되어야 합니다.
    validate_non_empty_string(
        check_name=(
            "answer is not empty"
        ),
        actual=state.get(
            "answer"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 7. Trace 식별자와 시간 검증
    # ---------------------------------------------------------

    validate_non_empty_string(
        check_name=(
            "trace_id is not empty"
        ),
        actual=state.get(
            "trace_id"
        ),
        failures=failures,
    )

    trace_duration_ms = (
        state.get(
            "trace_duration_ms"
        )
    )

    is_valid_duration = (
        isinstance(
            trace_duration_ms,
            Real,
        )
        and
        not isinstance(
            trace_duration_ms,
            bool,
        )
        and
        float(
            trace_duration_ms
        )
        >=
        0.0
    )

    add_check_result(
        condition=is_valid_duration,
        check_name=(
            "trace_duration_ms "
            "is zero or greater"
        ),
        expected=(
            "number >= 0.0"
        ),
        actual=trace_duration_ms,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 8. 오류 기록 검증
    # ---------------------------------------------------------

    errors = state.get(
        "errors",
        [],
    )

    # fallback 원인을 설명하는 오류가
    # 하나 이상 있어야 합니다.
    add_check_result(
        condition=(
            isinstance(
                errors,
                list,
            )
            and
            len(
                errors
            )
            >=
            1
        ),
        check_name=(
            "error_count is "
            "one or greater"
        ),
        expected=(
            "integer >= 1"
        ),
        actual=(
            len(
                errors
            )
            if isinstance(
                errors,
                list,
            )
            else
            "errors is not a list"
        ),
        failures=failures,
    )

    # 오류 목록이 단순히 존재하는지만 보지 않고,
    # 실제로 raw_sample 누락을 설명하는지 확인합니다.
    validate_message_list_contains(
        check_name=(
            "errors contain "
            "raw_sample reason"
        ),
        messages=errors,
        expected_text=(
            "raw_sample"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 9. 실제 LangGraph Trace 순서 검증
    # ---------------------------------------------------------

    actual_event_names = (
        get_trace_event_names(
            state
        )
    )

    add_check_result(
        condition=(
            actual_event_names
            ==
            EXPECTED_MISSING_SAMPLE_TRACE_EVENT_NAMES
        ),
        check_name=(
            "trace event order "
            "matches missing-sample "
            "fallback route"
        ),
        expected=(
            EXPECTED_MISSING_SAMPLE_TRACE_EVENT_NAMES
        ),
        actual=actual_event_names,
        failures=failures,
    )

    validate_trace_event_sequences(
        state=state,
        failures=failures,
    )

    return failures

def validate_unknown_state(
    *,
    state: AgentState,
    expected_question: str,
) -> list[str]:
    """
    지원하지 않는 질문에 대한
    실제 OpenAI unknown fallback E2E 결과를 검증합니다.

    이번 시나리오의 입력 예:

        오늘 점심 메뉴 추천해줘.

    이 질문은 제조 설비 고장 예측,
    AI4I 데이터셋 schema 설명과 관련이 없습니다.

    따라서 기대 결과는 다음과 같습니다.

        intent == "unknown"

        intent_source == "openai"

        prediction is None

        selected_route == "fallback"

        trace_status == "fallback"

        fallback_occurred is True


    주의
    ----
    unknown 질문은 시스템 오류가 아닙니다.

    따라서 errors가 반드시 1개 이상이어야 한다고
    강하게 검증하지 않습니다.

    대신 errors 필드가 list 구조인지 정도만 확인합니다.


    반환값
    ------
    빈 list:

        모든 검증 성공


    값이 있는 list:

        하나 이상의 검증 실패
    """

    failures: list[str] = []

    print()
    print(
        "[VALIDATION]"
    )

    # ---------------------------------------------------------
    # 1. 현재 question 유지 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "current question "
            "is preserved"
        ),
        actual=state.get(
            "question"
        ),
        expected=expected_question,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 2. 실제 OpenAI intent 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "intent == unknown"
        ),
        actual=state.get(
            "intent"
        ),
        expected="unknown",
        failures=failures,
    )

    # OpenAI 호출 자체는 성공해야 합니다.
    #
    # 만약 OpenAI 호출 실패 후
    # rule-based fallback이 사용되면
    # intent_source가 "fallback"이 될 수 있으므로
    # 이 검증에서 실패합니다.
    validate_equal(
        check_name=(
            "intent_source == openai"
        ),
        actual=state.get(
            "intent_source"
        ),
        expected="openai",
        failures=failures,
    )

    validate_confidence(
        confidence=state.get(
            "confidence"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "intent_reason is not empty"
        ),
        actual=state.get(
            "intent_reason"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 3. Prediction 미수행 검증
    # ---------------------------------------------------------
    #
    # unknown 질문은 설비 고장 예측 요청이 아니므로
    # PyTorch prediction을 실행하면 안 됩니다.

    validate_equal(
        check_name=(
            "raw_sample is null"
        ),
        actual=state.get(
            "raw_sample"
        ),
        expected=None,
        failures=failures,
    )

    validate_equal(
        check_name=(
            "prediction is null"
        ),
        actual=state.get(
            "prediction"
        ),
        expected=None,
        failures=failures,
    )

    validate_equal(
        check_name=(
            "probability is null"
        ),
        actual=state.get(
            "probability"
        ),
        expected=None,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 4. fallback route 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "selected_route == fallback"
        ),
        actual=state.get(
            "selected_route"
        ),
        expected="fallback",
        failures=failures,
    )

    validate_equal(
        check_name=(
            "trace_status == fallback"
        ),
        actual=state.get(
            "trace_status"
        ),
        expected="fallback",
        failures=failures,
    )

    validate_equal(
        check_name=(
            "fallback_occurred "
            "== true"
        ),
        actual=state.get(
            "fallback_occurred"
        ),
        expected=True,
        failures=failures,
    )

    # 사용자에게 반환할 fallback answer가
    # 생성되었는지 확인합니다.
    validate_non_empty_string(
        check_name=(
            "answer is not empty"
        ),
        actual=state.get(
            "answer"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 5. Trace 식별자와 시간 검증
    # ---------------------------------------------------------

    validate_non_empty_string(
        check_name=(
            "trace_id is not empty"
        ),
        actual=state.get(
            "trace_id"
        ),
        failures=failures,
    )

    trace_duration_ms = (
        state.get(
            "trace_duration_ms"
        )
    )

    is_valid_duration = (
        isinstance(
            trace_duration_ms,
            Real,
        )
        and
        not isinstance(
            trace_duration_ms,
            bool,
        )
        and
        float(
            trace_duration_ms
        )
        >=
        0.0
    )

    add_check_result(
        condition=is_valid_duration,
        check_name=(
            "trace_duration_ms "
            "is zero or greater"
        ),
        expected=(
            "number >= 0.0"
        ),
        actual=trace_duration_ms,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 6. errors 구조 검증
    # ---------------------------------------------------------
    #
    # unknown 질문은 시스템 오류가 아니므로
    # error_count를 0으로 강제하지 않습니다.
    #
    # 대신 errors가 list 구조인지 확인합니다.

    errors = state.get(
        "errors",
        [],
    )

    add_check_result(
        condition=isinstance(
            errors,
            list,
        ),
        check_name=(
            "errors is a list"
        ),
        expected="list",
        actual=(
            type(
                errors
            ).__name__
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 7. 실제 LangGraph Trace 순서 검증
    # ---------------------------------------------------------

    actual_event_names = (
        get_trace_event_names(
            state
        )
    )

    add_check_result(
        condition=(
            actual_event_names
            ==
            EXPECTED_UNKNOWN_TRACE_EVENT_NAMES
        ),
        check_name=(
            "trace event order "
            "matches unknown "
            "fallback route"
        ),
        expected=(
            EXPECTED_UNKNOWN_TRACE_EVENT_NAMES
        ),
        actual=actual_event_names,
        failures=failures,
    )

    validate_trace_event_sequences(
        state=state,
        failures=failures,
    )

    return failures

def validate_api_prediction_response(
    *,
    response_json: Any,
) -> list[str]:
    """
    실제 FastAPI failure prediction E2E의
    최종 JSON response를 검증합니다.

    검증 범위
    ---------
    1. JSON object 구조

    2. 실제 OpenAI intent

    3. 실제 PyTorch prediction

    4. probability / threshold 일관성

    5. 위험도와 사용자 답변

    6. Trace 응답

    7. TraceEvent 순서


    반환값
    ------
    빈 list:

        모든 검증 성공


    값이 있는 list:

        하나 이상의 검증 실패
    """

    failures: list[str] = []

    print()
    print(
        "[API RESPONSE VALIDATION]"
    )

    # ---------------------------------------------------------
    # 1. 최상위 JSON 구조 검증
    # ---------------------------------------------------------

    add_check_result(
        condition=isinstance(
            response_json,
            dict,
        ),
        check_name=(
            "response JSON is an object"
        ),
        expected="dict",
        actual=(
            type(
                response_json
            ).__name__
        ),
        failures=failures,
    )

    if not isinstance(
        response_json,
        dict,
    ):
        return failures

    # ---------------------------------------------------------
    # 2. 실제 OpenAI intent 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "API intent == "
            "failure_prediction"
        ),
        actual=response_json.get(
            "intent"
        ),
        expected=(
            "failure_prediction"
        ),
        failures=failures,
    )

    validate_equal(
        check_name=(
            "API intent_source == openai"
        ),
        actual=response_json.get(
            "intent_source"
        ),
        expected="openai",
        failures=failures,
    )

    validate_confidence(
        confidence=response_json.get(
            "confidence"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "API intent_reason "
            "is not empty"
        ),
        actual=response_json.get(
            "intent_reason"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 3. 실제 prediction 검증
    # ---------------------------------------------------------

    prediction = response_json.get(
        "prediction"
    )

    probability = response_json.get(
        "probability"
    )

    threshold = response_json.get(
        "threshold"
    )

    validate_binary_prediction(
        prediction=prediction,
        failures=failures,
    )

    validate_unit_interval_number(
        check_name=(
            "API probability is between "
            "0.0 and 1.0"
        ),
        actual=probability,
        failures=failures,
    )

    validate_unit_interval_number(
        check_name=(
            "API threshold is between "
            "0.0 and 1.0"
        ),
        actual=threshold,
        failures=failures,
    )

    validate_prediction_consistency(
        prediction=prediction,
        probability=probability,
        threshold=threshold,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 4. 위험도와 답변 검증
    # ---------------------------------------------------------

    risk_level = response_json.get(
        "risk_level"
    )

    allowed_risk_levels = {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    add_check_result(
        condition=(
            risk_level
            in
            allowed_risk_levels
        ),
        check_name=(
            "API risk_level is "
            "LOW, MEDIUM, or HIGH"
        ),
        expected=sorted(
            allowed_risk_levels
        ),
        actual=risk_level,
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "API recommended_action "
            "is not empty"
        ),
        actual=response_json.get(
            "recommended_action"
        ),
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "API answer is not empty"
        ),
        actual=response_json.get(
            "answer"
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 5. Trace 최종 상태 검증
    # ---------------------------------------------------------

    validate_equal(
        check_name=(
            "API trace_status == success"
        ),
        actual=response_json.get(
            "trace_status"
        ),
        expected="success",
        failures=failures,
    )

    validate_equal(
        check_name=(
            "API fallback_occurred "
            "== false"
        ),
        actual=response_json.get(
            "fallback_occurred"
        ),
        expected=False,
        failures=failures,
    )

    validate_non_empty_string(
        check_name=(
            "API trace_id is not empty"
        ),
        actual=response_json.get(
            "trace_id"
        ),
        failures=failures,
    )

    trace_duration_ms = (
        response_json.get(
            "trace_duration_ms"
        )
    )

    is_valid_duration = (
        isinstance(
            trace_duration_ms,
            Real,
        )
        and
        not isinstance(
            trace_duration_ms,
            bool,
        )
        and
        float(
            trace_duration_ms
        )
        >=
        0.0
    )

    add_check_result(
        condition=is_valid_duration,
        check_name=(
            "API trace_duration_ms "
            "is zero or greater"
        ),
        expected=(
            "number >= 0.0"
        ),
        actual=trace_duration_ms,
        failures=failures,
    )

    # ---------------------------------------------------------
    # 6. TraceEvent JSON 구조 검증
    # ---------------------------------------------------------

    trace_events = response_json.get(
        "trace_events"
    )

    add_check_result(
        condition=isinstance(
            trace_events,
            list,
        ),
        check_name=(
            "API trace_events is a list"
        ),
        expected="list",
        actual=(
            type(
                trace_events
            ).__name__
        ),
        failures=failures,
    )

    if not isinstance(
        trace_events,
        list,
    ):
        return failures

    # 각 TraceEvent는 FastAPI response에서
    # JSON object가 되어야 합니다.
    all_events_are_objects = all(
        isinstance(
            event,
            dict,
        )
        for event in trace_events
    )

    add_check_result(
        condition=(
            all_events_are_objects
        ),
        check_name=(
            "all API trace events "
            "are JSON objects"
        ),
        expected=True,
        actual=(
            all_events_are_objects
        ),
        failures=failures,
    )

    # ---------------------------------------------------------
    # 7. Trace 순서 검증
    # ---------------------------------------------------------

    actual_event_names = [
        event.get(
            "event_name"
        )
        for event in trace_events
        if isinstance(
            event,
            dict,
        )
    ]

    add_check_result(
        condition=(
            actual_event_names
            ==
            EXPECTED_API_PREDICTION_TRACE_EVENT_NAMES
        ),
        check_name=(
            "API trace event order "
            "matches prediction route"
        ),
        expected=(
            EXPECTED_API_PREDICTION_TRACE_EVENT_NAMES
        ),
        actual=actual_event_names,
        failures=failures,
    )

    actual_sequences = [
        event.get(
            "sequence"
        )
        for event in trace_events
        if isinstance(
            event,
            dict,
        )
    ]

    expected_sequences = list(
        range(
            1,
            len(
                trace_events
            )
            +
            1,
        )
    )

    add_check_result(
        condition=(
            actual_sequences
            ==
            expected_sequences
        ),
        check_name=(
            "API trace event sequence "
            "is continuous"
        ),
        expected=(
            expected_sequences
        ),
        actual=(
            actual_sequences
        ),
        failures=failures,
    )

    return failures

def run_prediction_scenario(
    return_state: bool = False,
) -> bool | AgentState:
    """
    Scenario 2를 실제로 실행합니다.

    실제 실행 범위
    -------------
    question

    -> 실제 OpenAI intent classification

    -> intent JSON validation

    -> LangGraph failure_prediction route

    -> 실제 PyTorch MLP prediction

    -> probability / threshold

    -> risk level

    -> final answer

    -> trace

    -> 최종 AgentState


    Returns
    -------
    bool

    True:
        모든 E2E 검증 성공

    False:
        하나 이상의 E2E 검증 실패
    """

    print_title(
        "[SCENARIO] "
        "Failure prediction real "
        "OpenAI + PyTorch E2E"
    )

    question = (
        "이 설비 조건이면 "
        "고장 위험이 높아?"
    )

    # 실제 AI4I failure prediction service에
    # 전달할 설비 입력값입니다.
    #
    # key:
    #   Agent/FastAPI에서 사용하는 입력 필드명
    #
    # value:
    #   현재 설비의 실제 입력값
    raw_sample: dict[str, Any] = {
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
    }

    print()
    print(
        "[REQUEST]"
    )
    print(
        f"question             : "
        f"{question}"
    )
    print(
        "raw_sample_provided  : "
        "true"
    )
    print(
        "chat_history_count   : "
        "0"
    )

    print()
    print(
        "[RAW SAMPLE]"
    )

    for (
        feature_name,
        feature_value,
    ) in raw_sample.items():
        print(
            f"{feature_name:<24}: "
            f"{feature_value}"
        )

    try:
        # 기존 프로젝트의 공식 LangGraph runner를 호출합니다.
        #
        # 실제 처리:
        #
        # 1. question과 raw_sample로 AgentState 생성
        #
        # 2. 실제 OpenAI gpt-4o-mini intent 분류
        #
        # 3. OpenAI JSON 검증
        #
        # 4. failure_prediction route 선택
        #
        # 5. 실제 PyTorch prediction service 실행
        #
        # 6. prediction / probability / threshold 생성
        #
        # 7. risk level과 권장 조치 생성
        #
        # 8. 최종 Agent answer 생성
        #
        # 9. Trace 종료
        #
        # 10. 최종 AgentState 반환
        state = run_failure_agent_graph(
            question=question,
            raw_sample=raw_sample,

            # 이번 Scenario 2의 주목적은:
            #
            # OpenAI
            #
            # -> LangGraph
            #
            # -> 실제 PyTorch prediction
            #
            # 연결 검증입니다.
            #
            # SHAP은 실제 prediction을 수행하기 위한
            # 필수 요소가 아니므로 우선 비활성화합니다.
            #
            # 이렇게 하면 E2E 범위를 명확하게 유지하고
            # 추가 설명 artifact 계산 시간을 줄일 수 있습니다.
            include_shap=False,

            # global permutation importance도
            # 핵심 prediction 실행에는 필수가 아니므로
            # 이번 시나리오에서는 비활성화합니다.
            include_global_importance=False,

            # Scenario 2는 single-turn 요청입니다.
            chat_history=None,
        )

    except Exception as exc:
        print()
        print(
            "[ERROR] "
            "Scenario execution raised "
            "an exception"
        )
        print(
            "exception_type      : "
            f"{type(exc).__name__}"
        )
        print(
            "exception_message   : "
            f"{exc}"
        )

        return False

    print_agent_result(
        state
    )

    print_trace_event_names(
        state
    )

    failures = (
        validate_prediction_state(
            state
        )
    )

    print()
    print(
        "[SCENARIO RESULT]"
    )

    if failures:
        print(
            "result              : "
            "FAILED"
        )
        print(
            "failure_count       : "
            f"{len(failures)}"
        )

        return False

    print(
        "result              : "
        "SUCCESS"
    )
    print(
        "failure_count       : "
        "0"
    )

    # return_state=False:
    #     기존 Day 17 실행 방식입니다.
    #     시나리오 성공 여부인 True를 반환합니다.
    #
    # return_state=True:
    #     Day 18 Benchmark에서 사용하는 방식입니다.
    #     실제 OpenAI, LangGraph, PyTorch 실행과 검증이 모두 끝난
    #     최종 AgentState를 반환합니다.
    if return_state:
        return state

    return True

def run_multi_turn_scenario() -> bool:
    """
    Scenario 3을 실제로 실행합니다.

    이전 데이터셋 대화와
    현재 후속 질문을 함께 전달하여
    실제 OpenAI가 문맥을 포함한 intent 분류를
    수행하는지 검증합니다.


    이전 대화
    ---------
    user:

        AI4I 데이터셋의 feature는 뭐야?


    assistant:

        현재 모델은 AI4I feature 6개를 사용합니다.


    현재 질문
    ---------
        그중 target은 뭐야?


    실제 실행 범위
    -------------
    chat_history

    +

    current question

    -> AgentState

    -> 실제 OpenAI gpt-4o-mini

    -> intent JSON validation

    -> dataset_schema_query

    -> LangGraph dataset schema route

    -> answer

    -> Trace

    -> 최종 AgentState


    Returns
    -------
    bool

    True:

        모든 E2E 검증 성공


    False:

        하나 이상의 E2E 검증 실패
    """

    print_title(
        "[SCENARIO] "
        "Multi-turn context real "
        "OpenAI E2E"
    )

    # 현재 OpenAI가 최종적으로
    # intent를 분류할 질문입니다.
    question = (
        "그중 target은 뭐야?"
    )

    # 이전 user/assistant 대화입니다.
    #
    # Day 15에서 정의한 ChatMessage 구조:
    #
    # {
    #     "role": "user" | "assistant",
    #     "content": "메시지 내용",
    # }
    #
    # chat_history는 이전 대화 문맥을
    # 이해하는 용도로만 사용합니다.
    #
    # 실제 prediction 입력값은 아닙니다.
    chat_history: list[
        ChatMessage
    ] = [
        {
            "role": "user",
            "content": (
                "AI4I 데이터셋의 "
                "feature는 뭐야?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "현재 모델은 "
                "AI4I feature 6개를 "
                "사용합니다."
            ),
        },
    ]

    print()
    print(
        "[REQUEST]"
    )
    print(
        f"question             : "
        f"{question}"
    )
    print(
        "raw_sample_provided  : "
        "false"
    )
    print(
        "chat_history_count   : "
        f"{len(chat_history)}"
    )

    print()
    print(
        "[CHAT HISTORY]"
    )

    for (
        index,
        message,
    ) in enumerate(
        chat_history,
        start=1,
    ):
        print(
            f"{index}. "
            f"{message['role']:<9}: "
            f"{message['content']}"
        )

    try:
        # 기존 LangGraph 공식 runner에:
        #
        # 현재 question
        #
        # +
        #
        # 이전 chat_history
        #
        # 를 함께 전달합니다.
        #
        # runner 내부 흐름:
        #
        # create_initial_agent_state()
        #
        # -> AgentState["chat_history"]
        #
        # -> classify_intent_node()
        #
        # -> classify_intent(
        #        question,
        #        chat_history=...
        #    )
        #
        # -> 실제 OpenAI
        #
        # -> LangGraph routing
        state = run_failure_agent_graph(
            question=question,

            # 데이터셋 schema 질문이므로
            # 실제 prediction 입력은 없습니다.
            raw_sample=None,

            # prediction route가 아니므로
            # SHAP은 사용하지 않습니다.
            include_shap=False,

            # global importance도 사용하지 않습니다.
            include_global_importance=False,

            # Day 15에서 추가한 keyword-only
            # multi-turn 입력입니다.
            chat_history=chat_history,
        )

    except Exception as exc:
        print()
        print(
            "[ERROR] "
            "Scenario execution raised "
            "an exception"
        )
        print(
            "exception_type      : "
            f"{type(exc).__name__}"
        )
        print(
            "exception_message   : "
            f"{exc}"
        )

        return False

    print_agent_result(
        state
    )

    print_trace_event_names(
        state
    )

    failures = (
        validate_multi_turn_state(
            state=state,
            expected_question=question,
            expected_chat_history=(
                chat_history
            ),
        )
    )

    print()
    print(
        "[SCENARIO RESULT]"
    )

    if failures:
        print(
            "result              : "
            "FAILED"
        )
        print(
            "failure_count       : "
            f"{len(failures)}"
        )

        return False

    print(
        "result              : "
        "SUCCESS"
    )
    print(
        "failure_count       : "
        "0"
    )

    return True

def run_missing_sample_scenario() -> bool:
    """
    Scenario 4를 실제로 실행합니다.

    이전 chat_history에는 설비 조건이 있지만
    현재 요청에는 raw_sample을 전달하지 않습니다.


    검증할 정책
    -------------
    chat_history:

        질문 문맥 이해용


    raw_sample:

        실제 PyTorch prediction 입력용


    이전 대화의 설비 조건:

        현재 raw_sample로 자동 재사용하지 않음


    기대 결과
    ---------
    intent:

        failure_prediction


    intent_source:

        openai


    prediction:

        None


    risk_level:

        UNKNOWN


    trace_status:

        fallback


    fallback_occurred:

        True
    """

    print_title(
        "[SCENARIO] "
        "Missing raw_sample real "
        "OpenAI fallback E2E"
    )

    question = (
        "그 조건으로 고장 위험을 "
        "다시 예측해줘."
    )

    # 이전 대화에는 실제 설비 조건이
    # 자연어로 들어 있습니다.
    #
    # 하지만 이 텍스트를 parsing해서
    # 현재 raw_sample로 자동 복원하지 않습니다.
    #
    # Day 15의 안전 정책을
    # 실제 E2E로 확인하기 위한 입력입니다.
    chat_history: list[
        ChatMessage
    ] = [
        {
            "role": "user",
            "content": (
                "공기 온도 303.0K, "
                "공정 온도 312.5K, "
                "회전 속도 1380rpm, "
                "토크 62.0Nm, "
                "공구 마모 220분, "
                "Type L 조건이면 "
                "고장 위험이 높아?"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "해당 설비 조건으로 "
                "고장 위험 예측을 "
                "요청하셨습니다."
            ),
        },
    ]

    print()
    print(
        "[REQUEST]"
    )
    print(
        f"question             : "
        f"{question}"
    )
    print(
        "raw_sample_provided  : "
        "false"
    )
    print(
        "chat_history_count   : "
        f"{len(chat_history)}"
    )

    print()
    print(
        "[CHAT HISTORY]"
    )

    for (
        index,
        message,
    ) in enumerate(
        chat_history,
        start=1,
    ):
        print(
            f"{index}. "
            f"{message['role']:<9}: "
            f"{message['content']}"
        )

    try:
        state = run_failure_agent_graph(
            question=question,

            # 핵심:
            #
            # 현재 요청에는 raw_sample을
            # 전달하지 않습니다.
            #
            # 이전 대화에 조건이 있더라도
            # prediction 입력으로 자동 재사용하지 않습니다.
            raw_sample=None,

            # 실제 prediction이 수행되지 않으므로
            # SHAP도 실행하지 않습니다.
            include_shap=False,

            # global importance도 실행하지 않습니다.
            include_global_importance=False,

            # 이전 대화는
            # intent 문맥 이해용으로만 전달합니다.
            chat_history=chat_history,
        )

    except Exception as exc:
        print()
        print(
            "[ERROR] "
            "Scenario execution raised "
            "an exception"
        )
        print(
            "exception_type      : "
            f"{type(exc).__name__}"
        )
        print(
            "exception_message   : "
            f"{exc}"
        )

        return False

    print_agent_result(
        state
    )

    print_trace_event_names(
        state
    )

    failures = (
        validate_missing_sample_state(
            state=state,
            expected_question=question,
            expected_chat_history=(
                chat_history
            ),
        )
    )

    print()
    print(
        "[SCENARIO RESULT]"
    )

    if failures:
        print(
            "result              : "
            "FAILED"
        )
        print(
            "failure_count       : "
            f"{len(failures)}"
        )

        return False

    print(
        "result              : "
        "SUCCESS"
    )
    print(
        "failure_count       : "
        "0"
    )

    return True

def run_unknown_scenario() -> bool:
    """
    Scenario 5를 실제로 실행합니다.

    지원하지 않는 질문을 입력하여
    실제 OpenAI가 unknown intent로 분류하고,
    LangGraph가 안전한 fallback answer를 반환하는지 검증합니다.


    입력 질문
    ---------
        오늘 점심 메뉴 추천해줘.


    기대 결과
    ---------
    intent:

        unknown


    intent_source:

        openai


    prediction:

        None


    selected_route:

        fallback


    trace_status:

        fallback


    fallback_occurred:

        True
    """

    print_title(
        "[SCENARIO] "
        "Unknown question real "
        "OpenAI fallback E2E"
    )

    question = (
        "오늘 점심 메뉴 추천해줘."
    )

    print()
    print(
        "[REQUEST]"
    )
    print(
        f"question             : "
        f"{question}"
    )
    print(
        "raw_sample_provided  : "
        "false"
    )
    print(
        "chat_history_count   : "
        "0"
    )

    try:
        state = run_failure_agent_graph(
            question=question,

            # 지원하지 않는 일반 질문이므로
            # prediction 입력값은 없습니다.
            raw_sample=None,

            # prediction route가 아니므로
            # SHAP은 실행하지 않습니다.
            include_shap=False,

            # global importance도 실행하지 않습니다.
            include_global_importance=False,

            # 이번 시나리오는 single-turn unknown 질문입니다.
            chat_history=None,
        )

    except Exception as exc:
        print()
        print(
            "[ERROR] "
            "Scenario execution raised "
            "an exception"
        )
        print(
            "exception_type      : "
            f"{type(exc).__name__}"
        )
        print(
            "exception_message   : "
            f"{exc}"
        )

        return False

    print_agent_result(
        state
    )

    print_trace_event_names(
        state
    )

    failures = (
        validate_unknown_state(
            state=state,
            expected_question=question,
        )
    )

    print()
    print(
        "[SCENARIO RESULT]"
    )

    if failures:
        print(
            "result              : "
            "FAILED"
        )
        print(
            "failure_count       : "
            f"{len(failures)}"
        )

        return False

    print(
        "result              : "
        "SUCCESS"
    )
    print(
        "failure_count       : "
        "0"
    )

    return True

def run_api_prediction_scenario(
    return_response: bool = False,
) -> bool | dict[str, Any]:
    """
    실제 FastAPI endpoint를 통한
    OpenAI + LangGraph + PyTorch E2E를 실행합니다.

    실제 흐름
    ---------
    TestClient

    -> POST /agent/langgraph-query

    -> LangGraphAgentQueryRequest

    -> 실제 OpenAI

    -> LangGraph

    -> 실제 PyTorch prediction

    -> AgentState

    -> LangGraphAgentQueryResponse

    -> HTTP JSON response


    TestClient를 사용하는 이유
    --------------------------
    별도 uvicorn 서버 프로세스를 실행하지 않아도
    FastAPI route와 Pydantic request/response 검증을
    실제로 수행할 수 있습니다.

    OpenAI와 PyTorch는 mock하지 않습니다.
    """

    print_title(
        "[SCENARIO] "
        "FastAPI prediction real "
        "OpenAI + PyTorch E2E"
    )

    # FastAPI application을 사용하는
    # 테스트용 HTTP client입니다.
    #
    # 이 객체는 endpoint 함수를 직접 호출하지 않고
    # 실제 HTTP 요청과 유사한 방식으로
    # FastAPI route를 실행합니다.
    client = TestClient(
        app
    )

    request_json = {
        "question": (
            "이 설비 조건이면 "
            "고장 위험이 높아?"
        ),
        "raw_sample": {
            "air_temperature": 303.0,
            "process_temperature": 312.5,
            "rotational_speed": 1380.0,
            "torque": 62.0,
            "tool_wear": 220.0,
            "type": "L",
        },

        # 이번 FastAPI E2E는
        # OpenAI -> LangGraph -> PyTorch -> API response
        # 연결 검증이 목적입니다.
        #
        # SHAP과 global importance는
        # prediction 자체에 필수 요소가 아니므로
        # 비활성화합니다.
        "include_shap": False,
        "include_global_importance": False,

        # single-turn API 요청입니다.
        "chat_history": [],
    }

    print()
    print(
        "[HTTP REQUEST]"
    )
    print(
        "method               : POST"
    )
    print(
        "path                 : "
        "/agent/langgraph-query"
    )
    print(
        "raw_sample_provided  : true"
    )
    print(
        "chat_history_count   : 0"
    )

    try:
        response = client.post(
            "/agent/langgraph-query",
            json=request_json,
        )

    except Exception as exc:
        print()
        print(
            "[ERROR] "
            "FastAPI scenario raised "
            "an exception"
        )
        print(
            "exception_type      : "
            f"{type(exc).__name__}"
        )
        print(
            "exception_message   : "
            f"{exc}"
        )

        return False

    print()
    print(
        "[HTTP RESPONSE]"
    )
    print(
        "status_code          : "
        f"{response.status_code}"
    )

    # HTTP 200이 아니더라도
    # 가능한 경우 response body를 출력하여
    # 실패 원인을 확인할 수 있게 합니다.
    try:
        response_json = (
            response.json()
        )

    except Exception as exc:
        print(
            "response_json       : "
            "PARSE ERROR"
        )
        print(
            "response_text       : "
            f"{response.text}"
        )
        print(
            "json_error          : "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        return False

    print(
        "intent               : "
        f"{format_value(response_json.get('intent'))}"
    )
    print(
        "intent_source        : "
        f"{format_value(response_json.get('intent_source'))}"
    )
    print(
        "prediction           : "
        f"{format_value(response_json.get('prediction'))}"
    )
    print(
        "probability          : "
        f"{format_value(response_json.get('probability'))}"
    )
    print(
        "risk_level           : "
        f"{format_value(response_json.get('risk_level'))}"
    )
    print(
        "trace_status         : "
        f"{format_value(response_json.get('trace_status'))}"
    )
    print(
        "fallback_occurred    : "
        f"{format_value(response_json.get('fallback_occurred'))}"
    )

    failures: list[str] = []

    print()
    print(
        "[HTTP VALIDATION]"
    )

    add_check_result(
        condition=(
            response.status_code
            ==
            200
        ),
        check_name=(
            "HTTP status_code == 200"
        ),
        expected=200,
        actual=(
            response.status_code
        ),
        failures=failures,
    )

    response_failures = (
        validate_api_prediction_response(
            response_json=response_json,
        )
    )

    failures.extend(
        response_failures
    )

    print()
    print(
        "[SCENARIO RESULT]"
    )

    if failures:
        print(
            "result              : "
            "FAILED"
        )
        print(
            "failure_count       : "
            f"{len(failures)}"
        )

        return False

    print(
        "result              : "
        "SUCCESS"
    )
    print(
        "failure_count       : "
        "0"
    )

    # return_response=False:
    #     기존 Day 17 동작을 유지하여 True를 반환합니다.
    #
    # return_response=True:
    #     Day 18 Benchmark가 이미 검증된 FastAPI JSON 응답에서
    #     intent, route, trace 등의 품질 지표를 수집할 수 있도록
    #     response_json 전체를 반환합니다.
    if return_response:
        return response_json

    return True

def validate_openai_environment() -> bool:
    """
    실제 OpenAI E2E 실행 전에
    OPENAI_API_KEY 존재 여부를 확인합니다.

    중요
    ----
    API key 값 자체는 출력하지 않습니다.

    아래 정보만 출력합니다.

        key가 설정되어 있는가?

    API key의 일부 문자열도 로그에 남기지 않습니다.
    """

    # 프로젝트 루트의 .env 파일을 읽어
    # 환경변수로 로드합니다.
    #
    # 이미 OS 환경변수에 값이 있다면
    # 기본적으로 기존 값을 유지합니다.
    load_dotenv()

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    is_configured = (
        isinstance(
            api_key,
            str,
        )
        and
        bool(
            api_key.strip()
        )
    )

    print()
    print(
        "[ENVIRONMENT CHECK]"
    )

    if is_configured:
        print(
            "[PASS] "
            "OPENAI_API_KEY is configured"
        )
        return True

    print(
        "[FAIL] "
        "OPENAI_API_KEY is not configured"
    )
    print()
    print(
        "Check the project root .env file."
    )
    print(
        "The API key value is never printed."
    )

    return False


def parse_arguments() -> argparse.Namespace:
    """
    PowerShell 명령행 인자를 읽습니다.

    현재 첫 번째 구현에서는
    schema 시나리오만 지원합니다.

    이후 Day 17 단계에서 다음 값을 추가할 예정입니다.

        prediction

        multi_turn

        missing_sample

        unknown

        all
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Day 17 real OpenAI "
            "E2E validation."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "schema",
            "prediction",
            "multi_turn",
            "missing_sample",
            "unknown",
            "api_prediction",
            "all",
        ],
        default="all",
        help=(
            "E2E scenario to run. "
            "Supported: "
            "schema, prediction, "
            "multi_turn, missing_sample, "
            "unknown, api_prediction, all"
        ),
    )

    return parser.parse_args()


def main() -> int:
    """
    Day 17 E2E validation의 실행 진입점입니다.

    반환값
    -----
    0:
        모든 검증 성공

    1:
        E2E 시나리오 실패

    2:
        실행 환경 검사 실패

    이 숫자는 process exit code로 사용됩니다.

    PowerShell에서는 실행 후 아래 명령으로 확인할 수 있습니다.

        $LASTEXITCODE
    """

    arguments = (
        parse_arguments()
    )

    print_title(
        "DAY 17 - "
        "REAL OPENAI E2E VALIDATION"
    )

    environment_is_valid = (
        validate_openai_environment()
    )

    if not environment_is_valid:
        print()
        print(
            "[FINAL RESULT]"
        )
        print(
            "result              : "
            "ENVIRONMENT ERROR"
        )

        return 2

    scenario_results: list[
        tuple[
            str,
            bool,
        ]
    ] = []

    # schema 또는 all을 선택하면
    # Dataset schema 실제 OpenAI E2E를 실행합니다.
    if (
        arguments.scenario
        in
        {
            "schema",
            "all",
        }
    ):
        scenario_results.append(
            (
                "schema",
                run_schema_scenario(),
            )
        )

    # prediction 또는 all을 선택하면
    # 실제 OpenAI + 실제 PyTorch
    # failure prediction E2E를 실행합니다.
    if (
        arguments.scenario
        in
        {
            "prediction",
            "all",
        }
    ):
        scenario_results.append(
            (
                "prediction",
                run_prediction_scenario(),
            )
        )

    # multi_turn 또는 all을 선택하면
    # 실제 chat_history와 실제 OpenAI를 사용하는
    # multi-turn E2E를 실행합니다.
    if (
        arguments.scenario
        in
        {
            "multi_turn",
            "all",
        }
    ):
        scenario_results.append(
            (
                "multi_turn",
                run_multi_turn_scenario(),
            )
        )

    # missing_sample 또는 all을 선택하면
    # 이전 대화 문맥은 있지만
    # 현재 raw_sample은 없는
    # 안전한 fallback E2E를 실행합니다.
    if (
        arguments.scenario
        in
        {
            "missing_sample",
            "all",
        }
    ):
        scenario_results.append(
            (
                "missing_sample",
                run_missing_sample_scenario(),
            )
        )

    # unknown 또는 all을 선택하면
    # 지원하지 않는 질문에 대한
    # 실제 OpenAI unknown fallback E2E를 실행합니다.
    if (
        arguments.scenario
        in
        {
            "unknown",
            "all",
        }
    ):
        scenario_results.append(
            (
                "unknown",
                run_unknown_scenario(),
            )
        )

    # api_prediction 또는 all을 선택하면
    # FastAPI request/response 계층까지 포함한
    # 실제 OpenAI + PyTorch E2E를 실행합니다.
    if (
        arguments.scenario
        in
        {
            "api_prediction",
            "all",
        }
    ):
        scenario_results.append(
            (
                "api_prediction",
                run_api_prediction_scenario(),
            )
        )

    passed_count = sum(
        1
        for (
            _,
            passed,
        )
        in scenario_results
        if passed
    )

    failed_count = (
        len(
            scenario_results
        )
        -
        passed_count
    )

    print()
    print_divider()
    print(
        "DAY 17 E2E "
        "VALIDATION SUMMARY"
    )
    print_divider()

    print(
        "completed scenarios : "
        f"{len(scenario_results)}"
    )
    print(
        "passed scenarios    : "
        f"{passed_count}"
    )
    print(
        "failed scenarios    : "
        f"{failed_count}"
    )

    all_passed = (
        bool(
            scenario_results
        )
        and
        failed_count
        ==
        0
    )

    print(
        "result              : "
        f"{'SUCCESS' if all_passed else 'FAILED'}"
    )

    return (
        0
        if all_passed
        else 1
    )


if __name__ == "__main__":
    # main()이 반환한 정수를
    # 운영체제 process exit code로 전달합니다.
    #
    # 예:
    #
    # 성공:
    #     0
    #
    # E2E 검증 실패:
    #     1
    #
    # 환경 설정 실패:
    #     2
    #
    # sys.exit()를 사용하면
    # PowerShell의 $LASTEXITCODE에서도
    # 결과를 확인할 수 있습니다.
    sys.exit(
        main()
    )