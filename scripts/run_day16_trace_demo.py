"""
Day 16 - LangGraph Trace / Observability 실행 데모

이 스크립트의 역할
-----------------
Day 16에서 구현한 내부 구조화 trace가
실제 LangGraph workflow에서 어떻게 기록되는지
콘솔에서 확인합니다.

Swagger를 열지 않고도 아래 흐름을 실행할 수 있습니다.

    사용자 질문

            │

            ▼

    run_failure_agent_graph()

            │

            ▼

    validate_question

            │

            ▼

    intent classification

            │

            ▼

    conditional routing

            │

            ▼

    prediction 또는 answer 생성

            │

            ▼

    trace finalize

            │

            ▼

    콘솔 출력


실행 시나리오
-------------
1. schema

    AI4I 데이터셋 질문

    예상 최종 상태:

        trace_status = success


2. prediction

    raw_sample이 포함된 고장 위험 예측

    예상 최종 상태:

        trace_status = success


3. missing-sample

    failure_prediction 질문이지만
    raw_sample을 제공하지 않음

    예상 최종 상태:

        trace_status = fallback


4. all

    위 세 시나리오를 순서대로 모두 실행


실행 예
-------
전체 실행:

    python -m scripts.run_day16_trace_demo


또는:

    python -m scripts.run_day16_trace_demo --scenario all


schema만 실행:

    python -m scripts.run_day16_trace_demo --scenario schema


정상 prediction만 실행:

    python -m scripts.run_day16_trace_demo --scenario prediction


raw_sample 누락 경로만 실행:

    python -m scripts.run_day16_trace_demo --scenario missing-sample


중요
----
이 스크립트는 trace 결과를 보기 위한 실행 데모입니다.

테스트 파일과 역할이 다릅니다.

pytest:

    자동 검증

    assert 실패 시 테스트 실패


이 스크립트:

    사람이 실제 trace 흐름을 읽고 확인

    포트폴리오 실행 화면과 면접 설명에 활용
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from src.agent.failure_agent_graph import (
    run_failure_agent_graph,
)
from src.agent.state import (
    AgentState,
)


# 콘솔 출력 구분선 길이입니다.
#
# 여러 시나리오를 연속 실행할 때
# 각 실행 결과를 눈으로 구분하기 쉽게 사용합니다.
DIVIDER_LENGTH = 100


# trace metadata에는 event 종류에 따라
# 서로 다른 값이 들어갑니다.
#
# 예:
#
# intent node:
#
#     intent
#
#     intent_source
#
#     confidence
#
#
# route event:
#
#     selected_route
#
#
# prediction node:
#
#     prediction_succeeded
#
#     prediction
#
#     risk_level
#
#
# 아래 tuple은 콘솔에서 우선 출력할
# 주요 metadata key의 순서를 정의합니다.
TRACE_METADATA_KEYS = (
    "question_valid",
    "question_length",
    "intent",
    "intent_source",
    "confidence",
    "selected_route",
    "raw_sample_provided",
    "prediction_succeeded",
    "prediction",
    "risk_level",
    "evidence_count",
    "warning_count",
    "error_count",
    "warnings_added",
    "errors_added",
    "answer_created",
    "has_errors",
)


def print_title(
    title: str,
) -> None:
    """
    실행 시나리오 제목을 출력합니다.

    예:

    ====================================================================================================

    [SCENARIO] Dataset schema query

    ====================================================================================================
    """

    print()
    print(
        "=" * DIVIDER_LENGTH
    )

    print(
        f"[SCENARIO] {title}"
    )

    print(
        "=" * DIVIDER_LENGTH
    )


def format_value(
    value: Any,
) -> str:
    """
    trace metadata 값을
    콘솔에서 읽기 쉬운 문자열로 변환합니다.

    None:

        null

    bool:

        true

        false

    나머지:

        str(value)


    왜 bool을 소문자로 출력하는가?
    -----------------------------
    Python은:

        True

        False

    를 사용합니다.

    JSON은:

        true

        false

    를 사용합니다.

    FastAPI JSON 응답과 비슷한 형태로 보기 위해
    콘솔에서도 소문자로 출력합니다.
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


def build_metadata_summary(
    metadata: dict[str, Any],
) -> str:
    """
    trace event metadata를
    한 줄 요약 문자열로 변환합니다.

    입력 예:

        {
            "intent": "failure_prediction",
            "intent_source": "openai",
            "confidence": 0.95,
            "warnings_added": 0,
            "errors_added": 0
        }


    출력 예:

        intent=failure_prediction,
        intent_source=openai,
        confidence=0.95,
        warnings_added=0,
        errors_added=0


    전체 metadata를 무조건 출력하지 않는 이유
    ---------------------------------------
    event마다 metadata 구조가 다르고,
    이후 확장 과정에서 값이 많아질 수 있습니다.

    실행 데모에서는 Day 16에서 중요한
    주요 key를 정해진 순서로 출력합니다.
    """

    summary_parts: list[str] = []

    for key in TRACE_METADATA_KEYS:
        if key not in metadata:
            continue

        value = metadata[
            key
        ]

        summary_parts.append(
            (
                f"{key}="
                f"{format_value(value)}"
            )
        )

    # 현재 우선 출력 목록에 없는 metadata만 존재해도
    # 빈 문자열만 반환하지 않도록
    # 전체 key 이름을 간단히 표시합니다.
    if not summary_parts and metadata:
        unknown_keys = ", ".join(
            metadata.keys()
        )

        return (
            f"metadata_keys=[{unknown_keys}]"
        )

    if not summary_parts:
        return "-"

    return ", ".join(
        summary_parts
    )


def print_trace_summary(
    state: AgentState,
) -> None:
    """
    요청 하나의 최종 결과와
    전체 trace 요약을 출력합니다.

    출력 항목
    ---------
    trace_id

    trace_status

    trace_started_at

    trace_finished_at

    trace_duration_ms

    fallback_occurred

    intent

    intent_source

    confidence

    prediction

    probability

    risk_level

    warning count

    error count
    """

    warnings = state.get(
        "warnings",
        [],
    )

    errors = state.get(
        "errors",
        [],
    )

    print()
    print(
        "[TRACE SUMMARY]"
    )

    print(
        (
            "trace_id            : "
            f"{format_value(state.get('trace_id'))}"
        )
    )

    print(
        (
            "trace_status        : "
            f"{format_value(state.get('trace_status'))}"
        )
    )

    print(
        (
            "trace_started_at    : "
            f"{format_value(state.get('trace_started_at'))}"
        )
    )

    print(
        (
            "trace_finished_at   : "
            f"{format_value(state.get('trace_finished_at'))}"
        )
    )

    print(
        (
            "trace_duration_ms   : "
            f"{format_value(state.get('trace_duration_ms'))}"
        )
    )

    print(
        (
            "fallback_occurred   : "
            f"{format_value(state.get('fallback_occurred'))}"
        )
    )

    print()
    print(
        "[AGENT RESULT]"
    )

    print(
        (
            "intent              : "
            f"{format_value(state.get('intent'))}"
        )
    )

    print(
        (
            "intent_source       : "
            f"{format_value(state.get('intent_source'))}"
        )
    )

    print(
        (
            "confidence          : "
            f"{format_value(state.get('confidence'))}"
        )
    )

    print(
        (
            "prediction          : "
            f"{format_value(state.get('prediction'))}"
        )
    )

    print(
        (
            "probability         : "
            f"{format_value(state.get('probability'))}"
        )
    )

    print(
        (
            "risk_level          : "
            f"{format_value(state.get('risk_level'))}"
        )
    )

    print(
        (
            "warning_count       : "
            f"{len(warnings)}"
        )
    )

    print(
        (
            "error_count         : "
            f"{len(errors)}"
        )
    )


def print_trace_events(
    state: AgentState,
) -> None:
    """
    trace_events를 실제 실행 순서대로 출력합니다.

    출력 예:

    seq | type  | status   | duration_ms | event_name
    -------------------------------------------------
      1 | node  | success  |       0.015 | validate_question
      2 | route | success  |       0.008 | route_after_validation


    각 event 아래에는
    주요 metadata도 함께 출력합니다.
    """

    trace_events = state.get(
        "trace_events",
        [],
    )

    print()
    print(
        "[TRACE EVENTS]"
    )

    if not trace_events:
        print(
            "trace event가 없습니다."
        )

        return

    header = (
        f"{'seq':>3} | "
        f"{'type':<5} | "
        f"{'status':<8} | "
        f"{'duration_ms':>11} | "
        f"event_name"
    )

    print(
        header
    )

    print(
        "-" * DIVIDER_LENGTH
    )

    for event in trace_events:
        sequence = event.get(
            "sequence",
            0,
        )

        event_type = event.get(
            "event_type",
            "unknown",
        )

        status = event.get(
            "status",
            "unknown",
        )

        duration_ms = event.get(
            "duration_ms",
            0.0,
        )

        event_name = event.get(
            "event_name",
            "unknown",
        )

        print(
            (
                f"{sequence:>3} | "
                f"{event_type:<5} | "
                f"{status:<8} | "
                f"{duration_ms:>11.3f} | "
                f"{event_name}"
            )
        )

        metadata = event.get(
            "metadata",
            {},
        )

        metadata_summary = (
            build_metadata_summary(
                metadata
            )
        )

        print(
            (
                "    metadata: "
                f"{metadata_summary}"
            )
        )


def print_messages(
    state: AgentState,
) -> None:
    """
    warning, error, answer를 출력합니다.

    trace만 보면 실행 흐름은 알 수 있지만,
    사용자가 실제로 어떤 응답을 받았는지도
    함께 확인해야 합니다.
    """

    warnings = state.get(
        "warnings",
        [],
    )

    errors = state.get(
        "errors",
        [],
    )

    answer = state.get(
        "answer",
        "",
    )

    if warnings:
        print()
        print(
            "[WARNINGS]"
        )

        for index, warning in enumerate(
            warnings,
            start=1,
        ):
            print(
                f"{index}. {warning}"
            )

    if errors:
        print()
        print(
            "[ERRORS]"
        )

        for index, error in enumerate(
            errors,
            start=1,
        ):
            print(
                f"{index}. {error}"
            )

    print()
    print(
        "[ANSWER]"
    )

    print(
        answer
        if answer
        else "생성된 answer가 없습니다."
    )


def run_demo_case(
    *,
    title: str,
    question: str,
    raw_sample: dict[str, Any] | None,
) -> bool:
    """
    하나의 데모 시나리오를 실행합니다.

    Returns
    -------
    bool

        True:
            Python 예외 없이 실행 완료

        False:
            workflow 실행 중
            예상하지 못한 Python 예외 발생


    왜 try/except를 사용하는가?
    --------------------------
    all 시나리오 실행 중
    첫 번째 경로에서 예상하지 못한 예외가 발생하더라도
    나머지 시나리오를 계속 확인할 수 있게 합니다.

    단, 예외를 숨기지는 않습니다.

    예외 type과 메시지를 콘솔에 출력하고
    최종 process exit code를 1로 반환합니다.
    """

    print_title(
        title
    )

    print()
    print(
        "[REQUEST]"
    )

    print(
        f"question             : {question}"
    )

    print(
        (
            "raw_sample_provided  : "
            f"{format_value(raw_sample is not None)}"
        )
    )

    try:
        state = run_failure_agent_graph(
            question=question,
            raw_sample=raw_sample,

            # trace 흐름을 빠르게 확인하기 위해
            # SHAP과 global importance 계산은 끕니다.
            #
            # Day 16의 핵심은
            # explainability 계산 자체가 아니라
            # node·route 관측성입니다.
            include_shap=False,
            include_global_importance=False,
        )

    except Exception as exc:
        print()
        print(
            "[UNEXPECTED EXCEPTION]"
        )

        print(
            (
                "exception_type      : "
                f"{type(exc).__name__}"
            )
        )

        print(
            (
                "exception_message   : "
                f"{exc}"
            )
        )

        return False

    print_trace_summary(
        state
    )

    print_trace_events(
        state
    )

    print_messages(
        state
    )

    return True


def run_schema_demo() -> bool:
    """
    AI4I dataset schema 정상 경로를 실행합니다.

    예상 event 흐름:

        validate_question

        -> route_after_validation

        -> classify_intent

        -> route_after_classification

        -> build_dataset_schema_answer
    """

    return run_demo_case(
        title=(
            "Dataset schema success path"
        ),
        question=(
            "AI4I 데이터셋의 "
            "feature와 target은 뭐야?"
        ),
        raw_sample=None,
    )


def run_prediction_demo() -> bool:
    """
    raw_sample이 포함된
    정상 failure prediction 경로를 실행합니다.

    예상 event 흐름:

        validate_question

        -> route_after_validation

        -> classify_intent

        -> route_after_classification

        -> call_failure_prediction

        -> route_after_prediction

        -> build_final_answer
    """

    raw_sample = {
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
    }

    return run_demo_case(
        title=(
            "Failure prediction success path"
        ),
        question=(
            "이 설비 조건이면 "
            "고장 위험이 높아?"
        ),
        raw_sample=raw_sample,
    )


def run_missing_sample_demo() -> bool:
    """
    failure_prediction 질문이지만
    raw_sample을 제공하지 않는 경로를 실행합니다.

    예상 event 흐름:

        validate_question

        -> route_after_validation

        -> classify_intent

        -> route_after_classification

        -> call_failure_prediction

        -> route_after_prediction

        -> build_fallback_answer


    예상 최종 trace:

        trace_status

        =

        fallback
    """

    return run_demo_case(
        title=(
            "Missing raw_sample fallback path"
        ),
        question=(
            "이 설비의 고장 위험을 "
            "예측해줘."
        ),
        raw_sample=None,
    )


def parse_arguments() -> argparse.Namespace:
    """
    command line option을 읽습니다.

    사용 예:

        --scenario all

        --scenario schema

        --scenario prediction

        --scenario missing-sample
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Day 16 LangGraph "
            "trace demonstration scenarios."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=(
            "all",
            "schema",
            "prediction",
            "missing-sample",
        ),
        default="all",
        help=(
            "Demo scenario to run. "
            "Default: all"
        ),
    )

    return parser.parse_args()


def main() -> int:
    """
    Day 16 trace demo의 시작 함수입니다.

    반환값
    -----
    0:

        선택한 모든 시나리오가
        예상하지 못한 Python 예외 없이 실행됨


    1:

        하나 이상의 시나리오에서
        예상하지 못한 Python 예외 발생


    주의
    ----
    AgentState에 errors가 존재하고
    fallback answer를 정상 생성한 경우는
    이 스크립트의 실행 실패로 보지 않습니다.

    예:

        raw_sample 없음

        -> errors 기록

        -> fallback answer 생성

        -> trace_status = fallback

    위 흐름은 의도된 Agent 동작이므로
    process exit code는 0입니다.
    """

    args = parse_arguments()

    scenario_functions = {
        "schema": run_schema_demo,
        "prediction": run_prediction_demo,
        "missing-sample": (
            run_missing_sample_demo
        ),
    }

    if args.scenario == "all":
        selected_scenarios = [
            "schema",
            "prediction",
            "missing-sample",
        ]

    else:
        selected_scenarios = [
            args.scenario
        ]

    results: list[bool] = []

    for scenario_name in selected_scenarios:
        scenario_function = (
            scenario_functions[
                scenario_name
            ]
        )

        result = (
            scenario_function()
        )

        results.append(
            result
        )

    print()
    print(
        "=" * DIVIDER_LENGTH
    )

    print(
        "[DAY 16 TRACE DEMO RESULT]"
    )

    print(
        "=" * DIVIDER_LENGTH
    )

    success_count = sum(
        results
    )

    total_count = len(
        results
    )

    print(
        (
            "completed scenarios : "
            f"{success_count}/{total_count}"
        )
    )

    if all(results):
        print(
            "result              : SUCCESS"
        )

        return 0

    print(
        "result              : FAILED"
    )

    return 1


if __name__ == "__main__":
    # Python에서 현재 파일을 직접 실행했을 때만
    # main()을 호출합니다.
    #
    # 다른 테스트 파일이 이 모듈을 import할 때는
    # 데모가 자동 실행되지 않습니다.
    sys.exit(
        main()
    )