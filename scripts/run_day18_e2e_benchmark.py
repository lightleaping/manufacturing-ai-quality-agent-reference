# scripts/run_day18_e2e_benchmark.py

"""
Day 18 - 실제 OpenAI E2E 반복 안정성 및 응답시간 Benchmark

이 스크립트의 목적
------------------
Day 17에서 만든 실제 OpenAI E2E 시나리오를 반복 실행하고,
각 실행의 성공 여부와 전체 실행시간을 구조화해서 저장한다.

Day 17과 Day 18의 차이
----------------------
Day 17:
    각 시나리오가 실제로 정상 동작하는지 한 번 검증한다.

Day 18:
    같은 시나리오를 여러 번 실행하고 다음 값을 측정한다.

    - 전체 실행 횟수
    - 성공 횟수
    - 실패 횟수
    - 성공률
    - 최소 응답시간
    - 최대 응답시간
    - 평균 응답시간
    - 중앙값
    - p95 참고값
    - intent 일치율
    - intent_source=openai 비율
    - route 일치율
    - trace_status 일치율
    - fallback 발생률

중요한 실행 원칙
----------------
1. 실제 OpenAI API를 호출한다.
2. 실제 네트워크 상태와 OpenAI 서비스 상태의 영향을 받는다.
3. 실제 API 비용이 발생할 수 있다.
4. 기본 pytest에는 자동 포함하지 않는다.
5. 결과는 특정 시점과 환경에서 측정한 참고값이다.
6. 이 결과를 운영 SLA로 해석하면 안 된다.

구조화 안정성 지표
------------------
schema, prediction, api_prediction 시나리오는
Day 17에서 이미 실행과 검증이 완료된 AgentState 또는
FastAPI response JSON을 선택적으로 반환받는다.

이를 통해 OpenAI를 중복 호출하지 않고 다음 값을 저장한다.

    - expected_intent
    - actual_intent
    - intent_match
    - intent_source
    - intent_source_is_openai
    - expected_route
    - actual_route
    - route_match
    - expected_trace_status
    - actual_trace_status
    - trace_status_match
    - expected_fallback_occurred
    - actual_fallback_occurred
    - fallback_match

현재 구조의 호환성
------------------
Day 17 함수는 기본 인자 없이 호출하면 기존처럼 bool을 반환한다.

Day 18에서는 핵심 시나리오만 다음 옵션을 사용한다.

    run_schema_scenario(return_state=True)
    run_prediction_scenario(return_state=True)
    run_api_prediction_scenario(return_response=True)

multi_turn, missing_sample, unknown 시나리오는
현재 기존 bool 반환 방식을 그대로 사용한다.

따라서 이 세 시나리오의 세부 안정성 지표는 null로 저장되며,
전체 시나리오 성공 여부와 응답시간은 정상적으로 측정된다.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from scripts.run_day17_e2e_openai_validation import (
    run_api_prediction_scenario,
    run_missing_sample_scenario,
    run_multi_turn_scenario,
    run_prediction_scenario,
    run_schema_scenario,
    run_unknown_scenario,
    validate_openai_environment,
)


# Day 18 Benchmark 결과가 기본으로 저장될 위치입니다.
#
# Path를 사용하면 Windows와 Linux에서 경로 구분자를
# 운영체제에 맞게 처리할 수 있습니다.
DEFAULT_OUTPUT_PATH = Path(
    "reports/artifacts/day18_e2e_benchmark.json"
)


# Benchmark에서 지원하는 단일 시나리오 이름입니다.
#
# Day 17의 CLI 시나리오 이름과 동일하게 유지해야
# Day 17과 Day 18의 용어가 달라지는 문제를 막을 수 있습니다.
SINGLE_SCENARIO_NAMES = (
    "schema",
    "prediction",
    "multi_turn",
    "missing_sample",
    "unknown",
    "api_prediction",
)


# Day 18의 기본 대표 Benchmark 묶음입니다.
#
# schema:
#     OpenAI intent 분류 + LangGraph schema 경로
#
# prediction:
#     OpenAI + LangGraph + 실제 PyTorch 예측
#
# api_prediction:
#     FastAPI TestClient부터 응답 검증까지 포함한 전체 E2E
CORE_SCENARIO_NAMES = (
    "schema",
    "prediction",
    "api_prediction",
)


# Day 17의 모든 시나리오를 실행할 때 사용하는 목록입니다.
ALL_SCENARIO_NAMES = SINGLE_SCENARIO_NAMES


# 시나리오 이름과 기존 Day 17 실행 함수를 연결하는 mapping입니다.
#
# 이 mapping은 아직 구조화 반환 옵션을 사용하지 않는
# multi_turn, missing_sample, unknown 시나리오에서 사용합니다.
#
# 핵심 시나리오인 schema, prediction, api_prediction은
# run_scenario_and_collect_result()에서 각각 구조화 반환 옵션을
# 명시적으로 사용합니다.
SCENARIO_RUNNERS: dict[str, Callable[[], bool]] = {
    "schema": run_schema_scenario,
    "prediction": run_prediction_scenario,
    "multi_turn": run_multi_turn_scenario,
    "missing_sample": run_missing_sample_scenario,
    "unknown": run_unknown_scenario,
    "api_prediction": run_api_prediction_scenario,
}


# 각 핵심 Benchmark 시나리오가 정상 실행되었을 때 기대하는
# 구조화 결과입니다.
#
# 실제 AgentState 또는 FastAPI response JSON과 비교하여
# intent, route, trace, fallback이 예상과 일치하는지 측정합니다.
#
# 현재는 Day 17에서 구조화 결과를 선택적으로 반환하도록 수정한
# 핵심 시나리오 3개만 등록합니다.
SCENARIO_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "schema": {
        "intent": "dataset_schema_query",
        "route": "dataset_schema",
        "trace_status": "success",
        "fallback_occurred": False,
    },
    "prediction": {
        "intent": "failure_prediction",
        "route": "final",
        "trace_status": "success",
        "fallback_occurred": False,
    },
    "api_prediction": {
        "intent": "failure_prediction",
        "route": "final",
        "trace_status": "success",
        "fallback_occurred": False,
    },
}


@dataclass
class BenchmarkRunResult:
    """
    반복 실행 한 번의 결과를 표현합니다.

    dataclass를 사용하는 이유
    ------------------------
    관련 데이터를 dictionary로 아무렇게나 저장하지 않고,
    하나의 명확한 구조로 관리할 수 있습니다.

    기본 실행 필드
    --------------
    scenario:
        실행한 시나리오 이름

    iteration:
        해당 시나리오의 몇 번째 반복 실행인지 표시

    success:
        Day 17 시나리오 검증과 Day 18 구조화 검증을
        모두 통과했는지 표시

    duration_ms:
        시나리오 함수 전체 실행시간

    구조화 안정성 필드
    ------------------
    expected_intent:
        해당 시나리오에서 기대한 intent

    actual_intent:
        실제 OpenAI 분류 결과

    intent_match:
        expected_intent와 actual_intent의 일치 여부

    intent_source:
        intent를 생성한 출처

    intent_source_is_openai:
        intent_source가 openai인지 여부

    expected_route:
        기대한 최종 LangGraph route

    actual_route:
        실제 선택된 최종 route

    route_match:
        expected_route와 actual_route의 일치 여부

    expected_trace_status:
        기대한 Trace 최종 상태

    actual_trace_status:
        실제 Trace 최종 상태

    trace_status_match:
        기대 상태와 실제 상태의 일치 여부

    expected_fallback_occurred:
        해당 시나리오에서 기대한 fallback 발생 여부

    actual_fallback_occurred:
        실제 fallback 발생 여부

    fallback_match:
        기대 fallback 값과 실제 값의 일치 여부

    오류 필드
    ---------
    error_type:
        예상하지 못한 예외가 발생한 경우 예외 클래스 이름

    error_message:
        예외 메시지
    """

    scenario: str
    iteration: int
    success: bool
    duration_ms: float

    expected_intent: str | None
    actual_intent: str | None
    intent_match: bool | None

    intent_source: str | None
    intent_source_is_openai: bool | None

    expected_route: str | None
    actual_route: str | None
    route_match: bool | None

    expected_trace_status: str | None
    actual_trace_status: str | None
    trace_status_match: bool | None

    expected_fallback_occurred: bool | None
    actual_fallback_occurred: bool | None
    fallback_match: bool | None

    error_type: str | None
    error_message: str | None


@dataclass
class ScenarioBenchmarkSummary:
    """
    시나리오 하나의 반복 실행 결과를 요약합니다.

    응답시간 통계는 성공한 실행만 사용합니다.

    실패한 실행을 응답시간 통계에서 제외하는 이유
    -------------------------------------------
    인증 오류나 네트워크 연결 실패는 정상 요청보다 매우 빠르게
    종료될 수 있습니다.

    이런 실패 시간을 정상 응답시간과 함께 평균내면 시스템이
    실제보다 빠르게 보이는 왜곡이 발생할 수 있습니다.

    구조화 비율 필드
    ----------------
    intent_match_rate:
        측정 가능한 실행 중 intent 일치 비율

    openai_source_rate:
        측정 가능한 실행 중 intent_source=openai 비율

    route_match_rate:
        측정 가능한 실행 중 route 일치 비율

    trace_status_match_rate:
        측정 가능한 실행 중 trace_status 일치 비율

    fallback_rate:
        측정 가능한 실행 중 실제 fallback이 발생한 비율

    fallback_match_rate:
        측정 가능한 실행 중 예상 fallback 값과
        실제 fallback 값이 일치한 비율

    구조화 결과를 아직 반환하지 않는 시나리오는
    해당 비율이 None으로 저장됩니다.
    """

    scenario: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    success_rate: float

    min_duration_ms: float | None
    max_duration_ms: float | None
    mean_duration_ms: float | None
    median_duration_ms: float | None
    p95_duration_ms: float | None

    intent_match_rate: float | None
    openai_source_rate: float | None
    route_match_rate: float | None
    trace_status_match_rate: float | None
    fallback_rate: float | None
    fallback_match_rate: float | None


def utc_now_iso() -> str:
    """
    현재 UTC 시각을 ISO 8601 문자열로 반환합니다.

    timezone.utc를 명시하는 이유
    ---------------------------
    실행 환경이 달라져도 저장된 시각의 기준을 분명하게 유지합니다.

    예:
        2026-07-10T02:30:00.123456+00:00
    """

    return datetime.now(timezone.utc).isoformat()


def validate_repeat_count(repeat: int) -> None:
    """
    반복 횟수가 1 이상인지 검증합니다.

    반복 횟수가 0이면 실제 실행이 한 번도 일어나지 않으므로
    Benchmark 결과를 만들 수 없습니다.

    Raises
    ------
    ValueError
        repeat가 1보다 작은 경우 발생
    """

    if repeat < 1:
        raise ValueError(
            "--repeat must be at least 1."
        )


def resolve_scenario_names(
    requested_scenario: str,
) -> list[str]:
    """
    CLI에서 받은 시나리오 선택값을 실제 실행 목록으로 변환합니다.

    예
    --
    requested_scenario == "prediction"
        -> ["prediction"]

    requested_scenario == "core"
        -> ["schema", "prediction", "api_prediction"]

    requested_scenario == "all"
        -> Day 17 시나리오 6개 전체
    """

    if requested_scenario == "core":
        return list(CORE_SCENARIO_NAMES)

    if requested_scenario == "all":
        return list(ALL_SCENARIO_NAMES)

    return [requested_scenario]


def calculate_percentile(
    values: list[float],
    percentile: float,
) -> float | None:
    """
    선형 보간 방식으로 percentile을 계산합니다.

    Parameters
    ----------
    values:
        percentile을 계산할 숫자 목록

    percentile:
        0.0부터 1.0 사이의 백분위 위치

        예:
            0.95 -> p95

    선형 보간이란?
    -------------
    원하는 위치가 정확한 데이터 index 사이에 있으면,
    앞뒤 값 사이를 비율에 따라 계산하는 방식입니다.

    주의
    ----
    반복 횟수가 3회 정도로 적으면 p95는 통계적으로 강한
    운영 지표가 아닙니다.

    Day 18의 p95는 통계 계산 구조를 구현하고 향후 반복 횟수를
    늘릴 수 있도록 준비하는 참고값입니다.
    """

    if not values:
        return None

    if not 0.0 <= percentile <= 1.0:
        raise ValueError(
            "percentile must be between 0.0 and 1.0."
        )

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    # 데이터가 n개일 때 index 범위는 0부터 n - 1입니다.
    #
    # 예:
    #     데이터 5개
    #     percentile 0.95
    #
    #     position = (5 - 1) * 0.95
    #              = 3.8
    position = (
        len(sorted_values) - 1
    ) * percentile

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    # position이 정확히 정수 index라면 보간할 필요가 없습니다.
    if lower_index == upper_index:
        return sorted_values[lower_index]

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    interpolation_weight = position - lower_index

    return lower_value + (
        upper_value - lower_value
    ) * interpolation_weight


def round_optional_number(
    value: float | None,
    digits: int = 2,
) -> float | None:
    """
    숫자가 있으면 반올림하고 None이면 그대로 반환합니다.

    통계 계산 대상이 없을 때는 None을 유지해야 JSON에서 null로
    저장할 수 있습니다.
    """

    if value is None:
        return None

    return round(value, digits)


def calculate_optional_boolean_rate(
    values: list[bool | None],
) -> float | None:
    """
    None을 제외한 bool 값의 True 비율을 계산합니다.

    예
    --
    [True, True, False]
        -> 2 / 3
        -> 0.6667

    [None, None]
        -> 계산 대상 없음
        -> None

    None을 제외하는 이유
    -------------------
    아직 구조화 결과 수집을 지원하지 않는 시나리오까지
    False로 계산하면 실제 품질 지표가 왜곡됩니다.

    따라서 측정 가능한 값만 사용하고,
    측정 대상이 전혀 없으면 None을 반환합니다.
    """

    measurable_values = [
        value
        for value in values
        if value is not None
    ]

    if not measurable_values:
        return None

    true_count = sum(
        1
        for value in measurable_values
        if value
    )

    return round(
        true_count / len(measurable_values),
        4,
    )


def run_scenario_and_collect_result(
    scenario_name: str,
) -> tuple[bool, dict[str, Any] | None]:
    """
    Day 17 시나리오를 실행하고 구조화 결과를 수집합니다.

    Returns
    -------
    tuple[bool, dict | None]

    첫 번째 값:
        Day 17 시나리오 성공 여부

    두 번째 값:
        구조화 결과를 받을 수 있으면 AgentState 또는
        FastAPI response JSON

        받을 수 없으면 None

    핵심 시나리오
    ------------
    schema:
        return_state=True를 사용하여 최종 AgentState를 받습니다.

    prediction:
        return_state=True를 사용하여 최종 AgentState를 받습니다.

    api_prediction:
        return_response=True를 사용하여 FastAPI response JSON을
        받습니다.

    비핵심 시나리오
    --------------
    multi_turn, missing_sample, unknown은 현재 기존 bool 반환을
    그대로 사용합니다.

    중요
    ----
    Day 17 함수는 검증에 실패하면 False를 반환합니다.

    검증에 성공하고 구조화 반환 옵션을 사용하면 dictionary 형태의
    AgentState 또는 response JSON을 반환합니다.
    """

    if scenario_name == "schema":
        scenario_result = run_schema_scenario(
            return_state=True,
        )

    elif scenario_name == "prediction":
        scenario_result = run_prediction_scenario(
            return_state=True,
        )

    elif scenario_name == "api_prediction":
        scenario_result = run_api_prediction_scenario(
            return_response=True,
        )

    else:
        scenario_runner = SCENARIO_RUNNERS[
            scenario_name
        ]

        scenario_result = scenario_runner()

    # bool은 int의 하위 타입이기도 하므로,
    # dictionary 검사보다 bool 검사를 먼저 수행합니다.
    #
    # Day 17 검증 실패:
    #     False
    #
    # 기존 bool 방식 성공:
    #     True
    if isinstance(scenario_result, bool):
        return scenario_result, None

    # AgentState와 response JSON은 dictionary처럼 .get()을
    # 사용할 수 있는 dict 객체로 반환됩니다.
    if isinstance(scenario_result, dict):
        return True, scenario_result

    # 예상하지 못한 반환형은 성공으로 처리하지 않습니다.
    raise TypeError(
        "Unsupported scenario result type: "
        f"{type(scenario_result).__name__}"
    )

def extract_actual_route(
    scenario_name: str,
    structured_result: dict[str, Any],
) -> str | None:
    """
    AgentState 또는 FastAPI 응답에서 실제 최종 route를 추출합니다.

    AgentState
    ----------
    schema와 prediction을 직접 실행하면 최종 AgentState에
    selected_route가 있으므로 해당 값을 그대로 사용합니다.

    FastAPI 응답
    ------------
    FastAPI response model에는 selected_route가 직접 포함되지 않을
    수 있습니다.

    이 경우 trace_events의 route trace 정보를 사용하거나,
    검증된 trace event 순서를 바탕으로 최종 route를 판정합니다.

    주의
    ----
    단순히 scenario_name만 보고 route를 반환하지 않습니다.

    실제 trace 결과가 prediction 정상 경로와 일치할 때만
    final로 판정합니다.
    """

    selected_route = structured_result.get(
        "selected_route"
    )

    if isinstance(selected_route, str):
        return selected_route

    trace_events = structured_result.get(
        "trace_events"
    )

    if not isinstance(trace_events, list):
        return None

    # 먼저 route trace event의 metadata에 실제 선택 route가
    # 기록되어 있는지 확인합니다.
    for trace_event in trace_events:
        if not isinstance(trace_event, dict):
            continue

        event_name = trace_event.get(
            "event_name"
        )

        metadata = trace_event.get(
            "metadata"
        )

        if not isinstance(metadata, dict):
            continue

        if event_name in {
            "route_after_classification",
            "route_after_prediction",
        }:
            metadata_route = (
                metadata.get("selected_route")
                or metadata.get("route")
                or metadata.get("result")
            )

            if isinstance(metadata_route, str):
                # route_after_classification에서는 prediction처럼
                # 중간 경로가 저장될 수 있으므로 계속 탐색합니다.
                #
                # route_after_prediction의 final/fallback이 발견되면
                # 최종 route로 사용할 수 있습니다.
                if event_name == "route_after_prediction":
                    return metadata_route

    # FastAPI response에 route 값이 직접 없고 metadata에도
    # 저장되지 않은 경우, 검증된 trace event 이름을 사용합니다.
    trace_event_names = [
        trace_event.get("event_name")
        for trace_event in trace_events
        if isinstance(trace_event, dict)
    ]

    prediction_final_trace = [
        "validate_question",
        "route_after_validation",
        "classify_intent",
        "route_after_classification",
        "call_failure_prediction",
        "route_after_prediction",
        "build_final_answer",
    ]

    prediction_fallback_trace = [
        "validate_question",
        "route_after_validation",
        "classify_intent",
        "route_after_classification",
        "call_failure_prediction",
        "route_after_prediction",
        "build_fallback_answer",
    ]

    schema_trace = [
        "validate_question",
        "route_after_validation",
        "classify_intent",
        "route_after_classification",
        "build_dataset_schema_answer",
    ]

    if trace_event_names == prediction_final_trace:
        return "final"

    if trace_event_names == prediction_fallback_trace:
        return "fallback"

    if trace_event_names == schema_trace:
        return "dataset_schema"

    return None

def extract_structured_metrics(
    scenario_name: str,
    structured_result: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    AgentState 또는 FastAPI JSON에서 안정성 지표를 추출합니다.

    구조화 결과가 없는 시나리오
    --------------------------
    모든 세부 지표를 None으로 반환합니다.

    이렇게 하면 JSON에서는 null로 저장되고,
    통계 계산 시 측정 대상에서 제외됩니다.
    """

    expectation = SCENARIO_EXPECTATIONS.get(
        scenario_name
    )

    if (
        structured_result is None
        or expectation is None
    ):
        return {
            "expected_intent": None,
            "actual_intent": None,
            "intent_match": None,
            "intent_source": None,
            "intent_source_is_openai": None,
            "expected_route": None,
            "actual_route": None,
            "route_match": None,
            "expected_trace_status": None,
            "actual_trace_status": None,
            "trace_status_match": None,
            "expected_fallback_occurred": None,
            "actual_fallback_occurred": None,
            "fallback_match": None,
        }

    expected_intent = expectation["intent"]
    actual_intent = structured_result.get(
        "intent"
    )

    intent_source = structured_result.get(
        "intent_source"
    )

    expected_route = expectation["route"]

    # AgentState에는 selected_route가 직접 있지만,
    # FastAPI 응답에는 없을 수 있으므로 trace_events까지 확인합니다.
    actual_route = extract_actual_route(
        scenario_name=scenario_name,
        structured_result=structured_result,
    )

    expected_trace_status = expectation[
        "trace_status"
    ]
    actual_trace_status = structured_result.get(
        "trace_status"
    )

    expected_fallback_occurred = expectation[
        "fallback_occurred"
    ]
    actual_fallback_occurred = structured_result.get(
        "fallback_occurred"
    )

    return {
        "expected_intent": expected_intent,
        "actual_intent": actual_intent,
        "intent_match": (
            actual_intent
            ==
            expected_intent
        ),
        "intent_source": intent_source,
        "intent_source_is_openai": (
            intent_source
            ==
            "openai"
        ),
        "expected_route": expected_route,
        "actual_route": actual_route,
        "route_match": (
            actual_route
            ==
            expected_route
        ),
        "expected_trace_status": (
            expected_trace_status
        ),
        "actual_trace_status": (
            actual_trace_status
        ),
        "trace_status_match": (
            actual_trace_status
            ==
            expected_trace_status
        ),
        "expected_fallback_occurred": (
            expected_fallback_occurred
        ),
        "actual_fallback_occurred": (
            actual_fallback_occurred
        ),
        "fallback_match": (
            actual_fallback_occurred
            ==
            expected_fallback_occurred
        ),
    }


def are_structured_metrics_successful(
    metrics: dict[str, Any],
) -> bool:
    """
    측정 가능한 구조화 지표가 모두 정상인지 확인합니다.

    구조화 결과가 없는 시나리오
    --------------------------
    intent_match가 None이면 기존 bool 성공 여부만 사용해야 하므로
    True를 반환합니다.

    구조화 결과가 있는 핵심 시나리오
    ------------------------------
    다음 값이 모두 True여야 합니다.

        intent_match
        intent_source_is_openai
        route_match
        trace_status_match
        fallback_match
    """

    if metrics["intent_match"] is None:
        return True

    return all(
        (
            metrics["intent_match"],
            metrics["intent_source_is_openai"],
            metrics["route_match"],
            metrics["trace_status_match"],
            metrics["fallback_match"],
        )
    )


def run_single_benchmark_iteration(
    scenario_name: str,
    iteration: int,
) -> BenchmarkRunResult:
    """
    시나리오를 한 번 실행하고 성공 여부와 실행시간을 반환합니다.

    실행시간 측정 범위
    ----------------
    perf_counter 시작

    -> Day 17 시나리오 함수
    -> 실제 OpenAI 호출
    -> LangGraph
    -> 필요하면 PyTorch
    -> Day 17 검증
    -> 구조화 결과 반환
    -> Day 18 구조화 지표 비교

    -> perf_counter 종료

    time.perf_counter()를 사용하는 이유
    -----------------------------------
    현재 날짜와 시각을 구하는 것이 아니라,
    두 시점 사이의 경과시간을 정밀하게 측정하기 위한
    단조 증가 고해상도 타이머이기 때문입니다.
    """

    print()
    print("-" * 100)
    print(
        f"[BENCHMARK RUN] "
        f"scenario={scenario_name}, "
        f"iteration={iteration}"
    )
    print("-" * 100)

    started_at = time.perf_counter()

    # 예외가 발생하더라도 BenchmarkRunResult를 만들 수 있도록
    # 기본 구조화 지표를 먼저 준비합니다.
    structured_metrics = (
        extract_structured_metrics(
            scenario_name=scenario_name,
            structured_result=None,
        )
    )

    try:
        (
            day17_success,
            structured_result,
        ) = run_scenario_and_collect_result(
            scenario_name=scenario_name,
        )

        structured_metrics = (
            extract_structured_metrics(
                scenario_name=scenario_name,
                structured_result=structured_result,
            )
        )

        structured_metrics_success = (
            are_structured_metrics_successful(
                structured_metrics
            )
        )

        # 최종 success는 단순히 Day 17 검증만 통과했다고
        # True가 되는 것이 아닙니다.
        #
        # 구조화 결과를 수집하는 핵심 시나리오는
        # intent, source, route, trace, fallback도 기대값과
        # 일치해야 최종 성공으로 처리합니다.
        success = (
            day17_success
            and structured_metrics_success
        )

        error_type = None
        error_message = None

    except Exception as error:
        # Day 17 함수 내부에서 처리하지 못한 예상 밖 예외가
        # Benchmark 전체 프로세스를 즉시 중단시키지 않도록
        # 실행 한 건의 실패 결과로 저장합니다.
        success = False
        error_type = type(error).__name__
        error_message = str(error)

        print(
            "[ERROR] "
            f"Benchmark iteration raised {error_type}: "
            f"{error_message}"
        )

    duration_ms = (
        time.perf_counter() - started_at
    ) * 1000.0

    result = BenchmarkRunResult(
        scenario=scenario_name,
        iteration=iteration,
        success=success,
        duration_ms=round(duration_ms, 2),

        expected_intent=(
            structured_metrics["expected_intent"]
        ),
        actual_intent=(
            structured_metrics["actual_intent"]
        ),
        intent_match=(
            structured_metrics["intent_match"]
        ),

        intent_source=(
            structured_metrics["intent_source"]
        ),
        intent_source_is_openai=(
            structured_metrics[
                "intent_source_is_openai"
            ]
        ),

        expected_route=(
            structured_metrics["expected_route"]
        ),
        actual_route=(
            structured_metrics["actual_route"]
        ),
        route_match=(
            structured_metrics["route_match"]
        ),

        expected_trace_status=(
            structured_metrics[
                "expected_trace_status"
            ]
        ),
        actual_trace_status=(
            structured_metrics[
                "actual_trace_status"
            ]
        ),
        trace_status_match=(
            structured_metrics[
                "trace_status_match"
            ]
        ),

        expected_fallback_occurred=(
            structured_metrics[
                "expected_fallback_occurred"
            ]
        ),
        actual_fallback_occurred=(
            structured_metrics[
                "actual_fallback_occurred"
            ]
        ),
        fallback_match=(
            structured_metrics["fallback_match"]
        ),

        error_type=error_type,
        error_message=error_message,
    )

    print()
    print("[BENCHMARK RUN RESULT]")
    print(
        f"scenario                    : "
        f"{result.scenario}"
    )
    print(
        f"iteration                   : "
        f"{result.iteration}"
    )
    print(
        f"success                     : "
        f"{result.success}"
    )
    print(
        f"duration_ms                 : "
        f"{result.duration_ms}"
    )
    print(
        f"expected_intent             : "
        f"{result.expected_intent}"
    )
    print(
        f"actual_intent               : "
        f"{result.actual_intent}"
    )
    print(
        f"intent_match                : "
        f"{result.intent_match}"
    )
    print(
        f"intent_source               : "
        f"{result.intent_source}"
    )
    print(
        f"intent_source_is_openai     : "
        f"{result.intent_source_is_openai}"
    )
    print(
        f"expected_route              : "
        f"{result.expected_route}"
    )
    print(
        f"actual_route                : "
        f"{result.actual_route}"
    )
    print(
        f"route_match                 : "
        f"{result.route_match}"
    )
    print(
        f"expected_trace_status       : "
        f"{result.expected_trace_status}"
    )
    print(
        f"actual_trace_status         : "
        f"{result.actual_trace_status}"
    )
    print(
        f"trace_status_match          : "
        f"{result.trace_status_match}"
    )
    print(
        f"expected_fallback_occurred  : "
        f"{result.expected_fallback_occurred}"
    )
    print(
        f"actual_fallback_occurred    : "
        f"{result.actual_fallback_occurred}"
    )
    print(
        f"fallback_match              : "
        f"{result.fallback_match}"
    )
    print(
        f"error_type                  : "
        f"{result.error_type}"
    )

    return result


def summarize_scenario_results(
    scenario_name: str,
    run_results: list[BenchmarkRunResult],
) -> ScenarioBenchmarkSummary:
    """
    특정 시나리오의 반복 결과를 통계로 요약합니다.
    """

    scenario_results = [
        result
        for result in run_results
        if result.scenario == scenario_name
    ]

    total_runs = len(scenario_results)

    successful_results = [
        result
        for result in scenario_results
        if result.success
    ]

    successful_runs = len(successful_results)
    failed_runs = total_runs - successful_runs

    success_rate = (
        successful_runs / total_runs
        if total_runs > 0
        else 0.0
    )

    successful_durations = [
        result.duration_ms
        for result in successful_results
    ]

    if successful_durations:
        min_duration_ms = min(successful_durations)
        max_duration_ms = max(successful_durations)

        # statistics.fmean은 float 평균 계산에 사용하는
        # Python 표준 라이브러리 함수입니다.
        mean_duration_ms = statistics.fmean(
            successful_durations
        )

        median_duration_ms = statistics.median(
            successful_durations
        )

        p95_duration_ms = calculate_percentile(
            values=successful_durations,
            percentile=0.95,
        )

    else:
        min_duration_ms = None
        max_duration_ms = None
        mean_duration_ms = None
        median_duration_ms = None
        p95_duration_ms = None

    intent_match_rate = (
        calculate_optional_boolean_rate(
            [
                result.intent_match
                for result in scenario_results
            ]
        )
    )

    openai_source_rate = (
        calculate_optional_boolean_rate(
            [
                result.intent_source_is_openai
                for result in scenario_results
            ]
        )
    )

    route_match_rate = (
        calculate_optional_boolean_rate(
            [
                result.route_match
                for result in scenario_results
            ]
        )
    )

    trace_status_match_rate = (
        calculate_optional_boolean_rate(
            [
                result.trace_status_match
                for result in scenario_results
            ]
        )
    )

    # fallback_rate는 fallback이 기대와 일치했는지가 아니라,
    # 실제 fallback이 얼마나 자주 발생했는지를 나타냅니다.
    fallback_rate = (
        calculate_optional_boolean_rate(
            [
                result.actual_fallback_occurred
                for result in scenario_results
            ]
        )
    )

    fallback_match_rate = (
        calculate_optional_boolean_rate(
            [
                result.fallback_match
                for result in scenario_results
            ]
        )
    )

    return ScenarioBenchmarkSummary(
        scenario=scenario_name,
        total_runs=total_runs,
        successful_runs=successful_runs,
        failed_runs=failed_runs,
        success_rate=round(success_rate, 4),

        min_duration_ms=round_optional_number(
            min_duration_ms
        ),
        max_duration_ms=round_optional_number(
            max_duration_ms
        ),
        mean_duration_ms=round_optional_number(
            mean_duration_ms
        ),
        median_duration_ms=round_optional_number(
            median_duration_ms
        ),
        p95_duration_ms=round_optional_number(
            p95_duration_ms
        ),

        intent_match_rate=intent_match_rate,
        openai_source_rate=openai_source_rate,
        route_match_rate=route_match_rate,
        trace_status_match_rate=(
            trace_status_match_rate
        ),
        fallback_rate=fallback_rate,
        fallback_match_rate=fallback_match_rate,
    )


def build_overall_summary(
    run_results: list[BenchmarkRunResult],
) -> dict[str, int | float | None]:
    """
    모든 시나리오를 합친 전체 성공률과 구조화 안정성 비율을
    계산합니다.
    """

    total_runs = len(run_results)

    successful_runs = sum(
        1
        for result in run_results
        if result.success
    )

    failed_runs = total_runs - successful_runs

    success_rate = (
        successful_runs / total_runs
        if total_runs > 0
        else 0.0
    )

    intent_match_rate = (
        calculate_optional_boolean_rate(
            [
                result.intent_match
                for result in run_results
            ]
        )
    )

    openai_source_rate = (
        calculate_optional_boolean_rate(
            [
                result.intent_source_is_openai
                for result in run_results
            ]
        )
    )

    route_match_rate = (
        calculate_optional_boolean_rate(
            [
                result.route_match
                for result in run_results
            ]
        )
    )

    trace_status_match_rate = (
        calculate_optional_boolean_rate(
            [
                result.trace_status_match
                for result in run_results
            ]
        )
    )

    fallback_rate = (
        calculate_optional_boolean_rate(
            [
                result.actual_fallback_occurred
                for result in run_results
            ]
        )
    )

    fallback_match_rate = (
        calculate_optional_boolean_rate(
            [
                result.fallback_match
                for result in run_results
            ]
        )
    )

    return {
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "failed_runs": failed_runs,
        "success_rate": round(success_rate, 4),
        "intent_match_rate": intent_match_rate,
        "openai_source_rate": openai_source_rate,
        "route_match_rate": route_match_rate,
        "trace_status_match_rate": (
            trace_status_match_rate
        ),
        "fallback_rate": fallback_rate,
        "fallback_match_rate": fallback_match_rate,
    }


def save_benchmark_artifact(
    output_path: Path,
    artifact: dict[str, object],
) -> None:
    """
    Benchmark 결과를 JSON 파일로 저장합니다.

    parents=True:
        reports/artifacts처럼 상위 폴더가 없어도 함께 생성합니다.

    exist_ok=True:
        폴더가 이미 존재해도 오류를 발생시키지 않습니다.

    ensure_ascii=False:
        한글을 \\uXXXX 형태로 바꾸지 않고 읽을 수 있게 저장합니다.

    indent=2:
        사람이 읽기 쉬운 들여쓰기 형태로 저장합니다.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_scenario_summary(
    summary: ScenarioBenchmarkSummary,
) -> None:
    """
    시나리오별 요약 통계를 콘솔에 출력합니다.
    """

    print()
    print("=" * 100)
    print(
        f"[SCENARIO BENCHMARK SUMMARY] "
        f"{summary.scenario}"
    )
    print("=" * 100)
    print(
        f"total_runs                  : "
        f"{summary.total_runs}"
    )
    print(
        f"successful_runs             : "
        f"{summary.successful_runs}"
    )
    print(
        f"failed_runs                 : "
        f"{summary.failed_runs}"
    )
    print(
        f"success_rate                : "
        f"{summary.success_rate:.4f}"
    )
    print(
        f"min_duration_ms             : "
        f"{summary.min_duration_ms}"
    )
    print(
        f"max_duration_ms             : "
        f"{summary.max_duration_ms}"
    )
    print(
        f"mean_duration_ms            : "
        f"{summary.mean_duration_ms}"
    )
    print(
        f"median_duration_ms          : "
        f"{summary.median_duration_ms}"
    )
    print(
        f"p95_duration_ms             : "
        f"{summary.p95_duration_ms}"
    )
    print(
        f"intent_match_rate           : "
        f"{summary.intent_match_rate}"
    )
    print(
        f"openai_source_rate          : "
        f"{summary.openai_source_rate}"
    )
    print(
        f"route_match_rate            : "
        f"{summary.route_match_rate}"
    )
    print(
        f"trace_status_match_rate     : "
        f"{summary.trace_status_match_rate}"
    )
    print(
        f"fallback_rate               : "
        f"{summary.fallback_rate}"
    )
    print(
        f"fallback_match_rate         : "
        f"{summary.fallback_match_rate}"
    )


def parse_arguments() -> argparse.Namespace:
    """
    PowerShell에서 전달한 CLI 옵션을 해석합니다.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Day 18 real OpenAI E2E reliability "
            "and response-time benchmark."
        )
    )

    parser.add_argument(
        "--scenario",
        choices=(
            *SINGLE_SCENARIO_NAMES,
            "core",
            "all",
        ),
        default="core",
        help=(
            "Benchmark scenario to run. "
            "'core' runs schema, prediction, and "
            "api_prediction. "
            "'all' runs all Day 17 scenarios."
        ),
    )

    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help=(
            "Number of repeated runs per scenario. "
            "Default: 3"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "JSON artifact output path. "
            "Default: "
            "reports/artifacts/day18_e2e_benchmark.json"
        ),
    )

    return parser.parse_args()


def main() -> int:
    """
    Day 18 Benchmark 전체 실행을 제어합니다.

    Process exit code
    -----------------
    0:
        모든 반복 실행 성공

    1:
        하나 이상의 반복 실행 실패

    2:
        CLI 입력 또는 OpenAI 환경 설정 문제
    """

    arguments = parse_arguments()

    try:
        validate_repeat_count(arguments.repeat)

    except ValueError as error:
        print(f"[ERROR] {error}")
        return 2

    print("=" * 100)
    print(
        "DAY 18 - REAL OPENAI E2E "
        "RELIABILITY AND PERFORMANCE BENCHMARK"
    )
    print("=" * 100)
    print()
    print("[CONFIGURATION]")
    print(f"scenario : {arguments.scenario}")
    print(f"repeat   : {arguments.repeat}")
    print(f"output   : {arguments.output}")

    # Day 17과 같은 환경 검증 기준을 재사용합니다.
    #
    # API key 값 자체는 출력하거나 JSON에 저장하지 않습니다.
    if not validate_openai_environment():
        return 2

    scenario_names = resolve_scenario_names(
        arguments.scenario
    )

    benchmark_started_at = utc_now_iso()
    benchmark_start_counter = time.perf_counter()

    run_results: list[BenchmarkRunResult] = []

    for scenario_name in scenario_names:
        for iteration in range(
            1,
            arguments.repeat + 1,
        ):
            run_result = run_single_benchmark_iteration(
                scenario_name=scenario_name,
                iteration=iteration,
            )

            run_results.append(run_result)

    benchmark_duration_ms = (
        time.perf_counter() - benchmark_start_counter
    ) * 1000.0

    benchmark_finished_at = utc_now_iso()

    scenario_summaries = [
        summarize_scenario_results(
            scenario_name=scenario_name,
            run_results=run_results,
        )
        for scenario_name in scenario_names
    ]

    for summary in scenario_summaries:
        print_scenario_summary(summary)

    overall_summary = build_overall_summary(
        run_results
    )

    artifact: dict[str, object] = {
        "benchmark_name": (
            "day18_e2e_reliability_performance"
        ),
        "generated_at": benchmark_finished_at,
        "configuration": {
            "requested_scenario": arguments.scenario,
            "resolved_scenarios": scenario_names,
            "repeat": arguments.repeat,
            "uses_real_openai": True,
            "included_in_pytest": False,
            "structured_metrics_scenarios": list(
                SCENARIO_EXPECTATIONS.keys()
            ),
        },
        "execution": {
            "started_at": benchmark_started_at,
            "finished_at": benchmark_finished_at,
            "total_duration_ms": round(
                benchmark_duration_ms,
                2,
            ),
        },
        "environment": {
            "openai_api_key_configured": True,
            "api_key_value_recorded": False,
        },
        "scenario_expectations": (
            SCENARIO_EXPECTATIONS
        ),
        "scenario_summaries": [
            asdict(summary)
            for summary in scenario_summaries
        ],
        "runs": [
            asdict(result)
            for result in run_results
        ],
        "overall_summary": overall_summary,
        "current_limitations": [
            (
                "Structured intent, route, trace, and "
                "fallback metrics are currently collected "
                "for schema, prediction, and "
                "api_prediction scenarios."
            ),
            (
                "Multi-turn, missing-sample, and unknown "
                "scenarios currently retain the original "
                "bool-only Day 17 return behavior."
            ),
            (
                "Response-time statistics include only "
                "successful runs."
            ),
            (
                "Small-sample p95 values are reference "
                "values and do not represent an SLA."
            ),
            (
                "FastAPI TestClient measurements do not "
                "represent deployed HTTP server latency."
            ),
        ],
        "disclaimer": (
            "This benchmark reflects a limited number of "
            "runs in a specific local environment and time. "
            "It must not be interpreted as production SLA "
            "performance."
        ),
    }

    try:
        save_benchmark_artifact(
            output_path=arguments.output,
            artifact=artifact,
        )

    except OSError as error:
        print(
            "[ERROR] Failed to save benchmark artifact: "
            f"{type(error).__name__}: {error}"
        )
        return 2

    print()
    print("=" * 100)
    print("[OVERALL BENCHMARK SUMMARY]")
    print("=" * 100)
    print(
        f"total_runs                  : "
        f"{overall_summary['total_runs']}"
    )
    print(
        f"successful_runs             : "
        f"{overall_summary['successful_runs']}"
    )
    print(
        f"failed_runs                 : "
        f"{overall_summary['failed_runs']}"
    )
    print(
        f"success_rate                : "
        f"{overall_summary['success_rate']:.4f}"
    )
    print(
        f"intent_match_rate           : "
        f"{overall_summary['intent_match_rate']}"
    )
    print(
        f"openai_source_rate          : "
        f"{overall_summary['openai_source_rate']}"
    )
    print(
        f"route_match_rate            : "
        f"{overall_summary['route_match_rate']}"
    )
    print(
        f"trace_status_match_rate     : "
        f"{overall_summary['trace_status_match_rate']}"
    )
    print(
        f"fallback_rate               : "
        f"{overall_summary['fallback_rate']}"
    )
    print(
        f"fallback_match_rate         : "
        f"{overall_summary['fallback_match_rate']}"
    )
    print(
        f"total_duration_ms           : "
        f"{round(benchmark_duration_ms, 2)}"
    )
    print(
        f"artifact                    : "
        f"{arguments.output}"
    )

    if overall_summary["failed_runs"] > 0:
        print(
            "result                      : "
            "FAILURE"
        )
        return 1

    print(
        "result                      : "
        "SUCCESS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())