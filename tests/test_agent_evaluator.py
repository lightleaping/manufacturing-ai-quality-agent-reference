"""
Day 21 Agent Evaluator 테스트입니다.

이 파일의 책임
--------------
src/evaluation/agent_evaluator.py가

기존 LangGraph Agent를 실행하고,

기대값과 실제값을 비교하고,

Case별 PASS / FAIL을 판정하고,

전체 평가 Summary를 계산하는지 검증합니다.


중요 원칙
---------
실제 OpenAI API를 호출하지 않습니다.

Evaluator 내부에서:

failure_agent_graph.classify_intent

를 deterministic fake classifier로 교체합니다.

고위험 Prediction Case는:

_run_failure_prediction_service

도 deterministic 결과로 교체합니다.

따라서 테스트는:

API 비용 없음

네트워크 의존 없음

실행 결과 재현 가능

조건을 유지합니다.
"""

import json

from dataclasses import replace

from src.evaluation import (
    build_day21_evaluation_cases,
    evaluate_agent_case,
    evaluate_agent_cases,
)


def _get_case(
    case_id: str,
):
    """
    Case ID로 Day 21 평가 Case 한 건을 찾습니다.

    Parameter
    ---------
    case_id:
        찾을 평가 Case의 고유 ID입니다.


    Return
    ------
    AgentEvaluationCase

        일치하는 평가 Case입니다.


    예외
    ----
    StopIteration

        존재하지 않는 ID를 전달한 경우입니다.

    현재 테스트에서는
    코드에 정의된 정상 ID만 사용합니다.
    """

    cases = (
        build_day21_evaluation_cases()
    )

    return next(
        case
        for case in cases
        if case.case_id == case_id
    )


def test_evaluate_dataset_schema_case_passes():
    """
    Dataset Schema 정상 경로를
    Evaluator가 PASS로 판정하는지 확인합니다.
    """

    case = _get_case(
        "dataset_schema_success"
    )

    result = evaluate_agent_case(
        case
    )

    assert result.passed is True

    assert (
        result.case_id
        == "dataset_schema_success"
    )

    assert (
        result.category
        == "routing"
    )

    assert (
        result.actual_output[
            "intent"
        ]
        == "dataset_schema_query"
    )

    assert (
        result.actual_output[
            "risk_level"
        ]
        is None
    )

    assert (
        result.actual_output[
            "fallback_occurred"
        ]
        is False
    )

    assert (
        result.actual_output[
            "evidence_count"
        ]
        >= 1
    )

    assert all(
        check.passed
        for check in result.checks
    )


def test_evaluate_missing_raw_sample_case_passes():
    """
    failure_prediction Intent이지만
    현재 raw_sample이 없는 경우를
    안전한 PASS 결과로 평가하는지 확인합니다.

    여기서 PASS는
    예측에 성공했다는 의미가 아닙니다.

    기대한 안전 정책:

    임의 예측 금지

    이전 입력 자동 재사용 금지

    fallback 실행

    risk_level UNKNOWN

    새 raw_sample 요청

    이 정상 동작했다는 의미입니다.
    """

    case = _get_case(
        "prediction_missing_raw_sample"
    )

    result = evaluate_agent_case(
        case
    )

    assert result.passed is True

    assert (
        result.actual_output[
            "intent"
        ]
        == "failure_prediction"
    )

    assert (
        result.actual_output[
            "prediction"
        ]
        is None
    )

    assert (
        result.actual_output[
            "probability"
        ]
        is None
    )

    assert (
        result.actual_output[
            "risk_level"
        ]
        == "UNKNOWN"
    )

    assert (
        result.actual_output[
            "fallback_occurred"
        ]
        is True
    )

    assert (
        len(
            result.actual_output[
                "errors"
            ]
        )
        == 1
    )


def test_evaluate_high_risk_prediction_case_passes():
    """
    deterministic 고위험 Prediction 결과가

    prediction

    probability

    threshold

    risk_level

    answer

    evidence

    에 일관되게 반영되는지 평가합니다.
    """

    case = _get_case(
        (
            "high_risk_"
            "prediction_consistency"
        )
    )

    result = evaluate_agent_case(
        case
    )

    assert result.passed is True

    assert (
        result.actual_output[
            "prediction"
        ]
        == 1
    )

    assert (
        result.actual_output[
            "probability"
        ]
        == 0.9929
    )

    assert (
        result.actual_output[
            "threshold"
        ]
        == 0.7
    )

    assert (
        result.actual_output[
            "risk_level"
        ]
        == "HIGH"
    )

    assert (
        result.actual_output[
            "fallback_occurred"
        ]
        is False
    )

    assert (
        result.actual_output[
            "evidence_count"
        ]
        >= 1
    )


def test_evaluate_agent_cases_returns_full_pass_summary():
    """
    Day 21 기본 평가 Case 6개를
    모두 실행했을 때
    전체 Summary가 올바른지 검증합니다.
    """

    cases = (
        build_day21_evaluation_cases()
    )

    summary = evaluate_agent_cases(
        cases
    )

    assert summary.total_count == 6

    assert summary.passed_count == 6

    assert summary.failed_count == 0

    assert summary.pass_rate == 100.0

    assert len(
        summary.results
    ) == 6

    assert all(
        result.passed
        for result in summary.results
    )


def test_evaluate_agent_cases_builds_category_summary():
    """
    영역별 평가 결과가
    올바르게 집계되는지 검증합니다.
    """

    cases = (
        build_day21_evaluation_cases()
    )

    summary = evaluate_agent_cases(
        cases
    )

    assert (
        summary.category_summary[
            "routing"
        ][
            "passed_count"
        ]
        == 1
    )

    assert (
        summary.category_summary[
            "routing"
        ][
            "total_count"
        ]
        == 1
    )

    assert (
        summary.category_summary[
            "safety"
        ][
            "passed_count"
        ]
        == 2
    )

    assert (
        summary.category_summary[
            "safety"
        ][
            "total_count"
        ]
        == 2
    )

    assert (
        summary.category_summary[
            "intent"
        ][
            "pass_rate"
        ]
        == 100.0
    )

    assert (
        summary.category_summary[
            "answer_consistency"
        ][
            "pass_rate"
        ]
        == 100.0
    )

    assert (
        summary.category_summary[
            "multi_turn"
        ][
            "pass_rate"
        ]
        == 100.0
    )


def test_evaluator_detects_incorrect_expected_intent():
    """
    Evaluator가 실제 불일치를
    FAIL로 감지하는지 검증합니다.

    왜 필요한가?
    ------------
    모든 정상 Case가 PASS하는 것만 확인하면
    Evaluator가 실제 비교 없이
    항상 PASS를 반환하는 버그를
    발견하지 못할 수 있습니다.

    따라서 정상 Dataset Schema Case의
    expected_intent만 의도적으로
    unknown으로 변경합니다.

    실제 Agent 결과:

        dataset_schema_query

    잘못된 기대값:

        unknown

    따라서 intent Check와
    Case 전체가 FAIL이어야 합니다.
    """

    original_case = _get_case(
        "dataset_schema_success"
    )

    incorrect_case = replace(
        original_case,
        expected_intent="unknown",
    )

    result = evaluate_agent_case(
        incorrect_case
    )

    assert result.passed is False

    intent_check = next(
        check
        for check in result.checks
        if check.check_name
        == "intent"
    )

    assert (
        intent_check.passed
        is False
    )

    assert (
        intent_check.expected
        == "unknown"
    )

    assert (
        intent_check.actual
        == "dataset_schema_query"
    )


def test_evaluation_summary_can_be_serialized_to_json():
    """
    전체 평가 Summary가
    JSON artifact로 저장 가능한 구조인지 검증합니다.

    Day 21 후속 단계에서는:

    reports/artifacts/
    day21_agent_evaluation.json

    파일로 평가 결과를 저장할 예정입니다.
    """

    cases = (
        build_day21_evaluation_cases()
    )

    summary = evaluate_agent_cases(
        cases
    )

    summary_dict = (
        summary.to_dict()
    )

    json_text = json.dumps(
        summary_dict,
        ensure_ascii=False,
        indent=2,
    )

    assert (
        summary_dict[
            "total_count"
        ]
        == 6
    )

    assert (
        summary_dict[
            "passed_count"
        ]
        == 6
    )

    assert (
        '"pass_rate": 100.0'
        in json_text
    )

    assert (
        "dataset_schema_success"
        in json_text
    )

    assert (
        "secret_request_safe_fallback"
        in json_text
    )