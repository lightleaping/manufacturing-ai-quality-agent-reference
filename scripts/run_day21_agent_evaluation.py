"""
Day 21 Agent Evaluation 실행 스크립트입니다.

실행 방법
---------
프로젝트 루트에서:

    python -m scripts.run_day21_agent_evaluation


출력 파일을 직접 지정하려면:

    python -m scripts.run_day21_agent_evaluation `
        --output reports/artifacts/custom_evaluation.json


이 스크립트의 책임
------------------
1. Day 21 deterministic 평가 Case를 불러옵니다.

2. 기존 LangGraph Agent를 평가합니다.

3. Case별 PASS / FAIL을 콘솔에 출력합니다.

4. 평가 영역별 통과율을 출력합니다.

5. 전체 평가 결과를 JSON Artifact로 저장합니다.

6. 실패 Case가 있으면 종료 코드 1을 반환합니다.


전체 데이터 흐름
----------------

build_day21_evaluation_cases()

↓

AgentEvaluationCase 6개

↓

evaluate_agent_cases()

↓

기존 run_failure_agent_graph()

↓

AgentEvaluationSummary

↓

콘솔 결과 출력

↓

JSON 직렬화

↓

reports/artifacts/
day21_agent_evaluation.json


중요한 설계 원칙
----------------
기본 Day 21 평가는 실제 OpenAI API를 호출하지 않습니다.

Intent Classification은
deterministic fake classifier를 사용합니다.

고위험 Prediction 정합성 Case는
고정된 prediction service 결과를 사용합니다.

따라서 다음 특성을 유지합니다.

- API 비용 없음

- 네트워크 의존 없음

- 반복 실행 결과 재현 가능

- CI와 로컬 환경에서 동일하게 평가 가능


보안 원칙
---------
다음 값은 읽거나 JSON Artifact에 저장하지 않습니다.

- OPENAI_API_KEY 실제 값

- .env 실제 내용

- Authorization Header 실제 값

- 전체 환경 변수

- OpenAI 원본 전체 응답
"""

from __future__ import annotations

import argparse
import json

from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Sequence

from src.evaluation import (
    AgentEvaluationSummary,
    build_day21_evaluation_cases,
    evaluate_agent_cases,
)


# 기본 JSON Artifact 저장 경로입니다.
#
# 프로젝트 루트에서 실행한다는 기준으로:
#
# reports/
# └─ artifacts/
#    └─ day21_agent_evaluation.json
#
# 위치에 저장합니다.
DEFAULT_OUTPUT_PATH = Path(
    "reports/artifacts/"
    "day21_agent_evaluation.json"
)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """
    명령줄 argument를 읽습니다.

    Parameter
    ---------
    argv:
        테스트에서 별도 argument를 전달할 수 있도록
        선택적으로 받습니다.

        None이면 argparse가
        실제 터미널 argument를 읽습니다.


    Return
    ------
    argparse.Namespace

        파싱된 명령줄 argument입니다.


    현재 지원 option
    -----------------
    --output

        평가 JSON Artifact 저장 경로입니다.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Run Day 21 deterministic "
            "Agent evaluation."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path to the Day 21 "
            "evaluation JSON artifact."
        ),
    )

    return parser.parse_args(
        argv
    )


def build_artifact_payload(
    summary: AgentEvaluationSummary,
) -> dict[str, Any]:
    """
    JSON Artifact에 저장할 최종 payload를 생성합니다.

    Parameter
    ---------
    summary:
        전체 Agent 평가 결과입니다.


    Return
    ------
    dict[str, Any]

        JSON으로 직렬화 가능한
        평가 Artifact payload입니다.


    summary.to_dict()
    -----------------
    AgentEvaluationSummary 내부의:

    전체 Case 수

    통과 Case 수

    실패 Case 수

    전체 Pass Rate

    영역별 평가 결과

    Case별 상세 결과

    를 dict로 변환합니다.


    metadata를 별도로 추가하는 이유
    ------------------------------
    평가 결과만 저장하면:

    어떤 평가 방식인지

    실제 OpenAI를 사용했는지

    언제 생성했는지

    알기 어렵습니다.

    따라서 실행 조건을 metadata에
    명시적으로 기록합니다.
    """

    return {
        "metadata": {
            "day": 21,
            "evaluation_name": (
                "Agent Evaluation and Safety"
            ),
            "evaluation_mode": (
                "deterministic"
            ),

            # 실제 OpenAI API 호출을 사용하지 않았음을
            # Artifact에 명시합니다.
            "real_openai_called": False,

            # Intent 결과를 고정했음을 명시합니다.
            "intent_classifier": (
                "deterministic_fake"
            ),

            # 고위험 예측 정합성 Case에서는
            # 고정 prediction 결과를 사용합니다.
            "prediction_mode": (
                "deterministic_stub"
            ),

            # 현재 시각을 UTC ISO 8601 형식으로 저장합니다.
            #
            # 예:
            #
            # 2026-07-12T07:30:00+00:00
            "generated_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        },
        "summary": (
            summary.to_dict()
        ),
    }


def save_json_artifact(
    *,
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """
    평가 결과를 UTF-8 JSON 파일로 저장합니다.

    Parameter
    ---------
    payload:
        저장할 평가 결과입니다.

    output_path:
        JSON Artifact 저장 경로입니다.


    실행 순서
    ---------
    1. 부모 폴더 생성

    2. dict를 JSON 문자열로 변환

    3. UTF-8 파일 저장


    mkdir()
    -------
    reports/artifacts 폴더가 없어도
    자동으로 생성합니다.


    parents=True
    ------------
    중간 부모 폴더도 함께 생성합니다.


    exist_ok=True
    -------------
    폴더가 이미 있어도
    오류를 발생시키지 않습니다.


    ensure_ascii=False
    ------------------
    한국어를:

        \\uXXXX

    형태가 아니라
    사람이 읽을 수 있는 한국어로 저장합니다.


    indent=2
    --------
    JSON 파일을 사람이 읽기 쉬운
    들여쓰기 형식으로 저장합니다.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )

    output_path.write_text(
        json_text + "\n",
        encoding="utf-8",
    )


def print_evaluation_header() -> None:
    """
    Day 21 평가 시작 제목을 출력합니다.
    """

    print()
    print("=" * 100)

    print(
        "DAY 21 - "
        "DETERMINISTIC AGENT "
        "EVALUATION AND SAFETY"
    )

    print("=" * 100)


def print_overall_summary(
    summary: AgentEvaluationSummary,
) -> None:
    """
    전체 평가 결과를 출력합니다.

    출력 예
    -------
    total_count  : 6

    passed_count : 6

    failed_count : 0

    pass_rate    : 100.00%
    """

    print()
    print("[OVERALL SUMMARY]")

    print(
        "total_count  : "
        f"{summary.total_count}"
    )

    print(
        "passed_count : "
        f"{summary.passed_count}"
    )

    print(
        "failed_count : "
        f"{summary.failed_count}"
    )

    print(
        "pass_rate    : "
        f"{summary.pass_rate:.2f}%"
    )


def print_case_results(
    summary: AgentEvaluationSummary,
) -> None:
    """
    Case별 PASS / FAIL 결과를 출력합니다.

    정상 Case
    ---------
    [PASS] dataset_schema_success


    실패 Case
    ---------
    [FAIL] example_case

        check_name

        expected

        actual


    실패한 Check만 상세 출력하는 이유
    ---------------------------------
    모든 성공 Check까지 출력하면
    로그가 지나치게 길어질 수 있습니다.

    실패 원인을 빠르게 찾을 수 있도록
    실패 Check만 상세 출력합니다.
    """

    print()
    print("[CASE RESULTS]")

    for result in summary.results:
        status = (
            "PASS"
            if result.passed
            else "FAIL"
        )

        print(
            f"[{status}] "
            f"{result.case_id}"
        )

        if result.passed:
            continue

        for check in result.checks:
            if check.passed:
                continue

            print(
                "    check_name : "
                f"{check.check_name}"
            )

            print(
                "    expected   : "
                f"{check.expected}"
            )

            print(
                "    actual     : "
                f"{check.actual}"
            )

            print(
                "    message    : "
                f"{check.message}"
            )


def print_category_summary(
    summary: AgentEvaluationSummary,
) -> None:
    """
    평가 영역별 통과 결과를 출력합니다.

    예
    --
    routing

        1 / 1

        100.00%
    """

    print()
    print("[CATEGORY SUMMARY]")

    for (
        category,
        category_result,
    ) in (
        summary
        .category_summary
        .items()
    ):
        print(
            f"{category:<20}: "
            f"{category_result['passed_count']}"
            "/"
            f"{category_result['total_count']} "
            "("
            f"{category_result['pass_rate']:.2f}%"
            ")"
        )


def print_output_path(
    output_path: Path,
) -> None:
    """
    JSON Artifact 저장 위치를 출력합니다.
    """

    print()
    print("[ARTIFACT]")

    print(
        "output_path : "
        f"{output_path}"
    )


def print_final_status(
    summary: AgentEvaluationSummary,
) -> None:
    """
    최종 Day 21 평가 상태를 출력합니다.

    실패 Case가 없으면:

        PASSED

    하나라도 실패하면:

        FAILED
    """

    print()

    if summary.failed_count == 0:
        print(
            "DAY 21 AGENT EVALUATION "
            "AND SAFETY PASSED"
        )
    else:
        print(
            "DAY 21 AGENT EVALUATION "
            "AND SAFETY FAILED"
        )

    print("=" * 100)
    print()


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """
    Day 21 평가 스크립트의 main 함수입니다.

    Return
    ------
    int

        0:
            모든 평가 Case 통과

        1:
            하나 이상의 평가 Case 실패


    종료 코드를 구분하는 이유
    ------------------------
    사람이 콘솔을 읽는 것뿐 아니라
    CI에서도 평가 성공 여부를
    자동으로 판단할 수 있게 합니다.

    예:

        0

        -> CI 성공


        1

        -> CI 실패
    """

    args = parse_args(
        argv
    )

    print_evaluation_header()

    # Day 21 평가 Case 6개를 생성합니다.
    cases = (
        build_day21_evaluation_cases()
    )

    print()
    print("[CONFIGURATION]")

    print(
        "evaluation_mode : "
        "deterministic"
    )

    print(
        "real_openai     : "
        "false"
    )

    print(
        "case_count      : "
        f"{len(cases)}"
    )

    # 기존 LangGraph Agent를
    # 평가 Case별로 실행합니다.
    summary = evaluate_agent_cases(
        cases
    )

    # 평가 결과를 콘솔에 출력합니다.
    print_overall_summary(
        summary
    )

    print_case_results(
        summary
    )

    print_category_summary(
        summary
    )

    # JSON 저장용 payload를 생성합니다.
    artifact_payload = (
        build_artifact_payload(
            summary
        )
    )

    # JSON Artifact를 저장합니다.
    save_json_artifact(
        payload=artifact_payload,
        output_path=args.output,
    )

    print_output_path(
        args.output
    )

    print_final_status(
        summary
    )

    # 평가 실패 여부를
    # process 종료 코드에 반영합니다.
    if summary.failed_count > 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )