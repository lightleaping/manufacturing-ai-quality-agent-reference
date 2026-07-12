"""
Day 21 Agent Evaluation 실행 스크립트 테스트입니다.

테스트 대상
-----------
scripts/run_day21_agent_evaluation.py


이 파일의 책임
--------------
Day 21 평가 실행 스크립트가 다음 기능을
정상적으로 수행하는지 검증합니다.

1. 기본 JSON Artifact 경로 사용

2. 사용자 지정 output 경로 처리

3. Artifact metadata 생성

4. UTF-8 JSON 파일 저장

5. 전체 평가 성공 시 종료 코드 0 반환

6. 평가 실패 시 종료 코드 1 반환


기존 Evaluator 테스트와의 차이
------------------------------
tests/test_agent_evaluator.py

    평가 Case 실행

    기대값과 실제값 비교

    PASS / FAIL 판정

    전체 Pass Rate 계산


현재 파일

    명령줄 argument 처리

    JSON Artifact 생성

    파일 저장

    콘솔 출력

    process 종료 코드


즉 Evaluator의 품질 판정 로직과
실행 Script의 입출력 책임을
서로 분리하여 테스트합니다.


보안 원칙
---------
테스트에서도 다음 값을 읽지 않습니다.

- 실제 OPENAI_API_KEY

- .env 실제 내용

- 실제 Authorization Header

- 전체 환경 변수

- OpenAI 원본 전체 응답
"""

from __future__ import annotations

import json

from pathlib import Path

from scripts import (
    run_day21_agent_evaluation
    as day21_script,
)
from src.evaluation import (
    AgentEvaluationSummary,
)


def test_parse_args_uses_default_output_path():
    """
    --output argument를 전달하지 않으면
    Day 21 기본 Artifact 경로를 사용해야 합니다.

    기대 경로
    ---------
    reports/
    └─ artifacts/
       └─ day21_agent_evaluation.json
    """

    args = day21_script.parse_args(
        []
    )

    assert (
        args.output
        == Path(
            "reports/artifacts/"
            "day21_agent_evaluation.json"
        )
    )


def test_parse_args_accepts_custom_output_path(
    tmp_path,
):
    """
    사용자가 --output을 전달하면
    해당 경로를 사용해야 합니다.

    tmp_path
    --------
    pytest가 테스트마다 생성하는
    임시 폴더입니다.

    실제 reports 폴더를 수정하지 않고
    안전하게 파일 경로를 테스트할 수 있습니다.
    """

    custom_output_path = (
        tmp_path
        / "custom"
        / "day21_result.json"
    )

    args = day21_script.parse_args(
        [
            "--output",
            str(
                custom_output_path
            ),
        ]
    )

    assert (
        args.output
        == custom_output_path
    )


def test_build_artifact_payload_contains_expected_metadata():
    """
    Artifact payload에
    Day 21 실행 조건과 평가 Summary가
    정상적으로 포함되는지 확인합니다.
    """

    summary = AgentEvaluationSummary(
        total_count=6,
        passed_count=6,
        failed_count=0,
        pass_rate=100.0,
        category_summary={
            "safety": {
                "total_count": 2,
                "passed_count": 2,
                "failed_count": 0,
                "pass_rate": 100.0,
            },
        },
        results=(),
    )

    payload = (
        day21_script
        .build_artifact_payload(
            summary
        )
    )

    metadata = payload[
        "metadata"
    ]

    saved_summary = payload[
        "summary"
    ]

    assert metadata[
        "day"
    ] == 21

    assert (
        metadata[
            "evaluation_name"
        ]
        == (
            "Agent Evaluation "
            "and Safety"
        )
    )

    assert (
        metadata[
            "evaluation_mode"
        ]
        == "deterministic"
    )

    assert (
        metadata[
            "real_openai_called"
        ]
        is False
    )

    assert (
        metadata[
            "intent_classifier"
        ]
        == "deterministic_fake"
    )

    assert (
        metadata[
            "prediction_mode"
        ]
        == "deterministic_stub"
    )

    # generated_at은
    # 실행 시점마다 달라지므로
    # 특정 문자열과 비교하지 않습니다.
    #
    # ISO 8601 시각 문자열이
    # 존재하는지만 확인합니다.
    assert isinstance(
        metadata[
            "generated_at"
        ],
        str,
    )

    assert (
        metadata[
            "generated_at"
        ]
    )

    assert (
        saved_summary[
            "total_count"
        ]
        == 6
    )

    assert (
        saved_summary[
            "passed_count"
        ]
        == 6
    )

    assert (
        saved_summary[
            "failed_count"
        ]
        == 0
    )

    assert (
        saved_summary[
            "pass_rate"
        ]
        == 100.0
    )


def test_save_json_artifact_creates_utf8_file(
    tmp_path,
):
    """
    save_json_artifact()가

    부모 폴더 생성

    UTF-8 JSON 저장

    한국어 유지

    를 정상적으로 수행하는지 확인합니다.
    """

    output_path = (
        tmp_path
        / "nested"
        / "artifacts"
        / "evaluation.json"
    )

    payload = {
        "message": (
            "Day 21 평가 통과"
        ),
        "passed": True,
    }

    day21_script.save_json_artifact(
        payload=payload,
        output_path=output_path,
    )

    # 부모 폴더가 처음에는 없어도
    # 함수가 자동 생성해야 합니다.
    assert (
        output_path.exists()
        is True
    )

    json_text = (
        output_path.read_text(
            encoding="utf-8"
        )
    )

    # ensure_ascii=False이므로
    # 한국어가 \\uXXXX 형태가 아니라
    # 사람이 읽을 수 있는 형태로
    # 저장되어야 합니다.
    assert (
        "Day 21 평가 통과"
        in json_text
    )

    loaded_payload = json.loads(
        json_text
    )

    assert (
        loaded_payload
        == payload
    )


def test_main_returns_zero_and_saves_artifact(
    tmp_path,
    capsys,
):
    """
    실제 deterministic Day 21 평가를
    실행했을 때:

    6개 Case 통과

    JSON Artifact 생성

    종료 코드 0

    최종 PASSED 출력

    조건을 만족하는지 확인합니다.


    실제 OpenAI 호출 여부
    ---------------------
    evaluate_agent_cases() 내부에서
    Intent Classifier를 deterministic 함수로
    교체하므로 실제 OpenAI API는 호출하지 않습니다.
    """

    output_path = (
        tmp_path
        / "day21_success.json"
    )

    exit_code = day21_script.main(
        [
            "--output",
            str(
                output_path
            ),
        ]
    )

    captured = (
        capsys.readouterr()
    )

    assert (
        exit_code
        == 0
    )

    assert (
        output_path.exists()
        is True
    )

    assert (
        (
            "DAY 21 AGENT "
            "EVALUATION AND "
            "SAFETY PASSED"
        )
        in captured.out
    )

    artifact = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    summary = artifact[
        "summary"
    ]

    assert (
        summary[
            "total_count"
        ]
        == 6
    )

    assert (
        summary[
            "passed_count"
        ]
        == 6
    )

    assert (
        summary[
            "failed_count"
        ]
        == 0
    )

    assert (
        summary[
            "pass_rate"
        ]
        == 100.0
    )


def test_main_returns_one_when_evaluation_fails(
    tmp_path,
    monkeypatch,
    capsys,
):
    """
    평가 Case가 하나라도 실패하면:

    종료 코드 1

    최종 FAILED 출력

    JSON Artifact 저장

    조건을 만족하는지 검증합니다.


    왜 실패 경로도 테스트하는가?
    ---------------------------
    성공 경로만 테스트하면
    CI가 평가 실패를 감지하지 못하는
    문제를 발견할 수 없습니다.

    따라서 evaluate_agent_cases()만
    실패 Summary를 반환하도록 교체합니다.
    """

    failed_summary = (
        AgentEvaluationSummary(
            total_count=6,
            passed_count=5,
            failed_count=1,
            pass_rate=83.33,
            category_summary={
                "safety": {
                    "total_count": 2,
                    "passed_count": 1,
                    "failed_count": 1,
                    "pass_rate": 50.0,
                },
            },
            results=(),
        )
    )

    def fake_evaluate_agent_cases(
        cases,
    ):
        """
        실제 Agent를 실행하지 않고
        의도적으로 실패 Summary를 반환합니다.

        cases parameter를 받는 이유
        --------------------------
        기존 evaluate_agent_cases()와
        함수 interface를 맞추기 위해서입니다.
        """

        return failed_summary

    monkeypatch.setattr(
        day21_script,
        "evaluate_agent_cases",
        fake_evaluate_agent_cases,
    )

    output_path = (
        tmp_path
        / "day21_failed.json"
    )

    exit_code = day21_script.main(
        [
            "--output",
            str(
                output_path
            ),
        ]
    )

    captured = (
        capsys.readouterr()
    )

    assert (
        exit_code
        == 1
    )

    assert (
        output_path.exists()
        is True
    )

    assert (
        (
            "DAY 21 AGENT "
            "EVALUATION AND "
            "SAFETY FAILED"
        )
        in captured.out
    )

    artifact = json.loads(
        output_path.read_text(
            encoding="utf-8"
        )
    )

    summary = artifact[
        "summary"
    ]

    assert (
        summary[
            "passed_count"
        ]
        == 5
    )

    assert (
        summary[
            "failed_count"
        ]
        == 1
    )

    assert (
        summary[
            "pass_rate"
        ]
        == 83.33
    )