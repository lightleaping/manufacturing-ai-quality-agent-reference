# tests/test_operational_explainer.py

"""
OpenAI 운영 해설 모듈 Unit Test입니다.

실제 OpenAI API를 호출하지 않습니다.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from src.agent.operational_explainer import (
    generate_operational_explanation_with_openai,
    validate_operational_explanation_payload,
)


def build_prediction_result() -> dict:
    """
    운영 해설 테스트용 Prediction 결과를 생성합니다.
    """

    return {
        "prediction": 1,
        "probability": 0.9929,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "recommended_action": (
            "설비 점검을 권장합니다."
        ),
        "evidence": [
            {
                "evidence_type": "rule_based",
                "title": "Tool wear 점검 신호",
                "summary": (
                    "공구 마모 시간이 높습니다."
                ),
                "feature": "Tool wear [min]",
                "value": 220.0,
                "severity": "UNKNOWN",
            }
        ],
        "warnings": [],
        "limitations": [
            "학습용 모델입니다."
        ],
    }


def install_fake_dotenv(
    monkeypatch,
) -> None:
    """
    실제 .env를 읽지 않는 Fake dotenv를 설치합니다.
    """

    fake_dotenv = SimpleNamespace(
        load_dotenv=lambda: None,
    )

    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        fake_dotenv,
    )


def install_fake_openai(
    monkeypatch,
    *,
    raw_content: str,
    captured: dict,
) -> None:
    """
    실제 API 호출 없이 OpenAI 응답을 재현합니다.
    """

    class FakeCompletions:

        def create(
            self,
            **kwargs,
        ):
            captured.update(
                kwargs,
            )

            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=raw_content,
                        )
                    )
                ]
            )

    class FakeOpenAI:

        def __init__(
            self,
            *,
            api_key: str,
        ) -> None:
            captured[
                "api_key"
            ] = api_key

            self.chat = SimpleNamespace(
                completions=(
                    FakeCompletions()
                )
            )

    fake_openai = SimpleNamespace(
        OpenAI=FakeOpenAI,
    )

    monkeypatch.setitem(
        sys.modules,
        "openai",
        fake_openai,
    )


def test_generate_operational_explanation_returns_valid_result(
    monkeypatch,
) -> None:
    """
    정상 Structured JSON 응답을 검증합니다.
    """

    install_fake_dotenv(
        monkeypatch,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    monkeypatch.setenv(
        "OPENAI_MODEL",
        "test-model",
    )

    expected_payload = {
        "summary": (
            "현재 고장 위험이 높게 예측되었습니다."
        ),
        "key_signals": [
            "공구 마모 시간을 확인하세요.",
            "토크 신호를 함께 확인하세요.",
        ],
        "recommended_checks": [
            "공구 상태를 점검하세요.",
            "운전 조건을 확인하세요.",
        ],
        "caution": (
            "SHAP는 실제 원인을 확정하지 않습니다."
        ),
    }

    captured: dict = {}

    install_fake_openai(
        monkeypatch,
        raw_content=json.dumps(
            expected_payload,
            ensure_ascii=False,
        ),
        captured=captured,
    )

    result = (
        generate_operational_explanation_with_openai(
            build_prediction_result()
        )
    )

    assert result.error is None

    assert (
        result.summary
        == expected_payload[
            "summary"
        ]
    )

    assert (
        result.key_signals
        == expected_payload[
            "key_signals"
        ]
    )

    assert (
        result.model
        == "test-model"
    )

    assert (
        captured[
            "api_key"
        ]
        == "test-key"
    )

    assert (
        captured[
            "response_format"
        ][
            "type"
        ]
        == "json_schema"
    )

    assert (
        captured[
            "temperature"
        ]
        == 0
    )


def test_generate_operational_explanation_handles_missing_api_key(
    monkeypatch,
) -> None:
    """
    API Key가 없으면 구조화 오류를 반환하는지 검증합니다.
    """

    install_fake_dotenv(
        monkeypatch,
    )

    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )

    result = (
        generate_operational_explanation_with_openai(
            build_prediction_result()
        )
    )

    assert (
        result.error
        is not None
    )

    assert (
        "missing_openai_api_key"
        in result.error
    )

    assert (
        result.summary
        == ""
    )


def test_generate_operational_explanation_handles_invalid_json(
    monkeypatch,
) -> None:
    """
    OpenAI가 잘못된 JSON을 반환해도
    예외를 외부로 던지지 않는지 검증합니다.
    """

    install_fake_dotenv(
        monkeypatch,
    )

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    captured: dict = {}

    install_fake_openai(
        monkeypatch,
        raw_content="not-json",
        captured=captured,
    )

    result = (
        generate_operational_explanation_with_openai(
            build_prediction_result()
        )
    )

    assert (
        result.error
        is not None
    )

    assert (
        result.error
        == (
            "OpenAI 운영 해설을 "
            "생성하지 못했습니다."
        )
    )

    assert (
        "JSONDecodeError"
        not in result.error
    )

    assert (
        "not-json"
        not in result.error
    )


def test_validate_operational_explanation_rejects_empty_required_content() -> None:
    """
    필수 해설 내용이 비어 있으면
    검증 오류를 반환하는지 확인합니다.
    """

    result = (
        validate_operational_explanation_payload(
            payload={
                "summary": "",
                "key_signals": [],
                "recommended_checks": [],
                "caution": "",
            },
            model="test-model",
        )
    )

    assert (
        result.error
        is not None
    )

    assert (
        "summary"
        in result.error
    )

    assert (
        "key_signals"
        in result.error
    )
