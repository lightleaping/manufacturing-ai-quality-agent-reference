# tests/test_api_failure_prediction_explanation.py

"""
Failure Prediction OpenAI 운영 해설 Endpoint Test입니다.

실제 OpenAI API는 호출하지 않습니다.
"""

from __future__ import annotations

from fastapi.testclient import (
    TestClient,
)

from src.api.main import app
from src.api.schemas import (
    FailurePredictionExplanationResponse,
)


client = TestClient(
    app,
)


def build_prediction_result() -> dict:
    """
    Endpoint 테스트용 기존 Prediction 결과입니다.
    """

    return {
        "prediction": 1,
        "probability": 0.9929,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "recommended_action": (
            "설비 점검을 권장합니다."
        ),
        "evidence": [],
        "answer": (
            "고장 위험이 높게 예측되었습니다."
        ),
        "warnings": [],
        "limitations": [
            "학습용 모델입니다."
        ],
    }


def test_failure_prediction_explanation_endpoint_returns_response(
    monkeypatch,
) -> None:
    """
    해설 Endpoint가 Request를 Service에 전달하고
    구조화 Response를 반환하는지 검증합니다.
    """

    captured: dict = {}

    def fake_generate(
        request,
    ):
        captured[
            "prediction"
        ] = (
            request
            .prediction_result
            .prediction
        )

        return (
            FailurePredictionExplanationResponse(
                summary=(
                    "현재 고장 위험이 높게 예측되었습니다."
                ),
                key_signals=[
                    "공구 마모 신호를 확인하세요."
                ],
                recommended_checks=[
                    "공구 상태를 우선 점검하세요."
                ],
                caution=(
                    "SHAP는 실제 원인을 확정하지 않습니다."
                ),
                source="openai",
                model="test-model",
                error=None,
            )
        )

    monkeypatch.setattr(
        (
            "src.api.failure_agent_api."
            "generate_failure_prediction_explanation"
        ),
        fake_generate,
    )

    response = client.post(
        (
            "/agent/failure-prediction/"
            "explanation"
        ),
        json={
            "prediction_result": (
                build_prediction_result()
            )
        },
    )

    assert (
        response.status_code
        == 200
    )

    payload = response.json()

    assert (
        payload[
            "source"
        ]
        == "openai"
    )

    assert (
        payload[
            "model"
        ]
        == "test-model"
    )

    assert (
        payload[
            "error"
        ]
        is None
    )

    assert (
        captured[
            "prediction"
        ]
        == 1
    )


def test_failure_prediction_explanation_endpoint_validates_request() -> None:
    """
    prediction_result가 없으면
    FastAPI Validation 오류가 발생하는지 검증합니다.
    """

    response = client.post(
        (
            "/agent/failure-prediction/"
            "explanation"
        ),
        json={},
    )

    assert (
        response.status_code
        == 422
    )
