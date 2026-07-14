# src/api/failure_explanation_service.py

"""
Failure Prediction OpenAI 운영 해설 API Service입니다.

Endpoint와 OpenAI Agent 모듈 사이를 연결합니다.

중요:
- 기존 Prediction Service를 다시 실행하지 않습니다.
- Model Artifact를 다시 로드하지 않습니다.
- SHAP를 다시 계산하지 않습니다.
- Request에 포함된 확정 Prediction 결과만 설명합니다.
"""

from __future__ import annotations

from src.agent.operational_explainer import (
    generate_operational_explanation_with_openai,
)
from src.api.schemas import (
    FailurePredictionExplanationRequest,
    FailurePredictionExplanationResponse,
)


def generate_failure_prediction_explanation(
    request: FailurePredictionExplanationRequest,
) -> FailurePredictionExplanationResponse:
    """
    확정된 Prediction 결과에 대한
    OpenAI 운영 해설을 생성합니다.
    """

    prediction_response = (
        request.prediction_result
    )

    if hasattr(
        prediction_response,
        "model_dump",
    ):
        prediction_dict = (
            prediction_response.model_dump(
                mode="json",
            )
        )

    else:
        prediction_dict = (
            prediction_response.dict()
        )

    explanation_result = (
        generate_operational_explanation_with_openai(
            prediction_dict,
        )
    )

    return (
        FailurePredictionExplanationResponse(
            **explanation_result.to_dict()
        )
    )
