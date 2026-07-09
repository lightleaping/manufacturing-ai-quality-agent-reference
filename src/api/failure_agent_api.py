# src/api/failure_agent_api.py

"""
Failure prediction Agent API router

Day 12 수정 방향
----------------
Day 10~11에서는 endpoint 안에서
prediction, SHAP, global importance, evidence 생성 로직이 길어질 수 있었다.

Day 12에서는 endpoint를 얇게 만든다.

endpoint의 역할:
    request를 받는다.
    service 함수를 호출한다.
    response를 반환한다.

복잡한 로직은:
    src/api/failure_agent_service.py

로 분리한다.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.failure_agent_service import run_failure_prediction_agent
from src.api.schemas import FailurePredictionRequest, FailurePredictionResponse


router = APIRouter(
    prefix="/agent",
    tags=["failure-agent"],
)


@router.post(
    "/failure-prediction",
    response_model=FailurePredictionResponse,
)
def predict_failure_agent(
    request: FailurePredictionRequest,
) -> FailurePredictionResponse:
    """
    설비 고장 예측 Agent endpoint.

    입력:
        raw 제조 sample

    출력:
        prediction
        probability
        threshold
        risk_level
        recommended_action
        evidence
        answer
        warnings
        limitations

    Day 12 핵심
    -----------
    endpoint 안에서 직접 artifact loading, SHAP 계산,
    evidence 조립을 하지 않는다.

    이유:
        endpoint가 너무 길어지면 테스트와 유지보수가 어려워진다.

    대신 service 함수에 위임한다.
    """
    return run_failure_prediction_agent(request)