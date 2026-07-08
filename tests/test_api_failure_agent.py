# tests/test_api_failure_agent.py

from dataclasses import dataclass, field


# FastAPI의 TestClient는 실제 서버를 켜지 않고도
# API endpoint를 테스트할 수 있게 해주는 도구입니다.
#
# 예:
# client.post("/agent/failure-prediction", json={...})
#
# 내부적으로는 FastAPI가 직접 HTTP 요청을 보내는 것이 아니라,
# Starlette의 TestClient를 사용합니다.
#
# 현재 설치된 Starlette 버전에서는 TestClient가 httpx2 패키지를 필요로 합니다.
# 그런데 .venv에 httpx2가 설치되어 있지 않으면,
# 테스트 실행 전에 import 단계에서 RuntimeError가 발생합니다.
#
# 해결:
# python -m pip install httpx2
#
# 그리고 requirements.txt에도 아래 줄을 추가합니다.
# httpx2
from fastapi.testclient import TestClient

from src.api.main import app


@dataclass
class FakePredictionResult:
    """
    Day 5 predict_failure_from_artifacts가 반환한다고 가정한 fake 결과입니다.

    API 테스트의 목적은 실제 모델 성능 검증이 아닙니다.

    여기서는 endpoint가
    prediction, probability, threshold, risk_level, evidence, answer를
    올바른 JSON 구조로 반환하는지 확인합니다.
    """

    prediction: int = 1
    probability: float = 0.993
    threshold: float = 0.7
    risk_level: str = "HIGH"
    recommended_action: str = "고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다."
    evidence: list[dict] = field(
        default_factory=lambda: [
            {
                "feature": "Torque [Nm]",
                "value": 62.0,
                "severity": "HIGH",
                "reason": "Torque가 rule 기준 점검 구간에 있습니다.",
            }
        ]
    )


@dataclass
class FakeAgentEvidence:
    """
    Day 9 AgentEvidence가 반환한다고 가정한 fake evidence입니다.
    """

    evidence_id: str
    evidence_type: str
    source: str
    title: str
    summary: str
    feature: str | None = None
    value: float | None = None
    direction: str | None = None
    contribution: float | None = None
    importance: float | None = None
    severity: str | None = None
    metadata: dict = field(default_factory=dict)


def test_failure_prediction_agent_api_returns_expected_structure(monkeypatch):
    """
    POST /agent/failure-prediction endpoint의 응답 구조를 검증합니다.

    이 테스트는 실제 PyTorch model을 로드하지 않습니다.
    대신 monkeypatch로 기존 함수들을 fake 함수로 바꿉니다.

    이유:
    - API layer 테스트는 API 구조 검증이 목적입니다.
    - 모델 로딩/추론 검증은 Day 5 테스트에서 이미 담당했습니다.
    - evidence builder 검증은 Day 9 테스트에서 이미 담당했습니다.
    """

    def fake_load_failure_model_artifacts(artifact_dir):
        return {"artifact_dir": artifact_dir}

    def fake_predict_failure_from_artifacts(artifacts, raw_sample):
        return FakePredictionResult()

    def fake_build_agent_evidence(
        prediction_result,
        shap_local_explanation=None,
        global_importance_items=None,
        shap_top_n=5,
    ):
        """
        실제 build_agent_evidence() 함수 signature에 맞춘 fake 함수입니다.

        실제 함수 정의:

        build_agent_evidence(
            prediction_result,
            shap_local_explanation=None,
            global_importance_items=None,
            shap_top_n=5,
        )

        테스트 fake 함수의 parameter 이름이 실제 함수와 다르면,
        테스트에서는 통과하지만 Swagger 실제 실행에서 실패할 수 있습니다.
        """
        return [
            FakeAgentEvidence(
                evidence_id="prediction_summary_001",
                evidence_type="prediction_summary",
                source="model_prediction",
                title="모델 예측 요약",
                summary="모델은 현재 sample의 고장 probability를 0.9930으로 예측했습니다.",
                severity="HIGH",
                metadata={
                    "prediction": 1,
                    "probability": 0.993,
                    "threshold": 0.7,
                },
            ),
            FakeAgentEvidence(
                evidence_id="rule_based_001",
                evidence_type="rule_based",
                source="rule_engine",
                title="Torque 점검 신호",
                summary="입력값 기준으로 Torque가 rule 기준 점검 신호로 표시되었습니다.",
                feature="Torque [Nm]",
                value=62.0,
                severity="HIGH",
            ),
        ]

    def fake_build_agent_answer(prediction_result, evidence_items):
        return (
            "모델은 현재 sample의 고장 probability를 99.30%로 예측했습니다. "
            "입력값 기준으로 Torque가 제조 rule에서 점검 신호로 표시되었습니다."
        )

    monkeypatch.setattr(
        "src.api.failure_agent_api.load_failure_model_artifacts",
        fake_load_failure_model_artifacts,
    )
    monkeypatch.setattr(
        "src.api.failure_agent_api.predict_failure_from_artifacts",
        fake_predict_failure_from_artifacts,
    )
    monkeypatch.setattr(
        "src.api.failure_agent_api.build_agent_evidence",
        fake_build_agent_evidence,
    )
    monkeypatch.setattr(
        "src.api.failure_agent_api.build_agent_answer",
        fake_build_agent_answer,
    )

    client = TestClient(app)

    response = client.post(
        "/agent/failure-prediction",
        json={
            "air_temperature": 303.0,
            "process_temperature": 312.5,
            "rotational_speed": 1380.0,
            "torque": 62.0,
            "tool_wear": 220.0,
            "type": "L",
            "include_shap": True,
            "include_global_importance": True,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] == 1
    assert data["probability"] == 0.993
    assert data["threshold"] == 0.7
    assert data["risk_level"] == "HIGH"
    assert "recommended_action" in data

    assert "evidence" in data
    assert isinstance(data["evidence"], list)
    assert len(data["evidence"]) >= 1

    evidence_types = {item["evidence_type"] for item in data["evidence"]}

    assert "prediction_summary" in evidence_types
    assert "rule_based" in evidence_types

    assert "answer" in data
    assert "Torque" in data["answer"]

    assert "warnings" in data
    assert "limitations" in data