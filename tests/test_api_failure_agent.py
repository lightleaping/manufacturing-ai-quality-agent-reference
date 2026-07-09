"""
Day 10/11 - FastAPI failure agent endpoint 테스트

이 테스트의 목적
----------------
POST /agent/failure-prediction endpoint의 request / response 구조를 검증합니다.

주의
----
이 테스트는 실제 PyTorch 모델 추론이나 실제 SHAP 계산을 수행하지 않습니다.

이유:
- API layer 테스트의 목적은 endpoint가 정상 등록되어 있고,
  request JSON을 받아 response JSON 구조를 반환하는지 확인하는 것입니다.
- 실제 모델 추론은 Day 5 테스트가 담당합니다.
- evidence builder는 Day 9 테스트가 담당합니다.
- 실제 SHAP 계산은 Day 8/Day 11의 interpretability 테스트 또는 Swagger 수동 실행에서 확인합니다.

Day 11 변경점
-------------
Day 11부터 API는 SHAP artifact를 로드하고,
include_shap=True일 때 SHAP local explanation helper를 호출합니다.

따라서 이 API 구조 테스트에서도 아래 함수들을 monkeypatch합니다.

- load_failure_model_artifacts
- predict_failure_from_artifacts
- load_shap_artifacts
- build_global_importance_items_from_map
- build_shap_local_explanation_for_sample
- build_agent_evidence
- build_agent_answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fastapi.testclient import TestClient

from src.api.main import app

from src.api.artifact_cache import clear_artifact_cache_for_tests


@dataclass
class FakePredictionResult:
    """
    Day 5 predict_failure_from_artifacts()가 반환하는 결과를 흉내 내는 fake 객체입니다.

    실제 테스트에서는 모델을 로드하지 않기 때문에,
    API response 구조를 만들 수 있을 정도의 필드만 제공합니다.
    """

    prediction: int = 1
    probability: float = 0.993
    threshold: float = 0.7
    risk_level: str = "HIGH"
    recommended_action: str = "고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다."
    evidence: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "feature": "Torque [Nm]",
                "value": 62.0,
                "reason": "입력값 기준으로 Torque가 rule 기준 점검 신호로 표시되었습니다.",
                "severity": "HIGH",
            }
        ]
    )


@dataclass
class FakeAgentEvidence:
    """
    Day 9 AgentEvidence를 흉내 내는 fake 객체입니다.

    실제 build_agent_evidence()는 AgentEvidence 또는 dict 형태의 evidence item을 반환할 수 있습니다.
    API layer 테스트에서는 response schema에 필요한 필드들이 있는지만 확인합니다.
    """

    evidence_id: str
    evidence_type: str
    source: str
    title: str
    summary: str
    feature: str | None = None
    value: float | str | None = None
    direction: str | None = None
    contribution: float | None = None
    importance: float | None = None
    severity: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeShapArtifacts:
    """
    Day 11 load_shap_artifacts()가 반환하는 ShapArtifacts를 흉내 내는 fake 객체입니다.

    실제 SHAP 계산을 하지 않으므로 background_tensor는 None으로 둡니다.
    build_shap_local_explanation_for_sample도 monkeypatch하기 때문에
    이 값이 실제 계산에 사용되지 않습니다.
    """

    background_tensor = None

    reference_values = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 100.0,
        "Type": 0.0,
    }

    global_importance_map = {
        "Torque [Nm]": 0.3309,
        "Air temperature [K]": 0.2725,
        "Rotational speed [rpm]": 0.2292,
    }


class FakeShapLocalExplanation:
    """
    SHAP local explanation 결과를 흉내 내는 fake 객체입니다.

    실제 build_agent_evidence()는 이 객체를 받아 shap_local evidence로 변환하지만,
    이 테스트에서는 build_agent_evidence()도 fake로 교체하므로
    최소 구조만 제공합니다.
    """

    summary = "SHAP local explanation fake result"
    contributions = []
    limitations = ["테스트에서는 실제 SHAP 계산을 수행하지 않습니다."]


def test_failure_prediction_agent_api_returns_expected_structure(monkeypatch):
    """
    POST /agent/failure-prediction endpoint의 응답 구조를 검증합니다.

    이 테스트는 실제 PyTorch model과 실제 SHAP 계산을 실행하지 않습니다.
    대신 monkeypatch로 외부 의존 함수를 fake 함수로 바꿉니다.

    이유:
    - API layer 테스트는 API 구조 검증이 목적입니다.
    - 모델 로딩/추론 검증은 Day 5 테스트에서 이미 담당했습니다.
    - evidence builder 검증은 Day 9 테스트에서 이미 담당했습니다.
    - SHAP 계산 검증은 Day 8/Day 11 interpretability 흐름에서 별도로 담당합니다.
    """

    def fake_load_failure_model_artifacts(artifact_dir):
        """
        실제 model.pt, scaler.joblib, metadata.json을 로드하지 않습니다.
        API layer 테스트에서는 artifact_dir이 전달되는지만 충분합니다.
        """
        return {"artifact_dir": artifact_dir}

    def fake_predict_failure_from_artifacts(artifacts, raw_sample):
        """
        실제 모델 추론 대신 고정된 prediction 결과를 반환합니다.
        """
        return FakePredictionResult()

    def fake_load_shap_artifacts(artifact_dir):
        """
        실제 shap_background.pt, shap_reference_values.json, global_importance.json을 로드하지 않습니다.
        Day 11 API 구조에서 load_shap_artifacts() 호출이 추가되었기 때문에 fake로 대체합니다.
        """
        return FakeShapArtifacts()

    def fake_build_global_importance_items_from_map(global_importance_map):
        """
        global_importance_map을 build_agent_evidence()에 넘길 수 있는 list 형태로 변환합니다.
        실제 runtime helper와 같은 역할을 단순화한 fake 함수입니다.
        """
        return [
            {
                "feature": feature,
                "importance": importance,
            }
            for feature, importance in global_importance_map.items()
        ]

    def fake_build_shap_local_explanation_for_sample(
        *,
        include_shap,
        artifacts,
        shap_artifacts,
        raw_sample,
        threshold,
        risk_level,
        top_k=5,
    ):
        """
        실제 SHAP 계산을 수행하지 않습니다.

        include_shap=False이면 실제 함수처럼 None을 반환하고,
        include_shap=True이면 fake local explanation 객체를 반환합니다.

        이 함수의 signature는 실제 build_shap_local_explanation_for_sample()과 맞춰야 합니다.
        그래야 테스트에서는 통과하지만 Swagger 실제 실행에서 깨지는 문제를 줄일 수 있습니다.
        """
        if not include_shap:
            return None

        return FakeShapLocalExplanation()

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
        evidence_items = [
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

        if shap_local_explanation is not None:
            evidence_items.append(
                FakeAgentEvidence(
                    evidence_id="shap_local_001",
                    evidence_type="shap_local",
                    source="shap",
                    title="SHAP local explanation",
                    summary=(
                        "SHAP 기준으로 Torque [Nm]는 모델의 고장 위험 logit을 "
                        "높이는 방향으로 작용했습니다."
                    ),
                    feature="Torque [Nm]",
                    value=62.0,
                    direction="positive",
                    contribution=5.1592,
                    importance=0.3309,
                    severity="HIGH",
                    metadata={
                        "note": "테스트에서는 실제 SHAP 계산을 수행하지 않습니다.",
                    },
                )
            )

        if global_importance_items:
            evidence_items.append(
                FakeAgentEvidence(
                    evidence_id="global_importance_001",
                    evidence_type="global_importance",
                    source="permutation_importance",
                    title="Global importance",
                    summary="전체 test set 기준으로 Torque [Nm] 중요도가 가장 높았습니다.",
                    feature="Torque [Nm]",
                    importance=0.3309,
                    severity="INFO",
                )
            )

        return evidence_items

    def fake_build_agent_answer(prediction_result, evidence_items):
        """
        실제 answer builder 대신 고정된 답변 문장을 반환합니다.
        """
        return (
            "모델은 현재 sample의 고장 probability를 99.30%로 예측했습니다. "
            "입력값 기준으로 Torque가 제조 rule에서 점검 신호로 표시되었습니다. "
            "SHAP 기준으로 Torque는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다."
        )

    clear_artifact_cache_for_tests()

    monkeypatch.setattr(
        "src.api.artifact_cache.load_failure_model_artifacts",
        fake_load_failure_model_artifacts,
    )
    monkeypatch.setattr(
        "src.api.failure_agent_service.predict_failure_from_artifacts",
        fake_predict_failure_from_artifacts,
    )
    monkeypatch.setattr(
        "src.api.artifact_cache.load_shap_artifacts",
        fake_load_shap_artifacts,
    )
    monkeypatch.setattr(
        "src.api.failure_agent_service.build_global_importance_items_from_map",
        fake_build_global_importance_items_from_map,
    )
    monkeypatch.setattr(
        "src.api.failure_agent_service.build_shap_local_explanation_for_sample",
        fake_build_shap_local_explanation_for_sample,
    )
    monkeypatch.setattr(
        "src.api.failure_agent_service.build_agent_evidence",
        fake_build_agent_evidence,
    )
    monkeypatch.setattr(
        "src.api.failure_agent_service.build_agent_answer",
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
    assert "answer" in data
    assert "warnings" in data
    assert "limitations" in data

    evidence_types = {
        evidence["evidence_type"]
        for evidence in data["evidence"]
    }

    assert "prediction_summary" in evidence_types
    assert "rule_based" in evidence_types
    assert "shap_local" in evidence_types
    assert "global_importance" in evidence_types

    shap_evidence = [
        evidence
        for evidence in data["evidence"]
        if evidence["evidence_type"] == "shap_local"
    ]

    assert len(shap_evidence) >= 1
    assert shap_evidence[0]["source"] == "shap"
    assert shap_evidence[0]["direction"] == "positive"


# tests/test_api_failure_agent.py 일부 추가

"""
Day 12 API 안정성 테스트

추가 테스트 목표
----------------
1. 정상 요청은 계속 성공해야 한다.
2. SHAP artifact 로드가 실패해도 API 전체는 200을 반환해야 한다.
3. SHAP 실패 시 warnings에 이유가 들어가야 한다.
4. shap_local evidence는 생략되어야 한다.
5. prediction_summary와 rule_based evidence는 유지되어야 한다.
"""

from fastapi.testclient import TestClient

from src.api.artifact_cache import clear_artifact_cache_for_tests
from src.api.main import app


client = TestClient(app)


def _sample_payload(
    *,
    include_shap: bool = True,
    include_global_importance: bool = True,
) -> dict:
    """
    테스트에서 반복해서 사용할 정상 입력 sample이다.
    """
    return {
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
        "include_shap": include_shap,
        "include_global_importance": include_global_importance,
    }


def test_failure_prediction_agent_returns_warning_when_shap_artifact_load_fails(
    monkeypatch,
) -> None:
    """
    SHAP artifact 로드 실패 테스트.

    기대 동작:
        API는 500으로 죽지 않는다.
        prediction은 반환된다.
        warnings에 SHAP artifact 실패 내용이 들어간다.
        shap_local evidence는 포함되지 않는다.

    왜 중요한가?
        SHAP는 부가 설명 기능이다.
        SHAP 실패 때문에 prediction API 전체가 실패하면 운영 안정성이 떨어진다.
    """
    clear_artifact_cache_for_tests()

    def fake_load_shap_artifacts(*args, **kwargs):
        raise FileNotFoundError("fake shap artifact missing")

    monkeypatch.setattr(
        "src.api.artifact_cache.load_shap_artifacts",
        fake_load_shap_artifacts,
    )

    response = client.post(
        "/agent/failure-prediction",
        json=_sample_payload(
            include_shap=True,
            include_global_importance=True,
        ),
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data
    assert "evidence" in data
    assert "warnings" in data

    assert len(data["warnings"]) >= 1
    assert any(
        "SHAP" in warning or "artifact" in warning
        for warning in data["warnings"]
    )

    evidence_types = {
        item["evidence_type"]
        for item in data["evidence"]
    }

    assert "prediction_summary" in evidence_types
    assert "rule_based" in evidence_types
    assert "shap_local" not in evidence_types


def test_failure_prediction_agent_skips_shap_when_include_shap_false() -> None:
    """
    include_shap=false 테스트.

    기대 동작:
        SHAP 계산을 하지 않는다.
        shap_local evidence가 없다.
        prediction은 정상 반환된다.
    """
    clear_artifact_cache_for_tests()

    response = client.post(
        "/agent/failure-prediction",
        json=_sample_payload(
            include_shap=False,
            include_global_importance=False,
        ),
    )

    assert response.status_code == 200

    data = response.json()

    evidence_types = {
        item["evidence_type"]
        for item in data["evidence"]
    }

    assert "prediction_summary" in evidence_types
    assert "rule_based" in evidence_types
    assert "shap_local" not in evidence_types
    assert "global_importance" not in evidence_types