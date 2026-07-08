from src.agent.answer_builder import build_agent_answer
from src.agent.evidence_builder import build_agent_evidence


def test_build_agent_answer_contains_prediction_summary() -> None:
    prediction_result = {
        "probability": 0.993,
        "threshold": 0.7,
        "prediction": 1,
        "risk_level": "HIGH",
        "recommended_action": "설비 점검 권장",
        "evidence": [],
    }

    evidence_items = build_agent_evidence(prediction_result)

    answer = build_agent_answer(
        prediction_result=prediction_result,
        evidence_items=evidence_items,
    )

    assert "설비 고장 위험 예측 결과" in answer
    assert "0.9930" in answer
    assert "risk_level=HIGH" in answer


def test_build_agent_answer_separates_rule_and_shap_evidence() -> None:
    prediction_result = {
        "probability": 0.993,
        "threshold": 0.7,
        "prediction": 1,
        "risk_level": "HIGH",
        "recommended_action": "설비 점검 권장",
        "evidence": [
            {
                "feature": "Tool wear [min]",
                "value": 220,
                "severity": "HIGH",
                "reason": "공구 마모 시간이 높습니다.",
            }
        ],
    }

    shap_items = [
        {
            "feature": "Torque [Nm]",
            "value": 62.0,
            "contribution": 5.1592,
            "direction": "positive",
        }
    ]

    evidence_items = build_agent_evidence(
        prediction_result=prediction_result,
        shap_local_explanation=shap_items,
    )

    answer = build_agent_answer(
        prediction_result=prediction_result,
        evidence_items=evidence_items,
    )

    assert "입력값 기준 점검 신호" in answer
    assert "모델 설명 기준 SHAP 근거" in answer
    assert "Tool wear [min]" in answer
    assert "Torque [Nm]" in answer

    # 잘못된 단정 표현이 들어가지 않도록 확인합니다.
    assert "때문에 고장" not in answer
    assert "원인입니다" not in answer