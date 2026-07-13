from src.agent.evidence_builder import (
    build_agent_evidence,
    build_prediction_summary_evidence,
    convert_rule_based_evidence,
    convert_shap_local_evidence,
    filter_evidence_by_type,
    group_evidence_by_type,
)


def test_build_prediction_summary_evidence() -> None:
    prediction_result = {
        "probability": 0.993,
        "threshold": 0.7,
        "prediction": 1,
        "risk_level": "HIGH",
        "recommended_action": "설비 점검 권장",
    }

    evidence = build_prediction_summary_evidence(prediction_result)

    assert evidence.evidence_type == "prediction_summary"
    assert evidence.source == "model_prediction"
    assert evidence.severity == "HIGH"
    assert evidence.metadata["probability"] == 0.993
    assert evidence.metadata["threshold"] == 0.7
    assert evidence.metadata["prediction"] == 1


def test_convert_rule_based_evidence() -> None:
    rule_items = [
        {
            "feature": "Tool wear [min]",
            "value": 220,
            "severity": "HIGH",
            "reason": "공구 마모 시간이 높습니다.",
        }
    ]

    evidence_items = convert_rule_based_evidence(rule_items)

    assert len(evidence_items) == 1
    assert evidence_items[0].evidence_type == "rule_based"
    assert evidence_items[0].source == "rule_engine"
    assert evidence_items[0].feature == "Tool wear [min]"
    assert evidence_items[0].severity == "HIGH"


def test_convert_shap_local_evidence_from_list() -> None:
    shap_items = [
        {
            "feature": "Torque [Nm]",
            "value": 62.0,
            "contribution": 5.1592,
            "direction": "positive",
            "reference_value": 40.0,
            "global_importance": 0.3309,
            "reason": "Torque가 모델 출력에 큰 양의 기여를 했습니다.",
        }
    ]

    evidence_items = convert_shap_local_evidence(shap_items)

    assert len(evidence_items) == 1
    assert evidence_items[0].evidence_type == "shap_local"
    assert evidence_items[0].source == "shap"
    assert evidence_items[0].feature == "Torque [Nm]"
    assert evidence_items[0].direction == "positive"
    assert evidence_items[0].contribution == 5.1592


def test_build_agent_evidence_keeps_sources_separated() -> None:
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

    prediction_evidence = filter_evidence_by_type(
        evidence_items,
        "prediction_summary",
    )
    rule_evidence = filter_evidence_by_type(
        evidence_items,
        "rule_based",
    )
    shap_evidence = filter_evidence_by_type(
        evidence_items,
        "shap_local",
    )

    assert len(prediction_evidence) == 1
    assert len(rule_evidence) == 1
    assert len(shap_evidence) == 1

    assert rule_evidence[0]["source"] == "rule_engine"
    assert shap_evidence[0]["source"] == "shap"


def test_group_evidence_by_type() -> None:
    prediction_result = {
        "probability": 0.9,
        "threshold": 0.7,
        "prediction": 1,
        "risk_level": "HIGH",
        "evidence": [],
    }

    evidence_items = build_agent_evidence(prediction_result)
    grouped = group_evidence_by_type(evidence_items)

    assert "prediction_summary" in grouped
    assert len(grouped["prediction_summary"]) == 1


def test_build_prediction_summary_evidence_normalizes_nan_probability() -> None:
    prediction_result = {
        "probability": float("nan"),
        "threshold": 0.7,
        "prediction": 1,
        "risk_level": "HIGH",
        "recommended_action": "설비 점검 권장",
    }

    evidence = build_prediction_summary_evidence(prediction_result)

    assert "nan" not in evidence.summary.lower()
    assert evidence.metadata["probability"] == 0.0


def test_build_prediction_summary_evidence_normalizes_none_prediction() -> None:
    prediction_result = {
        "probability": 0.5,
        "threshold": 0.7,
        "prediction": None,
        "risk_level": "MEDIUM",
        "recommended_action": "추가 점검 권장",
    }

    evidence = build_prediction_summary_evidence(prediction_result)

    assert "prediction=0" in evidence.summary
    assert evidence.metadata["prediction"] == 0


def test_build_prediction_summary_evidence_normalizes_none_risk_level() -> None:
    prediction_result = {
        "probability": 0.5,
        "threshold": 0.7,
        "prediction": 0,
        "risk_level": None,
        "recommended_action": "추가 확인 권장",
    }

    evidence = build_prediction_summary_evidence(prediction_result)

    assert "risk_level=UNKNOWN" in evidence.summary
    assert evidence.severity == "UNKNOWN"
    assert evidence.metadata["risk_level"] == "UNKNOWN"


def test_convert_shap_local_evidence_supports_increases_risk_direction() -> None:
    shap_items = [
        {
            "feature": "Torque [Nm]",
            "value": 62.0,
            "contribution": 5.1592,
            "direction": "increases_risk",
        }
    ]

    evidence_items = convert_shap_local_evidence(shap_items)

    assert (
        "모델의 고장 위험 logit을 높이는 방향"
        in evidence_items[0].summary
    )


def test_convert_shap_local_evidence_supports_decreases_risk_direction() -> None:
    shap_items = [
        {
            "feature": "Rotational speed [rpm]",
            "value": 1550.0,
            "contribution": -2.3145,
            "direction": "decreases_risk",
        }
    ]

    evidence_items = convert_shap_local_evidence(shap_items)

    assert (
        "모델의 고장 위험 logit을 낮추는 방향"
        in evidence_items[0].summary
    )


def test_convert_shap_local_evidence_keeps_missing_global_importance_as_none() -> None:
    shap_items = [
        {
            "feature": "Torque [Nm]",
            "value": 62.0,
            "contribution": 5.1592,
            "direction": "positive",
        }
    ]

    evidence_items = convert_shap_local_evidence(shap_items)

    assert evidence_items[0].importance is None
