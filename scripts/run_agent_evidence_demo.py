import json

from src.agent.answer_builder import build_agent_answer
from src.agent.evidence_builder import build_agent_evidence


def main() -> None:
    """
    Day 9 Agent evidence 통합 demo script입니다.

    실제 프로젝트에서는 prediction_result는 Day 5의
    predict_failure_from_artifacts 함수 결과를 사용하고,

    shap_local_explanation은 Day 8의 SHAP explanation 결과,
    또는 format_local_explanation_as_evidence 결과를 사용하게 됩니다.

    여기서는 Day 9 구조 확인을 위해 sample dict를 사용합니다.
    """
    prediction_result = {
        "probability": 0.9930,
        "threshold": 0.7000,
        "prediction": 1,
        "risk_level": "HIGH",
        "recommended_action": "고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다.",
        "evidence": [
            {
                "feature": "Tool wear [min]",
                "value": 220.0,
                "severity": "HIGH",
                "reason": "공구 마모 시간이 높아 점검이 필요합니다.",
            },
            {
                "feature": "Torque [Nm]",
                "value": 62.0,
                "severity": "HIGH",
                "reason": "토크 값이 높은 편이므로 부하 상태 확인이 필요합니다.",
            },
        ],
    }

    shap_local_explanation = [
        {
            "feature": "Torque [Nm]",
            "value": 62.0,
            "reference_value": 40.0033625,
            "contribution": 5.1592,
            "direction": "positive",
            "global_importance": 0.3309,
            "reason": "이 sample에서 Torque는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.",
        },
        {
            "feature": "Tool wear [min]",
            "value": 220.0,
            "reference_value": 107.685,
            "contribution": 2.8238,
            "direction": "positive",
            "global_importance": 0.1213,
            "reason": "이 sample에서 Tool wear는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.",
        },
        {
            "feature": "Process temperature [K]",
            "value": 312.5,
            "reference_value": 310.0060625,
            "contribution": -1.2535,
            "direction": "negative",
            "global_importance": 0.1651,
            "reason": "이 sample에서 Process temperature는 모델의 고장 위험 logit을 낮추는 방향으로 작용했습니다.",
        },
    ]

    global_importance_items = [
        {
            "feature": "Torque [Nm]",
            "importance": 0.3309,
            "permuted_score": 0.1734,
        },
        {
            "feature": "Air temperature [K]",
            "importance": 0.2725,
            "permuted_score": 0.2319,
        },
        {
            "feature": "Rotational speed [rpm]",
            "importance": 0.2292,
            "permuted_score": 0.2752,
        },
    ]

    evidence_items = build_agent_evidence(
        prediction_result=prediction_result,
        shap_local_explanation=shap_local_explanation,
        global_importance_items=global_importance_items,
        shap_top_n=3,
    )

    answer = build_agent_answer(
        prediction_result=prediction_result,
        evidence_items=evidence_items,
    )

    print("=" * 80)
    print("[INFO] Agent Evidence")
    print("=" * 80)
    print(json.dumps(evidence_items, ensure_ascii=False, indent=2))

    print()
    print("=" * 80)
    print("[INFO] Agent Answer")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()