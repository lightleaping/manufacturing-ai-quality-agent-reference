from __future__ import annotations

from typing import Any, Iterable

from src.agent.evidence_builder import filter_evidence_by_type


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    probability, threshold 같은 값을 안전하게 float로 바꿉니다.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_percent(value: float) -> str:
    """
    0.9930 같은 probability를 99.30%처럼 보여주기 위한 함수입니다.
    """
    return f"{value * 100:.2f}%"


def _get_prediction_metadata(
    evidence_items: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """
    evidence 리스트에서 prediction_summary evidence의 metadata만 꺼냅니다.
    """
    prediction_evidence = filter_evidence_by_type(
        evidence_items=evidence_items,
        evidence_type="prediction_summary",
    )

    if not prediction_evidence:
        return {}

    return prediction_evidence[0].get("metadata", {})


def _format_rule_section(
    rule_evidence_items: list[dict[str, Any]],
) -> str:
    """
    rule-based evidence 문단을 만듭니다.

    rule-based evidence는 제조 기준으로 입력값을 본 참고 근거입니다.
    모델 내부 판단 근거라고 말하지 않습니다.
    """
    if not rule_evidence_items:
        return (
            "## 입력값 기준 점검 신호\n"
            "제조 rule 기준으로 별도의 위험 입력값은 표시되지 않았습니다."
        )

    lines = ["## 입력값 기준 점검 신호"]

    lines.append(
        "아래 항목은 사람이 정한 제조 rule 기준으로 표시된 참고 근거입니다."
    )

    for item in rule_evidence_items:
        feature = item.get("feature", "unknown")
        value = item.get("value")
        severity = item.get("severity", "UNKNOWN")
        summary = item.get("summary", "")

        lines.append(
            f"- `{feature}` = `{value}` / severity={severity}\n"
            f"  - {summary}"
        )

    return "\n".join(lines)


def _format_shap_section(
    shap_evidence_items: list[dict[str, Any]],
) -> str:
    """
    SHAP local evidence 문단을 만듭니다.

    SHAP evidence는 모델 output에 대한 feature contribution입니다.
    실제 고장 원인이라고 단정하지 않습니다.
    """
    if not shap_evidence_items:
        return (
            "## 모델 설명 기준 SHAP 근거\n"
            "현재 응답에는 SHAP local explanation 결과가 포함되지 않았습니다."
        )

    lines = ["## 모델 설명 기준 SHAP 근거"]

    lines.append(
        "아래 항목은 개별 sample에서 각 feature가 모델 output에 기여한 방향입니다."
    )

    for item in shap_evidence_items:
        feature = item.get("feature", "unknown")
        value = item.get("value")
        direction = item.get("direction", "unknown")
        contribution = _safe_float(item.get("contribution"))
        summary = item.get("summary", "")

        lines.append(
            f"- `{feature}` = `{value}` / direction={direction} / "
            f"contribution={contribution:.4f}\n"
            f"  - {summary}"
        )

    return "\n".join(lines)


def _format_global_importance_section(
    global_evidence_items: list[dict[str, Any]],
) -> str:
    """
    global importance 문단을 만듭니다.

    permutation importance는 전체 test set 기준 설명입니다.
    개별 sample의 직접적인 이유가 아니라는 점을 반드시 분리합니다.
    """
    if not global_evidence_items:
        return ""

    lines = ["## 전체 데이터 기준 참고 중요도"]

    lines.append(
        "아래 항목은 전체 test set 기준 feature importance이며, "
        "개별 sample의 직접적인 예측 이유는 아닙니다."
    )

    for item in global_evidence_items:
        feature = item.get("feature", "unknown")
        importance = _safe_float(item.get("importance"))
        summary = item.get("summary", "")

        lines.append(
            f"- `{feature}` / importance={importance:.4f}\n"
            f"  - {summary}"
        )

    return "\n".join(lines)


def build_agent_answer(
    prediction_result: dict[str, Any],
    evidence_items: list[dict[str, Any]],
) -> str:
    """
    evidence 기반 Agent 답변을 만듭니다.

    이 함수의 핵심 원칙:

    1. prediction은 모델의 최종 판단으로 말한다.
    2. rule-based evidence는 입력값 기준 참고 근거로 말한다.
    3. SHAP evidence는 모델 output에 대한 contribution으로 말한다.
    4. 어떤 feature도 실제 고장의 원인이라고 단정하지 않는다.
    """
    prediction_metadata = _get_prediction_metadata(evidence_items)

    probability = _safe_float(
        prediction_metadata.get(
            "probability",
            prediction_result.get("probability"),
        )
    )
    threshold = _safe_float(
        prediction_metadata.get(
            "threshold",
            prediction_result.get("threshold"),
        )
    )
    prediction = prediction_metadata.get(
        "prediction",
        prediction_result.get("prediction"),
    )
    risk_level = prediction_metadata.get(
        "risk_level",
        prediction_result.get("risk_level", "UNKNOWN"),
    )
    recommended_action = prediction_metadata.get(
        "recommended_action",
        prediction_result.get("recommended_action"),
    )

    rule_evidence_items = filter_evidence_by_type(
        evidence_items=evidence_items,
        evidence_type="rule_based",
    )
    shap_evidence_items = filter_evidence_by_type(
        evidence_items=evidence_items,
        evidence_type="shap_local",
    )
    global_evidence_items = filter_evidence_by_type(
        evidence_items=evidence_items,
        evidence_type="global_importance",
    )

    lines: list[str] = []

    lines.append("# 설비 고장 위험 예측 결과")

    lines.append(
        f"모델은 현재 sample의 고장 probability를 "
        f"{probability:.4f}({_format_percent(probability)})로 예측했습니다."
    )

    lines.append(
        f"운영 threshold {threshold:.4f} 기준으로 "
        f"prediction={prediction}, risk_level={risk_level}입니다."
    )

    if recommended_action:
        lines.append(f"권장 조치: {recommended_action}")

    lines.append("")
    lines.append(_format_rule_section(rule_evidence_items))

    lines.append("")
    lines.append(_format_shap_section(shap_evidence_items))

    global_section = _format_global_importance_section(global_evidence_items)
    if global_section:
        lines.append("")
        lines.append(global_section)

    lines.append("")
    lines.append("## 해석 시 주의점")
    lines.append(
        "위 설명은 모델 예측 결과, 제조 rule 기준 입력값 점검 신호, "
        "SHAP 기반 모델 output contribution을 함께 보여주는 것입니다."
    )
    lines.append(
        "따라서 특정 feature를 실제 고장의 물리적 원인으로 단정하지 않고, "
        "모델 기준 위험 예측을 높이거나 낮춘 방향으로 해석해야 합니다."
    )

    return "\n".join(lines)