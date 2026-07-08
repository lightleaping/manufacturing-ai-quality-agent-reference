from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal


EvidenceType = Literal[
    "prediction_summary",
    "rule_based",
    "shap_local",
    "global_importance",
]

EvidenceSource = Literal[
    "model_prediction",
    "rule_engine",
    "shap",
    "permutation_importance",
]

Severity = Literal["LOW", "MEDIUM", "HIGH", "UNKNOWN"]


@dataclass
class AgentEvidence:
    """
    Agent가 최종 답변을 만들 때 사용할 표준 evidence 구조입니다.

    Day 5, Day 6, Day 8에서 나온 근거들은 성격이 서로 다릅니다.

    1. prediction_summary
       - 모델의 최종 예측 결과 요약
       - probability, threshold, prediction, risk_level 등을 담습니다.

    2. rule_based
       - 사람이 정한 제조 기준으로 입력값을 해석한 근거
       - 예: Tool wear가 높다, Torque가 높다
       - 모델 내부 판단 근거가 아니라 입력값 기준 참고 근거입니다.

    3. shap_local
       - SHAP으로 계산한 개별 sample 기준 feature contribution
       - 현재 FailureMLP는 Sigmoid가 없는 logit 출력 모델이므로
         SHAP value는 probability가 아니라 logit 기준 contribution입니다.

    4. global_importance
       - permutation importance처럼 전체 test set 기준 설명
       - 개별 sample의 직접적인 예측 이유로 말하면 안 됩니다.
    """

    evidence_id: str

    # 이 evidence가 어떤 종류인지 구분합니다.
    evidence_type: EvidenceType

    # 이 evidence가 어디서 온 것인지 구분합니다.
    source: EvidenceSource

    # 사람이 읽기 쉬운 짧은 제목입니다.
    title: str

    # Agent 답변에 넣을 수 있는 설명 문장입니다.
    summary: str

    # feature 단위 evidence일 경우 feature 이름을 넣습니다.
    feature: str | None = None

    # feature 값입니다.
    value: Any | None = None

    # SHAP direction 또는 rule 방향입니다.
    # 예: positive, negative, neutral
    direction: str | None = None

    # SHAP contribution 값입니다.
    contribution: float | None = None

    # permutation importance 같은 global importance 값입니다.
    importance: float | None = None

    # rule 기준 위험 정도입니다.
    severity: Severity = "UNKNOWN"

    # 추가 정보를 안전하게 담기 위한 공간입니다.
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        dataclass 객체를 dict로 바꿉니다.

        FastAPI 응답, LangGraph state, JSON 저장에서는
        dataclass보다 dict가 다루기 쉽습니다.
        """
        return asdict(self)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    dict와 dataclass/object를 모두 처리하기 위한 helper 함수입니다.

    Day 5 predict_failure_from_artifacts 결과는 dict일 가능성이 높고,
    Day 7 LocalExplanationResult / LocalFeatureContribution은
    dataclass일 가능성이 있습니다.

    그래서 obj[key]만 쓰지 않고,
    dict면 get,
    object면 getattr로 값을 꺼냅니다.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """
    None, 문자열, numpy float 등 다양한 값을 float로 안전하게 바꿉니다.

    Agent 답변에서는 probability, threshold, contribution 같은 값을
    소수점 포맷으로 출력해야 하므로 float 변환이 필요합니다.
    """
    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_severity(value: Any) -> Severity:
    """
    severity 값을 LOW / MEDIUM / HIGH / UNKNOWN 중 하나로 정리합니다.
    """
    text = str(value or "UNKNOWN").upper()

    if text in {"LOW", "MEDIUM", "HIGH"}:
        return text  # type: ignore[return-value]

    return "UNKNOWN"


def _direction_from_contribution(contribution: float) -> str:
    """
    SHAP contribution 값으로 방향을 정합니다.

    양수:
        모델 output, 여기서는 고장 위험 logit을 높이는 방향

    음수:
        모델 output, 여기서는 고장 위험 logit을 낮추는 방향

    0 근처:
        영향이 거의 없는 방향
    """
    if contribution > 0:
        return "positive"

    if contribution < 0:
        return "negative"

    return "neutral"


def _direction_to_korean(direction: str | None) -> str:
    """
    SHAP direction을 Agent 답변용 한국어 표현으로 바꿉니다.
    """
    if direction == "positive":
        return "모델의 고장 위험 logit을 높이는 방향"

    if direction == "negative":
        return "모델의 고장 위험 logit을 낮추는 방향"

    return "모델의 고장 위험 logit에 거의 영향을 주지 않는 방향"


def build_prediction_summary_evidence(
    prediction_result: dict[str, Any],
) -> AgentEvidence:
    """
    Day 5의 predict_failure_from_artifacts 결과에서
    모델 예측 요약 evidence를 만듭니다.

    이 evidence는 feature 단위 근거가 아니라,
    전체 예측 결과를 요약하는 evidence입니다.

    예:
        probability=0.9930
        threshold=0.7000
        prediction=1
        risk_level=HIGH
    """
    probability = _safe_float(_safe_get(prediction_result, "probability"))
    threshold = _safe_float(_safe_get(prediction_result, "threshold"))
    prediction = int(_safe_get(prediction_result, "prediction", 0))
    risk_level = str(_safe_get(prediction_result, "risk_level", "UNKNOWN"))

    summary = (
        f"모델은 고장 probability를 {probability:.4f}로 예측했고, "
        f"threshold {threshold:.4f} 기준 prediction={prediction}, "
        f"risk_level={risk_level}로 판단했습니다."
    )

    return AgentEvidence(
        evidence_id="prediction_summary_001",
        evidence_type="prediction_summary",
        source="model_prediction",
        title="모델 예측 요약",
        summary=summary,
        severity=_normalize_severity(risk_level),
        metadata={
            "probability": probability,
            "threshold": threshold,
            "prediction": prediction,
            "risk_level": risk_level,
            "recommended_action": _safe_get(
                prediction_result,
                "recommended_action",
            ),
        },
    )


def convert_rule_based_evidence(
    rule_evidence_items: Iterable[dict[str, Any]],
) -> list[AgentEvidence]:
    """
    Day 5의 rule-based evidence를 AgentEvidence로 변환합니다.

    rule-based evidence는 사람이 정한 제조 기준으로 입력값을 해석한 것입니다.

    주의:
        이 evidence를 "모델 내부 판단 근거"라고 말하면 안 됩니다.
        "입력값 기준 점검 신호"라고 말해야 합니다.
    """
    converted: list[AgentEvidence] = []

    for index, item in enumerate(rule_evidence_items, start=1):
        feature = _safe_get(item, "feature", "unknown")
        value = _safe_get(item, "value")
        severity = _normalize_severity(_safe_get(item, "severity", "UNKNOWN"))

        reason = (
            _safe_get(item, "reason")
            or _safe_get(item, "message")
            or _safe_get(item, "description")
            or "제조 rule 기준으로 점검이 필요한 입력값입니다."
        )

        summary = (
            f"입력값 기준으로 {feature}={value} 항목이 "
            f"제조 rule에서 점검 신호로 표시되었습니다. {reason}"
        )

        converted.append(
            AgentEvidence(
                evidence_id=f"rule_based_{index:03d}",
                evidence_type="rule_based",
                source="rule_engine",
                title="Rule 기반 입력값 점검 근거",
                summary=summary,
                feature=str(feature),
                value=value,
                severity=severity,
                metadata={
                    "raw_rule_evidence": item,
                },
            )
        )

    return converted


def _extract_shap_contributions(
    shap_local_explanation: Any,
) -> list[Any]:
    """
    SHAP local explanation 입력에서 contributions 리스트만 꺼냅니다.

    Day 8 이후 입력 형태는 여러 가지일 수 있습니다.

    1. LocalExplanationResult dataclass
       - obj.contributions

    2. dict
       - obj["contributions"]

    3. 이미 list
       - [contribution1, contribution2, ...]

    이 함수는 세 경우를 모두 처리합니다.
    """
    if shap_local_explanation is None:
        return []

    if isinstance(shap_local_explanation, list):
        return shap_local_explanation

    contributions = _safe_get(shap_local_explanation, "contributions", [])

    if contributions is None:
        return []

    return list(contributions)


def convert_shap_local_evidence(
    shap_local_explanation: Any,
    top_n: int | None = None,
) -> list[AgentEvidence]:
    """
    Day 8의 SHAP local explanation 결과를 AgentEvidence로 변환합니다.

    SHAP local evidence는 특정 sample 하나에서
    각 feature가 모델 output에 어떤 방향으로 기여했는지 보여줍니다.

    현재 FailureMLP는 마지막에 Sigmoid가 없으므로 raw output은 logit입니다.
    따라서 SHAP contribution은 probability가 아니라 logit 기준 contribution입니다.
    """
    contributions = _extract_shap_contributions(shap_local_explanation)

    if top_n is not None:
        contributions = contributions[:top_n]

    converted: list[AgentEvidence] = []

    for index, item in enumerate(contributions, start=1):
        feature = _safe_get(item, "feature", "unknown")
        value = _safe_get(item, "value")
        contribution = _safe_float(_safe_get(item, "contribution"))
        direction = _safe_get(item, "direction")

        if direction is None:
            direction = _direction_from_contribution(contribution)

        reference_value = _safe_get(item, "reference_value")
        global_importance = _safe_get(item, "global_importance")
        reason = _safe_get(item, "reason")

        direction_text = _direction_to_korean(str(direction))

        summary = (
            f"SHAP 기준으로 {feature}={value}는 "
            f"{direction_text}으로 작용했습니다. "
            f"SHAP contribution={contribution:.4f}입니다."
        )

        if reason:
            summary = f"{summary} {reason}"

        converted.append(
            AgentEvidence(
                evidence_id=f"shap_local_{index:03d}",
                evidence_type="shap_local",
                source="shap",
                title="SHAP 기반 개별 예측 설명",
                summary=summary,
                feature=str(feature),
                value=value,
                direction=str(direction),
                contribution=contribution,
                importance=_safe_float(global_importance, default=0.0)
                if global_importance is not None
                else None,
                severity="UNKNOWN",
                metadata={
                    "reference_value": reference_value,
                    "global_importance": global_importance,
                    "raw_shap_evidence": item,
                    "important_note": (
                        "SHAP contribution은 실제 고장의 물리적 원인 단정이 아니라 "
                        "현재 모델 output에 대한 feature contribution입니다."
                    ),
                },
            )
        )

    return converted


def convert_global_importance_evidence(
    global_importance_items: Iterable[dict[str, Any]],
    top_n: int = 3,
) -> list[AgentEvidence]:
    """
    Day 6 permutation importance 결과를 AgentEvidence로 변환합니다.

    이 evidence는 전체 test set 기준 설명입니다.
    개별 sample 하나의 직접적인 예측 이유로 말하면 안 됩니다.

    Agent 답변에서는 필요할 때 보조 설명으로만 사용합니다.
    """
    converted: list[AgentEvidence] = []

    for index, item in enumerate(list(global_importance_items)[:top_n], start=1):
        feature = _safe_get(item, "feature", "unknown")
        importance = _safe_float(_safe_get(item, "importance"))
        permuted_score = _safe_get(item, "permuted_score")

        summary = (
            f"전체 test set 기준 permutation importance에서 "
            f"{feature}의 importance는 {importance:.4f}입니다. "
            f"이는 해당 feature를 섞었을 때 모델 성능이 그만큼 감소했다는 뜻입니다."
        )

        converted.append(
            AgentEvidence(
                evidence_id=f"global_importance_{index:03d}",
                evidence_type="global_importance",
                source="permutation_importance",
                title="전체 데이터 기준 feature 중요도",
                summary=summary,
                feature=str(feature),
                importance=importance,
                metadata={
                    "permuted_score": permuted_score,
                    "raw_global_importance": item,
                    "important_note": (
                        "global importance는 전체 test set 기준 설명이며, "
                        "개별 sample의 직접적인 예측 이유가 아닙니다."
                    ),
                },
            )
        )

    return converted


def build_agent_evidence(
    prediction_result: dict[str, Any],
    shap_local_explanation: Any | None = None,
    global_importance_items: Iterable[dict[str, Any]] | None = None,
    shap_top_n: int | None = 5,
) -> list[dict[str, Any]]:
    """
    Day 9의 핵심 함수입니다.

    Day 5 prediction_result 안에는 보통 다음 값이 들어 있습니다.

    - prediction
    - probability
    - threshold
    - risk_level
    - recommended_action
    - evidence  # rule-based evidence

    이 함수는 다음 순서로 evidence를 합칩니다.

    1. prediction summary evidence
    2. rule-based evidence
    3. SHAP local evidence
    4. optional global importance evidence

    최종 반환은 list[dict]입니다.
    LangGraph state, FastAPI response, JSON 저장에 바로 넣기 좋습니다.
    """
    evidence_objects: list[AgentEvidence] = []

    evidence_objects.append(
        build_prediction_summary_evidence(prediction_result)
    )

    rule_evidence_items = _safe_get(prediction_result, "evidence", []) or []
    evidence_objects.extend(
        convert_rule_based_evidence(rule_evidence_items)
    )

    if shap_local_explanation is not None:
        evidence_objects.extend(
            convert_shap_local_evidence(
                shap_local_explanation=shap_local_explanation,
                top_n=shap_top_n,
            )
        )

    if global_importance_items is not None:
        evidence_objects.extend(
            convert_global_importance_evidence(
                global_importance_items=global_importance_items,
            )
        )

    return [evidence.to_dict() for evidence in evidence_objects]


def filter_evidence_by_type(
    evidence_items: Iterable[dict[str, Any]],
    evidence_type: EvidenceType,
) -> list[dict[str, Any]]:
    """
    evidence 리스트에서 특정 evidence_type만 골라냅니다.

    answer_builder에서 rule evidence와 SHAP evidence를
    서로 다른 문단으로 나누기 위해 사용합니다.
    """
    return [
        item
        for item in evidence_items
        if item.get("evidence_type") == evidence_type
    ]


def group_evidence_by_type(
    evidence_items: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    evidence를 evidence_type별로 묶습니다.

    예:
        {
            "prediction_summary": [...],
            "rule_based": [...],
            "shap_local": [...],
            "global_importance": [...],
        }
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    for item in evidence_items:
        evidence_type = str(item.get("evidence_type", "unknown"))
        grouped.setdefault(evidence_type, []).append(item)

    return grouped