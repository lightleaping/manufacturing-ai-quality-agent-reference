from dataclasses import dataclass, field
from typing import Any, Literal


# 하나의 feature가 개별 예측에 어떤 방향으로 작용했는지를 나타냅니다.
#
# increases_risk:
#   이 feature가 현재 샘플의 고장 위험 예측을 높이는 방향으로 작용했다는 뜻입니다.
#
# decreases_risk:
#   이 feature가 현재 샘플의 고장 위험 예측을 낮추는 방향으로 작용했다는 뜻입니다.
#
# neutral:
#   이 feature의 영향이 거의 없거나, 해석상 중립에 가깝다는 뜻입니다.
ContributionDirection = Literal[
    "increases_risk",
    "decreases_risk",
    "neutral",
]


# local explanation 결과가 어떤 방법에서 나온 것인지 표시합니다.
#
# local_proxy:
#   아직 SHAP를 붙이기 전, 임시/설계용 local explanation입니다.
#
# shap:
#   이후 Day 8에서 SHAP 값을 실제로 계산해서 넣을 때 사용할 수 있습니다.
#
# manual:
#   테스트나 수동 확인용으로 만든 설명일 수 있습니다.
ExplanationMethod = Literal[
    "local_proxy",
    "shap",
    "manual",
]


# Agent 답변에 넣을 evidence 종류를 구분하기 위한 타입입니다.
#
# rule_based:
#   제조 기준 rule로 만든 evidence입니다.
#   예: Tool wear가 200분 이상이면 위험 신호로 표시
#
# permutation_importance:
#   전체 test set 기준 feature importance입니다.
#
# local_explanation:
#   개별 샘플 하나에 대한 feature contribution 설명입니다.
EvidenceSource = Literal[
    "rule_based",
    "permutation_importance",
    "local_explanation",
]


# risk_level은 inference 결과와 Agent 답변에서 계속 사용하는 값입니다.
#
# LOW:
#   고장 위험이 낮다고 설명할 때 사용합니다.
#
# MEDIUM:
#   고장 위험이 중간 수준이라고 설명할 때 사용합니다.
#
# HIGH:
#   고장 위험이 높다고 설명할 때 사용합니다.
#
# UNKNOWN:
#   예측 결과를 만들 수 없거나 위험도를 판단하기 어려울 때 사용합니다.
RiskLevel = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
    "UNKNOWN",
]


@dataclass(frozen=True)
class LocalFeatureContribution:
    """
    LocalFeatureContribution은 개별 sample 하나에서
    feature 하나가 예측에 어떤 방향으로 영향을 줬는지 담는 자료 구조입니다.

    예를 들어 어떤 샘플에 대해 모델이 고장 확률을 0.82로 예측했다고 가정합니다.

    이때 다음과 같은 설명을 담을 수 있습니다.

    - Torque [Nm]는 고장 위험을 높이는 방향으로 작용했다.
    - Tool wear [min]도 고장 위험을 높이는 방향으로 작용했다.
    - Type은 영향이 거의 없었다.

    중요한 점:
    이 dataclass 자체가 SHAP를 계산하는 것은 아닙니다.
    SHAP, local proxy, 수동 테스트 결과 등에서 나온 값을
    공통 형식으로 담기 위한 구조입니다.
    """

    # 설명 대상 feature 이름입니다.
    #
    # 예:
    # "Torque [Nm]"
    # "Tool wear [min]"
    # "Rotational speed [rpm]"
    feature: str

    # 현재 샘플에서 해당 feature가 가진 실제 값입니다.
    #
    # 예:
    # Torque [Nm] = 65.0
    # Tool wear [min] = 220.0
    value: float | int | str | None

    # 개별 예측에 대한 feature 기여도 값입니다.
    #
    # SHAP를 사용하는 경우:
    #   SHAP value가 들어갈 수 있습니다.
    #
    # 아직 SHAP를 사용하지 않는 경우:
    #   테스트용 임시 contribution 값이나 proxy 값이 들어갈 수 있습니다.
    #
    # 양수:
    #   고장 위험을 높이는 방향
    #
    # 음수:
    #   고장 위험을 낮추는 방향
    #
    # 0에 가까움:
    #   영향이 작거나 중립
    contribution: float

    # contribution 값을 해석한 방향입니다.
    #
    # contribution > 0  -> increases_risk
    # contribution < 0  -> decreases_risk
    # contribution ~= 0 -> neutral
    direction: ContributionDirection

    # 비교 기준값입니다.
    #
    # SHAP에서는 base value, reference value 같은 개념이 등장합니다.
    # 여기서는 이후 확장을 위해 optional 값으로 둡니다.
    #
    # 예:
    # 해당 feature의 train 평균값
    # 정상 샘플 평균값
    # 기준 샘플의 값
    reference_value: float | int | str | None = None

    # 전체 데이터 기준 feature importance 값입니다.
    #
    # Day 6에서 계산한 permutation importance 값을 여기에 함께 넣을 수 있습니다.
    #
    # 단, 이것은 local contribution과 다릅니다.
    # local contribution은 개별 샘플 기준 영향이고,
    # importance는 전체 test set 기준 중요도입니다.
    global_importance: float | None = None

    # 사람이 읽을 수 있는 간단한 설명 문장입니다.
    reason: str = ""


@dataclass(frozen=True)
class LocalExplanationResult:
    """
    LocalExplanationResult는 개별 sample 하나에 대한 설명 결과 전체를 담습니다.

    이 구조는 Agent 답변 생성 직전에 사용하기 좋습니다.

    예:
    - prediction: 1
    - probability: 0.82
    - threshold: 0.70
    - risk_level: HIGH
    - contributions:
        - Torque [Nm]가 위험을 높임
        - Tool wear [min]가 위험을 높임
        - Type은 거의 영향 없음

    이후 Agent는 이 구조를 evidence로 변환해서 답변에 넣을 수 있습니다.
    """

    # 모델의 최종 예측값입니다.
    #
    # 0:
    #   정상으로 예측
    #
    # 1:
    #   고장 위험으로 예측
    prediction: int

    # 모델이 출력한 고장 확률입니다.
    #
    # 예:
    # 0.82는 82% 확률로 고장 위험이 있다고 본다는 뜻입니다.
    probability: float

    # prediction을 결정할 때 사용한 threshold입니다.
    #
    # 예:
    # probability >= 0.70 이면 prediction = 1
    threshold: float

    # 사람이 이해하기 쉬운 위험 등급입니다.
    risk_level: RiskLevel

    # 설명 방식입니다.
    #
    # Day 7에서는 local_proxy 또는 manual로 사용하고,
    # Day 8 이후에는 shap로 확장할 수 있습니다.
    explanation_method: ExplanationMethod

    # feature별 local contribution 목록입니다.
    contributions: list[LocalFeatureContribution] = field(default_factory=list)

    # local explanation 전체 요약 문장입니다.
    summary: str = ""

    # 설명 방법의 한계를 명시합니다.
    #
    # 예:
    # - 이 값은 SHAP가 아니라 local explanation schema 검증용 proxy입니다.
    # - 개별 contribution과 전체 importance는 다른 개념입니다.
    limitations: list[str] = field(default_factory=list)


def determine_contribution_direction(
    contribution: float,
    neutral_tolerance: float = 1e-6,
) -> ContributionDirection:
    """
    contribution 값을 보고 방향을 판단합니다.

    contribution이 양수이면:
        고장 위험을 높이는 방향입니다.

    contribution이 음수이면:
        고장 위험을 낮추는 방향입니다.

    contribution이 0에 매우 가까우면:
        영향이 거의 없다고 보고 neutral로 처리합니다.

    neutral_tolerance가 필요한 이유:
        float 계산에서는 0처럼 보이는 값도
        실제로는 0.0000000001처럼 아주 작은 값일 수 있습니다.

        그래서 정확히 contribution == 0만 보는 대신,
        일정 범위 안이면 neutral로 처리합니다.
    """

    if contribution > neutral_tolerance:
        return "increases_risk"

    if contribution < -neutral_tolerance:
        return "decreases_risk"

    return "neutral"


def create_local_feature_contribution(
    feature: str,
    value: float | int | str | None,
    contribution: float,
    reference_value: float | int | str | None = None,
    global_importance: float | None = None,
    reason: str | None = None,
) -> LocalFeatureContribution:
    """
    feature 하나에 대한 LocalFeatureContribution 객체를 만듭니다.

    이 함수를 따로 두는 이유:
        매번 direction을 직접 계산해서 넣으면 실수하기 쉽습니다.

    그래서 contribution 값만 넣으면,
    내부에서 increases_risk / decreases_risk / neutral을 자동으로 정합니다.
    """

    direction = determine_contribution_direction(contribution)

    if reason is None:
        reason = build_default_contribution_reason(
            feature=feature,
            contribution=contribution,
            direction=direction,
        )

    return LocalFeatureContribution(
        feature=feature,
        value=value,
        contribution=contribution,
        direction=direction,
        reference_value=reference_value,
        global_importance=global_importance,
        reason=reason,
    )


def build_default_contribution_reason(
    feature: str,
    contribution: float,
    direction: ContributionDirection,
) -> str:
    """
    feature contribution에 대한 기본 설명 문장을 만듭니다.

    이 문장은 Agent 답변에 바로 넣기보다는,
    evidence message나 디버깅용 설명으로 사용하는 것을 목표로 합니다.
    """

    if direction == "increases_risk":
        return (
            f"{feature} 값은 현재 샘플의 고장 위험 예측을 "
            f"높이는 방향으로 작용했습니다."
        )

    if direction == "decreases_risk":
        return (
            f"{feature} 값은 현재 샘플의 고장 위험 예측을 "
            f"낮추는 방향으로 작용했습니다."
        )

    return (
        f"{feature} 값은 현재 샘플의 예측에 큰 영향을 주지 않는 "
        f"것으로 해석됩니다."
    )


def get_top_local_contributions(
    contributions: list[LocalFeatureContribution],
    top_k: int = 3,
    only_risk_increasing: bool = False,
) -> list[LocalFeatureContribution]:
    """
    local contribution 목록에서 중요한 feature만 골라냅니다.

    정렬 기준:
        abs(contribution), 즉 contribution의 절댓값입니다.

    절댓값을 쓰는 이유:
        contribution이 +0.30이면 위험을 높이는 강한 영향이고,
        contribution이 -0.30이면 위험을 낮추는 강한 영향입니다.

        방향은 다르지만 둘 다 예측에 크게 영향을 준 feature입니다.

    only_risk_increasing=True이면:
        고장 위험을 높이는 feature만 남깁니다.

        Agent 답변에서
        "왜 위험하다고 봤는가?"
        를 설명할 때 유용합니다.
    """

    if top_k <= 0:
        return []

    filtered_contributions = contributions

    if only_risk_increasing:
        filtered_contributions = [
            contribution
            for contribution in contributions
            if contribution.direction == "increases_risk"
        ]

    sorted_contributions = sorted(
        filtered_contributions,
        key=lambda contribution: abs(contribution.contribution),
        reverse=True,
    )

    return sorted_contributions[:top_k]


def build_local_explanation_result(
    prediction: int,
    probability: float,
    threshold: float,
    risk_level: RiskLevel,
    contributions: list[LocalFeatureContribution],
    explanation_method: ExplanationMethod = "local_proxy",
) -> LocalExplanationResult:
    """
    개별 sample에 대한 local explanation 결과를 만듭니다.

    이 함수는 다음 정보를 하나로 묶습니다.

    1. 모델 예측 결과
    2. 고장 확률
    3. threshold
    4. risk_level
    5. feature별 contribution
    6. 설명 방식
    7. 한계 설명

    이 구조가 필요한 이유:
        Agent 답변에서는 단순히 probability만 말하면 부족합니다.

        예를 들어:
        "고장 확률은 82%입니다."

        보다 다음처럼 말하는 것이 더 좋습니다.

        "고장 확률은 82%이며, threshold 70%를 넘었기 때문에
        고장 위험으로 판단했습니다. 개별 설명 기준으로는
        Torque와 Tool wear가 위험 예측을 높이는 방향으로 작용했습니다."
    """

    summary = build_local_explanation_summary(
        prediction=prediction,
        probability=probability,
        threshold=threshold,
        risk_level=risk_level,
        contributions=contributions,
    )

    limitations = [
        "local contribution은 개별 샘플 기준 설명입니다.",
        "global importance와 local contribution은 서로 다른 개념입니다.",
    ]

    if explanation_method == "local_proxy":
        limitations.append(
            "현재 결과는 SHAP 계산값이 아니라, SHAP 연동 전 구조 검증용 local explanation입니다."
        )

    return LocalExplanationResult(
        prediction=prediction,
        probability=probability,
        threshold=threshold,
        risk_level=risk_level,
        explanation_method=explanation_method,
        contributions=contributions,
        summary=summary,
        limitations=limitations,
    )


def build_local_explanation_summary(
    prediction: int,
    probability: float,
    threshold: float,
    risk_level: RiskLevel,
    contributions: list[LocalFeatureContribution],
) -> str:
    """
    LocalExplanationResult에 들어갈 요약 문장을 만듭니다.
    """

    top_risk_features = get_top_local_contributions(
        contributions=contributions,
        top_k=3,
        only_risk_increasing=True,
    )

    if top_risk_features:
        feature_names = ", ".join(
            contribution.feature for contribution in top_risk_features
        )
    else:
        feature_names = "뚜렷한 위험 증가 feature 없음"

    if prediction == 1:
        return (
            f"모델은 고장 확률을 {probability:.4f}로 예측했고, "
            f"threshold {threshold:.4f} 이상이므로 고장 위험으로 판단했습니다. "
            f"local explanation 기준 주요 위험 증가 feature는 {feature_names}입니다. "
            f"risk_level은 {risk_level}입니다."
        )

    return (
        f"모델은 고장 확률을 {probability:.4f}로 예측했고, "
        f"threshold {threshold:.4f} 미만이므로 정상으로 판단했습니다. "
        f"local explanation 기준 주요 위험 증가 feature는 {feature_names}입니다. "
        f"risk_level은 {risk_level}입니다."
    )


def format_local_explanation_as_evidence(
    result: LocalExplanationResult,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    LocalExplanationResult를 Agent answer에 넣기 좋은 evidence 형식으로 바꿉니다.

    Agent evidence에서 중요한 점:
        사람이 읽을 수 있어야 하고,
        동시에 프로그램이 다루기 쉬운 dict 구조여야 합니다.

    반환 예시:

    [
        {
            "source": "local_explanation",
            "evidence_type": "prediction_summary",
            "prediction": 1,
            "probability": 0.82,
            "threshold": 0.7,
            "risk_level": "HIGH",
            "message": "모델은 고장 확률을 0.82로 예측..."
        },
        {
            "source": "local_explanation",
            "evidence_type": "feature_contribution",
            "feature": "Torque [Nm]",
            "value": 65.0,
            "contribution": 0.31,
            "direction": "increases_risk",
            "message": "Torque [Nm] 값은..."
        }
    ]
    """

    evidence: list[dict[str, Any]] = [
        {
            "source": "local_explanation",
            "evidence_type": "prediction_summary",
            "explanation_method": result.explanation_method,
            "prediction": result.prediction,
            "probability": result.probability,
            "threshold": result.threshold,
            "risk_level": result.risk_level,
            "message": result.summary,
        }
    ]

    top_contributions = get_top_local_contributions(
        contributions=result.contributions,
        top_k=top_k,
        only_risk_increasing=False,
    )

    for contribution in top_contributions:
        evidence.append(
            {
                "source": "local_explanation",
                "evidence_type": "feature_contribution",
                "explanation_method": result.explanation_method,
                "feature": contribution.feature,
                "value": contribution.value,
                "reference_value": contribution.reference_value,
                "contribution": contribution.contribution,
                "direction": contribution.direction,
                "global_importance": contribution.global_importance,
                "message": contribution.reason,
            }
        )

    return evidence