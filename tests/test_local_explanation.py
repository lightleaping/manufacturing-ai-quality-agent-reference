from src.interpretability.local_explanation import (
    build_local_explanation_result,
    create_local_feature_contribution,
    determine_contribution_direction,
    format_local_explanation_as_evidence,
    get_top_local_contributions,
)


def test_determine_contribution_direction() -> None:
    """
    contribution 값에 따라 방향이 제대로 정해지는지 확인합니다.
    """

    assert determine_contribution_direction(0.3) == "increases_risk"
    assert determine_contribution_direction(-0.2) == "decreases_risk"
    assert determine_contribution_direction(0.0) == "neutral"


def test_create_local_feature_contribution_sets_direction() -> None:
    """
    create_local_feature_contribution 함수가
    contribution 값으로 direction을 자동 설정하는지 확인합니다.
    """

    contribution = create_local_feature_contribution(
        feature="Torque [Nm]",
        value=65.0,
        contribution=0.31,
        reference_value=40.0,
        global_importance=0.3309,
    )

    assert contribution.feature == "Torque [Nm]"
    assert contribution.value == 65.0
    assert contribution.contribution == 0.31
    assert contribution.direction == "increases_risk"
    assert contribution.reference_value == 40.0
    assert contribution.global_importance == 0.3309
    assert "고장 위험 예측을 높이는 방향" in contribution.reason


def test_get_top_local_contributions_sorts_by_absolute_value() -> None:
    """
    local contribution은 절댓값 기준으로 정렬되어야 합니다.

    이유:
        +0.30은 위험을 높이는 강한 영향이고,
        -0.40은 위험을 낮추는 강한 영향입니다.

        방향은 다르지만 둘 다 예측에 크게 영향을 줬으므로
        abs(contribution) 기준으로 정렬합니다.
    """

    contributions = [
        create_local_feature_contribution(
            feature="Type",
            value=0,
            contribution=0.02,
        ),
        create_local_feature_contribution(
            feature="Torque [Nm]",
            value=65.0,
            contribution=0.31,
        ),
        create_local_feature_contribution(
            feature="Rotational speed [rpm]",
            value=1200.0,
            contribution=-0.40,
        ),
    ]

    top_contributions = get_top_local_contributions(
        contributions=contributions,
        top_k=2,
    )

    assert len(top_contributions) == 2
    assert top_contributions[0].feature == "Rotational speed [rpm]"
    assert top_contributions[1].feature == "Torque [Nm]"


def test_get_top_local_contributions_can_filter_risk_increasing() -> None:
    """
    only_risk_increasing=True이면
    고장 위험을 높이는 feature만 남겨야 합니다.
    """

    contributions = [
        create_local_feature_contribution(
            feature="Torque [Nm]",
            value=65.0,
            contribution=0.31,
        ),
        create_local_feature_contribution(
            feature="Rotational speed [rpm]",
            value=1200.0,
            contribution=-0.40,
        ),
        create_local_feature_contribution(
            feature="Tool wear [min]",
            value=220.0,
            contribution=0.25,
        ),
    ]

    top_contributions = get_top_local_contributions(
        contributions=contributions,
        top_k=3,
        only_risk_increasing=True,
    )

    assert len(top_contributions) == 2
    assert top_contributions[0].feature == "Torque [Nm]"
    assert top_contributions[1].feature == "Tool wear [min]"
    assert all(
        contribution.direction == "increases_risk"
        for contribution in top_contributions
    )


def test_build_local_explanation_result_creates_summary() -> None:
    """
    LocalExplanationResult가 prediction/probability/threshold/risk_level과
    feature contribution 정보를 함께 담는지 확인합니다.
    """

    contributions = [
        create_local_feature_contribution(
            feature="Torque [Nm]",
            value=65.0,
            contribution=0.31,
        ),
        create_local_feature_contribution(
            feature="Tool wear [min]",
            value=220.0,
            contribution=0.25,
        ),
    ]

    result = build_local_explanation_result(
        prediction=1,
        probability=0.82,
        threshold=0.70,
        risk_level="HIGH",
        contributions=contributions,
        explanation_method="local_proxy",
    )

    assert result.prediction == 1
    assert result.probability == 0.82
    assert result.threshold == 0.70
    assert result.risk_level == "HIGH"
    assert result.explanation_method == "local_proxy"
    assert len(result.contributions) == 2
    assert "고장 위험으로 판단" in result.summary
    assert "Torque [Nm]" in result.summary
    assert len(result.limitations) >= 1


def test_format_local_explanation_as_evidence() -> None:
    """
    LocalExplanationResult를 Agent evidence schema로 변환할 수 있는지 확인합니다.
    """

    contributions = [
        create_local_feature_contribution(
            feature="Torque [Nm]",
            value=65.0,
            contribution=0.31,
            reference_value=40.0,
            global_importance=0.3309,
        ),
        create_local_feature_contribution(
            feature="Tool wear [min]",
            value=220.0,
            contribution=0.25,
            reference_value=100.0,
            global_importance=0.1213,
        ),
    ]

    result = build_local_explanation_result(
        prediction=1,
        probability=0.82,
        threshold=0.70,
        risk_level="HIGH",
        contributions=contributions,
        explanation_method="local_proxy",
    )

    evidence = format_local_explanation_as_evidence(
        result=result,
        top_k=2,
    )

    assert len(evidence) == 3

    summary_evidence = evidence[0]

    assert summary_evidence["source"] == "local_explanation"
    assert summary_evidence["evidence_type"] == "prediction_summary"
    assert summary_evidence["prediction"] == 1
    assert summary_evidence["probability"] == 0.82
    assert summary_evidence["threshold"] == 0.70
    assert summary_evidence["risk_level"] == "HIGH"

    feature_evidence = evidence[1]

    assert feature_evidence["source"] == "local_explanation"
    assert feature_evidence["evidence_type"] == "feature_contribution"
    assert feature_evidence["feature"] == "Torque [Nm]"
    assert feature_evidence["direction"] == "increases_risk"
    assert feature_evidence["global_importance"] == 0.3309