import pytest
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.inference.model_artifacts import save_failure_model_artifacts
from src.inference.predict_failure import (
    build_recommended_action,
    build_rule_based_evidence,
    build_single_sample_dataframe,
    calculate_risk_level,
    predict_failure,
)
from src.models.failure_mlp import FailureMLP


def test_build_single_sample_dataframe_converts_type_string():
    """
    raw input의 Type 문자열이 학습 때와 같은 숫자 값으로 변환되는지 검증합니다.
    """

    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type",
    ]

    raw_sample = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0,
        "Type": "M",
    }

    sample_df = build_single_sample_dataframe(
        raw_sample=raw_sample,
        feature_columns=feature_columns,
    )

    assert list(sample_df.columns) == feature_columns
    assert sample_df.shape == (1, 6)
    assert sample_df.loc[0, "Type"] == 1


def test_build_single_sample_dataframe_raises_error_when_feature_missing():
    """
    필요한 feature가 누락되면 조용히 넘어가지 않고 에러를 내는지 검증합니다.
    """

    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type",
    ]

    raw_sample = {
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        # "Tool wear [min]" intentionally missing
        "Type": "M",
    }

    with pytest.raises(ValueError):
        build_single_sample_dataframe(
            raw_sample=raw_sample,
            feature_columns=feature_columns,
        )


def test_calculate_risk_level():
    """
    probability가 LOW / MEDIUM / HIGH로 변환되는지 검증합니다.
    """

    assert calculate_risk_level(0.10) == "LOW"
    assert calculate_risk_level(0.40) == "MEDIUM"
    assert calculate_risk_level(0.70) == "HIGH"


def test_predict_failure_returns_prediction_result(tmp_path):
    """
    저장된 model/scaler/metadata를 로드해서 단일 sample inference가 가능한지 검증합니다.

    이 테스트는 실제 모델 성능을 검증하는 테스트가 아닙니다.
    여기서 검증하는 것은 추론 파이프라인이 정상적으로 연결되는지입니다.

    검증 흐름:
    1. 테스트용 model 생성
    2. 테스트용 scaler fit
    3. artifact 저장
    4. predict_failure 호출
    5. probability / prediction / risk_level 반환 확인
    """

    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type",
    ]

    numeric_feature_columns = [
        column
        for column in feature_columns
        if column != "Type"
    ]

    model = FailureMLP(
        input_dim=6,
        hidden_dim=32,
        dropout_rate=0.2,
    )

    X = pd.DataFrame(
        [
            [300.0, 310.0, 1500.0, 40.0, 10.0, 0.0],
            [305.0, 315.0, 1600.0, 45.0, 20.0, 1.0],
            [310.0, 320.0, 1700.0, 50.0, 30.0, 2.0],
        ],
        columns=feature_columns,
    )

    scaler = StandardScaler()
    scaler.fit(X[numeric_feature_columns])

    save_failure_model_artifacts(
        model=model,
        scaler=scaler,
        threshold=0.6,
        feature_columns=feature_columns,
        artifact_dir=tmp_path,
        input_dim=6,
        hidden_dim=32,
        dropout_rate=0.2,
    )

    raw_sample = {
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 312.0,
        "Rotational speed [rpm]": 1550.0,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 15.0,
        "Type": "L",
    }

    result = predict_failure(
        raw_sample=raw_sample,
        artifact_dir=tmp_path,
    )

    assert 0.0 <= result.probability <= 1.0
    assert result.prediction in {0, 1}
    assert result.threshold == 0.6
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}
    
    assert isinstance(result.recommended_action, str)
    assert len(result.recommended_action) > 0
    assert isinstance(result.evidence, list)
    assert len(result.evidence) > 0

def test_build_recommended_action_for_low_risk_normal():
    """
    정상 예측과 LOW 위험도일 때 권장 조치 문장이 반환되는지 검증합니다.
    """

    action = build_recommended_action(
        prediction=0,
        risk_level="LOW",
    )

    assert "정상" in action

def test_build_rule_based_evidence_returns_default_message():
    """
    위험 feature가 뚜렷하지 않은 sample에서는 기본 evidence가 반환되어야 합니다.
    """

    raw_sample = {
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 312.0,
        "Rotational speed [rpm]": 1550.0,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 15.0,
        "Type": "L",
    }

    evidence = build_rule_based_evidence(raw_sample)

    assert len(evidence) == 1
    assert evidence[0]["feature"] == "overall"


def test_build_rule_based_evidence_detects_high_tool_wear():
    """
    Tool wear 값이 높으면 evidence에 포함되는지 검증합니다.
    """

    raw_sample = {
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 312.0,
        "Rotational speed [rpm]": 1550.0,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 220.0,
        "Type": "L",
    }

    evidence = build_rule_based_evidence(raw_sample)

    assert any(
        item["feature"] == "Tool wear [min]"
        for item in evidence
    )