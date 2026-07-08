# tests/test_shap_explainer.py

"""
Day 8 - SHAP explainer 테스트

이 테스트 파일의 목적
--------------------
Day 8에서 만든 src/interpretability/shap_explainer.py가
정상적으로 동작하는지 확인한다.

중요한 점
---------
이 테스트는 모델 성능을 검증하는 테스트가 아니다.

즉,
"이 모델이 고장을 잘 맞히는가?"
를 보는 테스트가 아니라,

"SHAP 계산 코드가 정상적으로 실행되는가?"
"SHAP value가 LocalFeatureContribution으로 변환되는가?"
"최종적으로 LocalExplanationResult 구조가 만들어지는가?"
를 확인하는 테스트다.

테스트 범위
-----------
1. background tensor 생성 확인
2. SHAP value 반환 형태 normalize 확인
3. PyTorch FailureMLP + SHAP DeepExplainer 연결 확인
4. LocalExplanationResult 생성 확인
"""

import numpy as np
import pytest
import torch

# shap이 설치되어 있지 않으면 이 테스트 파일 전체를 skip한다.
#
# Day 8에서는 requirements.txt에 shap==0.51.0을 추가했으므로
# 정상 환경이라면 skip되지 않고 실행되어야 한다.
pytest.importorskip("shap")

from src.interpretability.shap_explainer import (
    build_background_tensor,
    build_shap_local_explanation_result,
    normalize_shap_values,
)
from src.models.failure_mlp import FailureMLP


def test_build_background_tensor_limits_size() -> None:
    """
    background_size가 전체 sample 수보다 작으면
    요청한 개수만큼만 background tensor를 만든다.

    예:
        전체 sample = 20개
        background_size = 5

    결과:
        background.shape == (5, 6)
    """

    # 20개 sample, feature 6개짜리 가짜 tensor를 만든다.
    X_tensor = torch.randn(20, 6)

    background = build_background_tensor(
        X_tensor=X_tensor,
        background_size=5,
        seed=42,
    )

    assert background.shape == (5, 6)

    # SHAP와 PyTorch 모델 입력을 안정적으로 맞추기 위해
    # dtype은 float32여야 한다.
    assert background.dtype == torch.float32


def test_build_background_tensor_uses_all_when_size_is_large() -> None:
    """
    background_size가 전체 sample 수보다 크면
    전체 sample만 사용해야 한다.

    예:
        전체 sample = 8개
        background_size = 100

    결과:
        background.shape == (8, 6)

    이유:
        없는 sample을 만들 수는 없기 때문에
        min(background_size, 전체 sample 수)를 사용한다.
    """

    X_tensor = torch.randn(8, 6)

    background = build_background_tensor(
        X_tensor=X_tensor,
        background_size=100,
        seed=42,
    )

    assert background.shape == (8, 6)
    assert background.dtype == torch.float32


def test_normalize_shap_values_from_1d_array() -> None:
    """
    SHAP value가 1차원 배열로 들어오는 경우를 확인한다.

    예:
        raw shape = (3,)

    이 경우 sample 하나에 대한 feature별 SHAP 값으로 보고,
    batch 차원을 추가해서 아래 형태로 바꿔야 한다.

        normalized shape = (1, 3)
    """

    raw_shap_values = np.array([0.1, -0.2, 0.3])

    normalized = normalize_shap_values(raw_shap_values)

    assert normalized.shape == (1, 3)

    # normalized는 SHAP value를 2차원 배열 형태로 정리한 결과입니다.
    #
    # 예를 들어 raw_shap_values가 아래처럼 들어왔다고 가정합니다.
    #
    #   raw_shap_values = np.array([0.1, -0.2, 0.3])
    #
    # normalize_shap_values 함수는 이 값을 sample 1개, feature 3개 구조로 바꿉니다.
    #
    #   normalized.shape == (1, 3)
    #
    # 즉 normalized는 아래와 같은 형태가 됩니다.
    #
    #   [[0.1, -0.2, 0.3]]
    #
    # normalized[0, 0]은
    #   0번째 sample의 0번째 feature SHAP value를 의미합니다.
    #
    # 따라서 여기서는 첫 번째 SHAP value가 0.1인지 확인합니다.
    #
    # pytest.approx(0.1)을 쓰는 이유:
    #   float 값은 컴퓨터 내부에서 아주 미세한 오차가 생길 수 있습니다.
    #   그래서 0.1과 정확히 같은지 비교하기보다,
    #   "거의 0.1이면 통과"하도록 pytest.approx를 사용합니다.
    assert normalized[0, 0] == pytest.approx(0.1)

    assert normalized[0, 1] == pytest.approx(-0.2)
    assert normalized[0, 2] == pytest.approx(0.3)


def test_normalize_shap_values_from_3d_array() -> None:
    """
    SHAP value가 3차원 배열로 들어오는 경우를 확인한다.

    SHAP 버전이나 모델 출력 구조에 따라
    아래처럼 마지막 output 차원이 붙을 수 있다.

        raw shape = (1, 3, 1)

    이 경우 마지막 차원을 제거해서 아래 형태로 바꿔야 한다.

        normalized shape = (1, 3)
    """

    raw_shap_values = np.array(
        [
            [[0.1], [-0.2], [0.3]],
        ]
    )

    normalized = normalize_shap_values(raw_shap_values)

    assert normalized.shape == (1, 3)
    assert normalized[0, 0] == pytest.approx(0.1)
    assert normalized[0, 1] == pytest.approx(-0.2)
    assert normalized[0, 2] == pytest.approx(0.3)


def test_build_shap_local_explanation_result() -> None:
    """
    실제 PyTorch FailureMLP 모델에 SHAP를 적용해서
    Day 7의 LocalExplanationResult 구조로 변환되는지 확인한다.

    여기서는 학습된 모델을 쓰지 않는다.
    랜덤 weight를 가진 작은 FailureMLP를 사용한다.

    이유:
        이 테스트의 목적은 모델 성능 검증이 아니라
        SHAP 연결 구조 검증이기 때문이다.

    확인하는 것:
        1. LocalExplanationResult가 생성되는가?
        2. prediction이 0 또는 1인가?
        3. probability가 0~1 사이인가?
        4. contribution 개수가 top_k와 맞는가?
        5. contribution의 feature 이름과 direction이 정상인가?
    """

    torch.manual_seed(42)

    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type",
    ]

    # 테스트 속도를 위해 hidden_dim을 작게 둔다.
    # dropout_rate=0.0으로 두면 테스트 재현성이 좋아진다.
    model = FailureMLP(
        input_dim=6,
        hidden_dim=8,
        dropout_rate=0.0,
    )

    # SHAP background data
    # shape = (background_sample_count, feature_count)
    background_tensor = torch.randn(10, 6)

    # 설명할 sample 하나
    # shape = (1, feature_count)
    sample_tensor = torch.randn(1, 6)

    result = build_shap_local_explanation_result(
        model=model,
        background_tensor=background_tensor,
        sample_tensor=sample_tensor,
        feature_columns=feature_columns,
        threshold=0.5,
        risk_level="LOW",
        top_k=3,
    )

    assert result.explanation_method == "shap_deep_explainer_logit"

    # prediction은 최종 0/1 판단이다.
    assert result.prediction in [0, 1]

    # probability는 sigmoid(logit)이므로 0~1 사이여야 한다.
    assert 0.0 <= result.probability <= 1.0

    # threshold는 함수에 넣은 값이 그대로 들어가야 한다.
    assert result.threshold == pytest.approx(0.5)

    # top_k=3으로 요청했으므로 contribution도 3개여야 한다.
    assert len(result.contributions) == 3

    # summary는 비어 있으면 안 된다.
    assert isinstance(result.summary, str)
    assert len(result.summary) > 0

    # limitations에는 logit 기준 SHAP라는 설명이 포함되어야 한다.
    assert any("logit" in limitation for limitation in result.limitations)

    for contribution in result.contributions:
        assert contribution.feature in feature_columns
        assert contribution.direction in ["positive", "negative", "neutral"]
        assert isinstance(contribution.contribution, float)
        assert isinstance(contribution.reason, str)
        assert len(contribution.reason) > 0


# pytest 실행 시 주의
# ------------------
# 현재 프로젝트에는 .venv와 .venv-1처럼 여러 가상환경이 있을 수 있습니다.
#
# shap을 한 가상환경에 설치해도,
# pytest가 다른 가상환경의 Python으로 실행되면 shap을 찾지 못합니다.
#
# 예:
#   .venv에 shap 설치
#   pytest는 .venv-1에서 실행
#   -> ModuleNotFoundError 또는 pytest.importorskip("shap") 때문에 skip 발생
#
# 따라서 패키지 설치와 pytest 실행은 반드시 같은 Python 환경에서 해야 합니다.
#
# 확인 명령:
#   python -c "import sys; print(sys.executable)"
#
# 목표 경로:
#   ...\manufacturing-ai-quality-agent-reference\.venv\Scripts\python.exe