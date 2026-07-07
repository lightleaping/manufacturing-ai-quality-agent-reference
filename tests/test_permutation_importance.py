import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from src.interpretability.permutation_importance import (
    calculate_confusion_counts,
    calculate_metric,
    calculate_permutation_importance,
    format_permutation_importance_as_evidence,
    get_top_important_features,
    permute_feature,
)


class ToyFailureModel(nn.Module):
    """
    permutation importance 테스트용으로 만든 단순 모델입니다.

    실제 학습된 FailureMLP를 사용하지 않고 Toy 모델을 쓰는 이유:
    - 실제 모델은 학습 상태나 random seed에 따라 결과가 달라질 수 있습니다.
    - 단위 테스트에서는 항상 예측 가능한 결과가 나와야 합니다.
    - 그래서 일부러 sensor_a만 보는 모델을 만들어,
      sensor_a의 중요도가 가장 높게 나오는지 검증합니다.

    이 모델의 의도:
    - 첫 번째 feature인 sensor_a만 사용합니다.
    - 두 번째 feature인 sensor_b는 완전히 무시합니다.
    - 따라서 sensor_a를 섞으면 모델 성능이 크게 떨어져야 합니다.
    - sensor_b를 섞어도 모델 성능은 거의 변하지 않아야 합니다.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 Tensor x에서 첫 번째 feature만 사용해 logit을 반환합니다.

        x의 예시 shape:
        - x.shape == (batch_size, feature_count)
        - 여기서는 feature_count가 2라고 가정합니다.

        예:
        x = [
            [-2.0, 10.0],
            [-1.5, 30.0],
            [ 1.0, 35.0],
            [ 2.0, 45.0],
        ]

        이때:
        - x[:, 0:1]은 모든 row에서 첫 번째 column만 가져옵니다.
        - 첫 번째 column은 sensor_a입니다.
        - 두 번째 column인 sensor_b는 사용하지 않습니다.

        x[:, 0]이 아니라 x[:, 0:1]을 쓰는 이유:
        - x[:, 0]은 shape이 (batch_size,)가 됩니다.
        - x[:, 0:1]은 shape이 (batch_size, 1)로 유지됩니다.
        - 실제 FailureMLP도 보통 (batch_size, 1) 형태의 logit을 반환하므로,
          테스트용 모델도 같은 형태를 맞추기 위해 0:1 slicing을 사용합니다.

        * 5.0을 하는 이유:
        - sensor_a 값을 더 큰 logit으로 만들기 위해서입니다.
        - sensor_a가 양수이면 logit도 큰 양수가 됩니다.
        - sensor_a가 음수이면 logit도 큰 음수가 됩니다.
        - 이후 sigmoid를 적용하면 양수는 고장 확률 1에 가깝고,
          음수는 고장 확률 0에 가까워집니다.
        - 그래서 threshold=0.5 기준으로 예측이 명확하게 나뉩니다.
        """

        # x[:, 0:1]
        # → 모든 샘플(row)에서 첫 번째 feature(column)만 선택합니다.
        #
        # 여기서 첫 번째 feature는 sensor_a입니다.
        #
        # x[:, 0]을 쓰면 결과 shape이 (batch_size,)가 됩니다.
        # x[:, 0:1]을 쓰면 결과 shape이 (batch_size, 1)로 유지됩니다.
        #
        # 실제 binary classification 모델은 보통 샘플마다 logit 1개를 반환하므로
        # shape을 (batch_size, 1)로 유지하는 것이 좋습니다.
        #
        # * 5.0
        # → sensor_a 값을 5배 키워 logit으로 만듭니다.
        #
        # sensor_a가 음수이면 logit도 큰 음수:
        # sigmoid(logit) ≈ 0
        # prediction = 0
        #
        # sensor_a가 양수이면 logit도 큰 양수:
        # sigmoid(logit) ≈ 1
        # prediction = 1
        #
        # 즉, 이 Toy 모델은 sensor_a만 보고
        # 정상/고장을 거의 확실하게 구분하도록 만든 테스트용 모델입니다.
        return x[:, 0:1] * 5.0

def make_toy_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    permutation importance 테스트용 toy dataset을 만듭니다.

    sensor_a:
    - 음수면 정상 0
    - 양수면 고장 1

    sensor_b:
    - 정답과 거의 관계없는 값

    ToyFailureModel은 sensor_a만 보도록 만들었으므로,
    sensor_a를 섞으면 성능이 크게 떨어져야 합니다.
    sensor_b를 섞으면 성능이 거의 변하지 않아야 합니다.
    """

    X = pd.DataFrame(
        {
            "sensor_a": [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
            "sensor_b": [10.0, 30.0, 20.0, 40.0, 15.0, 35.0, 25.0, 45.0],
        }
    )

    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    return X, y


def test_calculate_confusion_counts() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    counts = calculate_confusion_counts(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert counts["tn"] == 1
    assert counts["fp"] == 1
    assert counts["fn"] == 1
    assert counts["tp"] == 1


def test_calculate_metric_accuracy_precision_recall_f1() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    accuracy = calculate_metric(
        y_true=y_true,
        y_pred=y_pred,
        metric_name="accuracy",
    )
    precision = calculate_metric(
        y_true=y_true,
        y_pred=y_pred,
        metric_name="precision",
    )
    recall = calculate_metric(
        y_true=y_true,
        y_pred=y_pred,
        metric_name="recall",
    )
    f1 = calculate_metric(
        y_true=y_true,
        y_pred=y_pred,
        metric_name="f1",
    )

    assert accuracy == 0.5
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_calculate_metric_handles_zero_division() -> None:
    """
    고장이라고 예측한 샘플이 하나도 없는 경우를 테스트합니다.

    이때 precision은 다음처럼 됩니다.

    precision = TP / (TP + FP)
              = 0 / 0

    0으로 나누는 상황이므로,
    이 프로젝트에서는 안전하게 0.0으로 처리합니다.
    """

    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])

    precision = calculate_metric(
        y_true=y_true,
        y_pred=y_pred,
        metric_name="precision",
    )

    assert precision == 0.0


def test_permute_feature_keeps_same_values() -> None:
    X, _ = make_toy_dataset()

    rng = np.random.default_rng(42)

    X_permuted = permute_feature(
        X=X,
        feature_name="sensor_a",
        rng=rng,
    )

    # 원본 DataFrame과 shape은 같아야 합니다.
    assert X_permuted.shape == X.shape

    # sensor_a의 값 목록은 그대로 있어야 합니다.
    # permutation은 값을 새로 만드는 것이 아니라 순서만 섞는 것이기 때문입니다.
    assert sorted(X_permuted["sensor_a"].tolist()) == sorted(X["sensor_a"].tolist())

    # sensor_b는 섞지 않았으므로 원본과 같아야 합니다.
    assert X_permuted["sensor_b"].tolist() == X["sensor_b"].tolist()


def test_permute_feature_raises_error_when_feature_missing() -> None:
    X, _ = make_toy_dataset()
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        permute_feature(
            X=X,
            feature_name="missing_feature",
            rng=rng,
        )


def test_calculate_permutation_importance_returns_sorted_results() -> None:
    X, y = make_toy_dataset()
    model = ToyFailureModel()

    summary = calculate_permutation_importance(
        model=model,
        X=X,
        y=y,
        threshold=0.5,
        metric_name="f1",
        n_repeats=10,
        random_state=42,
    )

    assert summary.baseline_score == 1.0
    assert summary.metric_name == "f1"
    assert summary.threshold == 0.5
    assert len(summary.results) == 2

    first_result = summary.results[0]
    second_result = summary.results[1]

    # ToyFailureModel은 sensor_a만 보기 때문에
    # sensor_a를 섞었을 때 성능이 더 크게 떨어져야 합니다.
    assert first_result.feature_name == "sensor_a"
    assert first_result.importance_mean >= second_result.importance_mean


def test_get_top_important_features() -> None:
    X, y = make_toy_dataset()
    model = ToyFailureModel()

    summary = calculate_permutation_importance(
        model=model,
        X=X,
        y=y,
        threshold=0.5,
        metric_name="f1",
        n_repeats=10,
        random_state=42,
    )

    top_features = get_top_important_features(
        summary=summary,
        top_k=1,
    )

    assert len(top_features) == 1
    assert top_features[0].feature_name == "sensor_a"


def test_format_permutation_importance_as_evidence() -> None:
    X, y = make_toy_dataset()
    model = ToyFailureModel()

    summary = calculate_permutation_importance(
        model=model,
        X=X,
        y=y,
        threshold=0.5,
        metric_name="f1",
        n_repeats=10,
        random_state=42,
    )

    evidence = format_permutation_importance_as_evidence(
        summary=summary,
        top_k=1,
    )

    assert len(evidence) == 1
    assert evidence[0]["feature"] == "sensor_a"
    assert evidence[0]["value"] is None
    assert "importance" in evidence[0]
    assert "message" in evidence[0]