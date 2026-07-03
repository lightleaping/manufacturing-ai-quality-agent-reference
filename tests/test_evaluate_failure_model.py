import pandas as pd
import torch

from src.models.failure_mlp import FailureMLP
from src.training.evaluate_failure_model import (
    evaluate_failure_model,
    probabilities_to_predictions,
    predict_probabilities,
)

def make_eval_sample():
    """
    평가 테스트에 사용할 작은 샘플 데이터를 만듭니다.

    실제 성능을 검증하는 목적이 아니라,
    평가 함수의 입출력 구조가 정상인지 확인하기 위한 데이터입니다.
    """

    X = pd.DataFrame(
        {
            "Air temperature [K]": [298.1, 298.2, 299.1, 300.2],
            "Process temperature [K]": [308.6, 308.7, 309.1, 310.2],
            "Rotational speed [rpm]": [1551, 1408, 1300, 1200],
            "Torque [Nm]": [42.8, 46.3, 55.1, 60.2],
            "Tool wear [min]": [0, 3, 180, 220],
            "Type": [0, 1, 1, 2],
        }
    )

    y = pd.Series([0, 0, 1, 1])

    return X, y

def test_predict_probabilities_output_range():
    """
    predict_probabilities가 0과 1 사이의 확률을 반환하는지 확인합니다.
    """

    X, _ = make_eval_sample()

    model = FailureMLP(input_dim=6)

    probabilities = predict_probabilities(model, X)

    # 샘플 4개에 대해 확률 1개씩 반환해야 합니다.
    assert probabilities.shape == (4, 1)

    # sigmoid를 거친 값이므로 0 이상이어야 합니다.
    assert torch.all(probabilities >= 0)

    # sigmoid를 거친 값이므로 1 이하이어야 합니다.
    assert torch.all(probabilities <= 1)

def test_probabilities_to_predictions():
    """
    probability가 threshold 기준으로 prediction으로 변환되는지 확인합니다.
    """

    probabilities = torch.tensor(
        [[0.1], [0.49], [0.5], [0.9]],
        dtype=torch.float32,
    )

    predictions = probabilities_to_predictions(
        probabilities=probabilities,
        threshold=0.5,
    )

    # 0.5 이상이면 1, 미만이면 0입니다.
    assert predictions.tolist() == [[0], [0], [1], [1]]

def test_evaluate_failure_model_returns_metrics():
    """
    evaluate_failure_model이 주요 평가 지표를 반환하는지 확인합니다.

    이 테스트는 성능이 좋은지 검증한는 것이 아니라,
    평가 함수가 에러 없이 실행되고 metric 값들이 0~1 범위에 있는지 확인합니다.
    """

    X, y = make_eval_sample()

    model = FailureMLP(input_dim=6)

    result = evaluate_failure_model(
        model=model,
        X_test=X,
        y_test=y,
        threshold=0.5,
    )

    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert result.threshold == 0.5