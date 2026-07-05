import pandas as pd
import torch

from src.models.failure_mlp import FailureMLP
from src.training.evaluate_failure_model import (
    calculate_confusion_counts,
    compare_thresholds,
    create_threshold_grid,
    evaluate_failure_model,
    probabilities_to_predictions,
    select_best_threshold_by_f1,
    select_best_threshold_with_min_recall,
    predict_probabilities
)

def make_sample_X() -> pd.DataFrame:
    """
    테스트용 sample feature DataFrame을 만듭니다.

    실제 AI4I 전체 CSV를 쓰지 않고 작은 데이터를 직접 만드는 이유:
    - 테스트가 빠릅니다.
    - 외부 파일에 의존하지 않습니다.
    - 입력 shape과 함수 동작만 명확히 검증할 수 있습니다.
    """

    return pd.DataFrame(
        {
            "Air temperature [K]": [300.1, 301.2, 299.8, 305.0],
            "Process temperature [K]": [310.2, 311.1, 309.7, 315.0],
            "Rotational speed [rpm]": [1500, 1600, 1400, 1300],
            "Torque [Nm]": [40.0, 42.0, 39.5, 50.0],
            "Tool wear [min]": [10, 20, 30, 200],
            "Type": [0, 1, 2, 0],
        }
    )

def make_sample_y() -> pd.Series:
    """
    테스트용 정답 label입니다.
    
    0 = 정상
    1 = 고장
    """

    return pd.Series([0, 0, 0, 1])

def test_probabilities_to_predictions_uses_threshold() -> None:
    """
    probability가 threshold 이상이면 1,
    threshold보다 작으면 0으로 바뀌는지 확인합니다.
    """

    probabilities = torch.tensor([[0.10], [0.49], [0.50], [0.90]])

    predictions = probabilities_to_predictions(
        probabilities=probabilities,
        threshold=0.5,
    )

    expected = torch.tensor([[0], [0], [1], [1]], dtype=torch.int)

    assert torch.equal(predictions, expected)

def test_calculate_confusion_counts() -> None:
    """
    confusion matrix의 TN, FP, FN, TP 계산이 맞는지 확인합니다.

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    해석:
    - 첫 번째: 실제 0, 예측 0 -> TN
    - 두 번째: 실제 0, 예측 1 -> FP
    - 세 번째: 실제 1, 예측 0 -> FN
    - 네 번째: 실제 1, 예측 1 -> TP
    """

    y_true = torch.tensor([0, 0, 1, 1])
    y_pred = torch.tensor([0, 1, 0, 1])

    true_negative, false_positive, false_negative, true_positive = (
        calculate_confusion_counts(y_true, y_pred)
    )

    assert true_negative == 1
    assert false_positive == 1
    assert false_negative == 1
    assert true_positive == 1

def test_evaluate_failure_model_returns_metrics() -> None:
    """
    evaluate_failure_model이 EvaluationResult를 반환하고,
    각 mertric이 0 이상 1 이하 범위인지 확인합니다.

    여기서는 모델 성능이 좋은지를 테스트하는 것이 아닙니다.
    평가 함수가 에러 없이 실행되고,
    metric 구조를 정상적으로 반환하는지를 테스트합니다.
    """

    X = make_sample_X()
    y = make_sample_y()

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

    total_count = (
        result.true_negative
        + result.false_positive
        + result.false_negative
        + result.true_positive
    )

    assert total_count == len(y)

def test_compare_thresholds_returns_one_result_per_threshold() -> None:
    """
    threshold 0.3, 0.5, 0.7을 비교했을 때
    threshold 개수만큼 EvaluationResult가 반환되는지 확인합니다.
    """

    X = make_sample_X()
    y = make_sample_y()

    model = FailureMLP(input_dim=6)

    thresholds = [0.3, 0.5, 0.7]

    results = compare_thresholds(
        model=model,
        X_test=X,
        y_test=y,
        thresholds=thresholds,
    )

    assert len(results) == len(thresholds)

    returned_thresholds = [result.threshold for result in results]

    assert returned_thresholds == thresholds

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

def test_create_threshold_grid() -> None:
    """
    threshold grid가 start부터 end까지 step 간격으로 생성되는지 확인합니다.

    예:
    start=0.50
    end=0.70
    step=0.05

    expected:
    [0.50, 0.55, 0.60, 0.65, 0.70]
    """

    thresholds = create_threshold_grid(
        start=0.50,
        end=0.70,
        step=0.05,
    )

    assert thresholds == [0.50, 0.55, 0.60, 0.65, 0.70]


def test_select_best_threshold_by_f1() -> None:
    """
    f1-score가 가장 높은 EvaluationResult를 선택하는지 확인합니다.
    """

    X = make_sample_X()
    y = make_sample_y()
    model = FailureMLP(input_dim=6)

    results = compare_thresholds(
        model=model,
        X_test=X,
        y_test=y,
        thresholds=[0.3, 0.5, 0.7],
    )

    best_result = select_best_threshold_by_f1(results)

    f1_scores = [result.f1 for result in results]

    assert best_result.f1 == max(f1_scores)


def test_select_best_threshold_with_min_recall() -> None:
    """
    recall이 일정 기준 이상인 결과들 중 f1-score가 가장 높은 결과를 선택하는지 확인합니다.

    이 테스트에서는 실제 성능 자체를 검증하는 것이 아닙니다.
    함수가 조건 필터링과 best selection을 정상 수행하는지 확인합니다.
    """

    X = make_sample_X()
    y = make_sample_y()
    model = FailureMLP(input_dim=6)

    results = compare_thresholds(
        model=model,
        X_test=X,
        y_test=y,
        thresholds=[0.3, 0.5, 0.7],
    )

    # 테스트 샘플과 랜덤 초기화 모델에서는 recall 값이 상황마다 달라질 수 있습니다.
    # 그래서 min_recall을 0.0으로 두어 최소 한 개 이상의 후보가 생기도록 합니다.
    best_result = select_best_threshold_with_min_recall(
        results=results,
        min_recall=0.0,
    )

    candidate_results = [
        result
        for result in results
        if result.recall >= 0.0
    ]

    candidate_f1_scores = [result.f1 for result in candidate_results]

    assert best_result.f1 == max(candidate_f1_scores)