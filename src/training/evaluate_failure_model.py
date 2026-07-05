"""
evaluate_failure_model.py

학습된 FailureMLP 모델을 평가하는 함수들을 정의하는 파일입니다.

이 파일의 목표는 다음과 같습니다.

1. pandas DataFrame / Series를 Tensor로 변환합니다.
2. 학습된 모델에서 logits를 계산합니다.
3. logits를 sigmoid로 변환해 probability를 구합니다.
4. probability를 threshold와 비교해 prediction을 만듭니다.
5. accuracy, precision, recall, f1-score를 계산합니다.

제조 고장 예측 문제에서는 정상 데이터가 많고 고장 데이터가 적을 수 있으므로,
accuracy만 보는 것은 위험합니다.

따라서 precision, recall, f1-score도 함께 확인합니다.
"""

from dataclasses import dataclass

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.failure_mlp import FailureMLP
from src.training.train_failure_model import dataframe_to_tensor, target_to_tensor

@dataclass
class EvaluationResult:
    """
    모델 평가 결과를 담는 데이터 클래스입니다.
    
    즉, EvaluationResult는 모델 평가 결과를 이름으로 묶어두기 위한 자료 구조입니다.

    Attributes
    ----------
    accuracy:
        - 전체 데이터 중 모델이 맞힌 비율입니다.
        - 하지만 제조 고장 데이터처럼 정상 데이터가 훨씬 많으면 accuracy만으로는 위험합니다.

    precision:
        - 모델이 '고장'이라고 예측한 것 중 실제 고장인 비율입니다.
        - precision이 낮으면 정상인데 고장이라고 잘못 잡는 경우가 많다는 뜻입니다.
            
    recall:
        - 실제 고장 중 모델이 고장이라고 잡아낸 비율입니다.
        - 제조 고장 예측에서는 실제 고장을 놓치는 것이 위험하므로 recall이 중요합니다.
    
    f1:
        - precision과 recall의 균형을 나타내는 지표입니다.

    threshold:
        - probability를 0 또는 1 prediction으로 바꿀 대 사용하는 기준값입니다.
        - 기본 baseline은 0.5입니다.

    true_negative:
        - 실제 정상이고 모델도 정상이라고 예측한 개수입니다.

    true_positive:
        - 실제 고장이고 모델이 고장이라고 예측한 개수입니다.

    false_negative:
        - 실제 고장인데 모델이 정상이라고 예측한 개수입니다.
        - 제조 AI에서는 이 값이 특히 중요합니다.

    true_positive:
        - 실제 고장이고 모델도 고장이라고 예측한 개수입니다.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int


def predict_probabilities(
        model: FailureMLP,
        X: pd.DataFrame,
) -> torch.Tensor:
    """
    학습된 모델을 사용해 각 샘플의 고장 확률을 계산합니다.

    중요한 흐름:
    1. pandas DataFrmae을 PyTorch Tensor로 변환합니다.
    2. model.eval()로 평가 모드로 전환합니다.
    3. torch.no_grad()로 gradient 계산을 끕니다.
    4. 모델 출력 logit에 sigmoid를 적용해 probability로 변환합니다.

    왜 model.eval()을 쓰는가?
    - Dropout 같은 layer는 학습 중에는 무작위로 일부 뉴런을 끄지만,
        평가 / 추론 시에는 꺼져야 합니다.
    - model.eval()을 호출해야 평가 결과가 안정적입니다..

    왜 torch.no_grad()를 쓰는가?
    - 평가 단계에서는 weight를 업데이트하지 않습니다.
    - 따라서 gradient 계산이 필요 없습니다.
    - 메모리 사용량과 계산량을 줄일 수 있습니다.

    Parameters
    ----------
    model:
        학습된 FailureMLP 모델입니다.

    X:
        평가할 feature DataFrame입니다.
    
    Returns
    -------
    tonch.Tensor:
        각 샘플의 고장 probability입니다.
        shape은 [sample_count, 1]입니다.
    """

    # pandas DataFrame을 PyTorch Tensor로 변환합니다.
    X_tensor = dataframe_to_tensor(X)

    # 평가 시에는 Dropout이 꺼져야 하므로 eval 모드로 전환합니다.
    #
    # model.train():
    #   학습 모드입니다.
    #   Dropout이 활성화됩니다.
    #
    # model.eval():
    #   평가 / 추론 모드입니다.
    #   Dropout이 비활성화됩니다.
    model.eval()

    # 평가 / 추론 단계에서는 gradient 계산이 필요 없습니다.
    #
    # torch.no_grad()를 사용하면:
    # 1. 불필요한 gradient 추적을 하지 않습니다.
    # 2. 메모리를 덜 사용합니다.
    # 3. 추론 속도가 조금 더 효율적입니다.
    with torch.no_grad():
        # 모델 출력은 logit입니다.
        logits = model(X_tensor)

        # 모델은 sigmoid 전의 원시 함수인 logit을 출력합니다.
        # 따라서 추론 / 평가 단계에서는 sigmoid를 직접 적용해 확률로 바꿉니다.
        #
        # logit은 확률이 아니므로,
        # sigmoid를 적용해 0과 1 사이의 probability로 변환합니다.
        probabilities = torch.sigmoid(logits)

    return probabilities

def probabilities_to_predictions(
    probabilities: torch.Tensor,

    # threshold는 probability를 0 또는 1의 최종 예측 label로 바꾸는 기준입니다.
    #
    # 기본값 0.5는 이진 분류에서 가장 일반적인 baseline입니다.
    #
    # probability가 0.5 이상이면 class 1로,
    # 0.5 미만이면 class 0으로 판단합니다.
    #
    # 이 프로젝트에서는:
    #   class 0 = 정상
    #   class 1 = 고장
    #
    # 따라서:
    #   probability >= threshold -> 고장 예측
    #   probability <= threshold -> 정상 예측
    #
    # 다만 제조 고장 예측에서는 미탐과 오탐의 비용이 다르기 때문에
    # threshold는 고정값이 아니라 조정 대상입니다.
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    고장 확률 probability를 최종 0 또는 1 prediction으로 변환합니다.

    threshold = 0.5 기준:
    - probability >= 0.5 이면 1, 즉 고장
    - probability < 0.5 이면 0, 즉 정상

    probabilities는 Tensor이고 threshold는 Python float입니다.
    PyTorch는 Tensor의 모든 원소에 threshold를 비교합니다.

    예:
    probabilities = [[0.10], [0.49], [0.50], [0.90]]
    threshold = 0.5

    probabilities >= threshold
    -> [[False], [False], [True], [True]]

    .int()
    ->[[0], [0], [1], [1]]

    Parameters
    ----------
    probabilities:
        sigmoid를 거친 고장 확률 Tensor입니다.
        shape은 보통 [sample_count, 1]입니다.

    threshold:
        고장으로 판단할 기준값입니다.
        기본값 0.5는 probability가 50% 이상이면
        고장으로 판단한다는 뜻입니다.

        Returns
        -------
        torch.Tensor:
            0 또는 1로 구성된 예측 label Tensor입니다.
            0 = 정상
            1 = 고장
    """

    # probability가 threshold 이상이면 1, 아니면 0으로 예측합니다.
    #
    # 예:
    # probability = 0.82, threshold = 0.5
    # -> prediction = 1 -> 고장
    #
    # probability = 0.21, threshold = 0.5
    # -> prediction = 0 -> 정상
    #
    # threshold를 낮추면:
    #   더 많은 샘플을 고장으로 예측합니다.
    #   recall은 올라갈 수 있지만 precision은 낮아질 수 있습니다.
    #
    # threshold를 높이면:
    #   더 확실한 샘플만 고장으로 예측합니다.
    #   precision은 올라갈 수 있지만 recall은 낮아질 수 있습니다.

    # probabilities는 Tensor이고 threshold는 Python float입니다.
    #
    # 예:
    # probabilities = tensor([
    #   [0.10],
    #   [0.49],
    #   [0.50],
    #   [0.90],
    # ])
    #
    # threshold = 0.5
    #
    # PyTorch는 Tensor와 float을 비교할 때,
    # float 값 하나를 Tensor의 모든 원소에 적용합니다.
    #
    # 즉, 아래 비교는 내부적으로 각 원소별로 수행됩니다.
    #
    # probabilities >= threshold
    #
    # 예:
    # 0.10 >= 0.5 -> False
    # 0.49 >= 0.5 -> False
    # 0.50 >= 0.5 -> True
    # 0.90 >= 0.5 -> True
    #
    # 이처럼 Tensor 안의 각 값을 독립적으로 비교하는 방식을
    # element-wise operation, 즉 원소별 연산이라고 합니다.
    #
    # 비교 결과는 Boolean Tensor입니다.
    #
    # 예:
    # tensor([
    #   [False],
    #   [False],
    #   [True],
    #   [True],
    # ])
    #
    # 하지만 최종 prediction label은
    # 0 = 정상
    # 1 = 고장
    # 형태의 숫자로 다루는 것이 편합니다.
    #
    # 그래서 .int()를 붙여 Boolean 값을 정수로 변환합니다.
    #
    # False -> 0
    # True -> 1
    #
    # 최종 결과:
    # tensor([
    #   [0],
    #   [0],
    #   [1],
    #   [1],
    # ])
    #
    # shape은 그대로 유지됩니다.
    # probabilities shape이 [sample_count, 1]이면,
    # predictions shape도 [sample_count, 1]입니다.
    predictions = (probabilities >= threshold).int()

    return predictions

def calculate_confusion_counts(
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
) -> tuple [int, int, int, int]:
    """
    실제 정답 y_true와 모델 예측 y_pred를 비교해
    confusion matrix의 네 값을 직접 계산합니다.

    반환 순서:
    true_negative, false_positive, false_negative, true_positive

    y_true:
    - 실제 정답입니다.
    - 0이면 정상, 1이면 고장입니다.

    y_pred:
    - 모델 예측입니다.
    - 0이면 정상 예측, 1이면 고장 예측입니다.
    """

    # shape이 [N, 1]일 수도 있으므로 1차원으로 펼칩니다.
    y_true = y_true.view(-1).int()
    y_pred = y_pred.view(-1).int()

    true_negative = ((y_true == 0) & (y_pred == 0)).sum().item()
    false_positive = ((y_true == 0) & (y_pred == 1)).sum().item()
    false_negative = ((y_true == 1) & (y_pred == 0)).sum().item()
    true_positive = ((y_true == 1) & (y_pred == 1)).sum().item()

    return true_negative, false_positive, false_negative, true_positive


def evaluate_failure_model(
    model: FailureMLP,
    X_test: pd.DataFrame,
    y_test: pd.Series,

    # threshold는 모델이 출력한 probability를
    # 최종 prediction label로 바꾸기 위한 기준값입니다.
    #
    # 현재 모델은 학습 단계에서 logit을 출력합니다.
    # 평가 / 추론 단계에서는 torch.sigmoid(logits)를 적용해
    # 0과 1 사이의 probability로 변환합니다.
    #
    # 이 프로젝트에서는 target label을 다음처럼 해석합니다.
    #
    # 0 = 정상
    # 1 = 고장
    #
    # 따라서 probability는 "고장일 확률"처럼 해석할 수 있습니다.
    #
    # threshold=0.5는 이진 분류에서 가장 기본적인 baseline 기준입니다.
    #
    # probability >= 0.5 이면:
    #   class 1, 즉 고장 가능성이 어 크다고 보고 prediction = 1로 판단합니다.
    #
    # probability < 0.5 이면:
    #   class 0, 즉 정상 가능성이 더 크다고 보고 prediction = 0으로 판단합니다.
    #
    # 예:
    #   probability = 0.82 -> 0.5 이상 -> prediction = 1, 고장
    #   probability = 0.21 -> 0.5 미만 -> prediction = 0, 정상
    #
    # 단, 0.5가 항상 최적 기준이라는 뜻은 아닙니다.
    #
    # 제조 설비 고장 예측에서는 실제 고장을 정상으로 놓치는 미탐이
    # 큰 리스크가 될 수 있습니다.
    #
    # 미탐을 줄이고 recall을 높이고 싶다면 threshold를 0.3처럼 낮출 수 있습니다.
    # 이 경우 더 많은 샘플을 고장으로 잡기 때문에 recall은 올라갈 수 있지만,
    # 정상 샘플까지 고장으로 잘못 잡는 오탐이 느러 precision은 낮아질 수 있습니다.
    #
    # 반대로 오탐을 줄이고 precision을 높이고 싶다면 threshold를 0.7처럼 높일 수 있습니다.
    # 이 경우 고장 판단은 더 신중해지지만,
    # 실제 고장을 놓쳐 recall이 낮아질 수 있습니다.
    #
    # 따라서 현재 0.5는 초기 baseline 기준이고,
    # 이후 평가 결과와 제조 현장의 비용 기준에 따라 조정해야 하는 hyperparameter입니다.
    threshold: float = 0.5,
) -> EvaluationResult:
    """
    학습된 FailureMLP 모델을 test data로 평가합니다.

    전체 흐름:
    1. X_test를 모델에 넣어 probability를 계산합니다.
    2. threshold 기준으로 prediction을 만듭니다.
    3. y_test와 prediction을 비교합니다.
    4. accuracy, precision, recall, f1을 계산합니다.
    5. confusion matrix의 TN, FP, FN, TP를 계산합니다.

    Parameters
    ----------
    model:
        학습된 FailureMLP 모델입니다.

    X_test:
        평가용 feature DataFrame입니다.

    y_test:
        평가용 실제 정답 label입니다.
        0 = 정상
        1 = 고장

    threshold:
        probability를 prediction으로 변환할 기준값입니다.

    Returns
    -------
    EvaluationResult:
        accuracy, precision, recall, f1-score를 담은 평가 결과입니다.
    """

    # 모델로 고장 확률을 계산합니다.
    probabilities = predict_probabilities(model, X_test)

    # 확률을 threshold 기준으로 0 / 1 prediction으로 바꿉니다.
    predictions = probabilities_to_predictions(
        probabilities=probabilities,
        threshold=threshold,   
    )

    # sklearn metric 함수들은 보통 1차원 배열을 기대합니다.
    y_true_array = y_test.to_numpy().reshape(-1)
    y_pred_array = predictions.numpy().reshape(-1)

    accuracy = accuracy_score(y_true_array, y_pred_array)

    # y_test도 Tensor로 변환합니다.
    #
    # target_to_tensor를 쓰면 shape이 [sample_count, 1]이 됩니다.
    y_tensor = target_to_tensor(y_test)

    # sklearn metric 함수들은 보통 1차원 배열을 기대하므로
    # .view(-1)을 사용해 [sample_count] 형태로 펼칩니다.
    y_true = y_tensor.view(-1).numpy()
    y_pred = predictions.view(-1).numpy()

    # 전체 중 맞힌 비율입니다.
    accuracy = accuracy_score(y_true, y_pred)

    # 고장이라고 예측한 것 중 실제 고장인 비율입니다.
    #
    # zero_division=0:
    #   모델이 고장이라고 예측한 샘플이 하나도 없을 때 precision 계산에서 분모가 0이 될 수 있습니다.
    #   이때 에러를 내지 않고 precision 계산에서 0으로 처리합니다.
    precision = precision_score(y_true, y_pred, zero_division=0)

    # 실제 고장 중 모델이 고장이라고 잡아낸 비율입니다.
    recall = recall_score(y_true, y_pred, zero_division=0)

    # precision과 recall의 조화 평균입니다.
    f1 = f1_score(y_true, y_pred, zero_division=0)

    y_true_tensor = torch.tensor(y_true_array, dtype=torch.int)
    y_pred_tensor = torch.tensor(y_pred_array, dtype=torch.int)

    true_negative, false_positive, false_negative, true_positive = (
        calculate_confusion_counts(y_true_tensor, y_pred_tensor)
    )

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )

def compare_thresholds(
        model: FailureMLP,
        X_test: pd.DataFrame,
        y_test:pd.Series,
        thresholds: list[float] | None = None,
) -> list[EvaluationResult]:
    """
    여러 threshold를 비교합니다.

    왜 필요한가?
    - threshold 0.5는 기본 baseline일 뿐입니다.
    - 제조 고장 예측에서는 고장을 놓치지 않는 것이 중요할 수 있습니다.
    - threshold를 낮추면 고장으로 판단하는 기준이 완화되어 recall이 올라갈 수 있습니다.
    - False positive가 늘어나 precision이 떨어질 수 있습니다.

    기본 비교:
    - 0.3: 고장으로 더 쉽게 판단
    - 0.5: 기본 baseline
    - 0.7: 고장으로 더 엄격하게 판단
    """

    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    results = []

    for threshold in thresholds:
        result = evaluate_failure_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            threshold=threshold,
        )
        results.append(result)

    return results

def create_threshold_grid(
    start: float = 0.50,
    end: float = 0.90,
    step: float = 0.05,
) -> list[float]:
    """
    threshold 비교에 사용할 후보값 리스트를 만듭니다.

    기존에는 threshold를 [0.3, 0.5, 0.7]처럼 듬성듬성 비교했습니다.

    하지만 실제 운영 기준을 정할 때는
    0.50, 0.55, 0.60, 0.65처럼 더 촘촘히 비교하는 것이 좋습니다.

    예:
    start = 0.50
    end = 0.90
    step = 0.05

    결과:
    [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

    왜 round를 사용하는가?
    - Python float 계산에서는 0.1 + 0.2가 정확히 0.3이 아니라
      0.30000000000000004처럼 표현될 수 있습니다.
    - threshold 출력과 비교를 깔끔하게 하기 위해 소수점 둘째 자리로 반올림합니다.
    """

    thresholds = []

    current = start

    # end도 포함하기 위해 current <= end 조건을 사용합니다.
    #
    # 예:
    # start=0.50, end=0.90이면
    # 0.90까지 포함되어야 합니다.
    #
    # Python float은 0.05, 0.1 같은 소수를 내부적으로 정확히 표현하지 못할 수 있습니다.
    #
    # 예를 들어 사람이 보기에는 current가 0.90이어야 하는데,
    # 실제 내부 값은 0.9000000000000001처럼 아주 조금 커질 수 있습니다.
    #
    # 이 상태에서 while current <= end 라고만 쓰면,
    #
    # 0.9000000000000001 <= 0.90
    #
    # 이 비교가 False가 되어 마지막 threshold 0.90이 누락될 수 있습니다.
    #
    # 그래서 end에 아주 작은 여유값 1e-9를 더합니다.
    #
    # 1e-9 = 0.000000001
    #
    # 이 값은 threshold 의미를 바꿀 정도로 큰 값이 아니라,
    # float 계산 오차를 허용하기 위한 작은 안전장치입니다.
    while current <= end + 1e-9:

        # round(current, 2)를 사용하는 이유:
        # - 0.7000000000000001 같은 값을 0.70처럼 깔끔하게 만들기 위해서입니다.
        # - threshold 출력과 테스트 비교를 안정적으로 하기 위해 사용합니다.
        thresholds.append(round(current, 2))
        current += step

    return thresholds

# 정수 기반으로 반복 횟수를 다루는 것도 가능합니다.
# def create_threshold_grid(
#     start: float = 0.50,
#     end: float = 0.90,
#     step: float = 0.05,
# ) -> list[float]:
#     """
#     float 누적 오차를 피하기 위해 반복 횟수를 정수로 계산하는 버전입니다.
#     """

#     count = int(round((end - start) / step))

#     thresholds = [
#         round(start + step * index, 2)
#         for index in range(count + 1)
#     ]

#     return thresholds

def select_best_threshold_by_f1(
    results: list[EvaluationResult],
) -> EvaluationResult:
    """
    여러 threshold 평가 결과 중 f1-score가 가장 높은 결과를 선택합니다.

    왜 f1 기준을 보는가?
    - precision은 고장이라고 예측한 것 중 실제 고장 비율입니다.
    - recall은 실제 고장 중 모델이 잡아낸 비율입니다.
    - f1은 precision과 recall의 균형을 보는 지표입니다.

    제조 고장 예측에서는 recall이 특히 중요하지만,
    precision이 너무 낮으면 정상 설비를 고장으로 너무 많이 오탐할 수 있습니다.

    그래서 우선 f1-score가 가장 높은 threshold를
    균형 후보로 확인합니다.

    주의:
    - f1이 가장 높은 threshold가 항상 현장 최적 기준이라는 뜻은 아닙니다.
    - 실제 현장에서는 false negative 비용과 false positive 비용을 함께 고려해야 합니다.
    """

    if not results:
        raise ValueError("threshold 평가 결과가 비어 있어 best threshold를 선택할 수 없습니다.")

    best_result = max(
        results,
        key=lambda result: result.f1,
    )

    return best_result

def select_best_threshold_with_min_recall(
    results: list[EvaluationResult],
    min_recall: float = 0.85,
) -> EvaluationResult:
    """
    recall이 일정 기준 이상인 threshold들 중 f1-score가 가장 높은 결과를 선택합니다.

    제조 고장 예측에서는 실제 고장을 정상으로 놓치는 FN이 위험할 수 있습니다.
    그래서 단순히 f1이 가장 높은 threshold만 보는 것이 아니라,
    recall이 최소 기준 이상인 후보 중에서 균형이 좋은 threshold를 찾을 수 있습니다.

    예:
    min_recall = 0.85

    의미:
    - 실제 고장 중 최소 85% 이상은 잡는 threshold 후보만 남깁니다.
    - 그 후보들 중 f1-score가 가장 높은 결과를 선택합니다.

    만약 조건을 만족하는 threshold가 없다면:
    - ValueError를 발생시킵니다.
    - 이 경우 min_recall 기준을 낮추거나,
      모델 성능 개선이 더 필요하다고 해석할 수 있습니다.
    """

    candidate_results = [
        result
        for result in results
        if result.recall >= min_recall
    ]

    if not candidate_results:
        raise ValueError(
            f"recall >= {min_recall} 조건을 만족하는 threshold가 없습니다."
        )

    best_result = max(
        candidate_results,
        key=lambda result: result.f1,
    )

    return best_result