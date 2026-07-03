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

    Attributes
    ----------
    accuracy:
        전체 샘플 중 모델이 맞힌 비율입니다.

    precision:
        모델이 고장이라고 예측한 것 중 실제 고장인 비율입니다.
    
    recall:
        실제 고장 샘플 중 모델이 고장이라고 잡아낸 비율입니다.
    
    f1:
        precision과 recall의 균형을 나타내는 지표입니다.

    threshold:
        probability를 prediction으로 바꿀 때 사용한 기준값입니다.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float

def predict_probabilities(
        model: FailureMLP,
        X: pd.DataFrame,
) -> torch.Tensor:
    """
    학습된 모델로 각 샘플의 고장 확류을 계산합니다.

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

    # pandas DataFrame을 PyTorch Tensor로 변환합니다.
    X_tensor = dataframe_to_tensor(X)

    # 평가 / 추론 단계에서는 gradient 계산이 필요 없습니다.
    #
    # torch.no_grad()를 사용하면:
    # 1. 불필요한 gradient 추적을 하지 않습니다.
    # 2. 메모리를 덜 사용합니다.
    # 3. 추론 속도가 조금 더 효율적입니다.
    with torch.no_grad():
        # 모델 출력은 logit입니다.
        logits = model(X_tensor)

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
    고장 확률 probability를 최종 prediction label로 변환합니다.

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

    # probability가 threshold 이상이면 1, 아니면 0으로 예측합니ㅏㄷ.
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
    #   모델이 고장이라고 예측한 샘플이 하나도 없을 때
    #   precision 계산에서 0으로 처리합니다.
    precision = precision_score(y_true, y_pred, zero_division=0)

    # 실제 고장 중 모델이 고장이라고 잡아낸 비율입니다.
    recall = recall_score(y_true, y_pred, zero_division=0)

    # precision과 recall의 조화 평균입니다.
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return EvaluationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
    )