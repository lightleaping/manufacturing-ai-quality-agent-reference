from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch import nn


# 모델 성능을 어떤 기준으로 볼지 정합니다.
#
# 제조 고장 예측에서는 accuracy만 보면 위험합니다.
# 왜냐하면 AI4I 데이터처럼 정상 비율이 매우 높으면,
# 전부 정상이라고 예측해도 accuracy가 높게 나올 수 있기 때문입니다.
#
# 그래서 permutation importance에서는 accuracy뿐 아니라
# precision, recall, f1도 선택할 수 있게 만듭니다.
MetricName = Literal["accuracy", "precision", "recall", "f1"]


@dataclass(frozen=True)
class PermutationImportanceResult:
    """
    feature 하나에 대한 permutation importance 결과입니다.

    예를 들어 feature_name이 "Tool wear [min]"이라면,
    이 객체는 Tool wear 컬럼을 섞었을 때
    모델 성능이 얼마나 떨어졌는지 저장합니다.
    """

    # 중요도를 계산한 feature 이름입니다.
    feature_name: str

    # feature를 섞기 전 모델의 원래 성능입니다.
    baseline_score: float

    # feature를 여러 번 섞어서 평가한 성능의 평균입니다.
    permuted_score_mean: float

    # feature를 여러 번 섞어서 평가한 성능의 표준편차입니다.
    #
    # permutation은 무작위 섞기이므로,
    # 한 번만 섞으면 우연에 영향을 받을 수 있습니다.
    # 그래서 여러 번 반복하고 평균과 표준편차를 함께 봅니다.
    permuted_score_std: float

    # 중요도 평균입니다.
    #
    # 계산식:
    # importance = baseline_score - permuted_score
    #
    # feature를 섞었을 때 성능이 크게 떨어지면
    # baseline_score - permuted_score 값이 커집니다.
    importance_mean: float

    # importance의 표준편차입니다.
    importance_std: float

    # 몇 번 반복해서 섞었는지 저장합니다.
    n_repeats: int

    # 어떤 성능 지표를 기준으로 중요도를 계산했는지 저장합니다.
    metric_name: MetricName


@dataclass(frozen=True)
class PermutationImportanceSummary:
    """
    전체 feature importance 계산 결과를 담는 객체입니다.

    baseline_score는 feature를 섞기 전 원래 모델 성능이고,
    results에는 각 feature별 중요도가 들어갑니다.
    """

    baseline_score: float
    threshold: float
    metric_name: MetricName
    results: list[PermutationImportanceResult]


def dataframe_to_tensor(X: pd.DataFrame) -> torch.Tensor:
    """
    pandas DataFrame을 PyTorch Tensor로 변환합니다.

    모델은 pandas DataFrame을 직접 입력받지 못합니다.
    PyTorch 모델은 Tensor를 입력으로 받기 때문에,
    DataFrame → numpy array → torch.Tensor 순서로 변환합니다.

    중요한 점:
    - dtype은 float32로 맞춥니다.
    - 학습 때도 float32 Tensor를 사용했기 때문입니다.
    """

    return torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32)


def predict_probabilities(
    model: nn.Module,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    모델의 logit 출력을 probability로 변환합니다.

    Day 2에서 FailureMLP의 마지막 layer에는 Sigmoid를 넣지 않았습니다.
    그래서 모델 출력은 probability가 아니라 logit입니다.

    따라서 추론 시에는 다음 순서가 필요합니다.

    logit
    → torch.sigmoid(logit)
    → probability
    """

    # 평가 모드로 전환합니다.
    #
    # Dropout이 있는 모델은 train mode와 eval mode에서 동작이 다릅니다.
    # 평가/추론할 때는 Dropout을 꺼야 하므로 model.eval()을 사용합니다.
    model.eval()

    # 평가할 DataFrame을 Tensor로 바꿉니다.
    X_tensor = dataframe_to_tensor(X)

    # 평가할 때는 gradient 계산이 필요 없습니다.
    #
    # torch.no_grad()를 사용하면:
    # 1. 메모리를 덜 사용하고
    # 2. 계산이 조금 더 빠르고
    # 3. 실수로 gradient가 누적되는 것을 막을 수 있습니다.
    with torch.no_grad():
        logits = model(X_tensor)

        # logits의 shape이 (N, 1)일 수 있으므로 마지막 차원을 제거합니다.
        #
        # 예:
        # [[0.1], [0.7], [-1.2]]
        # -> [0.1, 0.7, -1.2]
        logits = logits.squeeze(-1)

        probabilities = torch.sigmoid(logits)

    # 이후 metric 계산은 numpy로 처리하기 위해 ndarray로 변환합니다.
    return probabilities.cpu().numpy()


def probabilities_to_predictions(
    probabilities: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    probability를 threshold 기준으로 0/1 prediction으로 변환합니다.

    예:
    probability = 0.72
    threshold = 0.70
    -> prediction = 1

    probability = 0.52
    threshold = 0.70
    -> prediction = 0
    """

    return (probabilities >= threshold).astype(int)


def calculate_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """
    binary classification의 confusion matrix 구성 요소를 계산합니다.

    현재 프로젝트에서 class 의미:

    0 = 정상
    1 = 고장

    TN = 실제 정상이고, 정상이라고 예측
    FP = 실제 정상인데, 고장이라고 예측
    FN = 실제 고장인데, 정상이라고 예측
    TP = 실제 고장이고, 고장이라고 예측
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_name: MetricName,
) -> float:
    """
    accuracy, precision, recall, f1 중 하나를 계산합니다.

    zero division을 직접 처리합니다.

    예를 들어 모델이 고장을 하나도 예측하지 않으면:
    TP = 0
    FP = 0

    precision = TP / (TP + FP)
              = 0 / 0

    이 경우 계산이 불가능하므로 0.0으로 처리합니다.
    """

    counts = calculate_confusion_counts(y_true=y_true, y_pred=y_pred)

    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    tp = counts["tp"]

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    if metric_name == "accuracy":
        return float(accuracy)

    if metric_name == "precision":
        return float(precision)

    if metric_name == "recall":
        return float(recall)

    if metric_name == "f1":
        return float(f1)

    raise ValueError(f"지원하지 않는 metric_name입니다: {metric_name}")


def calculate_model_score(
    model: nn.Module,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    threshold: float,
    metric_name: MetricName,
) -> float:
    """
    모델의 성능 점수를 계산합니다.

    흐름:

    X_test
    -> model
    -> logits
    -> sigmoid
    -> probabilities
    -> threshold 비교
    -> predictions
    -> metric 계산
    """

    probabilities = predict_probabilities(model=model, X=X)
    predictions = probabilities_to_predictions(
        probabilities=probabilities,
        threshold=threshold,
    )

    y_true = np.asarray(y).astype(int)

    return calculate_metric(
        y_true=y_true,
        y_pred=predictions,
        metric_name=metric_name,
    )


def permute_feature(
    X: pd.DataFrame,
    feature_name: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    특정 feature 컬럼 하나만 무작위로 섞은 DataFrame을 반환합니다.

    원본 X는 직접 수정하지 않습니다.

    이유:
    permutation importance는 feature별로 반복 평가를 해야 합니다.
    원본 X를 직접 바꿔버리면 다음 feature 평가에도 영향을 줍니다.

    따라서 반드시 X.copy()로 복사본을 만든 뒤,
    복사본의 특정 feature만 섞습니다.
    """

    if feature_name not in X.columns:
        raise ValueError(f"X에 존재하지 않는 feature입니다: {feature_name}")

    X_permuted = X.copy()

    # rng.permutation은 값의 순서를 무작위로 섞습니다.
    #
    # 중요한 점:
    # 값 자체를 새로운 값으로 바꾸는 것이 아닙니다.
    # 기존 컬럼에 있던 값들의 순서만 바꿉니다.
    #
    # 예:
    # [10, 20, 30]
    # → [30, 10, 20]
    X_permuted[feature_name] = rng.permutation(X_permuted[feature_name].to_numpy())

    return X_permuted


def calculate_permutation_importance(
    model: nn.Module,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    threshold: float = 0.5,
    metric_name: MetricName = "f1",
    n_repeats: int = 5,
    random_state: int = 42,
) -> PermutationImportanceSummary:
    """
    전체 feature에 대해 permutation importance를 계산합니다.

    계산 순서:

    1. 원본 X로 baseline 성능을 계산합니다.
    2. feature 하나를 선택합니다.
    3. 해당 feature 컬럼만 무작위로 섞습니다.
    4. 섞인 데이터로 모델 성능을 다시 계산합니다.
    5. baseline 성능에서 섞은 후 성능을 뺍니다.
    6. 이 값을 feature importance로 봅니다.
    7. 모든 feature에 대해 반복합니다.

    중요:
    permutation importance는 개별 sample 하나의 설명이 아닙니다.
    test set 전체에서 어떤 feature가 모델 성능에 중요한지 보는 방법입니다.
    """

    if X.empty:
        raise ValueError("X가 비어 있습니다.")

    if len(X) != len(y):
        raise ValueError("X와 y의 길이가 서로 다릅니다.")

    if n_repeats <= 0:
        raise ValueError("n_repeats는 1 이상이어야 합니다.")

    if metric_name not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError(f"지원하지 않는 metric_name입니다: {metric_name}")

    rng = np.random.default_rng(random_state)

    baseline_score = calculate_model_score(
        model=model,
        X=X,
        y=y,
        threshold=threshold,
        metric_name=metric_name,
    )

    results: list[PermutationImportanceResult] = []

    for feature_name in X.columns:
        permuted_scores: list[float] = []

        for _ in range(n_repeats):
            X_permuted = permute_feature(
                X=X,
                feature_name=feature_name,
                rng=rng,
            )

            permuted_score = calculate_model_score(
                model=model,
                X=X_permuted,
                y=y,
                threshold=threshold,
                metric_name=metric_name,
            )

            permuted_scores.append(permuted_score)

        permuted_scores_array = np.asarray(permuted_scores, dtype=float)

        importances = baseline_score - permuted_scores_array

        result = PermutationImportanceResult(
            feature_name=feature_name,
            baseline_score=float(baseline_score),
            permuted_score_mean=float(permuted_scores_array.mean()),
            permuted_score_std=float(permuted_scores_array.std()),
            importance_mean=float(importances.mean()),
            importance_std=float(importances.std()),
            n_repeats=n_repeats,
            metric_name=metric_name,
        )

        results.append(result)

    # 중요도가 큰 feature가 위로 오도록 정렬합니다.
    #
    # importance_mean이 크다
    # = 그 feature를 섞었을 때 성능이 많이 떨어졌다
    # = 모델이 그 feature에 더 민감했을 가능성이 높다
    results = sorted(
        results,
        key=lambda item: item.importance_mean,
        reverse=True,
    )

    return PermutationImportanceSummary(
        baseline_score=float(baseline_score),
        threshold=threshold,
        metric_name=metric_name,
        results=results,
    )


def get_top_important_features(
    summary: PermutationImportanceSummary,
    top_k: int = 3,
) -> list[PermutationImportanceResult]:
    """
    중요도 상위 feature만 반환합니다.

    이후 Agent 답변이나 evidence에 붙일 때
    전체 feature를 모두 보여주면 너무 길어질 수 있습니다.

    그래서 상위 3개 정도만 선택할 수 있게 합니다.
    """

    if top_k <= 0:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    return summary.results[:top_k]


def format_permutation_importance_as_evidence(
    summary: PermutationImportanceSummary,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    permutation importance 결과를 evidence 형식으로 변환합니다.

    Day 5의 evidence 형식과 비슷하게 맞춰 둡니다.

    다만 여기서 value는 개별 sample의 feature 값이 아닙니다.
    permutation importance는 전체 test set 기준 설명이기 때문입니다.

    그래서 value는 None으로 두고,
    importance_mean과 message를 별도로 제공합니다.
    """

    top_features = get_top_important_features(
        summary=summary,
        top_k=top_k,
    )

    evidence: list[dict[str, Any]] = []

    for result in top_features:
        evidence.append(
            {
                "feature": result.feature_name,
                "value": None,
                "importance": result.importance_mean,
                "metric_name": result.metric_name,
                "message": (
                    f"{result.feature_name} 컬럼을 섞었을 때 "
                    f"{result.metric_name} 기준 성능이 평균 "
                    f"{result.importance_mean:.4f}만큼 감소했습니다."
                ),
            }
        )

    return evidence