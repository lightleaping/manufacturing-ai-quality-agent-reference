# src/interpretability/shap_explainer.py

"""
Day 8 - SHAP 기반 개별 예측 설명 모듈

이 파일의 목적
----------------
Day 6에서는 permutation importance로 전체 test set 기준 feature importance를 계산했다.
Day 7에서는 SHAP를 바로 적용하기 전에 LocalFeatureContribution,
LocalExplanationResult, format_local_explanation_as_evidence 구조를 먼저 만들었다.

Day 8에서는 실제 SHAP value를 계산해서
Day 7의 local explanation schema에 연결한다.

중요한 전제
----------------
현재 FailureMLP 모델은 마지막 layer에 Sigmoid가 없다.

따라서 model(sample)의 출력은 probability가 아니라 logit이다.

즉, 이 파일에서 계산하는 SHAP value는 기본적으로
"feature가 probability를 얼마나 올렸는가"가 아니라
"feature가 logit 출력을 얼마나 올리거나 내렸는가"를 의미한다.

정확한 표현
----------------
"SHAP 기준으로 이 feature는 고장 위험 logit을 높이는 방향으로 작용했다."

부정확한 표현
----------------
"모델은 이 feature 때문에 고장이라고 판단했다."

전체 흐름
----------------
background_tensor
sample_tensor
↓
shap.DeepExplainer(model, background_tensor)
↓
shap_values = explainer.shap_values(sample_tensor)
↓
feature별 SHAP contribution 계산
↓
LocalFeatureContribution으로 변환
↓
LocalExplanationResult로 변환
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import shap

from src.interpretability.local_explanation import (
    LocalExplanationResult,
    LocalFeatureContribution,
)


def sigmoid_float(logit: float) -> float:
    """
    logit 값을 probability로 변환한다.

    logit
    -----
    모델의 raw output이다.
    현재 FailureMLP는 마지막에 Sigmoid가 없으므로 model(x)는 logit을 반환한다.

    probability
    -----------
    logit에 sigmoid를 적용한 0~1 사이 값이다.

    추론에서는 이 probability를 threshold와 비교해서
    최종 prediction 0 또는 1을 만든다.

    공식
    ----
    probability = 1 / (1 + exp(-logit))
    """

    # logit 값이 너무 크거나 작으면 exp 계산에서 overflow가 날 수 있다.
    # 일반적인 모델 출력에서는 거의 문제되지 않지만,
    # 안전하게 -50~50 범위로 잘라준다.
    clipped_logit = np.clip(logit, -50, 50)

    return float(1.0 / (1.0 + np.exp(-clipped_logit)))


def ensure_2d_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """
    입력 tensor가 2차원인지 확인한다.

    이 프로젝트의 tabular input은 다음 형태여야 한다.

        (sample_count, feature_count)

    예:
        background_tensor.shape = (100, 6)
        sample_tensor.shape     = (1, 6)

    만약 sample 하나가 (6,) 형태로 들어오면 batch 차원이 없는 상태다.
    이 경우 unsqueeze(0)을 사용해서 (1, 6)으로 바꿔준다.
    """

    if tensor.ndim == 1:
        return tensor.unsqueeze(0)

    if tensor.ndim != 2:
        raise ValueError(
            f"{name} must be a 2D tensor with shape "
            f"(n_samples, n_features). Current shape: {tuple(tensor.shape)}"
        )

    return tensor


def build_background_tensor(
    X_tensor: torch.Tensor,
    background_size: int = 100,
    seed: int = 42,
) -> torch.Tensor:
    """
    SHAP 계산에 사용할 background tensor를 만든다.

    background data란?
    ------------------
    SHAP가 expected value를 계산하기 위해 사용하는 기준 sample 집합이다.

    쉽게 말하면:
        "평균적인 입력 상태와 비교했을 때,
         현재 sample의 각 feature가 모델 출력을 얼마나 움직였는가?"
    를 계산하기 위한 기준 데이터다.

    주의
    ----
    background sample을 너무 많이 쓰면 SHAP 계산이 느려진다.
    학습용 프로젝트에서는 보통 50~100개 정도로 시작하면 충분하다.

    입력
    ----
    X_tensor:
        이미 scaling이 끝난 feature tensor.
        모델이 실제로 받는 입력과 같은 형태여야 한다.

    background_size:
        background sample 개수.

    seed:
        랜덤 선택을 재현하기 위한 값.
    """

    X_tensor = ensure_2d_tensor(X_tensor, name="X_tensor")

    if len(X_tensor) == 0:
        raise ValueError("X_tensor must contain at least one sample.")

    # background_size가 전체 데이터 수보다 크면 전체 데이터만 사용한다.
    actual_size = min(background_size, len(X_tensor))

    generator = torch.Generator()
    generator.manual_seed(seed)

    # 0부터 len(X_tensor)-1까지의 index를 랜덤하게 섞는다.
    indices = torch.randperm(
        len(X_tensor),
        generator=generator,
    )[:actual_size]

    # detach()
    #   gradient 추적 그래프에서 분리한다.
    #
    # clone()
    #   원본 tensor와 메모리를 공유하지 않는 복사본을 만든다.
    #
    # SHAP background는 학습용 gradient update 대상이 아니므로
    # detach + clone을 사용해 안전하게 분리한다.
    background_tensor = X_tensor[indices].detach().clone()

    # SHAP와 PyTorch 모델 입력을 맞추기 위해 float32로 통일한다.
    return background_tensor.to(dtype=torch.float32)


def calculate_model_outputs(
    model: torch.nn.Module,
    sample_tensor: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    """
    sample 하나에 대해 logit, probability, prediction을 계산한다.

    이 함수가 필요한 이유
    --------------------
    SHAP value는 feature contribution이다.
    probability는 sigmoid를 통과한 고장 확률이다.
    prediction은 probability와 threshold를 비교한 최종 판단이다.

    이 세 가지를 섞으면 해석이 틀어진다.

    정리
    ----
    logit:
        model(sample)의 raw output

    probability:
        sigmoid(logit)

    prediction:
        probability >= threshold 이면 1, 아니면 0
    """

    sample_tensor = ensure_2d_tensor(
        sample_tensor,
        name="sample_tensor",
    ).to(dtype=torch.float32)

    # Dropout이 있는 모델은 평가 모드로 바꿔야 한다.
    # FailureMLP에는 Dropout이 있으므로 eval()이 중요하다.
    model.eval()

    # 추론에서는 weight 업데이트가 없으므로 gradient 계산이 필요 없다.
    with torch.no_grad():
        logits = model(sample_tensor)

    # FailureMLP 출력 shape은 보통 (1, 1)이다.
    # squeeze()로 크기가 1인 차원을 제거해 scalar처럼 다룬다.
    logit = float(logits.squeeze().detach().cpu().item())

    probability = sigmoid_float(logit)

    prediction = int(probability >= threshold)

    return {
        "logit": logit,
        "probability": probability,
        "prediction": prediction,
    }


def normalize_shap_values(raw_shap_values: Any) -> np.ndarray:
    """
    SHAP 라이브러리의 반환값을 프로젝트에서 쓰기 좋은 2D numpy array로 정리한다.

    SHAP 반환값이 복잡한 이유
    ------------------------
    모델 출력이 하나인지 여러 개인지,
    입력 sample이 하나인지 여러 개인지,
    SHAP 버전이 무엇인지에 따라 반환 형태가 달라질 수 있다.

    이 프로젝트의 목표 형태
    ---------------------
    sample 1개, feature 6개라면 최종 shape은 다음이어야 한다.

        (1, 6)

    즉:
        [[feature1_shap, feature2_shap, ..., feature6_shap]]

    처리하는 대표 case
    ------------------
    1. list 또는 tuple
    2. shape = (n_features,)
    3. shape = (n_samples, n_features, 1)
    4. shape = (n_samples, n_features)
    """

    # case 1.
    # SHAP가 list 또는 tuple을 반환하는 경우가 있다.
    # binary/single-output 모델에서는 보통 첫 번째 원소를 사용하면 된다.
    if isinstance(raw_shap_values, (list, tuple)):
        if len(raw_shap_values) == 0:
            raise ValueError("raw_shap_values is empty.")

        raw_shap_values = raw_shap_values[0]

    shap_values = np.asarray(raw_shap_values, dtype=float)

    # case 2.
    # sample 하나만 설명해서 shape이 (n_features,)인 경우
    # batch 차원을 추가해서 (1, n_features)로 만든다.
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    # case 3.
    # shape이 (n_samples, n_features, 1)인 경우
    # 마지막 output 차원을 제거해서 (n_samples, n_features)로 만든다.
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[:, :, 0]

    # 최종적으로는 반드시 2D여야 한다.
    if shap_values.ndim != 2:
        raise ValueError(
            "SHAP values must be normalized to shape "
            f"(n_samples, n_features). Current shape: {shap_values.shape}"
        )

    return shap_values


def extract_expected_value(explainer: shap.DeepExplainer) -> float:
    """
    SHAP explainer에서 expected value를 꺼낸다.

    expected value란?
    ----------------
    background data 기준 평균 모델 출력값이다.

    현재 구현에서는 모델 output이 logit이므로,
    expected_value도 probability가 아니라 logit 기준 값이다.

    SHAP의 기본 관계
    ----------------
    대략적으로 다음 관계가 성립한다.

        expected_value + sum(shap_values) ≈ model_output

    여기서 model_output은 현재 우리가 설명하는 출력이다.
    이 프로젝트에서는 logit이다.
    """

    expected_value = explainer.expected_value

    # expected_value도 SHAP 버전이나 모델 출력 구조에 따라
    # list, tuple, numpy array, scalar 등으로 올 수 있다.
    if isinstance(expected_value, (list, tuple)):
        if len(expected_value) == 0:
            raise ValueError("expected_value is empty.")

        expected_value = expected_value[0]

    expected_value_array = np.asarray(expected_value, dtype=float)

    return float(expected_value_array.reshape(-1)[0])


def calculate_shap_values(
    model: torch.nn.Module,
    background_tensor: torch.Tensor,
    sample_tensor: torch.Tensor,
) -> tuple[np.ndarray, float]:
    """
    PyTorch 모델에 SHAP DeepExplainer를 적용해 SHAP value를 계산한다.

    입력
    ----
    model:
        학습된 FailureMLP 모델.

    background_tensor:
        SHAP 기준값 계산에 사용할 sample 묶음.
        shape = (background_size, feature_count)

    sample_tensor:
        설명할 개별 sample.
        shape = (1, feature_count)

    반환
    ----
    shap_values:
        shape = (1, feature_count)

    expected_value:
        background 기준 평균 model output.
        현재는 logit 기준 expected value다.
    """

    background_tensor = ensure_2d_tensor(
        background_tensor,
        name="background_tensor",
    ).to(dtype=torch.float32)

    sample_tensor = ensure_2d_tensor(
        sample_tensor,
        name="sample_tensor",
    ).to(dtype=torch.float32)

    if background_tensor.shape[1] != sample_tensor.shape[1]:
        raise ValueError(
            "background_tensor and sample_tensor must have the same feature count. "
            f"background feature count: {background_tensor.shape[1]}, "
            f"sample feature count: {sample_tensor.shape[1]}"
        )

    # Dropout을 끄기 위해 평가 모드로 전환한다.
    model.eval()

    # DeepExplainer는 PyTorch nn.Module과 torch.Tensor background를 받을 수 있다.
    explainer = shap.DeepExplainer(
        model,
        background_tensor,
    )

    # SHAP 0.51.0 기준으로 check_additivity 인자를 사용할 수 있다.
    # 일부 PyTorch 모델에서는 additivity check가 엄격하게 실패할 수 있으므로
    # 설명 구조 검증 단계에서는 False로 둔다.
    #
    # 만약 특정 SHAP 버전에서 check_additivity 인자를 지원하지 않으면
    # TypeError가 발생할 수 있어 fallback을 둔다.
    try:
        raw_shap_values = explainer.shap_values(
            sample_tensor,
            check_additivity=False,
        )
    except TypeError:
        raw_shap_values = explainer.shap_values(sample_tensor)

    shap_values = normalize_shap_values(raw_shap_values)

    expected_value = extract_expected_value(explainer)

    return shap_values, expected_value


def determine_shap_direction(shap_value: float) -> str:
    """
    SHAP value의 방향을 문자열로 바꾼다.

    positive
    --------
    feature가 모델 output을 높이는 방향으로 작용했다는 뜻이다.
    이번 구현에서는 logit을 설명하므로,
    positive는 고장 위험 logit을 높이는 방향이다.

    negative
    --------
    feature가 모델 output을 낮추는 방향으로 작용했다는 뜻이다.

    neutral
    -------
    영향이 거의 없다고 볼 수 있는 경우다.
    """

    if shap_value > 0:
        return "positive"

    if shap_value < 0:
        return "negative"

    return "neutral"


def build_shap_reason(
    feature: str,
    shap_value: float,
    direction: str,
) -> str:
    """
    Agent evidence에 들어갈 수 있는 설명 문장을 만든다.

    주의
    ----
    "이 feature 때문에 고장이라고 판단했다"라고 쓰면 안 된다.

    정확한 표현은:
    "SHAP 기준으로 이 feature는 고장 위험 logit을 높이는 방향으로 작용했다."
    이다.
    """

    if direction == "positive":
        return (
            f"SHAP 기준으로 {feature}는 모델의 고장 위험 logit을 "
            f"높이는 방향으로 작용했습니다. "
            f"SHAP value={shap_value:.4f}"
        )

    if direction == "negative":
        return (
            f"SHAP 기준으로 {feature}는 모델의 고장 위험 logit을 "
            f"낮추는 방향으로 작용했습니다. "
            f"SHAP value={shap_value:.4f}"
        )

    return (
        f"SHAP 기준으로 {feature}는 모델의 고장 위험 logit에 "
        f"거의 영향을 주지 않았습니다. "
        f"SHAP value={shap_value:.4f}"
    )


def map_shap_values_to_local_contributions(
    feature_columns: list[str],
    sample_tensor: torch.Tensor,
    shap_values: np.ndarray,
    raw_sample_values: dict[str, float] | None = None,
    reference_values: dict[str, float] | None = None,
    global_importance_map: dict[str, float] | None = None,
) -> list[LocalFeatureContribution]:
    """
    SHAP value를 Day 7의 LocalFeatureContribution 구조로 변환한다.

    feature_columns
    ---------------
    모델 입력 feature 이름 목록.

    예:
        [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Type",
        ]

    sample_tensor
    -------------
    모델에 실제로 들어간 sample tensor.
    보통 StandardScaler로 scaling된 값이다.

    shap_values
    -----------
    feature별 SHAP contribution.
    shape = (1, feature_count)

    raw_sample_values
    -----------------
    사람이 보기 좋은 원본 입력값이다.

    예:
        {"Torque [Nm]": 65.0}

    이 값이 있으면 evidence 표시에는 raw value를 사용한다.
    없으면 scaled tensor 값을 사용한다.

    reference_values
    ----------------
    비교 기준값이다.
    가능하면 train set의 raw 평균값을 넣는다.

    global_importance_map
    ---------------------
    Day 6 permutation importance 결과를 feature별 dict로 만든 값이다.

    예:
        {"Torque [Nm]": 0.3309}
    """

    sample_tensor = ensure_2d_tensor(
        sample_tensor,
        name="sample_tensor",
    ).to(dtype=torch.float32)

    if shap_values.shape[0] != 1:
        raise ValueError(
            "This helper currently expects one sample at a time. "
            f"Current shap_values shape: {shap_values.shape}"
        )

    if len(feature_columns) != shap_values.shape[1]:
        raise ValueError(
            "feature_columns length must match shap_values feature count. "
            f"feature_columns: {len(feature_columns)}, "
            f"shap feature count: {shap_values.shape[1]}"
        )

    raw_sample_values = raw_sample_values or {}
    reference_values = reference_values or {}
    global_importance_map = global_importance_map or {}

    contributions: list[LocalFeatureContribution] = []

    for feature_index, feature_name in enumerate(feature_columns):
        shap_value = float(shap_values[0, feature_index])

        direction = determine_shap_direction(shap_value)

        # evidence에 표시할 value는 가능하면 raw value를 사용한다.
        # raw value가 없으면 모델에 들어간 scaled tensor 값을 사용한다.
        display_value = raw_sample_values.get(
            feature_name,
            float(sample_tensor[0, feature_index].detach().cpu().item()),
        )

        reference_value = reference_values.get(feature_name)

        global_importance = global_importance_map.get(feature_name)

        reason = build_shap_reason(
            feature=feature_name,
            shap_value=shap_value,
            direction=direction,
        )

        contribution = LocalFeatureContribution(
            feature=feature_name,
            value=display_value,
            contribution=shap_value,
            direction=direction,
            reference_value=reference_value,
            global_importance=global_importance,
            reason=reason,
        )

        contributions.append(contribution)

    # 절댓값이 큰 순서대로 정렬한다.
    #
    # 이유:
    #   SHAP value가 +0.3이면 모델 출력을 높이는 영향이 크다는 뜻이고,
    #   SHAP value가 -0.3이면 모델 출력을 낮추는 영향이 크다는 뜻이다.
    #
    #   local explanation에서는 방향뿐 아니라 영향 크기도 중요하므로
    #   abs(contribution)가 큰 feature를 먼저 보여준다.
    contributions.sort(
        key=lambda item: abs(item.contribution),
        reverse=True,
    )

    return contributions


def build_shap_summary(
    probability: float,
    prediction: int,
    top_contributions: list[LocalFeatureContribution],
) -> str:
    """
    LocalExplanationResult에 들어갈 summary 문장을 만든다.
    """

    if not top_contributions:
        return (
            f"모델의 예측 probability는 {probability:.4f}입니다. "
            "다만 표시할 SHAP contribution이 없습니다."
        )

    top_features = ", ".join(
        contribution.feature for contribution in top_contributions[:3]
    )

    prediction_text = "고장 위험" if prediction == 1 else "정상"

    return (
        f"모델은 이 sample을 {prediction_text}으로 예측했습니다. "
        f"예측 probability는 {probability:.4f}입니다. "
        f"SHAP 기준으로 영향이 큰 feature는 {top_features}입니다."
    )


def build_shap_local_explanation_result(
    model: torch.nn.Module,
    background_tensor: torch.Tensor,
    sample_tensor: torch.Tensor,
    feature_columns: list[str],
    threshold: float,
    risk_level: str,
    raw_sample_values: dict[str, float] | None = None,
    reference_values: dict[str, float] | None = None,
    global_importance_map: dict[str, float] | None = None,
    top_k: int = 5,
) -> LocalExplanationResult:
    """
    SHAP 계산 결과를 Day 7의 LocalExplanationResult로 변환한다.

    이 함수가 Day 8의 핵심 함수다.

    전체 흐름
    --------
    1. 모델로 logit, probability, prediction 계산
    2. SHAP value 계산
    3. feature별 SHAP value를 LocalFeatureContribution으로 변환
    4. contribution 절댓값 기준으로 정렬
    5. 상위 top_k개 contribution 선택
    6. LocalExplanationResult 반환

    주의
    ----
    여기서 contribution은 probability contribution이 아니다.
    현재 모델 출력이 logit이므로 SHAP value도 logit 기준 contribution이다.
    """

    model_outputs = calculate_model_outputs(
        model=model,
        sample_tensor=sample_tensor,
        threshold=threshold,
    )

    shap_values, expected_value = calculate_shap_values(
        model=model,
        background_tensor=background_tensor,
        sample_tensor=sample_tensor,
    )

    contributions = map_shap_values_to_local_contributions(
        feature_columns=feature_columns,
        sample_tensor=sample_tensor,
        shap_values=shap_values,
        raw_sample_values=raw_sample_values,
        reference_values=reference_values,
        global_importance_map=global_importance_map,
    )

    top_contributions = contributions[:top_k]

    summary = build_shap_summary(
        probability=float(model_outputs["probability"]),
        prediction=int(model_outputs["prediction"]),
        top_contributions=top_contributions,
    )

    limitations = [
        "현재 SHAP value는 probability가 아니라 logit 기준 contribution입니다.",
        "모델 입력이 StandardScaler로 scaling된 값이므로 SHAP 계산도 scaling된 feature space 기준입니다.",
        "background data 선택에 따라 expected value와 SHAP value가 달라질 수 있습니다.",
        "SHAP value는 feature contribution이며, 특정 feature 하나가 고장을 단독으로 발생시켰다는 의미는 아닙니다.",
        f"background expected logit={expected_value:.4f}",
    ]

    return LocalExplanationResult(
        prediction=int(model_outputs["prediction"]),
        probability=float(model_outputs["probability"]),
        threshold=threshold,
        risk_level=risk_level,
        explanation_method="shap_deep_explainer_logit",
        contributions=top_contributions,
        summary=summary,
        limitations=limitations,
    )