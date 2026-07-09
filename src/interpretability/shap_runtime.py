"""
API 실행 시점에서 SHAP local explanation을 생성하는 runtime helper입니다.

이 파일이 필요한 이유
--------------------
FastAPI endpoint 안에 SHAP 계산 과정을 길게 쓰면 endpoint가 너무 복잡해집니다.

그래서 API endpoint는:
    request 받기
    prediction 실행
    SHAP runtime helper 호출
    evidence/answer 생성
    response 반환

정도만 담당하게 합니다.

실제 SHAP local explanation 생성 과정은 이 파일이 담당합니다.
"""

from __future__ import annotations

from typing import Any

import torch

from src.inference.predict_failure import (
    build_single_sample_dataframe,
    dataframe_to_single_tensor,
    normalize_type_value,
    scale_single_sample_dataframe,
    validate_raw_sample,
)
from src.interpretability.shap_artifacts import ShapArtifacts
from src.interpretability.shap_explainer import build_shap_local_explanation_result


def normalize_sample_for_model_input(
    raw_sample: dict[str, Any],
) -> dict[str, Any]:
    """
    raw sample의 Type 값을 모델 입력용 숫자값으로 변환합니다.

    예:
        "L" -> 0
        "M" -> 1
        "H" -> 2

    SHAP 계산은 모델 입력 tensor 기준으로 수행되므로,
    문자열 Type을 그대로 둘 수 없습니다.
    """
    normalized_sample = dict(raw_sample)

    type_value = normalized_sample.get("Type")

    if isinstance(type_value, str):
        normalized_sample["Type"] = normalize_type_value(type_value)
    else:
        normalized_sample["Type"] = int(type_value)

    return normalized_sample


def sample_to_model_tensor(
    raw_sample: dict[str, Any],
    feature_columns: list[str],
    artifacts: Any,
) -> torch.Tensor:
    """
    raw sample 하나를 모델 입력 tensor로 변환합니다.

    이 함수는 Day 5 단일 추론 흐름과 같은 전처리 함수를 재사용합니다.

    흐름:
        raw_sample
        -> validate_raw_sample
        -> normalize_sample_for_model_input
        -> build_single_sample_dataframe
        -> scale_single_sample_dataframe
        -> dataframe_to_single_tensor

    중요한 이유:
        SHAP 설명은 prediction에 사용된 것과 동일한 입력 기준으로 계산되어야 합니다.
        전처리 방식이 다르면 prediction과 SHAP explanation이 서로 다른 기준이 됩니다.
    """
    validate_raw_sample(
        raw_sample=raw_sample,
        feature_columns=feature_columns,
    )

    normalized_sample = normalize_sample_for_model_input(raw_sample)

    sample_df = build_single_sample_dataframe(
        raw_sample=normalized_sample,
        feature_columns=feature_columns,
    )

    scaled_sample_df = scale_single_sample_dataframe(
        sample_df=sample_df,
        artifacts=artifacts,
    )

    sample_tensor = dataframe_to_single_tensor(scaled_sample_df)

    return sample_tensor.to(dtype=torch.float32)


def build_raw_sample_values(
    raw_sample: dict[str, Any],
    feature_columns: list[str],
) -> dict[str, float]:
    """
    evidence에 표시할 원본 feature 값을 만듭니다.

    SHAP 계산 자체는 scaling된 tensor 기준입니다.
    하지만 사용자가 읽는 evidence에는 원본 단위 값이 더 이해하기 쉽습니다.

    예:
        Torque [Nm] = 62.0
        Tool wear [min] = 220.0

    단, Type은 모델 입력 기준에 맞춰 숫자로 변환합니다.
    """
    normalized_sample = normalize_sample_for_model_input(raw_sample)

    return {
        feature: float(normalized_sample[feature])
        for feature in feature_columns
    }


def build_global_importance_items_from_map(
    global_importance_map: dict[str, float],
) -> list[dict[str, float | str]]:
    """
    global_importance_map을 build_agent_evidence()가 받기 쉬운 list 구조로 변환합니다.

    global importance는 Day 6 permutation importance 결과입니다.
    전체 test set 기준 모델 민감도이며,
    개별 sample의 직접 원인이 아닙니다.
    """
    return [
        {
            "feature": feature,
            "importance": importance,
        }
        for feature, importance in global_importance_map.items()
    ]


def build_shap_local_explanation_for_sample(
    *,
    include_shap: bool,
    artifacts: Any,
    shap_artifacts: ShapArtifacts,
    raw_sample: dict[str, Any],
    threshold: float,
    risk_level: str,
    top_k: int = 5,
):
    """
    include_shap=True일 때만 SHAP local explanation을 생성합니다.

    include_shap=False이면 None을 반환합니다.

    중요한 해석:
        SHAP value는 probability가 아닙니다.
        현재 FailureMLP는 마지막에 Sigmoid가 없으므로,
        SHAP value는 logit 기준 contribution입니다.

        positive SHAP value:
            모델의 고장 위험 logit을 높이는 방향

        negative SHAP value:
            모델의 고장 위험 logit을 낮추는 방향

    주의:
        SHAP 결과는 실제 고장의 물리적 원인을 단정하지 않습니다.
        현재 모델 출력에 대해 feature가 어느 방향으로 기여했는지 설명합니다.
    """
    if not include_shap:
        return None

    feature_columns = list(artifacts.feature_columns)

    sample_tensor = sample_to_model_tensor(
        raw_sample=raw_sample,
        feature_columns=feature_columns,
        artifacts=artifacts,
    )

    raw_sample_values = build_raw_sample_values(
        raw_sample=raw_sample,
        feature_columns=feature_columns,
    )

    return build_shap_local_explanation_result(
        model=artifacts.model,
        background_tensor=shap_artifacts.background_tensor,
        sample_tensor=sample_tensor,
        feature_columns=feature_columns,
        threshold=threshold,
        risk_level=risk_level,
        raw_sample_values=raw_sample_values,
        reference_values=shap_artifacts.reference_values,
        global_importance_map=shap_artifacts.global_importance_map,
        top_k=top_k,
    )