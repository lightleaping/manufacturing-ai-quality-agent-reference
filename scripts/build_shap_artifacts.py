"""
Day 11 - SHAP artifact 생성 스크립트

이 스크립트의 목적
------------------
운영 환경에 가까운 구조를 만들기 위해
API 요청 시점에 SHAP background를 만들지 않고,
미리 SHAP explanation용 artifact를 생성해 저장합니다.

생성되는 파일:
    models/failure_mlp/shap_background.pt
    models/failure_mlp/shap_reference_values.json
    models/failure_mlp/global_importance.json

중요한 원칙
-----------
이 스크립트는 API endpoint에서 실행하지 않습니다.

실행 시점:
    모델 학습 완료 후
    또는 배포 준비 단계

API에서는 이 스크립트가 만든 artifact를 로드만 합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.load_ai4i import load_ai4i_csv
from src.data.preprocess_ai4i import preprocess_ai4i_dataframe
from src.inference.model_artifacts import load_failure_model_artifacts
from src.inference.predict_failure import (
    build_single_sample_dataframe,
    dataframe_to_single_tensor,
    normalize_type_value,
    scale_single_sample_dataframe,
    validate_raw_sample,
)
from src.interpretability.shap_artifacts import save_shap_artifacts
from src.interpretability.shap_explainer import build_background_tensor


AI4I_CSV_PATH = Path("data/raw/ai4i/ai4i_2020.csv")
ARTIFACT_DIR = Path("models/failure_mlp")


def normalize_sample_for_model_input(
    raw_sample: dict[str, Any],
) -> dict[str, Any]:
    """
    raw sample의 Type 값을 모델 입력용 숫자값으로 변환합니다.

    사용자가 직접 만든 sample에서는 Type이 "L", "M", "H" 문자열일 수 있습니다.
    반면 preprocess된 X_train에서는 Type이 이미 0, 1, 2 숫자일 수 있습니다.

    그래서:
    - 문자열이면 normalize_type_value() 사용
    - 이미 숫자면 int로 변환
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
    sample 하나를 모델 입력 tensor로 변환합니다.

    이 함수는 Day 5의 전처리 함수를 재사용합니다.

    이유:
    - scaler를 직접 호출하면 컬럼 순서나 Type 처리 문제로 에러가 날 수 있습니다.
    - Day 5 추론과 Day 11 SHAP explanation이 같은 전처리 기준을 공유해야 합니다.
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


def build_background_source_tensor(
    X_train: pd.DataFrame,
    feature_columns: list[str],
    artifacts: Any,
    candidate_size: int = 300,
    seed: int = 42,
) -> torch.Tensor:
    """
    SHAP background 후보 tensor를 생성합니다.

    운영형 구조에서는 이 작업을 API 요청 시점에 하지 않습니다.
    이 스크립트를 통해 미리 생성한 뒤 shap_background.pt로 저장합니다.

    흐름:
    1. X_train에서 candidate_size개 row 샘플링
    2. 각 row를 sample dict로 변환
    3. Day 5 전처리 함수로 tensor 변환
    4. torch.cat으로 하나의 tensor로 결합
    """
    actual_candidate_size = min(candidate_size, len(X_train))

    sampled_X_train = X_train.sample(
        n=actual_candidate_size,
        random_state=seed,
    )

    tensors: list[torch.Tensor] = []

    for _, row in sampled_X_train.iterrows():
        sample = {
            feature: row[feature]
            for feature in feature_columns
        }

        sample_tensor = sample_to_model_tensor(
            raw_sample=sample,
            feature_columns=feature_columns,
            artifacts=artifacts,
        )

        tensors.append(sample_tensor)

    if not tensors:
        raise ValueError("No background tensors were created.")

    return torch.cat(tensors, dim=0).to(dtype=torch.float32)


def build_reference_values(
    X_train: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, float]:
    """
    feature별 reference value를 생성합니다.

    여기서는 train set 평균값을 사용합니다.

    reference value는 SHAP 계산 자체보다
    Agent evidence에서 현재 sample 값을 기준값과 비교해서 보여줄 때 사용합니다.
    """
    return {
        feature: float(X_train[feature].mean())
        for feature in feature_columns
    }


def build_global_importance_map() -> dict[str, float]:
    """
    Day 6 permutation importance 결과를 저장용 dict로 만듭니다.

    현재는 Day 6에서 얻은 실제 결과를 명시적으로 저장합니다.
    이후에는 reports나 별도 artifact에서 자동 로드하는 구조로 개선할 수 있습니다.

    주의:
    global importance는 전체 test set 기준 모델 민감도입니다.
    개별 sample의 직접 원인이 아닙니다.
    """
    return {
        "Torque [Nm]": 0.3309,
        "Air temperature [K]": 0.2725,
        "Rotational speed [rpm]": 0.2292,
        "Process temperature [K]": 0.1651,
        "Tool wear [min]": 0.1213,
        "Type": 0.0186,
    }


def main() -> None:
    """
    SHAP artifact를 생성하고 models/failure_mlp 디렉터리에 저장합니다.
    """
    print("[INFO] SHAP artifact build started")

    artifacts = load_failure_model_artifacts(ARTIFACT_DIR)

    feature_columns = list(artifacts.feature_columns)

    print(f"[INFO] artifact_dir    : {ARTIFACT_DIR}")
    print(f"[INFO] feature_columns : {feature_columns}")

    raw_df = load_ai4i_csv(AI4I_CSV_PATH)
    preprocessed = preprocess_ai4i_dataframe(raw_df)

    X_train = preprocessed.X_train

    background_source_tensor = build_background_source_tensor(
        X_train=X_train,
        feature_columns=feature_columns,
        artifacts=artifacts,
        candidate_size=300,
        seed=42,
    )

    background_tensor = build_background_tensor(
        X_tensor=background_source_tensor,
        background_size=100,
        seed=42,
    )

    reference_values = build_reference_values(
        X_train=X_train,
        feature_columns=feature_columns,
    )

    global_importance_map = build_global_importance_map()

    save_shap_artifacts(
        artifact_dir=ARTIFACT_DIR,
        background_tensor=background_tensor,
        reference_values=reference_values,
        global_importance_map=global_importance_map,
    )

    print(f"[INFO] background_source_tensor shape: {tuple(background_source_tensor.shape)}")
    print(f"[INFO] shap_background_tensor shape  : {tuple(background_tensor.shape)}")
    print("[INFO] saved: models/failure_mlp/shap_background.pt")
    print("[INFO] saved: models/failure_mlp/shap_reference_values.json")
    print("[INFO] saved: models/failure_mlp/global_importance.json")
    print("[INFO] SHAP artifact build completed")


if __name__ == "__main__":
    main()