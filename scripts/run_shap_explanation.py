# scripts/run_shap_explanation.py

"""
Day 8 - 실제 저장 모델 기준 SHAP explanation 실행 스크립트

이 스크립트의 목적
------------------
Day 5에서 저장한 FailureMLP 모델 artifact를 재사용해서,
raw sample 하나에 대한 SHAP 기반 local explanation을 생성한다.

Day 8 전체 흐름
---------------
1. Day 5 artifact 로드
2. AI4I train data에서 SHAP background tensor 생성
3. raw sample 하나 생성
4. Day 5 추론 함수로 prediction/probability/risk_level 확인
5. 같은 sample을 SHAP 입력 tensor로 변환
6. SHAP value 계산
7. Day 7 LocalExplanationResult 구조로 변환
8. Agent evidence 형식으로 변환
9. 결과 출력

중요한 원칙
-----------
이 스크립트는 새 로드/스케일링 로직을 직접 만들지 않는다.

Day 5에서 이미 만든 함수를 재사용한다.

재사용하는 함수:
    load_failure_model_artifacts
    validate_raw_sample
    normalize_type_value
    build_single_sample_dataframe
    scale_single_sample_dataframe
    dataframe_to_single_tensor
    calculate_risk_level
    predict_failure_from_artifacts

이렇게 해야 Day 5 단일 추론 흐름과
Day 8 SHAP 설명 흐름이 같은 전처리/추론 기준을 공유한다.

중요한 해석
-----------
현재 FailureMLP는 마지막에 Sigmoid가 없다.

따라서:
    model(sample) = logit

그리고:
    probability = sigmoid(logit)

이번 SHAP value는 probability 자체가 아니라
logit 기준 feature contribution이다.

정확한 표현:
    "SHAP 기준으로 이 feature는 고장 위험 logit을 높이는 방향으로 작용했습니다."

부정확한 표현:
    "모델은 이 feature 때문에 고장이라고 판단했습니다."
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
    calculate_risk_level,
    dataframe_to_single_tensor,
    normalize_type_value,
    predict_failure_from_artifacts,
    scale_single_sample_dataframe,
    validate_raw_sample,
)
from src.interpretability.local_explanation import (
    format_local_explanation_as_evidence,
)
from src.interpretability.shap_explainer import (
    build_background_tensor,
    build_shap_local_explanation_result,
    calculate_model_outputs,
)


AI4I_CSV_PATH = Path("data/raw/ai4i/ai4i_2020.csv")
ARTIFACT_DIR = Path("models/failure_mlp")


def normalize_sample_for_model_input(
    raw_sample: dict[str, Any],
) -> dict[str, Any]:
    """
    raw sample을 모델 입력용 sample로 변환한다.

    이 함수가 필요한 이유
    --------------------
    사용자가 입력하는 raw_sample의 Type은 보통 "L", "M", "H" 같은 문자열이다.

    예:
        {"Type": "L"}

    하지만 모델은 문자열을 직접 받을 수 없다.
    Day 1 전처리에서 Type은 아래처럼 숫자로 mapping했다.

        L -> 0
        M -> 1
        H -> 2

    따라서 SHAP 입력 tensor를 만들기 전에도
    Type을 숫자값으로 바꿔야 한다.

    주의
    ----
    AI4I 전처리 결과인 X_train 안의 Type은 이미 숫자다.
    반면 사용자가 직접 만든 raw_sample의 Type은 문자열일 수 있다.

    그래서:
        - 문자열이면 normalize_type_value를 사용한다.
        - 이미 숫자면 int로 변환한다.
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
    sample 하나를 모델 입력 tensor로 변환한다.

    이 함수는 Day 5의 전처리 함수를 재사용한다.

    전체 흐름
    --------
    raw_sample
    ↓
    validate_raw_sample
    ↓
    normalize_sample_for_model_input
    ↓
    build_single_sample_dataframe
    ↓
    scale_single_sample_dataframe
    ↓
    dataframe_to_single_tensor

    중요한 점
    --------
    scaler.transform을 이 스크립트에서 직접 호출하지 않는다.

    이유:
        저장된 scaler가 어떤 컬럼에 fit되었는지,
        Type을 scale하는지 제외하는지,
        컬럼 순서를 어떻게 맞추는지는
        Day 5의 scale_single_sample_dataframe이 책임져야 한다.

    이전 에러 원인
    --------------
    아래처럼 직접 호출하면 문제가 생길 수 있다.

        scaler.transform(X_train[feature_columns])

    실제로 저장된 scaler는 Type 없이 fit되었는데,
    feature_columns에는 Type이 포함되어 있어서
    "Feature names unseen at fit time: Type" 에러가 발생했다.

    따라서 이 스크립트에서는 반드시 기존 함수를 재사용한다.
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
    SHAP background 후보 tensor를 만든다.

    background data란?
    ------------------
    SHAP가 expected value를 계산하기 위해 사용하는 기준 sample 묶음이다.

    쉽게 말해:
        "평균적인 데이터와 비교했을 때,
         현재 sample의 feature들이 모델 출력을 얼마나 움직였는가?"
    를 계산하기 위한 기준이다.

    왜 scaler.transform을 직접 쓰지 않는가?
    --------------------------------------
    저장된 scaler는 Type 컬럼 없이 fit되었을 수 있다.

    그런데 feature_columns에는 Type이 포함되어 있다.

    그래서 아래처럼 직접 쓰면 에러가 날 수 있다.

        scaler.transform(X_train[feature_columns])

    따라서 background tensor를 만들 때도
    Day 5의 sample 전처리 함수들을 재사용한다.

    동작 방식
    --------
    1. X_train에서 일부 row를 뽑는다.
    2. 각 row를 sample dict로 바꾼다.
    3. sample_to_model_tensor 함수로 모델 입력 tensor를 만든다.
    4. tensor들을 이어붙인다.

    주의
    ----
    전체 8000개 train row를 모두 변환할 필요는 없다.
    SHAP background는 계산 비용이 크기 때문에
    후보 300개 정도에서 시작하고,
    이후 build_background_tensor에서 100개를 뽑는다.
    """

    actual_candidate_size = min(candidate_size, len(X_train))

    sampled_X_train = X_train.sample(
        n=actual_candidate_size,
        random_state=seed,
    )

    tensors: list[torch.Tensor] = []

    for _, row in sampled_X_train.iterrows():
        # row에는 모델 feature 6개가 들어 있다.
        # Type은 이미 0/1/2 숫자일 가능성이 높다.
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

    # 각 sample_tensor는 shape이 보통 (1, 6)이다.
    # torch.cat(dim=0)을 하면 (candidate_size, 6)이 된다.
    background_source_tensor = torch.cat(
        tensors,
        dim=0,
    )

    return background_source_tensor.to(dtype=torch.float32)


def main() -> None:
    """
    실제 저장 모델을 기준으로 sample 하나의 SHAP explanation을 생성한다.
    """

    print("[INFO] Day 8 SHAP explanation started")

    # -------------------------------------------------------------------------
    # 1. Day 5 artifact 로드
    # -------------------------------------------------------------------------
    #
    # 실제 존재하는 함수:
    #   load_failure_model_artifacts
    #
    # 이 함수는 Day 5에서 만든 model_artifacts.py의 로드 함수다.
    #
    # 여기서 직접 torch.load, joblib.load, json.load를 하지 않는다.
    # artifact 저장/로드 책임은 model_artifacts.py가 갖는 것이 맞다.
    artifacts = load_failure_model_artifacts(ARTIFACT_DIR)

    model = artifacts.model
    threshold = float(artifacts.threshold)
    feature_columns = list(artifacts.feature_columns)

    print(f"[INFO] artifact_dir: {ARTIFACT_DIR}")
    print(f"[INFO] threshold   : {threshold}")
    print(f"[INFO] features    : {feature_columns}")

    # -------------------------------------------------------------------------
    # 2. AI4I train data 로드
    # -------------------------------------------------------------------------
    #
    # SHAP background는 train data 일부를 기준으로 만든다.
    #
    # 주의:
    #   raw_df를 그대로 쓰지 않고,
    #   Day 1 전처리 함수 preprocess_ai4i_dataframe을 통과한 X_train을 사용한다.
    #
    # 이유:
    #   X_train은 이미 feature/target 분리와 Type encoding이 끝난 상태이기 때문이다.
    raw_df = load_ai4i_csv(AI4I_CSV_PATH)

    preprocessed = preprocess_ai4i_dataframe(raw_df)

    X_train = preprocessed.X_train

    # -------------------------------------------------------------------------
    # 3. SHAP background tensor 생성
    # -------------------------------------------------------------------------
    #
    # background_source_tensor:
    #   X_train 일부를 Day 5 전처리 함수로 변환한 모델 입력 tensor
    #
    # background_tensor:
    #   SHAP DeepExplainer에 실제로 넣을 기준 tensor
    #
    # 여기서도 scaler.transform을 직접 호출하지 않는다.
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

    print(
        f"[INFO] background_source_tensor shape: "
        f"{tuple(background_source_tensor.shape)}"
    )
    print(f"[INFO] background_tensor shape       : {tuple(background_tensor.shape)}")

    # -------------------------------------------------------------------------
    # 4. 설명할 raw sample 하나 만들기
    # -------------------------------------------------------------------------
    #
    # 이 sample은 고장 위험이 높아 보이도록 구성한 예시다.
    #
    # 주의:
    #   이 값들이 곧바로 모델 입력으로 들어가는 것은 아니다.
    #   Day 5 추론 흐름처럼 Type encoding과 scaling을 거쳐야 한다.
    raw_sample = {
        "Air temperature [K]": 303.0,
        "Process temperature [K]": 312.5,
        "Rotational speed [rpm]": 1380.0,
        "Torque [Nm]": 62.0,
        "Tool wear [min]": 220.0,
        "Type": "L",
    }

    print("\n[INFO] Raw sample:")
    for key, value in raw_sample.items():
        print(f"    {key}: {value}")

    # -------------------------------------------------------------------------
    # 5. Day 5 predict_failure_from_artifacts로 단일 추론 결과 확인
    # -------------------------------------------------------------------------
    #
    # Day 8 SHAP explanation은 새로운 예측 로직이 아니다.
    # Day 5 단일 추론 결과를 설명하는 역할이다.
    #
    # 따라서 먼저 Day 5 함수로 prediction 결과를 만든다.
    prediction_result = predict_failure_from_artifacts(
        raw_sample=raw_sample,
        artifacts=artifacts,
    )

    print("\n[INFO] Prediction result from Day 5 inference flow")
    print(f"[INFO] probability        : {prediction_result.probability:.4f}")
    print(f"[INFO] threshold          : {prediction_result.threshold:.4f}")
    print(f"[INFO] prediction         : {prediction_result.prediction}")
    print(f"[INFO] risk_level         : {prediction_result.risk_level}")
    print(f"[INFO] recommended_action : {prediction_result.recommended_action}")

    # -------------------------------------------------------------------------
    # 6. 같은 raw_sample을 SHAP 입력 tensor로 변환
    # -------------------------------------------------------------------------
    #
    # predict_failure_from_artifacts는 최종 prediction 결과를 반환한다.
    # 하지만 SHAP DeepExplainer에는 실제 모델 입력 tensor가 필요하다.
    #
    # 그래서 같은 raw_sample을 Day 5 전처리 함수로 다시 tensor화한다.
    sample_tensor = sample_to_model_tensor(
        raw_sample=raw_sample,
        feature_columns=feature_columns,
        artifacts=artifacts,
    )

    print(f"[INFO] sample_tensor shape: {tuple(sample_tensor.shape)}")

    # -------------------------------------------------------------------------
    # 7. 모델 출력 기준 logit/probability/prediction 재확인
    # -------------------------------------------------------------------------
    #
    # build_shap_local_explanation_result 내부에서도 모델 출력을 계산한다.
    #
    # 하지만 여기서 한 번 출력해두면,
    # Day 5 prediction_result와 SHAP 입력 tensor 기준 결과가 일치하는지 확인할 수 있다.
    model_outputs = calculate_model_outputs(
        model=model,
        sample_tensor=sample_tensor,
        threshold=threshold,
    )

    logit = float(model_outputs["logit"])
    probability = float(model_outputs["probability"])
    prediction = int(model_outputs["prediction"])

    # risk_level은 직접 새 기준을 만들지 않고,
    # Day 5 predict_failure.py에 실제 존재하는 calculate_risk_level을 재사용한다.
    risk_level = calculate_risk_level(probability)

    print("\n[INFO] Model outputs for SHAP input tensor")
    print(f"[INFO] logit       : {logit:.4f}")
    print(f"[INFO] probability : {probability:.4f}")
    print(f"[INFO] prediction  : {prediction}")
    print(f"[INFO] risk_level  : {risk_level}")

    # -------------------------------------------------------------------------
    # 8. evidence 표시용 raw_sample_values 만들기
    # -------------------------------------------------------------------------
    #
    # SHAP 계산 자체는 scaling된 tensor 기준으로 한다.
    #
    # 하지만 Agent evidence에 표시할 때는
    # 사람이 이해하기 쉬운 원본 단위 값이 더 좋다.
    #
    # Type은 "L"보다 모델 입력 기준인 0으로 넣는 편이
    # LocalFeatureContribution value 타입과 맞다.
    normalized_sample = normalize_sample_for_model_input(raw_sample)

    raw_sample_values = {
        feature: normalized_sample[feature]
        for feature in feature_columns
    }

    # -------------------------------------------------------------------------
    # 9. reference_values 만들기
    # -------------------------------------------------------------------------
    #
    # reference value는 비교 기준값이다.
    # 여기서는 train set 평균값을 사용한다.
    #
    # 예:
    #   현재 Torque 값이 train 평균 Torque보다 높은지 낮은지
    #   설명에 활용할 수 있다.
    reference_values = {
        feature: float(X_train[feature].mean())
        for feature in feature_columns
    }

    # -------------------------------------------------------------------------
    # 10. Day 6 permutation importance 결과 연결
    # -------------------------------------------------------------------------
    #
    # Day 6 결과는 전체 test set 기준 global explanation이다.
    #
    # Day 8 SHAP는 sample 하나 기준 local explanation이다.
    #
    # 두 값은 역할이 다르다.
    #
    # 다만 LocalFeatureContribution 안에 global_importance를 함께 넣으면
    # "이 feature는 전체적으로도 중요했고,
    #  이 sample에서도 이런 방향으로 작용했다"
    # 는 식의 설명이 가능해진다.
    global_importance_map = {
        "Torque [Nm]": 0.3309,
        "Air temperature [K]": 0.2725,
        "Rotational speed [rpm]": 0.2292,
        "Process temperature [K]": 0.1651,
        "Tool wear [min]": 0.1213,
        "Type": 0.0186,
    }

    # -------------------------------------------------------------------------
    # 11. SHAP 기반 LocalExplanationResult 생성
    # -------------------------------------------------------------------------
    #
    # 이 함수 안에서 일어나는 일:
    #
    #   model(sample)로 logit 계산
    #   sigmoid(logit)으로 probability 계산
    #   threshold와 비교해 prediction 계산
    #   SHAP DeepExplainer로 feature별 SHAP value 계산
    #   LocalFeatureContribution 목록 생성
    #   LocalExplanationResult 생성
    #
    # risk_level은 "HIGH"로 고정하지 않고,
    # Day 5 기준 calculate_risk_level 결과를 넣는다.
    local_explanation = build_shap_local_explanation_result(
        model=model,
        background_tensor=background_tensor,
        sample_tensor=sample_tensor,
        feature_columns=feature_columns,
        threshold=threshold,
        risk_level=risk_level,
        raw_sample_values=raw_sample_values,
        reference_values=reference_values,
        global_importance_map=global_importance_map,
        top_k=5,
    )

    # -------------------------------------------------------------------------
    # 12. Local explanation 출력
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[INFO] Local explanation summary")
    print("=" * 80)
    print(local_explanation.summary)

    print("\n[INFO] Contributions:")
    for contribution in local_explanation.contributions:
        print(
            f"- feature={contribution.feature}, "
            f"value={contribution.value}, "
            f"shap={contribution.contribution:.4f}, "
            f"direction={contribution.direction}, "
            f"reference_value={contribution.reference_value}, "
            f"global_importance={contribution.global_importance}"
        )
        print(f"  reason={contribution.reason}")

    print("\n[INFO] Limitations:")
    for limitation in local_explanation.limitations:
        print(f"- {limitation}")

    # -------------------------------------------------------------------------
    # 13. Agent evidence 형식으로 변환
    # -------------------------------------------------------------------------
    #
    # Day 7에서 만든 format_local_explanation_as_evidence 함수를 사용한다.
    #
    # 목표:
    #   LocalExplanationResult
    #   ↓
    #   Agent 답변에 넣을 evidence list
    evidence = format_local_explanation_as_evidence(local_explanation)

    print("\n" + "=" * 80)
    print("[INFO] Agent evidence")
    print("=" * 80)
    for item in evidence:
        print(item)

    print("\n[INFO] Day 8 SHAP explanation completed")


if __name__ == "__main__":
    main()