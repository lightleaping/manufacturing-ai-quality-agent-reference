from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.inference.model_artifacts import (
    FailureModelArtifacts,
    load_failure_model_artifacts,
)


@dataclass(frozen=True)
class FailurePredictionResult:
    """
    단일 설비 입력에 대한 고장 예측 결과를 담는 dataclass입니다.

    probability:
    - 모델이 예측한 고장 확률입니다.
    - 0.0 ~ 1.0 사이 값입니다.

    prediction:
    - 최종 예측 결과입니다.
    - 0이면 정상, 1이면 고장 위험으로 판단합니다.

    threshold:
    - probability를 prediction으로 바꿀 때 사용한 기준값입니다.

    risk_level:
    - probability를 사람이 이해하기 쉬운 위험 등급으로 바꾼 값입니다.
    - LOW / MEDIUM / HIGH 중 하나입니다.

    recommended_action:
    - prediction과 risk_level을 바탕으로 운영자에게 제안하는 조치입니다.

    evidence:
    - 입력 feature 중 위험 판단에 참고할 수 있는 근거 목록입니다.
    - 현재는 rule-based evidence이며,
      이후 학습 단계에서 feature importance, anomaly score, SHAP로 확장할 예정입니다.
    """

    probability: float
    prediction: int
    threshold: float
    risk_level: str
    recommended_action: str
    evidence: list[dict[str, Any]]


def normalize_type_value(value: Any) -> int:
    """
    raw input의 Type 값을 모델이 사용할 수 있는 숫자 값으로 변환합니다.

    AI4I 데이터의 Type은 원래 문자열입니다.

    L: Low quality type
    M: Medium quality type
    H: High quality type

    Day 1 전처리에서는 이 값을 다음과 같이 mapping했습니다.

    L -> 0
    M -> 1
    H -> 2

    추론 시에도 학습 때와 같은 방식으로 변환해야 합니다.
    """

    if isinstance(value, str):
        type_mapping = {
            "L": 0,
            "M": 1,
            "H": 2,
        }

        normalized = value.strip().upper()

        if normalized not in type_mapping:
            raise ValueError(
                "Type 값은 'L', 'M', 'H' 중 하나여야 합니다. "
                f"입력값: {value}"
            )

        return type_mapping[normalized]

    if isinstance(value, int):
        if value not in {0, 1, 2}:
            raise ValueError(
                "숫자 Type 값은 0, 1, 2 중 하나여야 합니다. "
                f"입력값: {value}"
            )

        return value

    if isinstance(value, float):
        if value not in {0.0, 1.0, 2.0}:
            raise ValueError(
                "숫자 Type 값은 0, 1, 2 중 하나여야 합니다. "
                f"입력값: {value}"
            )

        return int(value)

    raise ValueError(
        "Type 값은 문자열 'L'/'M'/'H' 또는 숫자 0/1/2여야 합니다. "
        f"입력값: {value}"
    )


def validate_raw_sample(
    raw_sample: dict[str, Any],
    feature_columns: list[str],
) -> None:
    """
    raw_sample에 모델이 필요로 하는 feature가 모두 들어 있는지 검증합니다.

    왜 필요한가?
    - 모델은 학습 때 사용한 feature 순서와 개수를 그대로 기대합니다.
    - feature가 하나라도 빠지면 tensor shape이 달라질 수 있습니다.
    - 잘못된 입력을 조용히 처리하면 예측 결과를 믿을 수 없습니다.
    """

    missing_columns = [
        column
        for column in feature_columns
        if column not in raw_sample
    ]

    if missing_columns:
        raise ValueError(
            "추론 입력에 필요한 feature가 누락되었습니다. "
            f"missing_columns={missing_columns}"
        )


def build_single_sample_dataframe(
    raw_sample: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    단일 raw input을 모델 입력용 DataFrame으로 변환합니다.

    중요한 점:
    - DataFrame column 순서는 반드시 학습 때의 feature_columns와 같아야 합니다.
    - PyTorch 모델은 column 이름을 보지 않고 숫자 배열 순서만 봅니다.
    - 따라서 순서가 바뀌면 전혀 다른 의미의 입력이 될 수 있습니다.
    """

    validate_raw_sample(
        raw_sample=raw_sample,
        feature_columns=feature_columns,
    )

    row = {
        column: raw_sample[column]
        for column in feature_columns
    }

    df = pd.DataFrame(
        [row],
        columns=feature_columns,
    )

    if "Type" in df.columns:
        df["Type"] = df["Type"].apply(normalize_type_value)

    return df


def scale_single_sample_dataframe(
    sample_df: pd.DataFrame,
    artifacts: FailureModelArtifacts,
) -> pd.DataFrame:
    """
    단일 sample DataFrame에 feature scaling을 적용합니다.

    Day 4에서 Type은 scaling하지 않았습니다.

    이유:
    - Type은 L/M/H를 0/1/2로 바꾼 범주형 feature입니다.
    - 숫자처럼 보이지만 온도, 회전 속도, 토크처럼 연속적인 센서 값이 아닙니다.

    따라서 추론 시에도 학습 때와 동일하게 numeric sensor feature만 scaling합니다.
    """

    scaled_df = sample_df.copy()

    numeric_feature_columns = [
        column
        for column in artifacts.feature_columns
        if column != "Type"
    ]

    scaled_numeric_values = artifacts.scaler.transform(
        scaled_df[numeric_feature_columns]
    )

    scaled_df.loc[:, numeric_feature_columns] = scaled_numeric_values

    return scaled_df[artifacts.feature_columns]


def dataframe_to_single_tensor(sample_df: pd.DataFrame) -> torch.Tensor:
    """
    단일 sample DataFrame을 PyTorch Tensor로 변환합니다.

    모델은 pandas DataFrame을 직접 입력받지 않습니다.
    따라서 DataFrame의 숫자 값을 float32 tensor로 변환해야 합니다.

    최종 shape:
    - 단일 sample이라도 shape은 (1, input_dim)이어야 합니다.
    - 예: (1, 6)
    """

    return torch.tensor(
        sample_df.values,
        dtype=torch.float32,
    )


def calculate_risk_level(probability: float) -> str:
    """
    고장 확률을 사람이 이해하기 쉬운 위험 등급으로 변환합니다.

    기준:
    - probability >= 0.70: HIGH
    - probability >= 0.40: MEDIUM
    - 그 외: LOW

    threshold와 risk_level은 역할이 다릅니다.

    threshold:
    - 최종 prediction 0/1을 정하는 기준

    risk_level:
    - 사용자나 운영자가 보기 쉽게 위험도를 설명하는 기준
    """

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.40:
        return "MEDIUM"

    return "LOW"

def build_recommended_action(
        prediction: int,
        risk_level: str,
) -> str:
    """
    prediction과 risk_level을 바탕으로 운영자에게 보여줄 권장 조치를 만듭니다.

    prediction: 
    - thredhold 기준 최종 고장 여부입니다.

    risk_level:
    - probability 기준 위험 등급입니다.

    둘을 함께 보는 이유:
    - prediction은 0 / 1이라 단순합니다.
    - risk_level은 위험 정도를 사람이 이해하기 쉽게 보여줍니다.
    """

    if prediction == 1 and risk_level == "HIGH":
        return "고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다."

    if prediction == 1 and risk_level == "MEDIUM":
        return "고장 가능성이 있습니다. 센서 값과 설비 상태를 추가 확인하세요."

    if risk_level == "MEDIUM":
        return "즉시 고장으로 판단되지는 않지만, 상태 변화를 모니터링하세요."

    return "현재 입력 기준으로는 정상 범위로 판단됩니다."


def build_rule_based_evidence(
    raw_sample: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    raw input을 바탕으로 간단한 rule 기반 evidence를 생성합니다.
    
    주의:
    - 이 evidence는 모델 내부를 직접 설명하는 SHAP 값은 아닙니다.
    - 현재 단계에서는 입력 feature 중 제조적으로 의미 있는 위험 신호를
    사람이 이해하기 쉽게 정리하는 참고 근거입니다.

    이 프로젝트는 학습용 레퍼런스 프로젝트이므로,
    여기서 끝내지 않고 이후 단계에서 다음 evidence 방식을 추가로 다룹니다.

    1. rule-based evidence
    - 사람이 정한 기준으로 위험 feature를 설명합니다.
    - 예: Tool wear가 200 이상이면 공구 마모 위험 근거로 표시

    2. feature importance
    - 모델 또는 데이터 기준으로 어떤 feature가 예측에 많이 영향을 주는지 확인합니다.

    3. permutation importance
    - 특정 feature 값을 섞었을 때 성능이 얼마나 떨어지는지 보고 중요도를 판단합니다.

    4. anomaly score
    - 정상 패턴과 얼마나 다른지 점수화합니다.
    - 예: 평균적인 정상 설비 상태에서 멀수록 위험도가 높다고 판단

    5. SHAP
    - 개별 예측 결과에서 각 feature가 probability를 높였는지 낮췄는지 설명합니다.

    따라서 현재 rule-based evidence는 최종 설명 방식이 아니라,
    이후 feature importance, anomaly score, SHAP로 확장하기 위한 1단계 evidence입니다.
    """

    evidence: list[dict[str, Any]] = []

    torque = raw_sample.get("Torque [Nm]")
    tool_wear = raw_sample.get("Tool wear [min]")
    rotational_speed = raw_sample.get("Rotational speed [rpm]")
    air_temperature = raw_sample.get("Air temperature [K]")
    process_temperature = raw_sample.get("Process temperature [K]")

    if isinstance(tool_wear, int | float) and tool_wear >= 200:
        evidence.append(
            {
                "feature": "Tool wear [min]",
                "value": tool_wear,
                "message": "공구 마모 시간이 높아 고장 위험 판단에 참고됩니다.",
            }
        )

    if isinstance(torque, int | float) and torque >= 60:
        evidence.append(
            {
                "feature": "Torque [Nm]",
                "value": torque,
                "message": "토크 값이 높아 설비 부하 가능성을 확인해야 합니다.",
            }
        )

    if isinstance(rotational_speed, int | float) and rotational_speed <= 1300:
        evidence.append(
            {
                "feature": "Rotational speed [rpm]",
                "value": rotational_speed,
                "message": "회전 속도가 낮아 비정상 운전 가능성을 확인해야 합니다.",
            }
        )

    if (
        isinstance(air_temperature, int | float)
        and isinstance(process_temperature, int | float)
    ):
        temperature_gap = process_temperature - air_temperature

        if temperature_gap >= 12:
            evidence.append(
                {
                    "feature": "Process temperature [K] - Air temperature [K]",
                    "value": temperature_gap,
                    "message": "공정 온도와 대기 온도의 차이가 커서 열적 이상 가능성을 확인해야 합니다.",
                }
            )

    if not evidence:
        evidence.append(
            {
                "feature": "overall",
                "value": None,
                "message": "현재 rule 기준에서 뚜렷한 위험 feature는 발견되지 않았습니다.",
            }
        )

    return evidence

def predict_failure_from_artifacts(
    raw_sample: dict[str, Any],
    artifacts: FailureModelArtifacts,
) -> FailurePredictionResult:
    """
    이미 로드된 artifacts를 사용해 단일 설비 고장 예측을 수행합니다.

    이 함수는 실제 추론의 핵심 함수입니다.

    전체 흐름:
    1. raw_sample 검증
    2. Type 값 변환
    3. DataFrame 생성
    4. 학습 때 저장한 scaler로 scaling
    5. Tensor 변환
    6. model inference
    7. sigmoid로 probability 계산
    8. threshold로 prediction 계산
    9. risk_level 계산
    """

    sample_df = build_single_sample_dataframe(
        raw_sample=raw_sample,
        feature_columns=artifacts.feature_columns,
    )

    scaled_df = scale_single_sample_dataframe(
        sample_df=sample_df,
        artifacts=artifacts,
    )

    input_tensor = dataframe_to_single_tensor(scaled_df)

    artifacts.model.eval()

    with torch.no_grad():
        logits = artifacts.model(input_tensor)
        probability = torch.sigmoid(logits).item()

    prediction = int(probability >= artifacts.threshold)
    risk_level = calculate_risk_level(probability)

    recommended_action = build_recommended_action(
        prediction=prediction,
        risk_level=risk_level,
    )

    evidence = build_rule_based_evidence(raw_sample)

    return FailurePredictionResult(
        probability=probability,
        prediction=prediction,
        threshold=artifacts.threshold,
        risk_level=risk_level,
        recommended_action=recommended_action,
        evidence=evidence,
    )


def predict_failure(
    raw_sample: dict[str, Any],
    artifact_dir: str | Path = "models/failure_mlp",
) -> FailurePredictionResult:
    """
    저장된 artifact 폴더에서 model/scaler/metadata를 로드한 뒤,
    단일 설비 고장 예측을 수행하는 편의 함수입니다.

    FastAPI나 Agent에서는 이 함수를 호출하면 됩니다.

    예:
    - API 요청이 들어온다.
    - 요청 body를 raw_sample dict로 만든다.
    - predict_failure(raw_sample)를 호출한다.
    - probability, prediction, risk_level을 응답으로 반환한다.
    """

    artifacts = load_failure_model_artifacts(artifact_dir)

    return predict_failure_from_artifacts(
        raw_sample=raw_sample,
        artifacts=artifacts,
    )