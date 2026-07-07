from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# 현재 파일 위치:
# manufacturing-ai-quality-agent-reference/scripts/run_permutation_importance.py
#
# Path(__file__)
# → 현재 실행 중인 파일 경로
#
# Path(__file__).resolve()
# → 절대 경로로 변환
#
# Path(__file__).resolve().parents[0]
# → scripts 폴더
#
# Path(__file__).resolve().parents[1]
# → 프로젝트 루트 폴더
#
# 우리가 import하려는 src 폴더는 프로젝트 루트 바로 아래에 있습니다.
# 따라서 프로젝트 루트를 Python import 경로에 추가해야
# from src.data.schemas import ... 같은 import가 동작합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# sys.path는 Python이 module/package를 찾는 경로 목록입니다.
#
# 현재 스크립트를 scripts 폴더에서 실행하면
# Python이 src 폴더를 자동으로 찾지 못할 수 있습니다.
#
# 그래서 프로젝트 루트를 sys.path에 직접 추가합니다.
#
# str(PROJECT_ROOT)를 넣는 이유:
# sys.path에는 Path 객체가 아니라 문자열 경로를 넣는 것이 일반적이기 때문입니다.
if str(PROJECT_ROOT) not in sys.path:
    # sys.path는 Python이 import할 module/package를 찾는 경로 목록입니다.
    #
    # 예를 들어 아래 import를 실행한다고 하면:
    #
    # from src.data.schemas import AI4I_FEATURE_COLUMNS
    #
    # Python은 sys.path에 들어 있는 폴더들을 순서대로 확인하면서
    # 그 안에 src/data/schemas.py가 있는지 찾습니다.
    #
    # 그런데 이 파일은 scripts 폴더 안에 있습니다.
    #
    # 현재 파일 위치:
    # project_root/scripts/run_permutation_importance.py
    #
    # import하려는 src 폴더 위치:
    # project_root/src
    #
    # 즉, src는 scripts 폴더 안에 있는 것이 아니라
    # project_root 바로 아래에 있습니다.
    #
    # 그래서 python scripts/run_permutation_importance.py처럼 실행하면
    # Python이 project_root를 import 검색 경로로 잡지 못해서
    # ModuleNotFoundError: No module named 'src'가 날 수 있습니다.
    #
    # PROJECT_ROOT는 project_root 폴더 경로입니다.
    #
    # sys.path.insert(0, str(PROJECT_ROOT))는
    # Python의 import 검색 경로 맨 앞에 project_root를 추가하라는 뜻입니다.
    #
    # insert(0, ...)을 쓰는 이유:
    # - 0번 위치는 리스트의 맨 앞입니다.
    # - 맨 앞에 넣으면 Python이 이 프로젝트 폴더를 가장 먼저 확인합니다.
    #
    # str(PROJECT_ROOT)를 쓰는 이유:
    # - PROJECT_ROOT는 Path 객체입니다.
    # - sys.path에는 일반적으로 문자열 경로를 넣습니다.
    # - 그래서 Path 객체를 문자열로 바꿔 넣습니다.
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schemas import (
    AI4I_FEATURE_COLUMNS,
    AI4I_TARGET_COLUMN,
    AI4I_TYPE_MAPPING,
)

from src.interpretability.permutation_importance import (
    calculate_permutation_importance,
    format_permutation_importance_as_evidence,
)

from src.models.failure_mlp import FailureMLP

# 이 스크립트의 목적:
#
# Day 6-1에서는 ToyFailureModel을 사용해서
# permutation importance 개념 자체를 테스트했습니다.
#
# 이제는 실제 프로젝트 흐름에 연결합니다.
#
# 흐름:
# 1. AI4I CSV를 로드합니다.
# 2. Day 1과 같은 방식으로 feature/target을 분리합니다.
# 3. Day 4~5에서 사용한 scaler를 로드합니다.
# 4. Day 5에서 저장한 FailureMLP 모델을 로드합니다.
# 5. X_test에 대해 permutation importance를 계산합니다.
# 6. feature별 중요도를 출력합니다.
#
# 중요한 점:
# permutation importance는 "개별 샘플 하나의 설명"이 아닙니다.
# test set 전체 기준으로 모델 성능에 어떤 feature가 중요한지 보는 방법입니다.


AI4I_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ai4i"

MODEL_DIR = PROJECT_ROOT / "models" / "failure_mlp"
MODEL_PATH = MODEL_DIR / "model.pt"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


def find_ai4i_csv_path() -> Path:
    """
    data/raw/ai4i 폴더 안에서 CSV 파일을 찾습니다.

    왜 파일명을 직접 고정하지 않는가?

    사용자가 Kaggle에서 받은 파일명이 다음처럼 다를 수 있기 때문입니다.

    - ai4i2020.csv
    - predictive_maintenance.csv
    - ai4i_2020.csv

    그래서 data/raw/ai4i 폴더 안의 csv 파일을 자동으로 찾습니다.
    """

    csv_files = sorted(AI4I_RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"AI4I CSV 파일을 찾을 수 없습니다. 위치를 확인하세요: {AI4I_RAW_DIR}"
        )

    # 보통 이 폴더에는 AI4I CSV 하나만 있어야 합니다.
    # 여러 개가 있으면 첫 번째 파일을 사용합니다.
    return csv_files[0]


def load_metadata() -> dict:
    """
    Day 5에서 저장한 metadata.json을 로드합니다.

    metadata에는 추론과 해석에 필요한 설정이 들어 있습니다.

    예:
    - input_dim
    - hidden_dim
    - dropout_rate
    - threshold
    - feature_columns

    특히 feature_columns가 중요합니다.

    모델은 학습 당시의 feature 순서에 민감합니다.
    같은 feature라도 순서가 바뀌면 모델 입력 의미가 달라집니다.
    """

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"metadata.json을 찾을 수 없습니다: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_scaler():
    """
    Day 5에서 저장한 scaler.joblib을 로드합니다.

    이 scaler는 Day 4에서 train set에 fit된 StandardScaler입니다.

    중요한 원칙:
    - train set에는 fit_transform 가능
    - test set에는 transform만 가능

    현재는 이미 저장된 scaler를 불러오기 때문에,
    여기서 다시 fit을 하면 안 됩니다.
    """

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"scaler.joblib을 찾을 수 없습니다: {SCALER_PATH}")

    return joblib.load(SCALER_PATH)


def load_failure_model(metadata: dict) -> FailureMLP:
    """
    저장된 PyTorch 모델 weight를 FailureMLP 구조에 로드합니다.

    PyTorch 모델 로드 순서:

    1. 먼저 모델 구조를 다시 만듭니다.
    2. 저장된 weight/bias인 state_dict를 불러옵니다.
    3. model.load_state_dict()로 weight/bias를 주입합니다.
    4. model.eval()로 평가 모드로 전환합니다.

    왜 구조를 먼저 만들어야 하는가?

    model.pt에는 보통 weight/bias 값만 저장합니다.
    weight/bias는 담겨 있지만,
    어떤 layer 구조였는지는 코드에서 다시 만들어줘야 합니다.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model.pt를 찾을 수 없습니다: {MODEL_PATH}")

    input_dim = int(metadata.get("input_dim", 6))
    hidden_dim = int(metadata.get("hidden_dim", 32))
    dropout_rate = float(metadata.get("dropout_rate", 0.2))

    model = FailureMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )

    loaded = torch.load(MODEL_PATH, map_location="cpu")

    # 저장 방식에 따라 model.pt 구조가 다를 수 있습니다.
    #
    # 방식 1:
    # torch.save(model.state_dict(), path)
    # -> loaded 자체가 state_dict입니다.
    #
    # 방식 2:
    # torch.save({"model_state_dict": model.state_dict()}, path)
    # -> loaded["model_state_dict"]가 state_dict입니다.
    #
    # 두 경우 모두 처리할 수 있게 만듭니다.
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    else:
        state_dict = loaded

    model.load_state_dict(state_dict)

    # permutation importance는 평가 작업입니다.
    # Dropout이 꺼져야 하므로 eval mode로 전환합니다.
    model.eval()

    return model


def encode_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    AI4I의 Type 컬럼을 L/M/H 문자열에서 숫자로 변환합니다.

    Day 1에서 정리한 기준:

    L → 0
    M → 1
    H → 2

    현재 레퍼런스 프로젝트에서는 Type을 label encoding 방식으로 처리했습니다.
    이후 최종 프로젝트에서는 one-hot encoding으로 개선할 수 있습니다.
    """

    df = df.copy()

    if "Type" not in df.columns:
        raise ValueError("AI4I 데이터에 Type 컬럼이 없습니다.")

    unknown_values = set(df["Type"].unique()) - set(AI4I_TYPE_MAPPING.keys())

    if unknown_values:
        raise ValueError(f"Type 컬럼에 알 수 없는 값이 있습니다: {unknown_values}")

    df["Type"] = df["Type"].map(AI4I_TYPE_MAPPING)

    return df


def prepare_test_data(
    metadata: dict,
    scaler,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    AI4I CSV에서 X_test, y_test를 준비합니다.

    여기서는 Day 1~4와 같은 split 기준을 사용해야 합니다.

    이유:
    permutation importance는 test set 기준으로 계산하는 것이 일반적입니다.
    학습에 사용한 train set으로 중요도를 계산하면
    모델이 이미 본 데이터에 대한 설명이 되어버립니다.

    따라서:
    - train/test split을 다시 만들고
    - X_test만 사용합니다.
    """

    csv_path = find_ai4i_csv_path()

    print(f"[INFO] AI4I CSV path: {csv_path}")

    df = pd.read_csv(csv_path)
    df = encode_type_column(df)

    feature_columns = metadata.get("feature_columns", AI4I_FEATURE_COLUMNS)

    missing_features = set(feature_columns) - set(df.columns)

    if missing_features:
        raise ValueError(f"데이터에 필요한 feature가 없습니다: {missing_features}")

    if AI4I_TARGET_COLUMN not in df.columns:
        raise ValueError(f"target 컬럼이 없습니다: {AI4I_TARGET_COLUMN}")

    X = df[feature_columns].copy()
    y = df[AI4I_TARGET_COLUMN].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Day 4~5 기준:
    # numeric sensor feature만 scaling하고,
    # Type은 범주형 mapping 값이므로 scaling하지 않습니다.
    #
    # 현재 feature_columns가 다음과 같다고 가정합니다.
    #
    # Air temperature [K]
    # Process temperature [K]
    # Rotational speed [rpm]
    # Torque [Nm]
    # Tool wear [min]
    # Type
    numeric_feature_columns = [
        column for column in feature_columns if column != "Type"
    ]

    X_test_scaled = X_test.copy()

    # 저장된 scaler는 train set의 numeric feature에 fit되어 있습니다.
    # 따라서 test set에는 transform만 적용합니다.
    X_test_scaled[numeric_feature_columns] = scaler.transform(
        X_test_scaled[numeric_feature_columns]
    )

    # model 입력 column 순서를 metadata 기준으로 다시 고정합니다.
    # 이 줄은 매우 중요합니다.
    X_test_scaled = X_test_scaled[feature_columns]

    return X_test_scaled, y_test


def print_importance_summary(summary) -> None:
    """
    permutation importance 결과를 보기 좋게 출력합니다.
    """

    print()
    print("=" * 80)
    print("[INFO] Permutation importance summary")
    print("=" * 80)
    print(f"[INFO] baseline_score: {summary.baseline_score:.4f}")
    print(f"[INFO] threshold     : {summary.threshold:.4f}")
    print(f"[INFO] metric_name   : {summary.metric_name}")
    print()

    print(
        "rank | feature                         | importance | permuted_score | std"
    )
    print("-" * 80)

    for rank, result in enumerate(summary.results, start=1):
        print(
            f"{rank:>4} | "
            f"{result.feature_name:<31} | "
            f"{result.importance_mean:>10.4f} | "
            f"{result.permuted_score_mean:>14.4f} | "
            f"{result.importance_std:>6.4f}"
        )


def print_evidence(summary) -> None:
    """
    permutation importance 결과를 evidence 형식으로 바꿔 출력합니다.

    이 evidence는 Day 5의 rule-based evidence와 다릅니다.

    Day 5 rule-based evidence:
    - 개별 입력 sample의 값 기준
    - 예: Tool wear가 230이라 높다

    Day 6 permutation evidence:
    - 전체 test set 기준 모델 중요도
    - 예: Tool wear를 섞었을 때 f1이 크게 떨어졌다
    """

    evidence = format_permutation_importance_as_evidence(
        summary=summary,
        top_k=3,
    )

    print()
    print("=" * 80)
    print("[INFO] Model-based evidence candidates")
    print("=" * 80)

    for item in evidence:
        print(f"- feature   : {item['feature']}")
        print(f"  importance: {item['importance']:.4f}")
        print(f"  message   : {item['message']}")


def main() -> None:
    """
    실제 실행 진입점입니다.

    전체 흐름:

    1. metadata 로드
    2. scaler 로드
    3. FailureMLP 모델 로드
    4. AI4I test set 준비
    5. permutation importance 계산
    6. 결과 출력
    """

    print("[INFO] Day 6 permutation importance started")

    metadata = load_metadata()
    scaler = load_scaler()
    model = load_failure_model(metadata=metadata)

    X_test, y_test = prepare_test_data(
        metadata=metadata,
        scaler=scaler,
    )

    threshold = float(metadata.get("threshold", 0.7))

    print(f"[INFO] X_test shape: {X_test.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")
    print(f"[INFO] threshold   : {threshold:.4f}")

    summary = calculate_permutation_importance(
        model=model,
        X=X_test,
        y=y_test,
        threshold=threshold,
        metric_name="f1",
        n_repeats=10,
        random_state=42,
    )

    print_importance_summary(summary)
    print_evidence(summary)

    print()
    print("[INFO] Day 6 permutation importance completed")


if __name__ == "__main__":
    main()