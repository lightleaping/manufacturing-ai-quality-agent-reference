"""
AI4I 제조 데이터를 모델 학습용 형태로 전처리하는 파일입니다.

전처리의 목표는 원본 CSV 데이터를 PyTorch 모델이 학습할 수 있는
X_Train, X_test, y_train, y-test 형태로 바꾸는 것입니다.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.schemas import (
    AI4I_CATEGORICAL_COLUMNS,
    AI4I_FEATURE_COLUMNS,
    AI4I_TARGET_COLUMN,
    AI4I_TYPE_MAPPING,
)

@dataclass
class PreprocessedAI4IData:
    """
    전처리 결과를 담는 데이터 클래스입니다.

    여러 값을 튜플로 반환하면 순서를 헷갈리기 쉽습니다.
    그래서 이름이 있는 필드로 묶어서 반환합니다.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def validate_ai4i_columns(df: pd.DataFrame) -> None:
    """
    AI4I 데이터에 필요한 컬럼이 모두 있는지 확인합니다.

    필요한 컬럼이 없으면 모델 학습 전에 바로 에러를 발생시킵니다.
    """

    # 모델 입력 feature, 범주형 컬럼, target 컬럼을 하나의 리스트로 합칩니다.
    required_columns = (
        AI4I_FEATURE_COLUMNS
        + AI4I_CATEGORICAL_COLUMNS
        + [AI4I_TARGET_COLUMN]
    )

    # 필요한 컬럼 중 DataFrame에 없는 컬럼만 찾습니다.
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    # 누락된 컬럼이 하나라도 있으면 에러를 발생시킵니다.
    if missing_columns:
        raise ValueError(f"AI4I 데이터에 필요한 컬럼이 없습니다: {missing_columns}")

def encode_type_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Type 컬럼을 문자열에서 숫자로 변환합니다.

    encode: 정보를 특정 형식으로 변환하거나 암호화한다.

    예:
    L -> 0
    M -> 1
    H -> 2

    Type 컬럼은 원본 CSV에 따라 공백이 섞여 있을 수 있으므로,
    먼저 문자열로 변환한 뒤 앞뒤 공백을 제거하고 대문자로 통일합니다.
    """

    # 원본 DataFrame을 직접 수정하지 않기 위해 copy를 만듭니다.
    # 이렇게 하면 함수 밖의 원본 데이터가 의도치 않게 바뀌지 않습니다.
    df = df.copy()

    # # Type 컬럼의 L, M, H 값을 숫자로 변환합니다.
    # df["Type"] = df["Type"].map(AI4I_TYPE_MAPPING)

    # Type 컬럼 값을 문자열로 변환합니다.
    # 혹시 숫자나 다른 타입으로 들어온 값이 있어도 문자열 처리 기준을 통일하기 위함입니다.
    df["Type"] = df["Type"].astype(str)

    # Type 컬럼의 앞뒤 공백을 제거하고 대문자로 통일합니다.
    #
    # 예:
    # " L" -> "L"
    # "i" -> "L"
    #
    # 이렇게 해두면 원본 데이터에 사소한 공백이나 소문자가 있어도
    # mapping이 안정적으로 동작합니다.
    df["Type"] = df["Type"].str.strip().str.upper()

    # Type 컬럼의 L, M, H 값을 숫자로 변환합니다.
    #
    # AI4I_TYPE_MAPPING:
    # {
    #     "L": 0,
    #     "M": 1,
    #     "H": 2,
    # }
    #
    # map()은 mapping에 없는 값을 만나면 NaN으로 바꿉니다.
    # 예를 들어 Type에 "UNKNOWN"이 있으면 숫자로 바꿀 수 없습니다.
    if df["Type"].isna().any():
        raise ValueError(
            "Type 컬럼에 L, M, H 외의 알 수 없는 값이 포함되어 있습니다."
            "df['Type'].unique()로 실제 값을 확인하세요."
        )



    # 변환된 DataFrmae을 반환합니다.
    return df

def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    DataFrame에서 feature X와 target y를 분리합니다.

    x는 모델 입력값이고, y는 모델이 맞춰야 하는 정답입니다.
    """

    # 숫자로 변환된 Type 컬럼까지 feature에 포함합니다.
    feature_columns = AI4I_FEATURE_COLUMNS + AI4I_CATEGORICAL_COLUMNS

    # 모델 입력 feature입니다.
    x = df[feature_columns]

    # 모델이 예측할 target입니다.
    y = df[AI4I_TARGET_COLUMN]

    # x와 y를 반환합니다.
    return x, y

def preprocess_ai4i_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    ) -> PreprocessedAI4IData:
    """
    AI4I DataFrame을 학습용 데이터로 전처리압니다.

    Args:
        df:
            원본 AI4I DataFrame입니다.
        test_size:
            전체 데이터 중 test set으로 사용할 비율입니다.
            기본값 0.2는 20%를 test set으로 사용한다는 뜻입니다.
        random_state:
            train / test split 결과를 재현 가능하게 만들기 위한 seed입니다.

    Returns:
        PreprocessedAI4IData:
            X_train, X_text, y_train, y_test를 담은 객체입니다.
    """

    # 1. 필요한 컬럼이 모두 있는지 확인합니다.
    validate_ai4i_columns(df)

    # 2. Type 컬럼을 문자열에서 숫자로 변환합니다.
    df = encode_type_column(df)

    # 3. feature X와 target y를 분리합니다.
    X, y = split_features_and_target(df)

    # 4. train / test split을 수행합니다.
    # stratify=y는 train / test 양쪽에 고장 / 정상 비율이 비슷하게 들어가도록 돕습니다.
    # 제조 고장 데이터는 정상 데이터가 훨씬 많을 수 있으므로 비율 유지가 중요합니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 5. 전처리 결과를 데이터 클래스로 묶어 반환합니다.
    return PreprocessedAI4IData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
