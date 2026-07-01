"""
AI4I 데이터 전처리 흐름을 테스트하는 파일입니다.

이 테스트는 실제 CSV 파일에 의존하지 않고,
작은 샘플 DataFrame을 직접 만들어 전처리 함수가 의도대로 동작하는지 확인합니다.
"""

import pandas as pd
import pytest

from src.data.preprocess_ai4i import (
    encode_type_column,
    preprocess_ai4i_dataframe,
    split_features_and_target,
    validate_ai4i_columns,
)

def make_sample_ai4i_dataframe() -> pd.DataFrame:
    """
    테스트용 AI4I 샘플 DataFrame을 만듭니다.

    실제 AI4I 전체 CSV를 테스트에 직접 사용하지 않는 이유:
    1. 테스트 속도를 빠르게 유지하기 위해
    2. 외부 데이터 파일이 없어도 테스트가 가능하게 하기 위해
    3. 어떤 값으로 어떤 결과가 나와야 하는지 명확히 보기 위해

    주의:
    stratify=y를 사용하려면 각 class가 최소 2개 이상 있어야 합니다.
    그래서 Machine failure 0과 2을 각각 3개씩 넣었습니다.
    """
    return pd.DataFrame(
        {
            # UDI와 Product ID는 식별자 컬럼입니다.
            # preprocess_ai4i_dataframe에서는 feature로 사용하지 않습니다.
            "UDI": [1, 2, 3, 4, 5, 6],
            "Product ID": ["A1", "A2", "A3", "A4", "A5", "A6"],

            # Type은 범주형 컬럼입니다.
            # encode_type_column 함수에서 L/M/H를 0/1/2로 변환합니다.
            "Type": ["L", "M", "H", "L", "M", "H"],

            # 아래 5개는 숫자형 센서/공정 feature입니다.
            "Air temperature [K]": [298.1, 298.2, 298.3, 299.1, 299.2, 299.3],
            "Process temperature [K]": [308.6, 308.7, 308.8, 309.1, 309.2, 309.3],
            "Rotational speed [rpm]": [1551, 1408, 1498, 1600, 1300, 1450],
            "Torque [Nm]": [42.8, 46.3, 49.4, 40.1, 50.2, 45.0],
            "Tool wear [min]": [0, 3, 5, 200, 210, 220],

            # Machine failure는 target입니다.
            # 0은 정상, 1은 고장으로 가정합니다.
            "Machine failure": [0, 0, 0, 1, 1, 1],
        }
    )

def test_validate_ai4i_columns_success():
    """
    필요한 컬럼이 모두 있으면 validate_ai4i_column가 에러 없이 통과해야 합니다.
    """

    # 테스트용 DateFrame을 만듭니다.
    df = make_sample_ai4i_dataframe()

    # 일부러 target 컬럼을 제거합니다.
    # Machine failure는 모델이 학습해야 하는 정답 label이므로 반드시 필요합니다.
    df = df.drop(columns=["Machine failure"])

    # target 컬럼이 없으므로 validate_ai4i_columns에서 ValueError가 발생해야 합니다.
    with pytest.raises(ValueError):
        validate_ai4i_columns(df)

def test_encode_type_column():
    """
    Type 컬럼의 L, M, H가 각각 0, 1, 2로 변환되는지 확인합니다.
    """

    # 테스트용 DataFrame을 만듭니다.
    df = make_sample_ai4i_dataframe()

    # Type 컬럼을 숫자로 변환합니다.
    encoded_df = encode_type_column(df)

    # 변환 결과가 기대한 값과 같은지 확인합니다.
    assert encoded_df["Type"].tolist() == [0, 1, 2, 0, 1, 2]


def test_split_features_and_target():
    """
    feature X와 target y가 올바르게 분리되는지 확인합니다.
    """

    # 테스트용 DataFrame을 만듭니다.
    df = make_sample_ai4i_dataframe()

    # split_features_and_target은 Type이 이미 숫자로 변환되어 있다고 가정합니다.
    # 그래서 먼저 Type encoding을 수행합니다.
    df = encode_type_column(df)

    # X와 y를 분리합니다.
    X, y = split_features_and_target(df)

    # X에는 target 컬럼이 들어가면 안 됩니다.
    assert "Machine failure" not in X.columns

    # y는 Machine failure 값과 같아야 합니다.
    assert y.tolist() == [0, 0, 0, 1, 1, 1]

    # X에는 최종 feature 6개가 있어야 합니다.
    # 숫자형 feature 5개 + Type 1개 = 총 6개입니다.
    assert X.shape[1] == 6


def test_preprocess_ai4i_dataframe():
    """
    전체 전처리 함수가 X_train, X_test, y_train, y_test를 올바르게 반환하는지 확인합니다.
    """

    # 테스트용 DataFrame을 만듭니다.
    df = make_sample_ai4i_dataframe()

    # 전체 전처리 흐름을 실행합니다.
    # test_size=0.5이므로 6개 중 3개는 train, 3개는 test로 나뉩니다.
    result = preprocess_ai4i_dataframe(
        df,
        test_size=0.5,
        random_state=42,
    )

    # train/test split 결과가 비어 있지 않은지 확인합니다.
    assert len(result.X_train) > 0
    assert len(result.X_test) > 0
    assert len(result.y_train) > 0
    assert len(result.y_test) > 0

    # feature에는 target 컬럼이 포함되면 안 됩니다.
    assert "Machine failure" not in result.X_train.columns

    # Type 컬럼이 문자열이 아니라 숫자로 변환되었는지 확인합니다.
    assert result.X_train["Type"].dtype != "object"

    # feature 개수는 6개여야 합니다.
    assert result.X_train.shape[1] == 6
    assert result.X_test.shape[1] == 6