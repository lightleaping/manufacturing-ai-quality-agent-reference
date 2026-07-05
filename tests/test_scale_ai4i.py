import numpy as np
import pandas as pd

from src.data.scale_ai4i import scale_train_test_features
from src.data.schemas import AI4I_FEATURE_COLUMNS


def make_sample_X_train() -> pd.DataFrame:
    """
    scaling 테스트용 train feature를 만듭니다.

    실제 CSV를 사용하지 않는 이유:
    - 테스트가 빠릅니다.
    - 외부 파일에 의존하지 않습니다.
    - scaling 동작만 명확히 검증할 수 있습니다.
    """

    return pd.DataFrame(
        {
            "Air temperature [K]": [300.0, 302.0, 304.0, 306.0],
            "Process temperature [K]": [310.0, 312.0, 314.0, 316.0],
            "Rotational speed [rpm]": [1400.0, 1500.0, 1600.0, 1700.0],
            "Torque [Nm]": [30.0, 40.0, 50.0, 60.0],
            "Tool wear [min]": [0.0, 50.0, 100.0, 150.0],
            "Type": [0, 1, 2, 0],
        }
    )


def make_sample_X_test() -> pd.DataFrame:
    """
    scaling 테스트용 test feature를 만듭니다.

    test set은 scaler.fit에 사용되면 안 됩니다.
    train set에서 학습된 평균과 표준편차로 transform만 되어야 합니다.
    """

    return pd.DataFrame(
        {
            "Air temperature [K]": [301.0, 305.0],
            "Process temperature [K]": [311.0, 315.0],
            "Rotational speed [rpm]": [1450.0, 1650.0],
            "Torque [Nm]": [35.0, 55.0],
            "Tool wear [min]": [25.0, 125.0],
            "Type": [1, 2],
        }
    )


def test_scale_train_test_features_keeps_shape_and_columns() -> None:
    """
    scaling 후에도 X_train, X_test의 shape과 컬럼 구성이 유지되는지 확인합니다.

    scaling은 feature 값의 scale만 바꾸는 작업입니다.
    따라서 행 개수, 컬럼 개수, 컬럼 이름, 컬럼 순서가 바뀌면 안 됩니다.
    """

    X_train = make_sample_X_train()
    X_test = make_sample_X_test()

    result = scale_train_test_features(
        X_train=X_train,
        X_test=X_test,
    )

    assert result.X_train.shape == X_train.shape
    assert result.X_test.shape == X_test.shape

    # result.X_train.columns는 Python list가 아니라 pandas Index 객체입니다.
    #
    # 예:
    # Index(["Air temperature [K]", "Process temperature [K]", ...])
    #
    # 여기서는 scaling 후에도 컬럼 이름과 컬럼 순서가 그대로 유지되었는지
    # 비교하고 싶기 때문에 list()로 변환합니다.
    #
    # list(df.columns)의 의미:
    # - DataFrame의 컬럼 이름들을 Python list로 바꿉니다.
    #
    # 이 테스트는 값이 아니라 "컬럼 구조"를 확인합니다.
    assert list(result.X_train.columns) == list(X_train.columns)
    assert list(result.X_test.columns) == list(X_test.columns)


def test_numeric_train_features_scaled_to_mean_zero() -> None:
    """
    train set의 numeric feature들이 scaling 후 평균 0에 가까운지 확인합니다.

    StandardScaler는 train set 기준으로 평균 0, 표준편차 1이 되도록 변환합니다.
    """

    X_train = make_sample_X_train()
    X_test = make_sample_X_test()

    result = scale_train_test_features(
        X_train=X_train,
        X_test=X_test,
    )

    train_numeric_values = result.X_train[AI4I_FEATURE_COLUMNS].to_numpy()

    # axis=0:
    # - 컬럼별 평균을 계산합니다.
    #
    # np.allclose:
    # - floating point 계산에서는 정확히 0이 아니라
    #   아주 작은 오차가 생길 수 있으므로 근사 비교를 합니다.
    assert np.allclose(train_numeric_values.mean(axis=0), 0.0)


def test_numeric_train_features_scaled_to_std_one() -> None:
    """
    train set의 numeric feature들이 scaling 후 표준편차 1에 가까운지 확인합니다.

    주의:
    - StandardScaler는 population standard deviation 기준을 사용합니다.
    - numpy std의 기본 ddof=0과 같습니다.
    """

    X_train = make_sample_X_train()
    X_test = make_sample_X_test()

    result = scale_train_test_features(
        X_train=X_train,
        X_test=X_test,
    )

    train_numeric_values = result.X_train[AI4I_FEATURE_COLUMNS].to_numpy()

    assert np.allclose(train_numeric_values.std(axis=0), 1.0)


def test_type_column_is_not_scaled() -> None:
    """
    Type 컬럼은 scaling하지 않는지 확인합니다.

    현재 Type은 L/M/H를 0/1/2로 mapping한 범주형 feature입니다.
    숫자처럼 보이지만 온도, 회전속도, 토크처럼 연속적인 numeric sensor feature가 아닙니다.

    따라서 지금 단계에서는 Type을 그대로 두고,
    이후 one-hot encoding으로 개선할 수 있습니다.
    """

    X_train = make_sample_X_train()
    X_test = make_sample_X_test()

    result = scale_train_test_features(
        X_train=X_train,
        X_test=X_test,
    )

    # result.X_train["Type"]은 pandas Series입니다.
    #
    # 예:
    # 0    0
    # 1    1
    # 2    2
    # 3    0
    # Name: Type, dtype: int64
    #
    # 여기서는 Series 객체 자체가 아니라,
    # 그 안에 들어 있는 값들이 scaling 전과 같은지 비교하고 싶습니다.
    #
    # .tolist()의 의미:
    # - pandas Series 안의 값을 Python list로 꺼냅니다.
    #
    # 예:
    # result.X_train["Type"].tolist()
    # → [0, 1, 2, 0]
    #
    # 이 테스트는 Type 컬럼이 StandardScaler에 의해 바뀌지 않았는지 확인합니다.
    assert result.X_train["Type"].tolist() == X_train["Type"].tolist()
    assert result.X_test["Type"].tolist() == X_test["Type"].tolist()