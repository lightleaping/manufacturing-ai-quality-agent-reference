from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.schemas import AI4I_FEATURE_COLUMNS


@dataclass
class ScaledAI4IData:
    """
    scaling이 끝난 train/test feature와 scaler를 함께 보관하는 자료 구조입니다.

    X_train:
    - scaling이 적용된 학습 feature입니다.

    X_test:
    - train set 기준으로 학습된 scaler를 사용해 변환된 test feature입니다.

    scaler:
    - X_train에 fit된 StandardScaler 객체입니다.
    - 나중에 API나 추론 코드에서 새 입력값을 변환할 때 같은 scaler를 재사용해야 합니다.

    왜 scaler를 같이 반환하는가?
    - 학습 때 사용한 scaling 기준과 추론 때 사용하는 scaling 기준이 같아야 하기 때문입니다.
    - 만약 추론 때 새로운 scaler를 fit하면, 학습 때와 입력 분포가 달라져 모델 예측이 불안정해질 수 있습니다.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    scaler: StandardScaler


def scale_train_test_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> ScaledAI4IData:
    """
    AI4I feature에 StandardScaler를 적용합니다.

    StandardScaler:
    - 각 feature를 평균 0, 표준편차 1에 가깝게 변환합니다.

    변환 공식:
    scaled_value = (value - train_mean) / train_std

    중요한 원칙:
    - scaler.fit()은 반드시 X_train에만 수행합니다.
    - X_test에는 transform()만 수행합니다.

    이유:
    - test set은 모델이 처음 보는 데이터라고 가정해야 합니다.
    - test set의 평균/표준편차를 scaling 기준에 사용하면,
      test 정보가 학습 과정에 새어 들어가는 data leakage가 됩니다.

    현재는 AI4I_FEATURE_COLUMNS에 포함된 5개 numeric feature만 scaling합니다.

    Type 컬럼은 여기서 scaling하지 않습니다.
    이유:
    - 현재 Type은 L/M/H를 0/1/2로 단순 mapping한 범주형 feature입니다.
    - 숫자처럼 보이지만 실제 의미는 범주입니다.
    - 따라서 우선 numeric sensor feature만 scaling하고,
      Type은 이후 one-hot encoding으로 개선하는 것이 더 자연스럽습니다.
    """

    # 원본 DataFrame을 직접 수정하지 않기 위해 복사합니다.
    #
    # 함수 안에서 X_train["컬럼"] = ... 식으로 수정하면
    # 바깥에서 사용하는 원본 DataFrame까지 바뀔 수 있습니다.
    #
    # 이런 부작용을 줄이기 위해 copy()를 사용합니다.
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    # StandardScaler 객체를 생성합니다.
    #
    # 이 객체는 각 feature의 train 평균과 train 표준편차를 기억합니다.
    scaler = StandardScaler()

    # fit_transform:
    # - fit: X_train의 평균과 표준편차를 계산합니다.
    # - transform: 계산한 평균과 표준편차로 X_train을 변환합니다.
    #
    # 주의:
    # fit_transform은 train set에만 사용합니다.
    X_train_scaled[AI4I_FEATURE_COLUMNS] = scaler.fit_transform(
        X_train[AI4I_FEATURE_COLUMNS]
    )

    # transform:
    # - 이미 X_train에서 계산한 평균과 표준편차를 사용해 X_test를 변환합니다.
    #
    # 주의:
    # X_test에는 fit을 하면 안 됩니다.
    # test set의 정보를 scaling 기준에 사용하면 data leakage가 됩니다.
    X_test_scaled[AI4I_FEATURE_COLUMNS] = scaler.transform(
        X_test[AI4I_FEATURE_COLUMNS]
    )

    return ScaledAI4IData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        scaler=scaler,
    )