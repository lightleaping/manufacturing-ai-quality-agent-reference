"""
이 프로젝트에서 AI4I를 어떻게 볼 것인지 정하는 기준표.

AI4I 제조 데이터에서 사용할 컬럼 이름과 기본 설정을 정의하는 파일입니다.

이 파일은 데이터 전처리 코드에서 공통으로 사용할 기준표 역할을 합니다.
컬럼명을 여러 파일에 직접 반복해서 쓰지 않고, 한 곳에서 관리하기 위해 분리했습니다.
"""

# 모델 입력에 사용할 센서 / 공정 feature 컬럼들입니다.
# 이 값들이 이후 PyTorch 모델의 입력 x가 됩니다.
AI4I_FEATURE_COLUMNS = [
    "Air temperature [K]",       # 주변 공기 온도
    "Process temperature [K]",   # 실제 공정 온도
    "Rotational speed [rpm]",    # 회전 속도
    "Torque [Nm]",               # 토크
    "Tool wear [min]",           # 공구 마모 시간
]

# 범주형 feature입니다.
# Type은 M, M, H 같은 문자값이므로 모델에 넣기 전에 숫자로 변환해야 합니다.
AI4I_CATEGORICAL_COLUMNS = [
    "Type"
]

# 모델이 예측해야 하는 target 컬럼입니다.
# 이 프로젝트의 첫 번째 모델은 설비 고장 여부를 예측합니다.
AI4I_TARGET_COLUMN = "Machine failure"

# 모델 학습에서 제외할 컬럼입니다.
# UDI, Product ID는 식별자에 가까워서 고장 예측의 일반적인 feature로 쓰지 않습니다.
# 이런 값을 그대로 사용하면 모델이 실제 고장 패턴을 학습하기보다
# 특정 ID를 외우는 방향으로 overfitting될 수 있습니다.
AI4I_DROP_COLUMNS = [
    "UDI",
    "Product ID"
]

# 고장 유형을 나타내는 컬럼들입니다.
# Day 1에서는 사용하지 않지만, 이후 failure type analysis에서 사용할 수 있습니다.
# 초기에는 Machine failure를 target으로 두고 정상 / 이진 분류부터 시작했습니다.
# 이후에는 TWF, HDF, PWF, OSF, RNF 컬럼을 활용해 고장 유형 분석이나
# multi-label classificaiton으로 확장할 수 있습니다.
AI4I_FAILURE_TYPE_COLUMNS = [
    "TWF",  # Tool wear Failure: 공구 마모 고장
    "HDF",  # Heat Dissipation Failure: 열 방출 고장
    "PWF",  # Power Failure: 전력 고장
    "OSF",  # Overstrain Failure: 과부하 고장
    "RNF",  # Random Failure: 랜덤 고장
]

# Type 컬럼의 문자값을 숫자로 바꾸기 위한 mapping입니다.
# 딥러닝 모델은 문자열을 바로 계산할 수 없기 때문에 숫자로 바꿔야 합니다.
#
# 초기 레퍼런스 프로젝트에서는 구조 이해를 위해 Type 컬럼을 간단히 숫자로 mapping했습니다.
# 다만 이 방식은 L, M, H 사이에 순서 관계가 있는 것처럼 모델이 해석할 수 있으므로,
# 최종 프로젝트에서는 one-hot encoding으로 개선할 수 있습니다.
AI4I_TYPE_MAPPING = {
    "L": 0,
    "M": 1,
    "H": 2,
}