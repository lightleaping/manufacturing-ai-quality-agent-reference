"""
failure_mlp.py

AI4I 제조 설비 데이터를 이용해
설비 고장 여부를 예측하는 PyTorch MLP 모델을 정의하는 파일입니다.

Day 1에서 만든 전처리 결과 X_train은
다음과 같은 6개 feature를 가집니다.

1. Air temperature [K]
2. Process temperature [K]
3. Rotational speed [rpm]
4. Torque [Nm]
5. Tool wear [min]
6. Type

이 모델은 위 6개 입력값을 받아
Machine failure가 발생할 확률을 예측합니다.
"""

import torch
from torch import nn

class FailureMLP(nn.Module):
    """
    FailureMLP는 설비 고장 여부를 예측하기 위한
    기본적인 Multi-Layer Perception 모델입니다.

    MLP는 표 형태의 정형 데이터(tabular data)를 입력으로 받아
    여러 개의 Linear layer를 통과시키며 패턴을 학습하는 신경망입니다.

    이 프로젝트에서는 AI4I 데이터의 6개 feature를 입력으로 받아
    최종적으로 고장 확률 하나를 출력합니다.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        dropout_rate: float = 0.2,
    ) -> None:
        """
        모델 구조를 정의하는 초기화 함수입니다.

        Parameters
        ----------
        input_dim:
            모델이 입력으로 받는 feature 개수입니다.
            AI4I 데이터에서는 숫자 feature 5개와 Type encoding 1개를 합쳐 총 6개입니다.
        
        hidden_dim:
            은닉층의 뉴런 개수입니다.
            입력 데이터를 더 풍부한 내부 표현으로 바꾸기 위한 중간 차원입니다.

        dropout_rate:
            학습 중 일부 뉴런을 임의로 꺼서 과적합을 줄이기 위한 비율입니다.
        """

        # nn.Module을 상속받은 클래스에서는 부모 클래스 초기화가 필요합니다.
        # PyTorch가 모델 내부 layer와 parameter를 추적할 수 있게 해줍니다.
        super().__init__()

        # self.network는 실제 계산 layer들을 순서대로 묶은 구조입니다.
        # nn.Sequential을 사용하면 forward에서 layer를 하나씩 직접 호출하지 않아도 됩니다.
        self.network = nn.Sequential(
        
            # 첫 번째 Linear layer입니다.
            # 입력 feature 6개를 hidden_dim 크기의 내부 표현으로 변환합니다.
            nn.Linear(input_dim, hidden_dim),
            
            # ReLU는 비선형 활성화 함수입니다.
            # Linear layer만 여러 개 쌓으면 결국 하나의 선형 계산과 비슷해지기 때문에,
            # 복잡한 비선형 패턴을 학습하기 어렵습니다.
            # ReLU를 넣어 복잡한 패턴을 학습할 수 있게 합니다.
            nn.ReLU(),

            # Dropout은 학습 중 일부 뉴런 출력을 0으로 만들어
            # 특정 뉴런에 지나치게 의존하는 것을 줄입니다.
            #
            # train 모드에서는 Dropout이 작동하고,
            # eval 모드에서는 Dropout이 꺼집니다.
            nn.Dropout(dropout_rate),

            # 두 번째 Linear layer입니다.
            # hidden_dim 크기의 표현을 다시 더 작은 hidden_dim // 2 크기로 압축합니다.
            nn.Linear(hidden_dim, hidden_dim // 2),

            # 두 번째 비선형 변환입니다.
            nn.ReLU(),

            # 마지막 Linear layer입니다.
            # hidden_dim // 2 크기의 표현을 최종 출력 1개로 변환합니다.
            # 여기서 출력되는 이 1개 값은 아직 확률이 아니라 logit입니다.
            #
            # Linear layer는 단순히 y = xW + b 형태의 계산을 하기 때문에,
            # 출력값이 음수일 수도 있고 1보다 클 수도 있습니다.
            #
            # 이렇게 Sigmoid를 통과하기 전의 원시 점수를 logit이라고 합니다.
            # 즉, logit은 "확률로 변환되기 전의 모델 점수"입니다.
            #
            # 이 프로젝트에서는 학습 시 BCEWithLogitsLoss를 사용할 예정입니다.
            # BCEWithLogitsLoss는 내부에서 Sigmoid 계산까지 함께 처리하므로,
            # 모델 마지막에 nn.Sigmoid()를 넣지 않는 것이 일반적입니다.
            #
            # 나중에 추론 단계에서 사람이 해석할 확률이 필요할 때만
            # torch.sigmoid(logits)를 적용해 0~1 사이 probability로 변환합니다.
            nn.Linear(hidden_dim // 2, 1),

            # Sigmoid는 logit을 0과 1 사이의 값으로 변환합니다.
            # 이진 분류에서는 Sigmoid 출력값을 보통 class 1에 속할 확률처럼 해석합니다.
            #
            # 이 프로젝트에서 target 컬럼은 "Machine failure"입니다.
            # AI4I 데이터의 Machine failure label은 다음 의미로 사용합니다.
            #
            # 0 = 고장 없음, 정상
            # 1 = 고장 발생
            #
            # 따라서 Sigmoid 출력값이 0에 가까우면
            # Machine failure = 0, 즉 정상일 가능성이 높다고 해석하고,
            #
            # Sigmoid 출력값이 1에 가까우면
            # Machine failure = 1, 즉 고장일 가능성이 높다고 해석합니다.
            #
            # 단, 0과 1의 의미는 데이터셋마다 다를 수 있으므로,
            # 실무에서는 항상 target label 정의를 먼저 확인해야 합니다.
            # nn.Sigmoid(),
       )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        모델의 순전파 계산을 정의합니다.

        Parameters
        ----------
        x:
            모델 입력 tensor입니다.
            shape은 보통 [batch_size, imput_dim]입니다.

            예:
            batch_size가 4이고 imput_dim이 6이면
            x.shape은 [4, 6]입니다.

        Returns
        -------
        torch.Tensor:
            각 샘플에 대한 logit을 반환합니다.
            shape은 [batch_size, input_dim]입니다.

            주의:
            이 값은 아직 확률이 아닙니다.
            학습 시에는 BCEWithLogitsLoss에 그대로 넣고,
            추론 시에는 torch.sigmoid(logits)를 적용해 확률로 변환합니다.
        """

        # 입력 x를 self.network에 통과시킵니다.
        # nn.Sequential 안에 정의된 layer들이 순서대로 실행됩니다.
        return self.network(x)