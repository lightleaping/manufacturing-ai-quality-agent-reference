"""
train_failure_model.py

AI4I 전처리 데이터와 FailureMLP 모델을 연결해
설비 고장 예측 모델을 학습하는 함수들을 정의하는 파일입니다.

이 파일의 목표는 다음과 같습니다.

1. pandas DataFrame / Series를 PyTorch Tensor로 변환합니다.
2. FailureMLP 모델을 BCEWithLogitsLoss로 학습합니다.
3. 학습 과정의 loss 기록을 반환합니다.
4. 이후 평가, 추론, API, Agent 연결의 기반을 만듭니다.

또한 이 파일에서는 PyTorch의 TensorDataset과 DataLoader를 사용해
mini-batch 단위 학습을 수행합니다.

전체 데이터를 한 번에 모델에 넣는 방식도 가능하지만,
실제 딥러닝 학습에서는 데이터를 작은 batch로 나누어
여러 번 weight update를 수행하는 방식이 일반적입니다.
"""
from dataclasses import dataclass

import pandas as pd
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from src.models.failure_mlp import FailureMLP

@dataclass
class TrainingResult:
    """
    학습 결과를 담는 데이터 클래스입니다.

    모델 학습 후에는 단순히 model만 필요한 것이 아니라,
    학습 과정에서 loss가 어떻게 변했는지도 확인해야 합니다.

    Attributes
    ----------
    model:
        학습이 끝난 PyTorch 모델입니다.

    losses:
        epoch마다 계산된 평균 loss 값입니다.
        mini-batch 학습에서는 batch마다 loss가 계산되므로,
        한 epoch의 여러 batch loss를 평균내어 기록합니다.
        예: [0.6921, 0.6814, 0.6702]
    """

    model: FailureMLP
    losses: list[float]

def dataframe_to_tensor(X: pd.DataFrame) -> torch.Tensor:
    """
    pandas DataFrame 형태의 feature 데이터를
    PyTorch 모델 입력용 Tensor로 변환합니다.

    Parameters
    ----------
    X:
        feature 컬럼들로 구성된 pandas DataFrame입니다.
        예: X_train, X_test

    Returns
    -------
    torch.Tensor:
        PyTorch 모델에 입력할 수 있는 float32 Tensor입니다.
        shape은 [sample_count, feature_count]입니다.
    """

    # pandas DataFrame은 PyTorch 모델에 직접 넣을 수 없습니다.
    # 그래서 먼저 X.values를 사용해 numpy 배열 형태로 꺼냅니다.
    #
    # 이후 torch.tensor(..., dtype=torch.float32)를 사용해
    # PyTorch Tensor로 변환합니다.
    #
    # dtype=torch.float32를 사용하는 이유:
    # nn.Linear의 weight는 보통 float32이고,
    # 입력값도 float32여야 안정적으로 행렬 계산이 가능합니다.
    return torch.tensor(X.values, dtype=torch.float32)

def target_to_tensor(y: pd.Series) -> torch.Tensor:
    """
    pandas Series 형태의 target label을
    BCEWithLogitsLoss에 사용할 수 있는 Tensor로 변환합니다.

    Parameters
    ----------
    y:
        Machine failure 정답 label입니다.
        값은 0 또는 1입니다.

    Returns
    -------
    torch.Tensor:
        shape이 [sample_count, 1]인 float32 Tensor입니다.
    """

    # y.values는 보통 shape이 [sample_count]인 1차원 배열입니다.
    #
    # 예:
    # [0, 1, 0, 0, 1]
    #
    # 하지만 FailureMLP의 출력 logits shape은 [sample_count, 1]입니다.
    #
    # loss 계산 시 logits와 y의 shape을 맞추기 위해
    # view(-1, 1)을 사용해 [sample_count, 1] 형태로 바꿉니다.
    #
    # BCEWithLogitsLoss는 정답 label도 float Tensor를 기대하므로
    # dtype=torch.float32를 사용합니다.
    return torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

def create_dataloader(
        X: pd.DataFrame,
        y: pd.Series,
        batch_size: int = 32,
        shuffle: bool = True
) -> DataLoader:
    """
    pandas DataFrame / Series를 PyTorch DataLoader로 변환합니다.

    DataLoader는 전체 데이터를 mini-batch 단위로 나누어
    학습 루프에서 하나씩 꺼내 쓸 수 있게 해주는 객체입니다.

    Parameters
    ----------
    X:
        feature DataFrame입니다.
    y:
        target label Series입니다.

    batch_size:
        한 번에 넣을 샘플 개수입니다.
        예를 들어 batch_size=32이면 32개 샘플씩 나눠 학습합니다.

    shuffle:
        학습 전에 데이터 순서를 섞을지 여부입니다.
        학습 데이터는 보통 shuffle=True로 둡니다.

    Returns
    -------
    DataLoader:
        mini-batch 단위로 (X_batch, y_batch)를 반환하는 객체입니다.
    """

    # feature DataFrame을 Tensor로 변환합니다.
    X_tensor = dataframe_to_tensor(X)

    # target Series를 Tensor로 변환합니다.
    y_tensor = target_to_tensor(y)

    # TensorDataset은 X_tensor와 y_tensor를 하나의 dataset으로 묶습니다.
    #
    # 예:
    # X_tensor[0]과 y_tensor[0]이 하나의 학습 샘플이 됩니다.
    #
    # 즉, feature와 정답 label이 같은 index 기준으로 묶입니다.
    dataset = TensorDataset(X_tensor, y_tensor)

    # DataLoader는 TensorDataset을 batch_size 단위로 잘라서 반환합니다.
    #
    # shuffle=True이면 매 epoch마다 데이터 순서를 섞습니다.
    # 이렇게 하면 모델이 데이터 순서에 과하게 의존하는 것을 줄일 수 있습니다.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return dataloader

def train_one_epoch(
        model: FailureMLP,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
) -> float:
    """
    모델을 한 epoch 학습합니다.

    한 epoch은 학습 데이터 전체를 한 번 사용하는 단위입니다.

    mini-batch 학습에서는 한 epoch 안에서
    DataLoader가 여러 batch를 반환하고,
    각 batch마다 forward, loss 계산, backward, optimizer update를 수행합니다.

    Returns
    -------
    float:
        한 epoch 동안 계산된 batch loss들의 평균값입니다.
    """

    # 모델을 학습 모드로 전환합니다.
    #
    # Dropout, BatchNorm 같은 layer는 train / eval 모드에서 동작이 달라집니다.
    # 현재 FailureMLP에는 Dropout이 있으므로 학습 시 model.train()이 필요합니다.
    model.train()

    # 한 epoch 동안 batch별 loss를 저장할 리스트입니다.
    batch_losses: list[float] = []

    # DataLoader에서 mini-batch를 하나씩 꺼냅니다.
    #
    # X_batch shape:
    #   [batch_size, input_dim]
    #
    # y_batch shape:
    #   [batch_size, 1]
    for X_batch, y_batch in dataloader:

        # 이전 batch에서 계산된 gradient를 초기화합니다.
        #
        # PyTorch는 기본적으로 gradient를 누적합니다.
        # 따라서 매번 backward를 하기 전에 zero_grad()를 호출해야
        # 이전 gradient와 현재 gradient가 섞이지 않습니다.
        # 따라서 매번 loss.backward() 전에 gradient를 0으로 초기화합니다.
        optimizer.zero_grad()

        # 현재 batch를 모델에 넣어 logits를 계산합니다.
        #
        # 출력 logits는 아직 확률이 아닙니다.
        # shape은 [batch_size, 1]입니다.
        # 이후 BCEWithLogitsLoss는 logits를 직접 입력으로 받습니다.
        logits = model(X_batch)

        # 모델 출력 logits와 실제 정답 y_batch를 비교해 loss를 계산합니다.
        #
        # criterion은 BCEWithLogitsLoss입니다.
        # 내부적으로 sigmoid + binary cross entropy 계산을 수행합니다.
        loss = criterion(logits, y_batch)

        # loss를 기준으로 각 parameter가 loss에 얼마나 영향을 주었는지
        # gredient를 계산합니다.
        #
        # 이 단계에서는 아직 weight가 바뀌지 않습니다.
        # weight를 어떻게 바꿔야 하는지에 대한 기울기만 계산됩니다.
        # 각 parameter의 .grad에 gradient가 저장됩니다.
        loss.backward()

        # optimizer가 계산된 gredient를 사용해 model parameter를 업데이트합니다.
        #
        # 이 단계에서 실제로 weight와 bias 값이 바뀝니다.
        optimizer.step()

        # 현재 batch의 loss를 기록합니다.
        # loss는 Tensor이므로, 기록용 숫자로 사용하기 위해 item()으로 Python float로 변환합니다.
        batch_losses.append(loss.item())

    # 한 epoch 동안 여러 batch loss가 나왔으므로 평균 loss를 계산합니다.
    #
    # 예:
    # batch_losses = [0.71, 0.68, 0.66]
    # epoch_loss = 0.683
    epoch_loss = sum(batch_losses) / len(batch_losses)

    return epoch_loss

def train_failure_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        input_dim: int = 6,
        hidden_dim: int = 32,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
) -> TrainingResult:
    """
    FailureMLP 모델을 mini-batch 방식으로 학습하는 메인 함수입니다.

    Parameters
    ----------
    X_train:
        전처리된 학습 feature DataFrame입니다.
    
    y_train:
        학습 target label입니다.
        0 = 정상
        1 = 고장
    
    input_dim:
        모델 입력 feature 개수입니다.
        AI4I baseline에서는 6입니다.
    
    hidden_dim:
        hidden layer 크기입니다.
        즉, 은닉층 뉴런 개수입니다.
        baseline에서는 32로 시작합니다.
    
    dropout_rate:
        Dropout 비율입니다.
        baseline에서는 0.2로 시작합니다.

    learning_rate:
        optimizer가 parameter를 한 번에 얼마나 크게 수정할지 정하는 값입니다.
        Adam baseline에서는 0.001로 시작합니다.
    
    epochs:
        전체 학습 데이터를 몇 번 반복해서 학습할지 정하는 값입니다.

    batch_size:
        한 번에 모델에 넣을 샘플 개수입니다.
        데이터가 커질수록 전체 데이터를 한 번에 넣기보다
        batch_size 단위로 나누어 학습하는 것이 일반적입니다.

    Returns
    -------
    TrainingResult:
        학습된 모델과 epoch별 평균 loss 기록입니다.
    """

    # FailureMLP 모델을 생성합니다.
    model = FailureMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )

    # 이진 분류용 loss function입니다.
    #
    # 모델은 logit을 출력하고,
    # BCEWithLogitsLoss는 내부에서 sigmoid와 BCE 계산을 함께 수행합니다.
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer를 사용합니다.
    #
    # optimizer는 loss.backward()로 계산된 gradient를 이용해
    # 모델의 weight와 bias를 업데이트하는 역할을 합니다.
    
    # 다시 말해, optimizer는 모델의 weight와 bias를 실제로 업데이트하는 객체입니다.
    #
    # 모델 학습 과정은 다음 순서로 진행됩니다.
    #
    # 1. model(X_tensor)
    #   -> 현재 weight와 bias를 사용해 예측값(logits)을 계산합니다.
    #
    # 2. criterion(logits, y_tensor)
    #   -> 예측값과 실제 정답을 비교해 loss를 계산합니다.
    #
    # 3. loss.backward()
    #   -> loss를 줄이기 위해 각 weight롸 bias를 어느 방향으로 바꿔야 하는지
    #       gradient를 계산합니다.
    #
    # 4. optimizer.step()
    #   -> loss.backward()로 계산된 gradient를 이용해
    #       실제 weight와 bias 값을 업데이트합니다.
    #
    # 여기서 Adam은 PyTorch에서 제공하는 대표적인 optimizer 중 하나입니다.
    #
    # Adam은 기본적인 SGD보다 각 parameter의 업데이트 크기를
    # 조금 더 적응적으로 조절해주는 방식이라,
    # 딥러닝 baseline 학습에서 자주 사용됩니다.
    #
    # model.parameters()는 모델 내부의 학습 가능한 parameter들을 반환합니다.
    #
    # 여기서 parameter란 주로 nn.Linear layer 안의 weight와 bias입니다.
    #
    # 예를 들어 FailureMLP에는 다음 Linear layer들이 있습니다.
    #
    # - nn.Linear(input_dim, hidden_dim)
    # - nn.Linear(hidden_dim, hidden_dim // 2)
    # - nn.Linear(hidden_dim // 2, 1)
    #
    # 각 Linear layer는 weight와 bias를 가지고 있고,
    # model.parameters()는 이 값들을 optimizer에게 넘겨줍니다.
    #
    # optimizer는 이 parameter들을 loss가 줄어드는 방향으로 업데이트합니다.
    #
    # ReLU나 Dropout은 학습되는 weight가 없으므로
    # optimizer가 직접 업데이트할 parameter가 없습니다.
    #
    # lr은 learning rate의 줄임말입니다.
    #
    # learning_rate가 너무 크면 loss가 튀거나 학습이 불안정할 수 있고,
    # 너무 작으면 학습이 지나치게 느릴 수 있습니다.
    #
    # Adam에서는 0.001을 baseline learning rate로 자주 사용하므로,
    # 여기서는 learning_rate 기본값을 0.001로 두고 시작합니다.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    # 학습 데이터를 mini-batch 단위로 꺼내기 위한 DataLoader를 생성합니다.
    #
    # 학습 데이터는 보통 순서를 섞어서 학습하므로 shuffle=True를 사용합니다.
    train_dataloader = create_dataloader(
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    # epoch별 loss를 저장할 빈 리스트를 함수 안에서 생성합니다.
    #
    # Python에서는 함수 안에서 새로운 변수를 만들 수 있습니다.
    # 이렇게 함수 안에서 만든 변수는 local variable, 즉 지역 변수입니다.
    #
    # losses는 train_failure_model 함수가 실행되는 동안에만 사용됩니다.
    # 함수가 끝나면 그냥 사라지지만,
    # 아래 return에서 TrainingResult(losses=losses)로 넘겨주기 때문에
    # 학습 결과 객체 안에 loss 기록이 담겨 함수 밖으로 전달됩니다.
    #
    # list[float]는 타입 힌트입니다.
    # "이 리스트에는 float 값들이 들어갈 예정이다"라는 의미입니다.
    #
    # 실제 빈 리스트를 만드는 부분은 오른쪽의 []입니다.
    #
    # 예:
    # 처음에는 losses = []
    # 1 epoch 후 losses = [0.692]
    # 2 epoch 후 losses = [0.692, 0.681]
    # 3 epoch 후 losses = [0.692, 0.681, 0.674]
    losses: list[float] = []

    # 지정한 epoch 수만큼 학습을 반복합니다.
    for _ in range(epochs):
        # 한 epoch 동안 DataLoader의 모든 batch를 사용해 학습하고,
        # 그 epoch의 평균 loss를 반환받습니다.
        epoch_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
        )

        # epoch별 평균 loss를 기록합니다.
        losses.append(epoch_loss)

    # 학습용 모델과 loss 기록을 반환합니다.
    return TrainingResult(
        model=model,
        losses=losses,
    )