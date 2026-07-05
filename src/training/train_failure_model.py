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

def calculate_pos_weight(y: pd.Series) -> torch.Tensor:
    """
    불균형 이진 분류에서 positive class, 즉 고장 class에 줄 가중치를 계산합니다.

    현재 프로젝트의 target은 Machine failure입니다.

    Machine failure = 0
    - 정상
    - negative class

    Machine failure = 1
    - 고장
    - positive class
    - 모델이 특히 잡아내야 하는 대상

    현재 AI4I 데이터는 정상 데이터가 훨씬 많고, 고장 데이터가 매우 적습니다.
    예를 들어 실행 결과에서 test set은 다음과 같았습니다.

    정상 class 0: 약 96.6%
    고장 class 1: 약 3.4%

    이런 데이터에서는 모델이 대부분을 정상이라고 예측해도
    accuracy가 높게 나올 수 있습니다.

    실제로 baseline 모델에서는 다음과 같은 결과가 나왔습니다.

    accuracy  = 0.9660
    precision = 0.0000
    recall    = 0.0000
    f1        = 0.0000

    confusion matrix를 보면 실제 고장 68개를 모두 정상으로 예측했습니다.

    즉, 모델이 고장을 잘 맞힌 것이 아니라
    데이터 대부분이 정상이기 때문에 accuracy가 높게 보인 것입니다.

    그래서 BCEWithLogitsLoss의 pos_weight를 사용합니다.

    pos_weight의 의미:
    - positive class 1, 즉 고장 class의 loss를 더 크게 반영합니다.
    - 쉽게 말하면 고장 샘플을 틀렸을 때 더 크게 벌점을 줍니다.

    계산식:
    pos_weight = negative_count / positive_count

    예:
        정상 3개, 고장 1개라면
        pos_weight = 3 / 1 = 3.0
    
        정상 7729개, 고장 271개라면
        pos_weight = 7729 / 271 ≈ 28.5

    주의:
    - pos_weight를 적용하면 recall이 올라갈 수 있습니다.
    - 대신 모델이 고장을 더 적극적으로 예측하게 되어 false positive가 늘 수 있습니다.
    - 따라서 precision, recall, f1, confusion matrix를 함께 봐야 합니다.
    """

    # negative_count:
    # - negative class 0의 개수입니다.
    # - 현재 프로젝트에서는 정상 설비/정상 샘플의 개수입니다.
    negative_count = (y == 0).sum()

    # positive_count:
    # - positive class 1의 개수입니다.
    # - 현재 프로젝트에서는 고장 설비/고장 샘플의 개수입니다.
    # - 모델이 특히 잡아내야 하는 대상입니다.
    positive_count = (y == 1).sum()

    # positive class가 하나도 없으면 pos_weight를 계산할 수 없습니다.
    #
    # 예:
    # y = [0, 0, 0, 0]
    #
    # 이 경우 positive_count = 0이므로
    # negative_count / positive_count 계산에서 0으로 나누는 문제가 생깁니다.
    #
    # 또한 고장 샘플이 하나도 없는 데이터로는
    # 고장 class를 학습시키는 것도 불가능합니다.
    if positive_count == 0:
        raise ValueError(
            "positive class, 즉 Machine failure = 1인 고장 샘플이 없어 "
            "pos_weight를 계산할 수 없습니다."
        )
    
    # pos_weight는 정상 개수 / 고장 개수로 계산합니다.
    #
    # 정상 데이터가 많고 고장 데이터가 적을수록 pos_weight는 커집니다.
    # pos_weight가 커질수록 고장 class를 틀렸을 때 loss가 더 크게 반영됩니다.
    pos_weight_value = negative_count / positive_count

    # BCEWithLogitsLoss의 pos_weight에는 Tensor를 넣어야 합니다.
    #
    # torch.tensor([pos_weight_value])처럼 리스트로 감싸는 이유:
    # - 값 1개를 가진 1차원 Tensor를 만들기 위해서입니다.
    # - 이진 분류에서는 positive class가 하나뿐이므로 가중치도 1개입니다.
    #
    # 예:
    # torch.tensor([3.0]).shape
    # → torch.Size([1])
    #
    # dtype=torch.float32를 쓰는 이유:
    # - PyTorch 모델의 입력, weight, loss 계산은 보통 float32를 사용합니다.
    # - pos_weight도 loss 계산에 들어가므로 float32로 맞춥니다.
    return torch.tensor([pos_weight_value], dtype=torch.float32)

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
        use_pos_weight: bool = False,
) -> TrainingResult:
    """
    FailureMLP 모델을 mini-batch 방식으로 학습합니다.

    이 함수는 Day 1에서 만든 전처리 결과와
    Day 2에서 만든 FailureMLP 모델을 연결하는 학습 메인 함수입니다.

    전체 흐름:
    1. X_train, y_train을 DataLoader로 변환합니다.
    2. FailureMLP 모델을 생성합니다.
    3. loss function을 생성합니다.
    4. Adam optimizer를 생성합니다.
    5. epoch 수만큼 train_one_epoch를 반복합니다.
    6. 학습된 model과 epoch별 loss를 TrainingResult로 반환합니다.

    use_pos_weight:
    - False이면 기본 BCEWithLogitsLoss를 사용합니다.
    - True이면 class imbalance 대응을 위해
        y_train의 class 비율을 기준으로 pos_weight를 계산해 적용합니다.

    왜 use_pos_weight을 옵션으로 두는가?
    - baseline과 개선 모델을 비교하기 위해서입니다.
    - 처음부터 pos_weight를 무조건 적용하면,
        기본 baseline 모델이 어떤 문제가 있었는지 비교하기 어렵습니다.

    비교 구조:
    use_pos_weight=False
    -> 기본 baseline
    -> accuracy는 높지만 recall이 0일 수 있음

    use_pos_weight=True
    -> 고장 class에 더 큰 loss 부여
    -> recall 개선 가능
    -> 대신 precision 하락 또는 FP 증가 가능
        
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

    # DataLoader는 전체 데이터를 한 번에 모델에 넣지 않고
    # mini-batch 단위로 나누어 학습할 수 있게 해줍니다.
    #
    # X_train, y_train은 pandas DataFrame / Series이므로
    # create_dataloader 내부에서 PyTorch Tensor로 변환됩니다.
    # 학습 데이터는 보통 순서를 섞어서 학습하므로 shuffle=True를 사용합니다.
    dataloader = create_dataloader(
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    # FailureMLP 모델을 생성합니다.
    #
    # input_dim=6:
    # - AI4I feature 5개
    # - Type encoding 1개
    #
    #   1. Air temperature [K]
    #   2. Process temperature [K]
    #   3. Rotational speed [rpm]
    #   4. Torque [Nm]
    #   5. Tool wear [min]
    #   6. Type
    #
    # hidden_dim=32:
    # - hidden_dim은 hidden layer, 즉 은닉층의 뉴런 개수입니다.
    # - MLP는 입력층과 출력층 사이에 hidden layer를 두고,
    #   그 안에서 입력 feature를 더 풍부한 내부 표현으로 바꿉니다.
    #
    # - 현재 모델 구조에서는 hidden_dim=32이므로
    #   첫 번째 hidden layer는 32개의 뉴런을 가집니다.
    #
    # - 그다음 layer는 hidden_dim // 2를 사용하므로
    #   32 // 2 = 16개의 뉴런을 가집니다.
    #
    # - 즉, 전체 흐름은 대략 다음과 같습니다.
    #
    #   입력 feature 6개
    #   → hidden layer 32개
    #   → hidden layer 16개
    #   → 출력 logit 1개
    #
    # - hidden_dim=32는 "최적값"이라기보다 baseline 기본값입니다.
    # - baseline은 이후 개선 모델과 비교하기 위한 첫 번째 기준 모델입니다.
    #
    # - 현재 데이터는 이미지나 문장이 아니라 행과 열로 구성된 tabular data입니다.
    # - tabular data는 표 형태의 정형 데이터를 의미합니다.
    #
    # - feature가 6개인 작은 tabular 데이터에서 처음부터 hidden_dim을 너무 크게 잡으면
    #   모델이 데이터에 비해 과하게 복잡해질 수 있습니다.
    #
    # - 그래서 처음에는 hidden_dim=32 정도의 작은 MLP로 시작해
    #   학습 파이프라인과 평가 지표가 정상적으로 동작하는지 확인합니다.
    #
    # - 이후 성능을 보고 hidden_dim을 16, 32, 64, 128 등으로 조정할 수 있습니다.
    #
    # dropout_rate=0.2:
    # - Dropout은 학습 중 일부 뉴런의 출력을 무작위로 0으로 만드는 기법입니다.
    #
    # - dropout_rate=0.2는 학습 중 약 20%의 뉴런 출력을 무작위로 끈다는 뜻입니다.
    #
    # - Dropout을 사용하는 이유는 과적합을 줄이기 위해서입니다.
    #
    # - 과적합은 모델이 training data에는 잘 맞지만,
    #   처음 보는 test data에는 약한 상태를 말합니다.
    #
    # - Dropout은 모델이 특정 뉴런이나 특정 feature 조합에만 지나치게 의존하지 않도록
    #   일부 정보를 무작위로 가리면서 학습하게 만듭니다.
    #
    # - regularization은 과적합을 줄이기 위한 제어 기법을 의미합니다.
    # - Dropout은 대표적인 regularization 기법 중 하나입니다.
    #
    # - dropout_rate=0.2도 최적값이 아니라 baseline 기본값입니다.
    # - 이후 train/test 성능 차이, recall, precision, f1-score를 보면서
    #   0.0, 0.1, 0.2, 0.3, 0.5 등으로 조정할 수 있습니다.
    #
    # - 주의:
    #   model.train() 상태에서는 Dropout이 활성화됩니다.
    #   model.eval() 상태에서는 Dropout이 비활성화됩니다.
    #
    # - 따라서 학습할 때는 model.train(),
    #   평가/추론할 때는 model.eval()을 반드시 사용해야 합니다.
    model = FailureMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )

    # loss function을 설정합니다.
    #
    # 모델은 마지막에 Sigmoid를 적용하지 않고 logit을 출력합니다.
    # 따라서 loss는 BCEWithLogitsLoss를 사용합니다.
    #
    # BCEWithLogitsLoss는 내부에서 sigmoid와 binary cross entropy를
    # 함께 계산하므로 수치적으로 더 안정적입니다.
    if use_pos_weight:

        # y_train의 class 비율을 기준으로 pos_weight를 계산합니다.
        #
        # 예:
        # 정상 7729개, 고장 271개라면
        # pos_weight ≈ 28.5
        #
        # 이 값은 positive class 1, 즉 고장 class를 틀렸을 때
        # loss를 더 크게 반영하는 데 사용됩니다.
        pos_weight = calculate_pos_weight(y_train)

        # BCEWithLogitsLoss는 모델 출력 logit과 정답 y를 비교합니다.
        # pos_weight를 넣으면 positive class 1, 즉 고장 class의 loss를 더 크게 반영합니다.

        # pos_weight를 적용한 loss입니다.
        #
        # 이 설정의 목적:
        # - accuracy를 더 높이는 것이 아닙니다.
        # - 고장 class를 놓치는 false negative를 줄이는 것입니다.
        # - 즉, recall 개선이 목적입니다.
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        # 기본 baseline loss입니다.
        # class imbalance를 별도로 보정하지 않습니다.
        # 따라서 정상 class가 압도적으로 많으면
        # 모델이 대부분을 정상으로 예측하는 방향으로 학습될 수 있습니다.

        # 이진 분류용 loss function입니다.
        #
        # 모델은 logit을 출력하고,
        # BCEWithLogitsLoss는 내부에서 sigmoid와 BCE 계산을 함께 수행합니다.
        criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer를 생성합니다.
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

    # epoch별 평균 loss를 저장할 리스트입니다.
    #
    # 예:
    # losses = [2.215, 0.654, 0.301, ...]
    #
    # 이후 학습이 진행되면서 loss가 전체적으로 감소하는지 확인합니다.
    losses: list[float] = []

    # epochs 수만큼 전체 train set을 반복 학습합니다.
    #
    # epoch 1번:
    # - train set 전체를 mini-batch 단위로 한 번 학습
    #
    # epoch 10번:
    # - train set 전체를 총 10번 반복 학습
    for _ in range(epochs):
        # 한 epoch 동안 DataLoader의 모든 batch를 사용해 학습하고,
        # 그 epoch의 평균 loss를 반환받습니다.
        epoch_loss = train_one_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
        )

        # epoch별 평균 loss를 기록합니다.
        losses.append(epoch_loss)

    # 학습된 model과 loss 기록을 dataclass로 묶어 반환합니다.
    #
    # 튜플로 반환하면 result[0], result[1]처럼 접근해야 해서 헷갈릴 수 있습니다.
    # dataclass를 쓰면 result.model, result.losses처럼 이름으로 접근할 수 있습니다.
    return TrainingResult(
        model=model,
        losses=losses,
    )

# 현재 프로젝트는 이진 분류입니다.
#
# target:
# Machine failure
#
# label:
# 0 = 정상
# 1 = 고장
#
# 따라서 현재 positive class는 하나입니다.
# positive class = Machine failure 1 = 고장
#
# 그래서 pos_weight도 값 1개짜리 Tensor로 만듭니다.
#
# 예:
# pos_weight = torch.tensor([28.5], dtype=torch.float32)
#
# 만약 나중에 TWF, HDF, PWF, OSF, RNF처럼
# 여러 고장 유형을 각각 0/1로 예측하는 multi-label classification으로 확장한다면,
# positive class가 여러 개가 됩니다.
#
# 예:
# [TWF, HDF, PWF, OSF, RNF] = [1, 0, 0, 1, 0]
#
# 이 경우 모델 출력은 [batch_size, 5]가 되고,
# pos_weight도 label 개수에 맞게 5개를 넣을 수 있습니다.
#
# 예:
# pos_weight = torch.tensor([10.0, 20.0, 15.0, 8.0, 30.0])
#
# 반면 정상/TWF/HDF/PWF/OSF/RNF 중 하나만 고르는 multi-class classification이라면
# BCEWithLogitsLoss(pos_weight=...)가 아니라
# CrossEntropyLoss(weight=...)를 사용하는 것이 일반적입니다.
def calculate_pos_weight(y: pd.Series) -> torch.Tensor:
    """
    불균형 이진 분류에서 positive class, 즉 고장 class에 줄 가중치를 계산합니다.

    현재 문제:
    - 정상 class 0은 매우 많습니다.
    - 고장 class 1은 매우 적습니다.
    - 이 상태에서 그냥 학습하면 모델이 대부분 정상이라고 예측해도 loss가 낮아질 수 있습니다.

    pos_weight:
    - BCEWithLogitsLoss에서 positive class 1의 loss를 더 크게 반영하기 위한 값입니다.
    - 제조 고장 예측에서는 고장 class가 적기 때문에,
        고장 class를 틀렸을 때 더 크게 벌점을 주기 위해 사용합니다.

    계산식:
    pos_weight = negative_count / positive_count

    예:
    정상 7729개, 고장 271개라면
    pos_weight = 7729 / 271 ≈ 28.5

    주의:
    - pos_weight가 크면 recall은 올라갈 수 있습니다.
    - 대신 정상도 고장으로 잘못 예측하는 false positive가 늘어 precision이 낮아질 수 있습니다.
    """

    # negative_count:
    # - negative class 0의 개수입니다.
    # - 현재 프로젝트에서는 "정상 설비 / 정상 샘플"의 개수입니다.
    negative_count = (y == 0).sum()

    # positive_count:
    # - positive class 1의 개수입니다.
    # - 현재 프로젝트에서는 "고장 설비 / 고장 샘플"의 개수입니다.
    # - 제조 고장 예측에서 모델이 특히 잡아내야 하는 대상입니다.
    positive_count = (y == 1).sum()

    if positive_count == 0:
        raise ValueError(
            "positive class, 즉 고장 class 1이 하나도 없어 pos_weight를 계산할 수 없습니다."
        )
    
    # pos_weight는 positive class 1에 대한 가중치입니다.
    #
    # 현재 프로젝트에서 positive class는 "고장"입니다.
    # 즉, Machine failure = 1인 샘플입니다.
    #
    # 데이터에서 고장 샘플은 정상 샘플보다 훨씬 적기 때문에,
    # 그냥 학습하면 모델이 대부분을 정상으로 예측해도 loss가 낮게 나올 수 있습니다.
    #
    # 그래서 BCEWithLogitsLoss(pos_weight=...)를 사용해
    # 고장 class 1을 틀렸을 때의 손실을 더 크게 반영합니다.
    #
    # 쉽게 말하면:
    # "고장 데이터가 적으니까, 고장을 틀리면 더 크게 혼내자."
    pos_weight_value = negative_count / positive_count

    return torch.tensor([pos_weight_value], dtype=torch.float32)