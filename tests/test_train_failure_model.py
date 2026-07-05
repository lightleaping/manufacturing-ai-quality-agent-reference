import pandas as pd
import torch

from src.training.train_failure_model import (
    calculate_pos_weight,
    create_dataloader,
    dataframe_to_tensor,
    target_to_tensor,
    train_failure_model,
)
from test_evaluate_failure_model import make_sample_X, make_sample_y

def make_training_sample():
    """
    학습 테스트에 사용할 작은 샘플 데이터를 만듭니다.

    실제 AI4I CSV를 사용하지 않고 테스트용 DataFrame을 직접 만드는 이유:
    1. 테스트가 외부 파일에 의존하지 않습니다.
    2. 빠르게 실행됩니다.
    3. 학습 함수의 입출력 구조만 독립적으로 검증할 수 있습니다.
    """

    X = pd.DataFrame(
        {
            "Air temperature [K]": [298.1, 298.2, 299.1, 300.2],
            "Process temperature [K]": [308.6, 308.7, 309.1, 310.2],
            "Rotational speed [rpm]": [1551, 1408, 1300, 1200],
            "Torque [Nm]": [42.8, 46.3, 55.1, 60.2],
            "Tool wear [min]": [0, 3, 180, 220],
            "Type": [0, 1, 1, 2],
        }
    )

    y = pd.Series([0, 0, 1, 1])

    return X, y

def test_dataframe_to_tensor_shape():
    """
    feature DataFrame이 모델 입력 Tensor로 변환되는지 확인합니다.
    """

    X, _ = make_training_sample()

    X_tensor = dataframe_to_tensor(X)

    # 샘플 4개, feature 6개이므로 shape은 [4, 6]입니다.
    assert X_tensor.shape == (4, 6)


def test_target_to_tensor_shape():
    """
    target Series가 BCEWithLogitsLoss에 맞는 shape으로 변환되는지 확인합니다.
    """

    _, y = make_training_sample()

    y_tensor = target_to_tensor(y)

    # target은 모델 출력 logits와 shape을 맞추기 위해 [4, 1]이어야 합니다.
    assert y_tensor.shape == (4, 1)

def test_create_dataloader_returns_batches():
    """
    create_dataloader가 mini-batch 단위로
    X_batch, y_batch를 반환하는지 확인합니다.
    """

    X, y = make_training_sample()

    dataloader = create_dataloader(
        X=X,
        y=y,
        batch_size=2,
        shuffle=False,
    )

    batches = list(dataloader)

    # 데이터 4개를 batch_size=2로 나누면 batch는 2개가 됩니다.
    assert len(batches) == 2

    X_batch, y_batch = batches[0]

    # 첫 번째 batch는 샘플 2개, feature 6개입니다.
    assert X_batch.shape == (2, 6)

    # target도 샘플 2개에 대해 [2, 1] 형태여야 합니다.
    assert y_batch.shape == (2, 1)

    # TensorDataset / DataLoader를 거치면 Tensor가 반환되어야 합니다.
    assert isinstance(X_batch, torch.Tensor)
    assert isinstance(y_batch, torch.Tensor)


def test_train_failure_model_returns_losses():
    """
    train_failure_model이 학습된 모델과 epoch별 loss 기록을 반환하는지 확인합니다.

    이 테스트는 모델 성능을 검증하는 것이 아닙니다.
    mini-batch 학습 루프가 에러 없이 실행되고,
    지정한 epoch 수만큼 평균 loss가 기록되는지 확인합니다.
    """

    X, y = make_training_sample()

    result = train_failure_model(
        X_train=X,
        y_train=y,
        input_dim=6,
        epochs=3,
        batch_size=2,
    )

    # epochs=3으로 학습했으므로 loss도 3개 기록되어야 합니다.
    assert len(result.losses) == 3

    # 모든 epoch 평균 loss는 0 이상이어야 합니다.
    assert all(loss >= 0 for loss in result.losses)

    # 학습된 model이 result 안에 들어 있어야 합니다.
    assert result.model is not None

def test_calculate_pos_weight() -> None:
    """
    calculate_pos_weight 함수가 class imbalance 비율을 올바르게 계산하는지 확인합니다.
    pos_weight가 negative_count / positive_count로 계산되는지 확인합니다.

    예:
    y = [0, 0, 0, 1]

    negative class 0, 즉 정상 데이터 개수 = 3
    positive class 1, 즉 고장 데이터 개수 = 1

    negative_count = 3
    positive_count = 1

    pos_weight = negative_count / positive_count
    pos_weight = 3 / 1 = 3.0
    """

    y = pd.Series([0, 0, 0, 1])

    pos_weight = calculate_pos_weight(y)

    # pos_weight는 Python float이 아니라 PyTorch Tensor여야 합니다.
    #
    # BCEWithLogitsLoss는 PyTorch loss 함수이므로,
    # 내부 계산에 사용할 pos_weight도 Tensor 형태로 전달하는 것이 맞습니다.
    #
    # torch.Size([1])의 의미:
    # - 값 1개를 가진 1차원 Tensor라는 뜻입니다.
    # - 이진 분류에서는 positive class가 하나분이므로,
    #   positive class 1에 대한 가중치도 1개만 있으면 됩니다.
    #
    # 예:
    # torch.tensor([3.0]).shape == torch.Size([1])
    #
    # 반대로 torch.tensor(3.0)은 스칼라 Tensor라서
    # shape이 torch.Size([])가 됩니다.
    #
    # 이 테스트는 pos_weight가
    # "값 1개짜리 Tensor 형태"로 만들어졌는지 확인합니다.
    assert pos_weight.shape == torch.Size([1])

    # .item()은 값 1개짜리 Tensor 안에 들어 있는 Python 숫자를 꺼냅니다.
    #
    # pos_weight 자체는 tensor([3.]) 형태이고,
    # pos_weight.item()은 그 안의 실제 값 3.0입니다.
    #
    # 이 테스트는 pos_weight의 실제 계산값이
    # negative_count / positive_count = 3 / 1 = 3.0인지 확인합니다.
    assert pos_weight.item() == 3.0

def test_train_failure_model_with_pos_weight_returns_losses() -> None:
    """
    pos_weight를 적용한 상태에서도 학습 함수가 정상 실행되고,
    epoch 수만큼 loss를 반환하는지 확인합니다.

    여기서 성능이 좋은지를 검증하는 것은 아닙니다.
    목적은 pos_weight 옵션이 학습 루프와 연결되어 정상 동작하는지 확인하는 것입니다.
    """

    X = make_sample_X()
    y = make_sample_y()

    result = train_failure_model(
        X_train=X,
        y_train=y,
        input_dim=6,
        hidden_dim=8,
        dropout_rate=0.1,
        learning_rate=0.001,
        epochs=2,
        batch_size=2,
        use_pos_weight=True,
    )

    assert len(result.losses) == 2
    assert all(loss >= 0 for loss in result.losses)

def test_calculate_pos_weight() -> None:
    """
    calculate_pos_weight 함수가 class imbalance 비율을 올바르게 계산하는지 확인합니다.

    테스트 데이터:
    y = [0, 0, 0, 1]

    의미:
    - 정상 class 0이 3개
    - 고장 class 1이 1개

    계산:
     계산:
    negative_count = 3
    positive_count = 1

    pos_weight = negative_count / positive_count
    pos_weight = 3 / 1
    pos_weight = 3.0

    이 테스트의 목적:
    1. pos_weight가 Tensor인지 확인합니다.
    2. shape이 torch.Size([1])인지 확인합니다.
    3. 실제 값이 3.0인지 확인합니다.
    """

    y = pd.Series([0, 0, 0, 1])

    pos_weight = calculate_pos_weight(y)

    # pos_weight는 값 1개를 가진 Tensor여야 합니다.
    #
    # torch.Size([1])의 의미:
    # - 1차원 Tensor
    # - 그 안에 값이 1개 있음
    #
    # 이진 분류에서는 positive class가 하나뿐입니다.
    # 현재 프로젝트에서는 Machine failure = 1, 즉 고장 class 하나입니다.
    #
    # 따라서 positive class에 줄 가중치도 1개만 있으면 됩니다.
    assert pos_weight.shape == torch.Size([1])

    # .item()은 Tensor 안에 들어 있는 값 하나를 Python 숫자로 꺼냅니다.
    #
    # pos_weight는 tensor([3.]) 형태이고,
    # pos_weight.item()은 3.0입니다.
    assert pos_weight.item() == 3.0

def test_train_failure_model_with_pos_weight_returns_losses() -> None:
    """
    use_pos_weight=True 상태에서도 학습 함수가 정상 실행되는지 확인합니다.

    여기서 확인하는 것은 모델 성능이 아닙니다.

    테스트 목적:
    - train_failure_model 함수가 use_pos_weight 인자를 받을 수 있는가?
    - calculate_pos_weight가 loss function에 연결되는가?
    - 학습 루프가 에러 없이 실행되는가?
    - epochs 수만큼 loss가 반환되는가?

    즉, 이 테스트는 성능 테스트가 아니라 구조 테스트입니다.
    """

    X = make_sample_X()
    y = make_sample_y()

    result = train_failure_model(
        X_train=X,
        y_train=y,
        input_dim=6,
        hidden_dim=8,
        dropout_rate=0.1,
        learning_rate=0.001,
        epochs=2,
        batch_size=2,
        use_pos_weight=True,
    )

    # epochs=2로 학습했으므로 loss도 2개가 있어야 합니다.
    assert len(result.losses) == 2

    # loss는 음수가 될 수 없습니다.
    #
    # BCE 계열 loss는 예측과 정답의 차이를 나타내는 값이고,
    # 수학적으로 0 이상입니다.
    assert all(loss >= 0 for loss in result.losses)