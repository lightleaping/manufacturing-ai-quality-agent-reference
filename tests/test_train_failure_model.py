import pandas as pd
import torch

from src.training.train_failure_model import (
    create_dataloader,
    dataframe_to_tensor,
    target_to_tensor,
    train_failure_model,
)

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