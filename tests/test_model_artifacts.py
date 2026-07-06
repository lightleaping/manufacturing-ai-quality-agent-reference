import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.inference.model_artifacts import (
    load_failure_model_artifacts,
    save_failure_model_artifacts,
)
from src.models.failure_mlp import FailureMLP


def test_save_and_load_failure_model_artifacts(tmp_path):
    """
    학습된 model, scaler, metadata를 저장하고 다시 불러올 수 있는지 검증합니다.

    tmp_path:
    - pytest가 제공하는 임시 폴더입니다.
    - 실제 models/ 폴더를 더럽히지 않고 테스트할 수 있습니다.
    """

    feature_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type",
    ]

    # 테스트용 모델을 생성합니다.
    # 실제 학습된 모델은 아니지만,
    # 저장 → 로드 후 같은 구조와 같은 weight를 유지하는지 검증할 수 있습니다.
    model = FailureMLP(
        input_dim=6,
        hidden_dim=32,
        dropout_rate=0.2,
    )

    # 테스트용 scaler를 생성합니다.
    # StandardScaler는 fit이 되어 있어야 추론 시 transform을 사용할 수 있습니다.
    X = pd.DataFrame(
        [
            [300.0, 310.0, 1500.0, 40.0, 10.0, 0.0],
            [305.0, 315.0, 1600.0, 45.0, 20.0, 1.0],
        ],
        columns=feature_columns,
    )

    scaler = StandardScaler()
    scaler.fit(X)

    # 저장 전 모델 출력값을 기록합니다.
    # 저장 후 다시 로드한 모델이 같은 입력에 대해 같은 출력을 내야 합니다.
    sample_tensor = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4, 0.5, 1.0]],
        dtype=torch.float32,
    )

    model.eval()
    with torch.no_grad():
        original_logits = model(sample_tensor)

    # model, scaler, threshold, metadata를 임시 폴더에 저장합니다.
    paths = save_failure_model_artifacts(
        model=model,
        scaler=scaler,
        threshold=0.6,
        feature_columns=feature_columns,
        artifact_dir=tmp_path,
        input_dim=6,
        hidden_dim=32,
        dropout_rate=0.2,
    )

    # 실제 파일이 생성되었는지 확인합니다.
    assert paths.model_path.exists()
    assert paths.scaler_path.exists()
    assert paths.metadata_path.exists()

    # 저장된 artifact를 다시 로드합니다.
    artifacts = load_failure_model_artifacts(tmp_path)

    # metadata 값이 제대로 복원되었는지 확인합니다.
    assert artifacts.threshold == 0.6
    assert artifacts.input_dim == 6
    assert artifacts.hidden_dim == 32
    assert artifacts.dropout_rate == 0.2
    assert artifacts.feature_columns == feature_columns

    # 로드된 모델은 추론 모드여야 합니다.
    # load_failure_model_artifacts 내부에서 model.eval()을 호출했는지 검증합니다.
    assert artifacts.model.training is False

    # 로드된 모델이 저장 전 모델과 같은 출력을 내는지 확인합니다.
    with torch.no_grad():
        loaded_logits = artifacts.model(sample_tensor)

    assert torch.allclose(original_logits, loaded_logits)