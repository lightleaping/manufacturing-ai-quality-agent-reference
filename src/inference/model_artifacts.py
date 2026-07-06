from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import joblib
import torch
from sklearn.preprocessing import StandardScaler

from src.models.failure_mlp import FailureMLP


@dataclass(frozen=True)
class FailureModelArtifactPaths:
    """
    저장될 파일들의 경로를 하나로 묶어두는 dataclass입니다.

    model_path:
    - PyTorch 모델의 가중치가 저장되는 경로입니다.
    - 예: models/failure_mlp/model.pt

    scaler_path:
    - StandardScaler 객체가 저장되는 경로입니다.
    - 예: models/failure_mlp/scaler.joblib

    metadata_path:
    - threshold, input_dim, feature_columns 같은 설정 정보가 저장되는 경로입니다.
    - 예: models/failure_mlp/metadata.json
    """

    model_path: Path
    scaler_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class FailureModelArtifacts:
    """
    추론에 필요한 모든 구성요소를 하나로 묶은 dataclass입니다.

    실제 inference에서는 model만 있으면 부족합니다.

    이유:
    1. 학습 때 사용한 scaler를 그대로 써야 합니다.
    2. 학습 후 선택한 threshold를 그대로 써야 합니다.
    3. feature 컬럼 순서도 학습 때와 같아야 합니다.

    따라서 model, scaler, threshold, feature_columns를 함께 관리합니다.
    """

    model: FailureMLP
    scaler: StandardScaler
    threshold: float
    input_dim: int
    hidden_dim: int
    dropout_rate: float
    feature_columns: list[str]


def create_failure_artifact_paths(
    artifact_dir: str | Path = "models/failure_mlp",
) -> FailureModelArtifactPaths:
    """
    모델 관련 파일들이 저장될 경로를 생성합니다.

    artifact_dir:
    - 모델 저장 폴더입니다.
    - 기본값은 models/failure_mlp 입니다.

    이 함수는 실제 파일을 저장하지는 않고,
    저장할 위치만 정리해서 반환합니다.
    """

    artifact_dir = Path(artifact_dir)

    return FailureModelArtifactPaths(
        model_path=artifact_dir / "model.pt",
        scaler_path=artifact_dir / "scaler.joblib",
        metadata_path=artifact_dir / "metadata.json",
    )


def save_failure_model_artifacts(
    model: FailureMLP,
    scaler: StandardScaler,
    threshold: float,
    feature_columns: list[str],
    artifact_dir: str | Path = "models/failure_mlp",
    input_dim: int = 6,
    hidden_dim: int = 32,
    dropout_rate: float = 0.2,
) -> FailureModelArtifactPaths:
    """
    학습된 모델, scaler, threshold 정보를 저장합니다.

    저장하는 파일은 3개입니다.

    1. model.pt
       - PyTorch 모델의 state_dict 저장
       - state_dict는 모델 구조 전체가 아니라 학습된 weight/bias 값입니다.

    2. scaler.joblib
       - 학습 데이터에 fit된 StandardScaler 저장
       - 추론할 때도 같은 평균/표준편차로 scaling해야 하므로 반드시 저장합니다.

    3. metadata.json
       - threshold
       - input_dim
       - hidden_dim
       - dropout_rate
       - feature_columns
       같은 설정값을 저장합니다.

    왜 model만 저장하면 안 되는가?
    - 모델은 숫자 tensor만 입력받습니다.
    - 하지만 raw input을 모델 입력 tensor로 만들려면 scaler와 feature column 순서가 필요합니다.
    - 또한 probability를 prediction으로 바꾸려면 threshold가 필요합니다.
    """

    paths = create_failure_artifact_paths(artifact_dir)

    # 저장 폴더가 없으면 생성합니다.
    # parents=True: 상위 폴더까지 함께 생성
    # exist_ok=True: 이미 폴더가 있어도 에러를 내지 않음
    paths.model_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. PyTorch model 저장
    # ------------------------------------------------------------
    # model 전체를 저장하는 방식도 있지만,
    # 일반적으로는 state_dict를 저장하는 방식이 더 권장됩니다.
    #
    # state_dict:
    # - 모델의 학습된 parameter 값만 담은 dictionary입니다.
    # - 나중에 같은 모델 구조를 다시 만든 뒤 load_state_dict로 불러옵니다.
    torch.save(model.state_dict(), paths.model_path)

    # ------------------------------------------------------------
    # 2. scaler 저장
    # ------------------------------------------------------------
    # StandardScaler는 sklearn 객체입니다.
    # torch.save가 아니라 joblib.dump를 사용해 저장합니다.
    joblib.dump(scaler, paths.scaler_path)

    # ------------------------------------------------------------
    # 3. metadata 저장
    # ------------------------------------------------------------
    # JSON으로 저장하면 사람이 직접 열어서 확인하기 쉽습니다.
    metadata: dict[str, Any] = {
        "threshold": threshold,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate,
        "feature_columns": feature_columns,
    }

    with paths.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return paths


def load_failure_model_artifacts(
    artifact_dir: str | Path = "models/failure_mlp",
) -> FailureModelArtifacts:
    """
    저장된 model, scaler, metadata를 다시 불러옵니다.

    추론 서버나 Agent에서는 학습을 다시 하지 않습니다.

    일반적인 운영 흐름:
    1. 서버 시작
    2. 저장된 model/scaler/threshold 로드
    3. 사용자 입력이 들어올 때마다 inference 수행

    즉, 이 함수는 Day 5 이후 FastAPI나 Agent와 연결될 핵심 함수입니다.
    """

    paths = create_failure_artifact_paths(artifact_dir)

    # metadata를 먼저 읽습니다.
    # 이유:
    # - model 구조를 다시 만들려면 input_dim, hidden_dim, dropout_rate가 필요하기 때문입니다.
    with paths.metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    input_dim = int(metadata["input_dim"])
    hidden_dim = int(metadata["hidden_dim"])
    dropout_rate = float(metadata["dropout_rate"])
    threshold = float(metadata["threshold"])
    feature_columns = list(metadata["feature_columns"])

    # 저장 당시와 같은 구조의 모델을 다시 생성합니다.
    model = FailureMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
    )

    # 저장된 state_dict를 불러옵니다.
    # map_location="cpu":
    # - GPU에서 저장한 모델도 CPU 환경에서 안전하게 불러오기 위한 옵션입니다.
    state_dict = torch.load(paths.model_path, map_location="cpu")

    # 빈 모델 구조에 학습된 weight/bias 값을 채웁니다.
    model.load_state_dict(state_dict)

    # 추론 모드로 전환합니다.
    # Dropout 같은 학습 전용 동작을 끄기 위해 필요합니다.
    model.eval()

    # scaler도 다시 불러옵니다.
    scaler = joblib.load(paths.scaler_path)

    return FailureModelArtifacts(
        model=model,
        scaler=scaler,
        threshold=threshold,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout_rate=dropout_rate,
        feature_columns=feature_columns,
    )