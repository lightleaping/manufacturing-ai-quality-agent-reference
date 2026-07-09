"""
SHAP artifact 저장/로드 유틸리티입니다.

이 파일이 필요한 이유
--------------------
운영 환경에 가까운 구조에서는 API 요청이 들어올 때마다
SHAP background data를 새로 만들지 않습니다.

대신 모델 학습 또는 배포 준비 단계에서 아래 파일들을 미리 저장합니다.

- shap_background.pt
- shap_reference_values.json
- global_importance.json

그리고 API에서는 이 파일들을 로드해서 사용합니다.

이렇게 하면:
1. API 응답 시간이 줄어듭니다.
2. 매 요청마다 background sample이 달라지는 문제를 줄일 수 있습니다.
3. 모델 artifact와 explanation artifact를 함께 관리할 수 있습니다.
4. 운영 배포 구조에 더 가까워집니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


SHAP_BACKGROUND_FILENAME = "shap_background.pt"
SHAP_REFERENCE_VALUES_FILENAME = "shap_reference_values.json"
GLOBAL_IMPORTANCE_FILENAME = "global_importance.json"


@dataclass(frozen=True)
class ShapArtifacts:
    """
    SHAP explanation에 필요한 artifact 묶음입니다.

    background_tensor:
        SHAP DeepExplainer에 들어갈 기준 tensor입니다.

    reference_values:
        feature별 기준값입니다.
        보통 train set 평균값을 사용합니다.
        Agent evidence에서 현재 입력값과 비교할 때 사용합니다.

    global_importance_map:
        Day 6 permutation importance 결과입니다.
        전체 test set 기준 모델 민감도를 local explanation에 참고 정보로 붙일 때 사용합니다.
    """

    background_tensor: torch.Tensor
    reference_values: dict[str, float]
    global_importance_map: dict[str, float]


def save_json(path: Path, data: dict[str, Any]) -> None:
    """
    dict 데이터를 JSON 파일로 저장합니다.

    ensure_ascii=False:
        한글이 깨지지 않도록 저장합니다.

    indent=2:
        사람이 파일을 열어봤을 때 읽기 쉽게 저장합니다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_json(path: Path) -> dict[str, Any]:
    """
    JSON 파일을 dict로 로드합니다.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_shap_artifacts(
    artifact_dir: str | Path,
    background_tensor: torch.Tensor,
    reference_values: dict[str, float],
    global_importance_map: dict[str, float],
) -> None:
    """
    SHAP 관련 artifact를 저장합니다.

    저장 위치 예:
        models/failure_mlp/shap_background.pt
        models/failure_mlp/shap_reference_values.json
        models/failure_mlp/global_importance.json
    """
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        background_tensor.detach().cpu(),
        artifact_path / SHAP_BACKGROUND_FILENAME,
    )

    save_json(
        artifact_path / SHAP_REFERENCE_VALUES_FILENAME,
        reference_values,
    )

    save_json(
        artifact_path / GLOBAL_IMPORTANCE_FILENAME,
        global_importance_map,
    )


def load_shap_artifacts(
    artifact_dir: str | Path,
) -> ShapArtifacts:
    """
    저장된 SHAP artifact를 로드합니다.

    API에서는 이 함수를 사용해서
    미리 저장된 background tensor와 reference/global importance 정보를 가져옵니다.
    """
    artifact_path = Path(artifact_dir)

    background_path = artifact_path / SHAP_BACKGROUND_FILENAME
    reference_values_path = artifact_path / SHAP_REFERENCE_VALUES_FILENAME
    global_importance_path = artifact_path / GLOBAL_IMPORTANCE_FILENAME

    if not background_path.exists():
        raise FileNotFoundError(f"SHAP background artifact not found: {background_path}")

    if not reference_values_path.exists():
        raise FileNotFoundError(
            f"SHAP reference values artifact not found: {reference_values_path}"
        )

    if not global_importance_path.exists():
        raise FileNotFoundError(
            f"Global importance artifact not found: {global_importance_path}"
        )

    background_tensor = torch.load(
        background_path,
        map_location="cpu",
    ).to(dtype=torch.float32)

    raw_reference_values = load_json(reference_values_path)
    raw_global_importance_map = load_json(global_importance_path)

    reference_values = {
        str(feature): float(value)
        for feature, value in raw_reference_values.items()
    }

    global_importance_map = {
        str(feature): float(value)
        for feature, value in raw_global_importance_map.items()
    }

    return ShapArtifacts(
        background_tensor=background_tensor,
        reference_values=reference_values,
        global_importance_map=global_importance_map,
    )