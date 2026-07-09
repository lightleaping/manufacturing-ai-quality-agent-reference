# src/api/artifact_cache.py

"""
Day 12 - API artifact cache helper

이 파일의 목적
--------------
FastAPI endpoint가 요청을 받을 때마다
model.pt, scaler.joblib, metadata.json, shap_background.pt 등을
매번 다시 로드하지 않도록 cache하는 helper를 제공한다.

왜 필요한가?
------------
Day 11까지는 API 요청이 들어올 때마다 artifact를 로드하는 구조였다.

하지만 운영 환경에서는:
1. 모델 파일을 매 요청마다 읽으면 느리다.
2. SHAP background tensor도 매 요청마다 로드하면 비효율적이다.
3. endpoint 안에 로딩/예외 처리 코드가 많아지면 API 계층이 복잡해진다.

그래서 Day 12에서는 artifact loading을 endpoint 밖 helper로 분리한다.

중요한 구분
-----------
1. model artifact 실패
   - prediction 자체가 불가능하다.
   - 따라서 API는 명확한 error를 반환해야 한다.

2. SHAP artifact 실패
   - prediction은 가능하다.
   - 따라서 API 전체를 실패시키지 않고 warning으로 처리한다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


from src.inference.model_artifacts import load_failure_model_artifacts
from src.interpretability.shap_artifacts import load_shap_artifacts


# 이 프로젝트에서 학습된 FailureMLP artifact가 저장되는 기본 위치다.
# Day 5: model.pt, scaler.joblib, metadata.json
# Day 11: shap_background.pt, shap_reference_values.json, global_importance.json
ARTIFACT_DIR = Path("models/failure_mlp")


@lru_cache(maxsize=1)
def get_cached_failure_model_artifacts(
    artifact_dir: str = str(ARTIFACT_DIR),
) -> Any:
    """
    FailureMLP model artifact를 cache해서 로드한다.

    lru_cache란?
    --------------
    같은 인자로 함수를 다시 호출했을 때,
    함수 내부를 다시 실행하지 않고 이전 반환값을 재사용하는 기능이다.

    여기서는 artifact_dir가 같으면:
        load_failure_model_artifacts(...)
    를 매번 다시 실행하지 않는다.

    왜 model artifact는 실패하면 안 되는가?
    --------------------------------------
    model.pt, scaler.joblib, metadata.json이 없으면
    raw sample을 예측할 수 없다.

    따라서 이 함수에서 예외가 발생하면
    endpoint 쪽에서는 HTTP 500 또는 명확한 error response로 처리해야 한다.
    """
    return load_failure_model_artifacts(Path(artifact_dir))


@lru_cache(maxsize=1)
def get_cached_shap_artifacts(
    artifact_dir: str = str(ARTIFACT_DIR),
) -> Any:
    """
    SHAP artifact를 cache해서 로드한다.

    SHAP artifact 예시:
        shap_background.pt
        shap_reference_values.json
        global_importance.json

    주의
    ----
    이 함수는 SHAP artifact 로드 자체만 담당한다.
    실패를 warning으로 바꿀지, API error로 바꿀지는 service 계층에서 결정한다.

    이유:
        artifact_cache.py는 loading만 담당하고,
        API 응답 정책은 failure_agent_service.py에서 담당하는 것이 역할 분리에 좋다.
    """
    return load_shap_artifacts(Path(artifact_dir))


def clear_artifact_cache_for_tests() -> None:
    """
    테스트에서 cache를 초기화하기 위한 함수다.

    왜 필요한가?
    ------------
    lru_cache는 한 번 로드한 결과를 계속 들고 있다.

    테스트에서는 monkeypatch로 load 함수를 바꾸거나,
    SHAP artifact 실패 상황을 가짜로 만들 수 있다.

    그런데 cache가 남아 있으면
    monkeypatch가 적용되기 전에 저장된 값이 재사용되어
    테스트가 의도대로 동작하지 않을 수 있다.

    그래서 테스트 시작 전에 cache_clear()를 호출할 수 있게 한다.
    """
    get_cached_failure_model_artifacts.cache_clear()
    get_cached_shap_artifacts.cache_clear()