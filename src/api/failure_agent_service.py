# src/api/failure_agent_service.py

"""
Day 12 - Failure Agent API service layer

이 파일의 목적
--------------
FastAPI endpoint에서 직접 처리하던
prediction, SHAP, global importance, evidence, answer 생성 흐름을
별도 service 함수로 분리한다.

왜 service layer를 만드는가?
----------------------------
endpoint는 원래 얇게 유지하는 것이 좋다.

endpoint의 역할:
    1. HTTP request를 받는다.
    2. service 함수를 호출한다.
    3. service 결과를 response로 반환한다.

service의 역할:
    1. model artifact를 가져온다.
    2. prediction을 수행한다.
    3. SHAP 계산을 시도한다.
    4. 실패 가능한 부가 설명은 warning으로 처리한다.
    5. evidence와 answer를 만든다.

Day 12 핵심 정책
----------------
1. prediction 실패
   - 모델 artifact 또는 추론 자체가 실패한 것이다.
   - API 전체 실패로 처리한다.

2. SHAP 실패
   - 예측은 이미 성공했거나 성공 가능하다.
   - API 전체를 실패시키지 않는다.
   - warnings에 기록하고 shap_local evidence만 생략한다.

3. global importance 실패
   - 부가 설명 실패다.
   - API 전체를 실패시키지 않는다.
   - warnings에 기록하고 global_importance evidence만 생략한다.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from fastapi import HTTPException

from src.agent.answer_builder import build_agent_answer
from src.agent.evidence_builder import build_agent_evidence
from src.api.artifact_cache import (
    get_cached_failure_model_artifacts,
    get_cached_shap_artifacts,
)
from src.api.schemas import FailurePredictionRequest, FailurePredictionResponse
from src.inference.predict_failure import predict_failure_from_artifacts
from src.interpretability.shap_runtime import (
    build_global_importance_items_from_map,
    build_shap_local_explanation_for_sample,
)


def request_to_raw_sample(
    request: FailurePredictionRequest,
) -> dict[str, Any]:
    """
    API request schema를 Day 5 inference pipeline이 이해하는 raw_sample로 변환한다.

    왜 변환이 필요한가?
    -------------------
    API JSON에서는 사용하기 쉬운 snake_case 필드명을 쓴다.

    예:
        air_temperature
        process_temperature
        rotational_speed
        torque
        tool_wear
        type

    하지만 Day 5 inference pipeline은 AI4I 원본 컬럼명에 가까운 key를 사용한다.

    예:
        "Air temperature [K]"
        "Process temperature [K]"
        "Rotational speed [rpm]"
        "Torque [Nm]"
        "Tool wear [min]"
        "Type"

    따라서 API request를 모델 입력용 raw_sample로 바꿔줘야 한다.

    주의
    ----
    Pydantic 모델 내부에서는 JSON의 "type" 필드가
    machine_type 같은 이름으로 alias 처리되어 있을 가능성이 높다.

    그래서 request.type이 아니라 request.machine_type을 사용한다.
    Day 11에서 이 부분 때문에 AttributeError가 발생했었다.
    """
    return {
        "Air temperature [K]": request.air_temperature,
        "Process temperature [K]": request.process_temperature,
        "Rotational speed [rpm]": request.rotational_speed,
        "Torque [Nm]": request.torque,
        "Tool wear [min]": request.tool_wear,
        "Type": request.machine_type,
    }


def _to_dict(value: Any) -> Any:
    """
    dataclass, Pydantic model, dict를 response에 넣기 쉬운 dict로 바꾼다.

    왜 필요한가?
    ------------
    evidence_builder가 반환하는 값이
    dataclass일 수도 있고, dict일 수도 있다.

    API response에서는 JSON으로 변환 가능한 구조가 필요하다.
    그래서 최대한 안전하게 dict로 변환한다.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return value

    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return value


def _to_dict_list(items: list[Any]) -> list[dict[str, Any]]:
    """
    여러 evidence item을 JSON 응답에 넣기 쉬운 dict list로 변환한다.
    """
    return [_to_dict(item) for item in items]


def _get_global_importance_map_from_shap_artifacts(
    shap_artifacts: Any,
) -> dict[str, float]:
    """
    ShapArtifacts 객체에서 global importance map을 안전하게 꺼낸다.

    왜 getattr을 쓰는가?
    --------------------
    Day 11에서 만든 ShapArtifacts의 필드명이
    global_importance일 수도 있고,
    global_importance_map일 수도 있다.

    현재 코드와 약간 달라도 쉽게 맞출 수 있도록
    둘 다 확인한다.

    반환 예시:
        {
            "Torque [Nm]": 0.3309,
            "Air temperature [K]": 0.2725,
            ...
        }
    """
    global_importance = getattr(shap_artifacts, "global_importance", None)

    if global_importance is None:
        global_importance = getattr(shap_artifacts, "global_importance_map", None)

    if global_importance is None:
        return {}

    return dict(global_importance)


def _safe_build_shap_local_explanation(
    *,
    request: FailurePredictionRequest,
    model_artifacts: Any,
    shap_artifacts: Any | None,
    raw_sample: dict[str, Any],
    prediction_dict: dict[str, Any],
    warnings: list[str],
) -> Any | None:
    """
    SHAP local explanation을 안전하게 생성한다.

    핵심 정책
    ---------
    include_shap=False:
        SHAP 계산을 하지 않는다.

    include_shap=True인데 shap_artifacts가 없음:
        prediction은 유지한다.
        warnings에 이유를 남긴다.
        shap_local evidence는 생략한다.

    include_shap=True인데 SHAP 계산 중 오류:
        prediction은 유지한다.
        warnings에 이유를 남긴다.
        shap_local evidence는 생략한다.

    왜 이렇게 하는가?
    -----------------
    SHAP는 설명 기능이다.
    예측 결과 자체보다 부가적인 정보다.

    따라서 SHAP 실패 때문에 전체 API가 500으로 죽으면
    사용자 입장에서는 prediction까지 못 받게 된다.

    제조 AI API에서는 최소한의 prediction과 rule-based evidence는
    반환하는 쪽이 더 안정적인 설계다.
    """
    if not request.include_shap:
        return None

    if shap_artifacts is None:
        warnings.append(
            "include_shap=true였지만 SHAP artifact를 사용할 수 없어 "
            "shap_local evidence를 생략했습니다."
        )
        return None

    try:
        # 함수 인자 이름이 바뀌어도 덜 깨지도록 positional argument로 호출한다.
        # 예상 흐름:
        #   model_artifacts + shap_artifacts + raw_sample
        #   -> LocalExplanationResult
        return build_shap_local_explanation_for_sample(
            include_shap=request.include_shap,
            artifacts=model_artifacts,
            shap_artifacts=shap_artifacts,
            raw_sample=raw_sample,
            threshold=prediction_dict.get("threshold"),
            risk_level=prediction_dict.get("risk_level"),
            top_k=5,
        )

    except Exception as exc:
        warnings.append(
            "include_shap=true였지만 SHAP local explanation 계산 중 오류가 발생해 "
            f"shap_local evidence를 생략했습니다. error={type(exc).__name__}: {exc}"
        )
        return None


def _safe_build_global_importance_items(
    *,
    request: FailurePredictionRequest,
    shap_artifacts: Any | None,
    warnings: list[str],
) -> list[dict[str, Any]]:
    """
    global importance evidence용 item을 안전하게 생성한다.

    핵심 정책
    ---------
    include_global_importance=False:
        global importance를 응답에 넣지 않는다.

    include_global_importance=True인데 artifact가 없음:
        prediction은 유지한다.
        warnings에 이유를 남긴다.
        global_importance evidence는 생략한다.

    global importance 해석 주의
    ---------------------------
    global importance는 전체 test set 기준으로
    모델이 어떤 feature에 민감했는지 보여주는 값이다.

    개별 sample의 직접 원인이 아니다.
    """
    if not request.include_global_importance:
        return []

    if shap_artifacts is None:
        warnings.append(
            "include_global_importance=true였지만 global importance artifact를 "
            "사용할 수 없어 global_importance evidence를 생략했습니다."
        )
        return []

    try:
        global_importance_map = _get_global_importance_map_from_shap_artifacts(
            shap_artifacts
        )

        if not global_importance_map:
            warnings.append(
                "global importance artifact가 비어 있어 "
                "global_importance evidence를 생략했습니다."
            )
            return []

        return build_global_importance_items_from_map(global_importance_map)

    except Exception as exc:
        warnings.append(
            "global importance 변환 중 오류가 발생해 "
            f"global_importance evidence를 생략했습니다. "
            f"error={type(exc).__name__}: {exc}"
        )
        return []


def _load_optional_shap_artifacts(
    *,
    request: FailurePredictionRequest,
    warnings: list[str],
) -> Any | None:
    """
    SHAP/global importance가 필요한 경우에만 SHAP artifact를 로드한다.

    왜 조건부로 로드하는가?
    ----------------------
    사용자가 include_shap=false,
    include_global_importance=false로 요청했다면
    SHAP artifact는 필요 없다.

    이 경우 굳이 shap_background.pt 등을 로드하지 않는 것이 낫다.
    """
    needs_shap_artifacts = (
        request.include_shap or request.include_global_importance
    )

    if not needs_shap_artifacts:
        return None

    try:
        return get_cached_shap_artifacts()

    except Exception as exc:
        warnings.append(
            "SHAP/global importance artifact 로드에 실패했습니다. "
            "prediction과 rule_based evidence는 계속 반환합니다. "
            f"error={type(exc).__name__}: {exc}"
        )
        return None


def run_failure_prediction_agent(
    request: FailurePredictionRequest,
) -> FailurePredictionResponse:
    """
    Failure prediction Agent API의 핵심 실행 함수다.

    전체 흐름
    --------
    1. request를 raw_sample로 변환
    2. cached model artifacts 로드
    3. prediction 실행
    4. 필요한 경우 SHAP artifacts 로드
    5. SHAP local explanation 생성 시도
    6. global importance item 생성 시도
    7. Agent evidence 통합
    8. Agent answer 생성
    9. warnings / limitations 포함한 response 반환

    예외 처리 기준
    --------------
    model artifact 또는 prediction 실패:
        API 전체 실패

    SHAP 또는 global importance 실패:
        prediction은 유지하고 warnings에 기록
    """
    warnings: list[str] = []
    limitations: list[str] = [
        "이 결과는 AI4I 기반 학습용 예시 모델의 출력이며 실제 설비 진단을 대체하지 않습니다.",
        "rule_based evidence는 사람이 정한 제조 기준 기반 참고 신호입니다.",
        "shap_local evidence는 모델 logit 출력에 대한 feature contribution이며 실제 고장의 물리적 원인 단정이 아닙니다.",
        "global_importance는 전체 test set 기준 모델 민감도이며 개별 sample의 직접 원인이 아닙니다.",
    ]

    raw_sample = request_to_raw_sample(request)

    try:
        model_artifacts = get_cached_failure_model_artifacts()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "모델 artifact 로드에 실패해 prediction을 수행할 수 없습니다.",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "hint": "models/failure_mlp/model.pt, scaler.joblib, metadata.json이 존재하는지 확인하세요.",
            },
        ) from exc

    try:
        prediction_result = predict_failure_from_artifacts(
            artifacts=model_artifacts,
            raw_sample=raw_sample,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "모델 추론 중 오류가 발생했습니다.",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "hint": "입력 feature 이름, Type 값, scaler, metadata의 feature_columns를 확인하세요.",
            },
        ) from exc

    prediction_dict = _to_dict(prediction_result)

    shap_artifacts = _load_optional_shap_artifacts(
        request=request,
        warnings=warnings,
    )

    shap_local_explanation = _safe_build_shap_local_explanation(
        request=request,
        model_artifacts=model_artifacts,
        shap_artifacts=shap_artifacts,
        raw_sample=raw_sample,
        prediction_dict=prediction_dict,
        warnings=warnings,
    )

    global_importance_items = _safe_build_global_importance_items(
        request=request,
        shap_artifacts=shap_artifacts,
        warnings=warnings,
    )

    evidence_items = build_agent_evidence(
        prediction_result=prediction_dict,
        shap_local_explanation=shap_local_explanation,
        global_importance_items=global_importance_items,
        shap_top_n=5,
    )

    answer = build_agent_answer(
        prediction_result=prediction_dict,
        evidence_items=evidence_items,
    )

    evidence_dicts = _to_dict_list(evidence_items)

    return FailurePredictionResponse(
        prediction=prediction_dict["prediction"],
        probability=prediction_dict["probability"],
        threshold=prediction_dict["threshold"],
        risk_level=prediction_dict["risk_level"],
        recommended_action=prediction_dict["recommended_action"],
        evidence=evidence_dicts,
        answer=answer,
        warnings=warnings,
        limitations=limitations,
    )