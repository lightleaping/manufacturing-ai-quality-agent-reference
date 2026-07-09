"""
Day 10/11 - Failure prediction Agent API

이 파일의 역할
--------------
FastAPI endpoint를 통해 설비 고장 예측 Agent 응답을 제공합니다.

Endpoint:
    POST /agent/failure-prediction

전체 흐름
--------
1. 사용자가 raw sample JSON을 보냅니다.
2. Pydantic schema가 입력값을 검증합니다.
3. Day 5 model artifact를 로드합니다.
4. Day 5 predict_failure_from_artifacts()로 prediction을 수행합니다.
5. Day 11 SHAP artifact를 로드합니다.
6. include_shap=True이면 SHAP local explanation을 생성합니다.
7. include_global_importance=True이면 global importance evidence를 포함합니다.
8. Day 9 build_agent_evidence()로 evidence list를 만듭니다.
9. Day 9 build_agent_answer()로 최종 answer를 만듭니다.
10. API response JSON을 반환합니다.

중요한 설계 원칙
----------------
- API endpoint 안에 모델 추론 로직을 길게 쓰지 않습니다.
- API endpoint 안에 SHAP 계산 로직을 길게 쓰지 않습니다.
- endpoint는 request를 받고, 기존 함수들을 연결하는 얇은 계층으로 유지합니다.
- rule_based evidence, shap_local evidence, global_importance evidence를 구분합니다.
- SHAP value는 probability가 아니라 logit 기준 contribution입니다.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from src.agent.answer_builder import build_agent_answer
from src.agent.evidence_builder import build_agent_evidence
from src.api.schemas import FailurePredictionRequest
from src.inference.model_artifacts import load_failure_model_artifacts
from src.inference.predict_failure import predict_failure_from_artifacts
from src.interpretability.shap_artifacts import load_shap_artifacts
from src.interpretability.shap_runtime import (
    build_global_importance_items_from_map,
    build_shap_local_explanation_for_sample,
)


router = APIRouter()

ARTIFACT_DIR = Path("models/failure_mlp")


def _to_dict(value: Any) -> dict[str, Any]:
    """
    dataclass, Pydantic model, dict 객체를 일반 dict로 변환합니다.

    이 함수가 필요한 이유
    --------------------
    프로젝트 내부 함수들이 반환하는 객체 형태가 서로 다를 수 있습니다.

    예:
    - Day 5 PredictionResult dataclass
    - Day 9 AgentEvidence dataclass
    - 테스트 fake 객체
    - 이미 dict인 객체

    API response를 만들 때는 최종적으로 JSON 직렬화 가능한 dict 구조가 필요합니다.
    """

    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if is_dataclass(value):
        return asdict(value)

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    if hasattr(value, "__dict__"):
        return dict(value.__dict__)

    raise TypeError(f"Cannot convert object to dict: {type(value)}")


def _build_raw_sample_from_request(
    request: FailurePredictionRequest,
) -> dict[str, float | str]:
    """
    FastAPI request schema를 Day 5 추론 함수들이 이해하는 raw_sample dict로 변환합니다.

    중요한 점
    --------
    API JSON 입력에서는 "type"이라는 이름을 사용합니다.

    예:
        {
            "type": "L"
        }

    하지만 Pydantic schema 내부 필드명은 machine_type입니다.

    즉, Python 코드에서는 request.type이 아니라 request.machine_type으로 접근해야 합니다.

    변환 흐름:
        JSON "type"
        -> request.machine_type
        -> raw_sample["Type"]
        -> Day 5 inference pipeline

    Day 5 추론 함수는 AI4I 원본 feature 이름을 기준으로 동작하므로,
    API 입력 필드명을 AI4I 컬럼명으로 다시 맞춥니다.
    """

    return {
        "Air temperature [K]": request.air_temperature,
        "Process temperature [K]": request.process_temperature,
        "Rotational speed [rpm]": request.rotational_speed,
        "Torque [Nm]": request.torque,
        "Tool wear [min]": request.tool_wear,
        "Type": request.machine_type,
    }


def _normalize_evidence_item(
    evidence_item: Any,
) -> dict[str, Any]:
    """
    Agent evidence item을 API response에 넣기 좋은 dict로 변환합니다.

    evidence item은 상황에 따라 dataclass일 수도 있고 dict일 수도 있습니다.
    이 함수에서 구조를 통일합니다.
    """

    item = _to_dict(evidence_item)

    return {
        "evidence_id": item.get("evidence_id"),
        "evidence_type": item.get("evidence_type"),
        "source": item.get("source"),
        "title": item.get("title"),
        "summary": item.get("summary"),
        "feature": item.get("feature"),
        "value": item.get("value"),
        "direction": item.get("direction"),
        "contribution": item.get("contribution"),
        "importance": item.get("importance"),
        "severity": item.get("severity"),
        "metadata": item.get("metadata") or {},
    }


def _normalize_evidence_items(
    evidence_items: list[Any],
) -> list[dict[str, Any]]:
    """
    evidence list 전체를 JSON response에 넣을 수 있는 list[dict]로 변환합니다.
    """

    return [
        _normalize_evidence_item(evidence_item)
        for evidence_item in evidence_items
    ]


def _build_response_warnings(
    *,
    include_shap: bool,
    shap_local_explanation: Any | None,
) -> list[str]:
    """
    API response에 포함할 warnings를 만듭니다.

    warnings는 실패가 아니라, 사용자가 해석할 때 주의해야 하는 내용을 담습니다.
    """

    warnings: list[str] = []

    if include_shap and shap_local_explanation is None:
        warnings.append(
            "include_shap=True로 요청했지만 SHAP local explanation이 생성되지 않았습니다."
        )

    return warnings


def _build_response_limitations(
    *,
    include_shap: bool,
    include_global_importance: bool,
) -> list[str]:
    """
    API response에 포함할 limitations를 만듭니다.

    limitations는 모델/설명 결과의 해석 한계를 명확히 알려주기 위한 항목입니다.
    """

    limitations = [
        (
            "이 API의 prediction은 현재 학습된 FailureMLP 모델과 운영 threshold 기준의 "
            "예측 결과입니다."
        ),
        (
            "rule_based evidence는 입력값이 사람이 정한 제조 기준에서 점검 신호에 "
            "해당하는지 보여주는 참고 근거입니다."
        ),
        (
            "SHAP local evidence는 실제 고장의 물리적 원인을 단정하지 않고, "
            "현재 모델 출력에 대해 각 feature가 어느 방향으로 기여했는지 설명합니다."
        ),
        (
            "현재 FailureMLP는 마지막에 Sigmoid가 없으므로 SHAP value는 probability가 아니라 "
            "logit 기준 contribution으로 해석해야 합니다."
        ),
        (
            "global importance는 전체 test set 기준 모델 민감도이며, "
            "개별 sample의 직접 원인이 아닙니다."
        ),
    ]

    if not include_shap:
        limitations.append(
            "이번 요청에서는 include_shap=False이므로 SHAP local explanation 계산을 생략했습니다."
        )

    if not include_global_importance:
        limitations.append(
            "이번 요청에서는 include_global_importance=False이므로 global importance evidence를 제외했습니다."
        )

    return limitations


@router.post("/agent/failure-prediction")
def predict_failure_agent(
    request: FailurePredictionRequest,
) -> dict[str, Any]:
    """
    설비 고장 예측 Agent API endpoint입니다.

    입력:
        air_temperature
        process_temperature
        rotational_speed
        torque
        tool_wear
        type
        include_shap
        include_global_importance

    출력:
        prediction
        probability
        threshold
        risk_level
        recommended_action
        evidence
        answer
        warnings
        limitations

    Day 11 변경점
    -------------
    Day 10에서는 SHAP local explanation이 placeholder 상태였습니다.

    Day 11에서는:
    - models/failure_mlp/shap_background.pt
    - models/failure_mlp/shap_reference_values.json
    - models/failure_mlp/global_importance.json

    을 미리 생성해두고, API에서는 이 artifact를 로드해서 사용합니다.

    이 방식은 API 요청마다 train CSV를 다시 로드하거나
    SHAP background를 새로 만들지 않기 때문에 운영 환경에 더 가깝습니다.
    """

    # -------------------------------------------------------------------------
    # 1. API request를 Day 5 추론 함수가 이해하는 raw_sample로 변환
    # -------------------------------------------------------------------------
    raw_sample = _build_raw_sample_from_request(request)

    # -------------------------------------------------------------------------
    # 2. Day 5 model artifact 로드
    # -------------------------------------------------------------------------
    #
    # model.pt, scaler.joblib, metadata.json을 로드합니다.
    #
    # API endpoint 안에서 torch.load, joblib.load, json.load를 직접 하지 않습니다.
    # artifact 로딩 책임은 model_artifacts.py가 갖는 것이 맞습니다.
    artifacts = load_failure_model_artifacts(ARTIFACT_DIR)

    # -------------------------------------------------------------------------
    # 3. Day 5 추론 함수로 prediction 수행
    # -------------------------------------------------------------------------
    #
    # predict_failure_from_artifacts()는 기존 단일 sample 추론 흐름을 재사용합니다.
    #
    # 내부 흐름:
    #   validate_raw_sample
    #   build_single_sample_dataframe
    #   normalize_type_value
    #   scale_single_sample_dataframe
    #   dataframe_to_single_tensor
    #   model inference
    #   sigmoid(logit)
    #   threshold 비교
    #   risk_level 생성
    #   rule_based evidence 생성
    prediction_result = predict_failure_from_artifacts(
        artifacts=artifacts,
        raw_sample=raw_sample,
    )

    prediction_dict = _to_dict(prediction_result)

    # -------------------------------------------------------------------------
    # 4. Day 11 SHAP artifact 로드
    # -------------------------------------------------------------------------
    #
    # 운영형 구조에서는 API 요청마다 background를 새로 만들지 않습니다.
    # 미리 저장된 SHAP artifact를 로드합니다.
    #
    # include_shap 또는 include_global_importance 중 하나라도 True면 필요합니다.
    shap_artifacts = None

    if request.include_shap or request.include_global_importance:
        shap_artifacts = load_shap_artifacts(ARTIFACT_DIR)

    # -------------------------------------------------------------------------
    # 5. global importance items 생성
    # -------------------------------------------------------------------------
    #
    # 변수명은 global_importance_items로 통일합니다.
    #
    # 이전 에러 원인:
    #   어떤 곳에서는 global_importance_items,
    #   어떤 곳에서는 global_importance_evidence를 사용해서
    #   UnboundLocalError가 발생했습니다.
    global_importance_items: list[dict[str, Any]] = []

    if request.include_global_importance and shap_artifacts is not None:
        global_importance_items = build_global_importance_items_from_map(
            shap_artifacts.global_importance_map,
        )

    # -------------------------------------------------------------------------
    # 6. SHAP local explanation 생성
    # -------------------------------------------------------------------------
    #
    # include_shap=False이면 실제 SHAP 계산을 수행하지 않고 None을 반환합니다.
    #
    # include_shap=True이면:
    #   raw_sample을 Day 5 전처리 흐름으로 tensor화하고,
    #   저장된 shap_background.pt를 사용해
    #   build_shap_local_explanation_result()를 호출합니다.
    shap_local_explanation = None

    if request.include_shap and shap_artifacts is not None:
        shap_local_explanation = build_shap_local_explanation_for_sample(
            include_shap=request.include_shap,
            artifacts=artifacts,
            shap_artifacts=shap_artifacts,
            raw_sample=raw_sample,
            threshold=float(prediction_dict["threshold"]),
            risk_level=str(prediction_dict["risk_level"]),
            top_k=5,
        )

    # -------------------------------------------------------------------------
    # 7. Agent evidence 생성
    # -------------------------------------------------------------------------
    #
    # build_agent_evidence()의 실제 signature에 맞춰 keyword argument를 사용합니다.
    #
    # 실제 함수:
    #   build_agent_evidence(
    #       prediction_result,
    #       shap_local_explanation=None,
    #       global_importance_items=None,
    #       shap_top_n=5,
    #   )
    agent_evidence = build_agent_evidence(
        prediction_result=prediction_dict,
        shap_local_explanation=shap_local_explanation,
        global_importance_items=global_importance_items,
        shap_top_n=5,
    )

    normalized_evidence = _normalize_evidence_items(agent_evidence)

    # -------------------------------------------------------------------------
    # 8. Agent answer 생성
    # -------------------------------------------------------------------------
    answer = build_agent_answer(
        prediction_result=prediction_dict,
        evidence_items=agent_evidence,
    )

    # -------------------------------------------------------------------------
    # 9. warnings / limitations 생성
    # -------------------------------------------------------------------------
    warnings = _build_response_warnings(
        include_shap=request.include_shap,
        shap_local_explanation=shap_local_explanation,
    )

    limitations = _build_response_limitations(
        include_shap=request.include_shap,
        include_global_importance=request.include_global_importance,
    )

    # -------------------------------------------------------------------------
    # 10. 최종 API response 반환
    # -------------------------------------------------------------------------
    return {
        "prediction": prediction_dict["prediction"],
        "probability": prediction_dict["probability"],
        "threshold": prediction_dict["threshold"],
        "risk_level": prediction_dict["risk_level"],
        "recommended_action": prediction_dict["recommended_action"],
        "evidence": normalized_evidence,
        "answer": answer,
        "warnings": warnings,
        "limitations": limitations,
    }