# src/api/failure_agent_api.py

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.agent.answer_builder import build_agent_answer
from src.agent.evidence_builder import build_agent_evidence
from src.api.schemas import (
    AgentEvidenceResponse,
    FailurePredictionRequest,
    FailurePredictionResponse,
)
from src.inference.model_artifacts import load_failure_model_artifacts
from src.inference.predict_failure import predict_failure_from_artifacts


router = APIRouter(
    prefix="/agent",
    tags=["Failure Prediction Agent"],
)


DEFAULT_ARTIFACT_DIR = Path("models/failure_mlp")


def _to_dict(value: Any) -> dict[str, Any]:
    """
    dataclass 또는 dict 형태의 객체를 dict로 변환합니다.

    왜 필요한가?

    Day 5, Day 8, Day 9에서 만든 결과 객체가
    dataclass일 수도 있고 dict일 수도 있습니다.

    FastAPI response는 최종적으로 JSON으로 변환되어야 하므로,
    내부 객체를 dict 형태로 맞춰주는 보조 함수가 있으면 안전합니다.
    """

    if is_dataclass(value):
        return asdict(value)

    if isinstance(value, dict):
        return value

    raise TypeError(f"dict로 변환할 수 없는 타입입니다: {type(value)}")


def _convert_agent_evidence_to_response(
    evidence_items: list[Any],
) -> list[AgentEvidenceResponse]:
    """
    Day 9 AgentEvidence 객체 list를 API response schema list로 변환합니다.

    Day 9의 AgentEvidence는 내부 Python 객체입니다.
    API 응답은 Pydantic BaseModel 형태로 반환하는 것이 좋습니다.

    그래서 여기서 AgentEvidence -> AgentEvidenceResponse로 변환합니다.
    """

    response_items: list[AgentEvidenceResponse] = []

    for item in evidence_items:
        item_dict = _to_dict(item)
        response_items.append(AgentEvidenceResponse(**item_dict))

    return response_items


def _build_limitations(
    include_shap: bool,
    include_global_importance: bool,
) -> list[str]:
    """
    API 응답에 항상 포함할 해석상 한계 문장을 만듭니다.

    중요:
    제조 AI Agent에서는 evidence를 보여주더라도
    특정 feature를 실제 고장의 물리적 원인이라고 단정하면 안 됩니다.

    rule-based evidence:
        사람이 정한 제조 기준에 따른 점검 신호

    shap_local evidence:
        현재 모델 output에 대한 feature contribution

    global_importance evidence:
        전체 test set 기준 모델 민감도

    이 셋은 의미가 다릅니다.
    """

    limitations = [
        "prediction은 운영 threshold 기준의 모델 판단이며 실제 설비 고장을 확정하는 값은 아닙니다.",
        "rule_based evidence는 입력값을 사람이 정한 제조 기준으로 해석한 점검 신호입니다.",
        "SHAP evidence는 모델 출력에 대한 feature contribution이며 실제 고장의 물리적 원인을 단정하지 않습니다.",
    ]

    if not include_shap:
        limitations.append(
            "include_shap=false로 요청되어 SHAP local explanation은 응답에 포함하지 않았습니다."
        )

    if not include_global_importance:
        limitations.append(
            "include_global_importance=false로 요청되어 global importance evidence는 응답에 포함하지 않았습니다."
        )

    return limitations


def _build_global_importance_placeholder(
    include_global_importance: bool,
) -> list[dict[str, Any]]:
    """
    Day 6 permutation importance 결과를 API에 넣기 위한 임시 구조입니다.

    현재 단계에서는 Day 6 실행 결과를 파일에서 자동 로드하는 구조까지 만들지는 않습니다.
    Day 10의 핵심은 API 연결입니다.

    그래서 우선 Day 6에서 확인한 global importance 값을
    정적인 참고 evidence로 넣을 수 있게 구성합니다.

    추후 개선:
    - reports/day6_interpretability_summary.md에서 읽기
    - models/failure_mlp/global_importance.json으로 저장 후 로드
    - 학습 파이프라인에서 permutation importance 결과를 artifact로 저장
    """

    if not include_global_importance:
        return []

    return [
        {
            "feature": "Torque [Nm]",
            "importance": 0.3309,
            "summary": "전체 test set 기준 permutation importance가 가장 높게 나타난 feature입니다.",
        },
        {
            "feature": "Air temperature [K]",
            "importance": 0.2725,
            "summary": "전체 test set 기준 모델 성능에 큰 영향을 준 feature입니다.",
        },
        {
            "feature": "Rotational speed [rpm]",
            "importance": 0.2292,
            "summary": "전체 test set 기준 모델이 민감하게 반응한 feature입니다.",
        },
    ]


def _build_shap_evidence_placeholder(
    include_shap: bool,
) -> Any | None:
    """
    Day 8 SHAP local explanation 연결 위치입니다.

    현재 Day 10에서는 실제 SHAP 계산 함수를 아직 연결하지 않았습니다.

    build_agent_evidence() 함수는 두 번째 인자로
    shap_local_explanation을 받습니다.

    이 값은 이미 변환된 evidence list라기보다,
    Day 8에서 만든 LocalExplanationResult 같은
    local explanation 결과 객체가 들어오는 자리입니다.

    그래서 아직 실제 SHAP를 연결하지 않은 현재 단계에서는
    빈 list []보다 None을 반환하는 것이 더 자연스럽습니다.

    None 의미:
    - SHAP local explanation을 이번 응답에 포함하지 않는다.
    - API 구조는 유지하되, 실제 SHAP 연결은 다음 단계에서 진행한다.
    """

    if not include_shap:
        return None

    return None


@router.post(
    "/failure-prediction",
    response_model=FailurePredictionResponse,
)
def predict_failure_agent(
    request: FailurePredictionRequest,
) -> FailurePredictionResponse:
    """
    설비 고장 예측 Agent API endpoint입니다.

    이 함수는 HTTP 요청을 받는 FastAPI endpoint입니다.

    핵심 원칙:
    - endpoint 안에서 직접 scaling하지 않습니다.
    - endpoint 안에서 직접 torch model을 실행하지 않습니다.
    - endpoint 안에서 직접 evidence 문장을 길게 만들지 않습니다.

    대신 기존에 만든 함수들을 호출합니다.

    흐름:
    1. API request를 Day 5 raw_sample 형식으로 변환
    2. Day 5 model artifacts 로드
    3. Day 5 predict_failure_from_artifacts 호출
    4. Day 8 SHAP evidence 연결 준비
    5. Day 6 global importance evidence 연결
    6. Day 9 build_agent_evidence 호출
    7. Day 9 build_agent_answer 호출
    8. FastAPI response schema로 반환
    """

    raw_sample = request.to_raw_sample()

    try:
        artifacts = load_failure_model_artifacts(DEFAULT_ARTIFACT_DIR)
        prediction_result = predict_failure_from_artifacts(
            artifacts=artifacts,
            raw_sample=raw_sample,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "모델 artifact를 찾을 수 없습니다. "
                "먼저 학습을 실행해 models/failure_mlp/model.pt, "
                "scaler.joblib, metadata.json을 생성해야 합니다."
            ),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"예측 처리 중 알 수 없는 오류가 발생했습니다: {exc}",
        ) from exc

    # Day 5의 predict_failure_from_artifacts 결과를 dict로 변환합니다.
    #
    # 실제 build_agent_evidence() 함수 정의를 확인해보면,
    # 첫 번째 인자인 prediction_result 타입이 dict[str, Any]입니다.
    #
    # 즉, dataclass 형태의 prediction_result를 그대로 넘기면
    # 내부에서 prediction_result["probability"] 같은 방식으로 접근할 때 문제가 생길 수 있습니다.
    #
    # 그래서 build_agent_evidence()를 호출하기 전에
    # 먼저 _to_dict()로 dict 형태로 맞춥니다.
    prediction_dict = _to_dict(prediction_result)


    # Day 8 SHAP local explanation 연결 위치입니다.
    #
    # 현재 Day 10에서는 실제 SHAP 함수 연결 전이므로 None을 넘깁니다.
    #
    # 중요한 점:
    # build_agent_evidence()의 두 번째 parameter 이름은
    # shap_local_evidence가 아니라 shap_local_explanation입니다.
    #
    # 함수 signature:
    # build_agent_evidence(
    #     prediction_result,
    #     shap_local_explanation=None,
    #     global_importance_items=None,
    #     shap_top_n=5,
    # )
    #
    # 따라서 잘못된 이름인 shap_local_evidence=...를 쓰면
    # TypeError가 발생합니다.
    shap_local_explanation = _build_shap_evidence_placeholder(
        include_shap=request.include_shap,
    )


    # Day 6 permutation importance 결과를 global importance evidence로 넣습니다.
    #
    # build_agent_evidence()의 세 번째 parameter 이름은
    # global_importance_evidence가 아니라 global_importance_items입니다.
    #
    # 따라서 global_importance_items=... 라는 이름으로 넘겨야 합니다.
    global_importance_evidence = _build_global_importance_placeholder(
        include_global_importance=request.include_global_importance,
    )


    # Day 9에서 만든 build_agent_evidence()를 호출합니다.
    #
    # 여기서 중요한 점:
    # keyword argument 이름은 실제 함수 정의부의 parameter 이름과
    # 정확히 일치해야 합니다.
    #
    # 맞는 이름:
    # - prediction_result
    # - shap_local_explanation
    # - global_importance_items
    # - shap_top_n
    #
    # 틀린 이름:
    # - shap_local_evidence
    # - global_importance_evidence
    agent_evidence = build_agent_evidence(
        prediction_result=prediction_dict,
        shap_local_explanation=shap_local_explanation,
        global_importance_items=global_importance_evidence,
        shap_top_n=5,
    )


    # Day 9 answer builder를 호출합니다.
    #
    # prediction_result도 dict 형태로 넘기는 것이 안전합니다.
    # evidence_items는 build_agent_evidence()가 반환한 list[dict[str, Any]]입니다.
    answer = build_agent_answer(
        prediction_result=prediction_dict,
        evidence_items=agent_evidence,
    )

    return FailurePredictionResponse(
        prediction=prediction_dict["prediction"],
        probability=prediction_dict["probability"],
        threshold=prediction_dict["threshold"],
        risk_level=prediction_dict["risk_level"],
        recommended_action=prediction_dict["recommended_action"],
        evidence=_convert_agent_evidence_to_response(agent_evidence),
        answer=answer,
        warnings=[
            "evidence_type과 source를 확인해 rule 기반 근거, SHAP 근거, global importance를 구분해서 해석해야 합니다."
        ],
        limitations=_build_limitations(
            include_shap=request.include_shap,
            include_global_importance=request.include_global_importance,
        ),
    )