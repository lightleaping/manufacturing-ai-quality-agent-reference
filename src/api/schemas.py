# src/api/schemas.py

from typing import Any

from pydantic import BaseModel, Field


class FailurePredictionRequest(BaseModel):
    """
    POST /agent/failure-prediction 요청 body를 정의하는 Pydantic schema입니다.

    Pydantic schema는 API 입력값을 검증하는 역할을 합니다.

    예를 들어 사용자가 Swagger UI 또는 HTTP client로 아래 JSON을 보내면,

    {
      "air_temperature": 303.0,
      "process_temperature": 312.5,
      "rotational_speed": 1380.0,
      "torque": 62.0,
      "tool_wear": 220.0,
      "type": "L",
      "include_shap": true,
      "include_global_importance": true
    }

    FastAPI는 이 JSON을 FailurePredictionRequest 객체로 변환합니다.

    주의:
    - Python에서 type이라는 이름은 내장 함수 type과 겹칠 수 있습니다.
    - 그래서 내부 변수명은 machine_type으로 두고,
      API JSON에서는 alias="type"을 사용해 type이라는 이름으로 받습니다.
    """

    air_temperature: float = Field(
        ...,
        description="Air temperature [K]",
        examples=[303.0],
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature [K]",
        examples=[312.5],
    )

    rotational_speed: float = Field(
        ...,
        description="Rotational speed [rpm]",
        examples=[1380.0],
    )

    torque: float = Field(
        ...,
        description="Torque [Nm]",
        examples=[62.0],
    )

    tool_wear: float = Field(
        ...,
        description="Tool wear [min]",
        examples=[220.0],
    )

    machine_type: str = Field(
        ...,
        alias="type",
        description="AI4I product type. Usually one of L, M, H.",
        examples=["L"],
    )

    include_shap: bool = Field(
        default=True,
        description="Whether to include SHAP local explanation evidence.",
    )

    include_global_importance: bool = Field(
        default=True,
        description="Whether to include global permutation importance evidence.",
    )

    def to_raw_sample(self) -> dict[str, Any]:
        """
        API request field 이름을 Day 5 inference 함수가 기대하는 raw_sample 형식으로 변환합니다.

        Day 5 추론 흐름에서는 AI4I 원본 feature 이름을 그대로 사용했습니다.

        즉 API에서는 사용하기 쉬운 snake_case 이름을 받고,
        내부 inference 함수에는 기존 프로젝트 feature 이름으로 넘깁니다.

        API request:
            air_temperature

        내부 raw_sample:
            Air temperature [K]
        """

        return {
            "Air temperature [K]": self.air_temperature,
            "Process temperature [K]": self.process_temperature,
            "Rotational speed [rpm]": self.rotational_speed,
            "Torque [Nm]": self.torque,
            "Tool wear [min]": self.tool_wear,
            "Type": self.machine_type,
        }


class AgentEvidenceResponse(BaseModel):
    """
    API 응답에 포함될 evidence 한 개의 구조입니다.

    Day 9에서 만든 AgentEvidence dataclass를
    JSON으로 반환하기 좋게 Pydantic schema로 표현합니다.

    evidence_type 예:
    - prediction_summary
    - rule_based
    - shap_local
    - global_importance

    source 예:
    - model_prediction
    - rule_engine
    - shap
    - permutation_importance
    """

    evidence_id: str
    evidence_type: str
    source: str
    title: str
    summary: str

    feature: str | None = None
    value: Any | None = None
    direction: str | None = None
    contribution: float | None = None
    importance: float | None = None
    severity: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailurePredictionResponse(BaseModel):
    """
    POST /agent/failure-prediction 응답 schema입니다.

    이 response는 단순 모델 결과만 반환하지 않습니다.

    포함 항목:
    - prediction: threshold 기준 최종 0/1 판단
    - probability: 모델이 예측한 고장 확률
    - threshold: 운영 판단 기준
    - risk_level: 설명용 위험 등급
    - recommended_action: 권장 조치
    - evidence: prediction_summary, rule_based, shap_local, global_importance
    - answer: Agent가 사용자에게 보여줄 자연어 답변
    - warnings: 해석상 주의사항
    - limitations: 현재 시스템 한계
    """

    prediction: int
    probability: float
    threshold: float
    risk_level: str
    recommended_action: str

    evidence: list[AgentEvidenceResponse]
    answer: str

    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)