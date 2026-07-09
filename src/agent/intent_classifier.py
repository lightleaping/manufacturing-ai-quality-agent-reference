"""
Day 13 - OpenAI 기반 intent classifier

이 파일의 역할
----------------
사용자의 자연어 질문을 보고 Agent가 어떤 작업을 해야 하는지 intent로 분류합니다.

기존 manufacturing-mcp-agent에서는 rule-based 방식으로
"불량률", "센서", "라인 성능" 같은 키워드를 보고 tool을 선택했습니다.

이번 reference 프로젝트에서는 그 한계를 개선하기 위해
OpenAI gpt-4o-mini를 사용해서 자연어 질문을 intent JSON으로 분류합니다.

단, LLM 출력은 항상 100% 신뢰할 수 없기 때문에 다음 안전장치를 둡니다.

1. LLM은 최종 예측을 직접 하지 않는다.
2. LLM은 intent, confidence, reason만 JSON으로 반환한다.
3. 반환된 intent가 지원 목록에 없으면 unknown으로 바꾼다.
4. API key가 없거나, API 호출이 실패하거나, JSON 파싱이 실패하면 rule-based fallback을 사용한다.
5. 테스트에서는 실제 OpenAI API를 호출하지 않고 monkeypatch로 대체할 수 있게 함수 단위로 나눈다.

즉, 이 파일의 핵심은
"LLM intent classification + validation + fallback" 입니다.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any


# 현재 Day 13에서 지원할 intent 목록입니다.
#
# 처음부터 너무 많은 intent를 만들면 LangGraph 흐름과 테스트가 복잡해지므로,
# 현재 프로젝트에서 바로 연결 가능한 intent만 먼저 사용합니다.
#
# failure_prediction:
#   사용자가 설비 입력값을 바탕으로 고장 위험 예측을 요청한 경우
#
# dataset_schema_query:
#   사용자가 AI4I 데이터셋의 feature, target, 컬럼 의미를 질문한 경우
#
# unknown:
#   현재 Agent가 지원하지 않거나, 입력이 부족해서 판단하기 어려운 경우
SUPPORTED_INTENTS: set[str] = {
    "failure_prediction",
    "dataset_schema_query",
    "unknown",
}


DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class IntentClassificationResult:
    """
    intent 분류 결과를 담는 dataclass입니다.

    왜 dict만 쓰지 않고 dataclass를 쓰는가?
    ---------------------------------------
    dict만 사용하면 어떤 key가 있어야 하는지 코드만 보고 파악하기 어렵습니다.

    dataclass를 사용하면 intent 분류 결과가 항상 아래 필드를 가진다는 점을
    코드 구조로 명확하게 표현할 수 있습니다.

    Attributes
    ----------
    intent:
        분류된 intent입니다.
        SUPPORTED_INTENTS 중 하나여야 합니다.

    confidence:
        intent 분류에 대한 신뢰도입니다.
        0.0 ~ 1.0 사이 값으로 정규화합니다.

    reason:
        왜 이 intent로 판단했는지에 대한 짧은 설명입니다.

    source:
        어떤 방식으로 분류했는지 표시합니다.
        - "openai": OpenAI API 결과
        - "rule_based": fallback rule 기반 결과
        - "fallback": OpenAI 실패 후 rule-based로 대체된 결과
        - "validation": 입력값 검증 단계에서 생성된 결과

    raw_response:
        OpenAI가 반환한 원문 문자열입니다.
        디버깅용이며, 운영 응답에 그대로 노출할 필요는 없습니다.

    error:
        OpenAI 호출 실패, JSON 파싱 실패, 검증 실패 등의 오류 메시지입니다.
        실패하더라도 Agent 전체가 죽지 않도록 문자열로 보관합니다.
    """

    intent: str
    confidence: float
    reason: str
    source: str
    raw_response: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        dataclass 결과를 dict로 변환합니다.

        LangGraph AgentState, FastAPI response, 테스트 assertion에서는
        dict 형태가 더 다루기 쉬운 경우가 많습니다.
        """
        return asdict(self)


def classify_intent(
    question: str,
    *,
    use_openai: bool = True,
) -> IntentClassificationResult:
    """
    사용자 질문을 intent로 분류합니다.

    Parameters
    ----------
    question:
        사용자가 입력한 자연어 질문입니다.

    use_openai:
        True이면 OpenAI API를 먼저 사용합니다.
        False이면 rule-based classifier만 사용합니다.

        테스트에서는 실제 API 호출을 피하기 위해 False로 둘 수 있습니다.

    Returns
    -------
    IntentClassificationResult
        intent, confidence, reason, source 등을 포함한 분류 결과입니다.

    처리 흐름
    --------
    1. 질문이 비어 있으면 unknown 반환
    2. use_openai=True이면 OpenAI intent classification 시도
    3. OpenAI 성공 시 결과 반환
    4. OpenAI 실패 시 rule-based fallback 반환
    5. use_openai=False이면 바로 rule-based 결과 반환
    """

    normalized_question = question.strip() if isinstance(question, str) else ""

    if not normalized_question:
        return IntentClassificationResult(
            intent="unknown",
            confidence=0.0,
            reason="질문이 비어 있어 intent를 분류할 수 없습니다.",
            source="validation",
            raw_response=None,
            error="empty_question",
        )

    if not use_openai:
        return classify_intent_rule_based(normalized_question)

    openai_result = classify_intent_with_openai(normalized_question)

    # OpenAI 호출이 성공했고, 별도 error가 없다면 그대로 사용합니다.
    if openai_result.error is None:
        return openai_result

    # OpenAI 호출, JSON 파싱, schema 검증 중 하나라도 실패하면
    # 기존 manufacturing-mcp-agent에서 사용했던 접근처럼
    # rule-based fallback으로 최소한의 intent 분류를 수행합니다.
    fallback_result = classify_intent_rule_based(normalized_question)

    return IntentClassificationResult(
        intent=fallback_result.intent,
        confidence=fallback_result.confidence,
        reason=(
            f"OpenAI intent 분류가 실패하여 rule-based fallback을 사용했습니다. "
            f"Fallback reason: {fallback_result.reason}"
        ),
        source="fallback",
        raw_response=openai_result.raw_response,
        error=openai_result.error,
    )


def classify_intent_with_openai(question: str) -> IntentClassificationResult:
    """
    OpenAI API를 사용해 intent를 분류합니다.

    이 함수는 실제 외부 API를 호출합니다.

    주의
    ----
    - API key를 코드에 직접 쓰지 않습니다.
    - .env 또는 OS 환경변수에 OPENAI_API_KEY를 저장합니다.
    - 테스트에서는 이 함수를 직접 호출하지 않고 monkeypatch로 대체합니다.
    """

    # .env 파일을 사용하기 위해 python-dotenv를 지연 import합니다.
    #
    # 왜 파일 상단에서 import하지 않는가?
    # ---------------------------------
    # 테스트 환경이나 일부 실행 환경에서 python-dotenv가 아직 설치되지 않았을 수 있습니다.
    # 이 경우에도 rule-based fallback은 동작할 수 있어야 합니다.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # dotenv 로딩 실패는 치명적인 오류가 아닙니다.
        # OS 환경변수로 OPENAI_API_KEY가 이미 설정되어 있을 수도 있기 때문입니다.
        pass

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return IntentClassificationResult(
            intent="unknown",
            confidence=0.0,
            reason="OPENAI_API_KEY가 없어 OpenAI intent 분류를 수행할 수 없습니다.",
            source="openai",
            raw_response=None,
            error="missing_openai_api_key",
        )

    model_name = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": _build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": _build_user_prompt(question),
                },
            ],
            temperature=0,
            response_format=_build_response_format_schema(),
        )

        raw_content = response.choices[0].message.content

        if raw_content is None:
            return IntentClassificationResult(
                intent="unknown",
                confidence=0.0,
                reason="OpenAI 응답 content가 비어 있습니다.",
                source="openai",
                raw_response=None,
                error="empty_openai_response",
            )

        parsed = json.loads(raw_content)

        return validate_intent_payload(
            payload=parsed,
            source="openai",
            raw_response=raw_content,
        )

    except Exception as exc:
        return IntentClassificationResult(
            intent="unknown",
            confidence=0.0,
            reason="OpenAI intent 분류 중 오류가 발생했습니다.",
            source="openai",
            raw_response=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def classify_intent_rule_based(question: str) -> IntentClassificationResult:
    """
    rule-based 방식으로 intent를 분류합니다.

    이 함수는 OpenAI API가 실패했을 때 fallback으로 사용됩니다.

    기존 manufacturing-mcp-agent의 rule-based routing과 비슷한 개념이지만,
    여기서는 Day 13의 intent 3개에 맞춰 아주 단순하게 시작합니다.
    """

    q = question.lower()

    # 고장 예측 관련 표현
    failure_keywords = [
        "고장",
        "위험",
        "예측",
        "failure",
        "predict",
        "prediction",
        "risk",
        "설비",
        "장비",
        "machine",
        "maintenance",
        "tool wear",
        "torque",
        "rpm",
        "temperature",
    ]

    if any(keyword in q for keyword in failure_keywords):
        return IntentClassificationResult(
            intent="failure_prediction",
            confidence=0.65,
            reason="질문에 고장 예측 또는 설비 위험도와 관련된 키워드가 포함되어 있습니다.",
            source="rule_based",
        )

    # 데이터셋 스키마 관련 표현
    schema_keywords = [
        "데이터셋",
        "컬럼",
        "feature",
        "features",
        "target",
        "schema",
        "ai4i",
        "입력값",
        "변수",
        "특성",
    ]

    if any(keyword in q for keyword in schema_keywords):
        return IntentClassificationResult(
            intent="dataset_schema_query",
            confidence=0.65,
            reason="질문에 데이터셋 schema, feature, target과 관련된 키워드가 포함되어 있습니다.",
            source="rule_based",
        )

    return IntentClassificationResult(
        intent="unknown",
        confidence=0.3,
        reason="현재 지원하는 intent로 분류하기 어렵습니다.",
        source="rule_based",
    )


def validate_intent_payload(
    *,
    payload: dict[str, Any],
    source: str,
    raw_response: str | None = None,
) -> IntentClassificationResult:
    """
    OpenAI 또는 다른 classifier가 반환한 payload를 검증합니다.

    왜 검증이 필요한가?
    ------------------
    LLM에게 JSON으로 답하라고 요청해도,
    운영 환경에서는 다음 문제가 발생할 수 있습니다.

    1. intent key가 없을 수 있다.
    2. confidence가 문자열로 올 수 있다.
    3. 지원하지 않는 intent가 올 수 있다.
    4. reason이 비어 있을 수 있다.
    5. confidence가 0~1 범위를 벗어날 수 있다.

    따라서 LangGraph state에 넣기 전에 반드시 정리합니다.
    """

    if not isinstance(payload, dict):
        return IntentClassificationResult(
            intent="unknown",
            confidence=0.0,
            reason="intent payload가 dict 형식이 아닙니다.",
            source=source,
            raw_response=raw_response,
            error="invalid_payload_type",
        )

    raw_intent = payload.get("intent", "unknown")
    raw_confidence = payload.get("confidence", 0.0)
    raw_reason = payload.get("reason", "")

    intent = str(raw_intent).strip()

    if intent not in SUPPORTED_INTENTS:
        return IntentClassificationResult(
            intent="unknown",
            confidence=0.0,
            reason=f"지원하지 않는 intent가 반환되었습니다: {intent}",
            source=source,
            raw_response=raw_response,
            error="unsupported_intent",
        )

    confidence = _normalize_confidence(raw_confidence)

    reason = str(raw_reason).strip()
    if not reason:
        reason = "분류 이유가 제공되지 않았습니다."

    return IntentClassificationResult(
        intent=intent,
        confidence=confidence,
        reason=reason,
        source=source,
        raw_response=raw_response,
        error=None,
    )


def _normalize_confidence(value: Any) -> float:
    """
    confidence 값을 0.0 ~ 1.0 사이 float으로 정규화합니다.

    LLM이 confidence를 숫자 대신 문자열로 반환할 수도 있으므로
    float 변환을 시도합니다.
    """

    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if confidence < 0.0:
        return 0.0

    if confidence > 1.0:
        return 1.0

    return confidence


def _build_system_prompt() -> str:
    """
    OpenAI intent classifier용 system prompt를 생성합니다.

    핵심 원칙
    --------
    - 모델은 예측을 직접 하지 않습니다.
    - 모델은 intent만 분류합니다.
    - 모델은 반드시 JSON schema에 맞춰 응답해야 합니다.
    """

    return """
당신은 제조 AI Agent의 intent classifier입니다.

역할:
- 사용자의 자연어 질문을 보고 어떤 작업 intent인지 분류합니다.
- 고장 확률을 직접 예측하지 않습니다.
- feature contribution이나 SHAP 설명을 직접 생성하지 않습니다.
- 오직 intent, confidence, reason만 반환합니다.

지원 intent:
1. failure_prediction
   - 설비 입력값을 바탕으로 고장 위험 예측을 요청하는 질문
   - 예: "이 설비 조건이면 고장 위험이 높아?"
   - 예: "Torque 62, Tool wear 220이면 위험해?"

2. dataset_schema_query
   - AI4I 데이터셋의 feature, target, 컬럼 의미를 묻는 질문
   - 예: "AI4I feature가 뭐야?"
   - 예: "target 컬럼은 뭐야?"

3. unknown
   - 현재 지원하지 않는 질문
   - 입력이 부족해서 판단하기 어려운 질문
   - 제조 고장 예측 또는 데이터셋 설명과 관련이 약한 질문

반드시 JSON schema에 맞춰 응답하세요.
""".strip()


def _build_user_prompt(question: str) -> str:
    """
    사용자의 질문을 classifier에게 전달할 prompt로 변환합니다.
    """

    return f"""
다음 사용자 질문을 intent로 분류하세요.

사용자 질문:
{question}
""".strip()


def _build_response_format_schema() -> dict[str, Any]:
    """
    OpenAI Chat Completions API의 response_format에 전달할 JSON Schema입니다.

    이 schema를 사용하면 모델이 아래 형태에 맞춰 응답하도록 강하게 유도할 수 있습니다.

    {
      "intent": "failure_prediction",
      "confidence": 0.92,
      "reason": "사용자가 설비 조건 기반 고장 위험 예측을 요청했습니다."
    }
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "intent_classification_result",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_INTENTS),
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "reason": {
                        "type": "string",
                    },
                },
                "required": ["intent", "confidence", "reason"],
            },
        },
    }