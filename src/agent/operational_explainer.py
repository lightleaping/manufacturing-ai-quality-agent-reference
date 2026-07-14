# src/agent/operational_explainer.py

"""
OpenAI 기반 설비 고장 Prediction 운영 해설 모듈입니다.

역할:
- 이미 확정된 Prediction 결과를 읽습니다.
- Evidence를 현장 담당자가 이해하기 쉬운 한국어로 설명합니다.
- 주요 점검 신호와 우선 확인 항목을 구조화합니다.
- SHAP·Global Importance 해석 주의사항을 제공합니다.

중요:
- Prediction을 생성하지 않습니다.
- Probability를 다시 계산하지 않습니다.
- Threshold를 변경하지 않습니다.
- Risk Level을 다시 판단하지 않습니다.
- Evidence를 새로 계산하지 않습니다.
- OpenAI는 기존 결과의 설명 계층으로만 사용합니다.
"""

from __future__ import annotations

import json
import os
from dataclasses import (
    asdict,
    dataclass,
)
from typing import Any


DEFAULT_OPERATIONAL_EXPLANATION_MODEL = (
    "gpt-4o-mini"
)

MAX_CONTEXT_EVIDENCE_ITEMS = 12

MAX_LIST_ITEMS = 5

MAX_SUMMARY_LENGTH = 1200

MAX_ITEM_LENGTH = 400

MAX_CAUTION_LENGTH = 800


@dataclass
class OperationalExplanationResult:
    """
    OpenAI 운영 해설 결과입니다.

    error가 None이면 정상 결과입니다.

    error가 문자열이면
    OpenAI 호출 또는 응답 검증에 실패한 상태입니다.
    """

    summary: str

    key_signals: list[str]

    recommended_checks: list[str]

    caution: str

    source: str = "openai"

    model: str | None = None

    error: str | None = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        API Response 생성에 사용할 dict로 변환합니다.
        """

        return asdict(
            self,
        )


def _normalize_text(
    value: Any,
    *,
    max_length: int,
) -> str:
    """
    OpenAI 문자열을 안전한 표시 길이로 정규화합니다.
    """

    if not isinstance(
        value,
        str,
    ):
        return ""

    normalized = value.strip()

    if not normalized:
        return ""

    return normalized[
        :max_length
    ]


def _normalize_text_list(
    value: Any,
) -> list[str]:
    """
    OpenAI 문자열 목록을 정규화합니다.

    빈 항목은 제거하고,
    최대 항목 수와 항목 길이를 제한합니다.
    """

    if not isinstance(
        value,
        list,
    ):
        return []

    normalized_items: list[str] = []

    for item in value:
        normalized = _normalize_text(
            item,
            max_length=MAX_ITEM_LENGTH,
        )

        if not normalized:
            continue

        normalized_items.append(
            normalized,
        )

        if (
            len(
                normalized_items,
            )
            >= MAX_LIST_ITEMS
        ):
            break

    return normalized_items


def _build_error_result(
    *,
    error: str,
    model: str | None,
) -> OperationalExplanationResult:
    """
    OpenAI 실패를 구조화된 결과로 반환합니다.

    예외를 Dashboard까지 그대로 전파하지 않습니다.
    """

    return OperationalExplanationResult(
        summary="",
        key_signals=[],
        recommended_checks=[],
        caution="",
        source="openai",
        model=model,
        error=error,
    )


def validate_operational_explanation_payload(
    *,
    payload: Any,
    model: str | None,
) -> OperationalExplanationResult:
    """
    OpenAI JSON Payload를 검증하고 정규화합니다.

    LLM JSON을 바로 신뢰하지 않고
    필수 필드와 자료형을 확인합니다.
    """

    if not isinstance(
        payload,
        dict,
    ):
        return _build_error_result(
            error=(
                "invalid_operational_explanation_payload: "
                "OpenAI 응답의 최상위 구조가 JSON object가 아닙니다."
            ),
            model=model,
        )

    summary = _normalize_text(
        payload.get(
            "summary",
        ),
        max_length=MAX_SUMMARY_LENGTH,
    )

    key_signals = _normalize_text_list(
        payload.get(
            "key_signals",
        )
    )

    recommended_checks = (
        _normalize_text_list(
            payload.get(
                "recommended_checks",
            )
        )
    )

    caution = _normalize_text(
        payload.get(
            "caution",
        ),
        max_length=MAX_CAUTION_LENGTH,
    )

    missing_fields: list[str] = []

    if not summary:
        missing_fields.append(
            "summary",
        )

    if not key_signals:
        missing_fields.append(
            "key_signals",
        )

    if not recommended_checks:
        missing_fields.append(
            "recommended_checks",
        )

    if not caution:
        missing_fields.append(
            "caution",
        )

    if missing_fields:
        return _build_error_result(
            error=(
                "invalid_operational_explanation_payload: "
                "필수 내용이 비어 있습니다. "
                "fields="
                + ", ".join(
                    missing_fields,
                )
            ),
            model=model,
        )

    return OperationalExplanationResult(
        summary=summary,
        key_signals=key_signals,
        recommended_checks=(
            recommended_checks
        ),
        caution=caution,
        source="openai",
        model=model,
        error=None,
    )




def _build_system_prompt() -> str:
    """
    Build a beginner-first operational explanation prompt.

    The source prompt is intentionally written with ASCII characters only.
    OpenAI must return natural Korean for every user-facing field.
    """

    return """
You explain an already confirmed manufacturing AI prediction
to a person who has never used an AI equipment-risk system.

Assume that the reader does not know artificial intelligence,
manufacturing prediction, probability, thresholds, SHAP,
feature importance, model outputs, or evidence.

Follow every rule below.

1. Use only the supplied prediction result and evidence.
2. Never change prediction, probability, threshold, risk level,
   recommended action, or evidence.
3. Never calculate a new probability, score, or risk level.
4. Never say that a failure definitely happened.
5. Never present an input value as a proven physical failure cause.
6. Write every user-facing field in natural Korean.
7. Use short sentences and familiar words.
8. Explain the practical meaning before any technical idea.
9. Do not use unexplained English technical terms.
10. Do not use the words logit, contribution, permutation,
    or sensitivity in user-facing text.
11. Explain rule-based evidence as a value that matched
    a person-defined inspection rule.
12. Explain local SHAP evidence as the direction in which
    the current input raised or lowered the AI risk judgment.
13. Explain global importance as an input that the AI often
    relied on across the whole reference dataset.
14. Do not use category labels, bracketed prefixes, headings,
    markdown headings, or tags inside key signal items.
15. Do not use emojis or decorative symbols.
16. Use a simple comparison or example when it helps,
    but clearly state that the comparison is only for understanding.
17. Explain that the result is an AI prediction,
    not a confirmed equipment diagnosis.
18. Make each item understandable on its own.
19. Return only JSON that matches the required JSON Schema.
""".strip()


def _build_response_format_schema() -> dict[str, Any]:
    """
    OpenAI Structured Output JSON Schema를 생성합니다.
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": (
                "failure_operational_explanation"
            ),
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                    },
                    "key_signals": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "recommended_checks": {
                        "type": "array",
                        "items": {
                            "type": "string",
                        },
                    },
                    "caution": {
                        "type": "string",
                    },
                },
                "required": [
                    "summary",
                    "key_signals",
                    "recommended_checks",
                    "caution",
                ],
                "additionalProperties": False,
            },
        },
    }


def _build_prediction_context(
    prediction_result: dict[str, Any],
) -> str:
    """
    OpenAI에 전달할 Prediction Context를 생성합니다.

    전체 내부 객체를 무제한 전달하지 않고,
    운영 해설에 필요한 필드만 선택합니다.
    """

    raw_evidence = prediction_result.get(
        "evidence",
    )

    if not isinstance(
        raw_evidence,
        list,
    ):
        raw_evidence = []

    selected_evidence: list[
        dict[str, Any]
    ] = []

    for item in raw_evidence[
        :MAX_CONTEXT_EVIDENCE_ITEMS
    ]:
        if not isinstance(
            item,
            dict,
        ):
            continue

        selected_evidence.append(
            {
                "evidence_type": (
                    item.get(
                        "evidence_type",
                    )
                ),
                "title": item.get(
                    "title",
                ),
                "summary": item.get(
                    "summary",
                ),
                "feature": item.get(
                    "feature",
                ),
                "value": item.get(
                    "value",
                ),
                "direction": item.get(
                    "direction",
                ),
                "contribution": item.get(
                    "contribution",
                ),
                "importance": item.get(
                    "importance",
                ),
                "severity": item.get(
                    "severity",
                ),
            }
        )

    context = {
        "prediction": (
            prediction_result.get(
                "prediction",
            )
        ),
        "probability": (
            prediction_result.get(
                "probability",
            )
        ),
        "threshold": (
            prediction_result.get(
                "threshold",
            )
        ),
        "risk_level": (
            prediction_result.get(
                "risk_level",
            )
        ),
        "recommended_action": (
            prediction_result.get(
                "recommended_action",
            )
        ),
        "evidence": selected_evidence,
        "warnings": (
            prediction_result.get(
                "warnings",
                [],
            )
        ),
        "limitations": (
            prediction_result.get(
                "limitations",
                [],
            )
        ),
    }

    return json.dumps(
        context,
        ensure_ascii=False,
        indent=2,
        default=str,
    )




def _build_user_prompt(
    prediction_result: dict[str, Any],
) -> str:
    """
    Build a detailed but beginner-first prompt for the confirmed result.
    """

    prediction_context = (
        _build_prediction_context(
            prediction_result,
        )
    )

    return f"""
Explain the confirmed result below
to a first-time user with no AI or manufacturing background.

Write every user-facing field in natural Korean.

General writing rules:
- Use short sentences.
- Put the conclusion before technical detail.
- Prefer familiar Korean words.
- Do not expose internal field names such as
  summary, metadata, key_signals, recommended_checks, or caution.
- Do not use category labels, bracketed prefixes, markdown headings,
  tags, emojis, or decorative symbols inside any returned string.
- Do not repeat the same sentence in multiple fields.
- Do not change any confirmed prediction value.
- Do not invent a new measurement, threshold, diagnosis, or cause.

summary:
- Write 4 to 6 short sentences.
- Sentence 1: state the current result in plain language.
- Sentence 2: explain that the AI model score crossed
  or did not cross the decision threshold.
- Sentence 3: give one simple 100-point comparison
  only when both score and threshold exist.
- Sentence 4: explain what the user should do next.
- Final sentence: clearly say this is an AI prediction,
  not proof that a failure already happened.
- Clearly say that the displayed model percentage
  is not necessarily the real-world failure probability.

key_signals:
- Provide 2 to 4 items.
- Each item must contain 2 or 3 short Korean sentences.
- First sentence: state the observed value or condition.
- Second sentence: explain the meaning in everyday language.
- Third sentence when needed: explain why it is worth checking.
- Never call the item a confirmed physical failure cause.
- Do not start with brackets, labels, headings, or evidence type names.

recommended_checks:
- Provide 2 to 4 items in practical order.
- Each item must contain 2 short Korean sentences.
- First sentence: state exactly what to check.
- Second sentence: explain why that check is relevant
  to the confirmed evidence.
- Do not claim that a check confirms a diagnosis.
- Do not invent a measurement procedure not supplied in the context.

caution:
- Write 2 or 3 short Korean sentences.
- Explain that AI shows a prediction,
  not a confirmed equipment diagnosis.
- Explain that input influence and dataset importance
  are interpretation references,
  not proof of a physical failure cause.
- Explain that actual action should also follow
  the equipment manual and the site's inspection standard.

Confirmed Prediction Context:
{prediction_context}
""".strip()


def generate_operational_explanation_with_openai(
    prediction_result: dict[str, Any],
) -> OperationalExplanationResult:
    """
    OpenAI API를 사용하여 운영 해설을 생성합니다.

    실패 시 예외를 외부로 던지지 않고
    error가 포함된 구조화 결과를 반환합니다.
    """

    try:
        from dotenv import load_dotenv

        load_dotenv()

    except Exception:
        pass

    model_name = (
        os.getenv(
            "OPENAI_MODEL",
            DEFAULT_OPERATIONAL_EXPLANATION_MODEL,
        )
        or DEFAULT_OPERATIONAL_EXPLANATION_MODEL
    ).strip()

    api_key = (
        os.getenv(
            "OPENAI_API_KEY",
        )
        or ""
    ).strip()

    if not api_key:
        return _build_error_result(
            error=(
                "missing_openai_api_key: "
                "OPENAI_API_KEY가 없어 "
                "AI 운영 해설을 생성할 수 없습니다."
            ),
            model=model_name,
        )

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
        )

        response = (
            client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            _build_system_prompt()
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            _build_user_prompt(
                                prediction_result,
                            )
                        ),
                    },
                ],
                temperature=0,
                response_format=(
                    _build_response_format_schema()
                ),
            )
        )

        raw_content = (
            response
            .choices[0]
            .message
            .content
        )

        if raw_content is None:
            return _build_error_result(
                error=(
                    "empty_openai_response: "
                    "OpenAI 운영 해설 응답이 비어 있습니다."
                ),
                model=model_name,
            )

        parsed = json.loads(
            raw_content,
        )

        return (
            validate_operational_explanation_payload(
                payload=parsed,
                model=model_name,
            )
        )

    except Exception as exc:
        return _build_error_result(
            error=(
                "OpenAI 운영 해설 생성 중 "
                "오류가 발생했습니다. "
                f"{type(exc).__name__}: {exc}"
            ),
            model=model_name,
        )
