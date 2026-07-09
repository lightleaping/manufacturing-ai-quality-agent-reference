"""
Day 13 - intent_classifier 테스트

이 테스트 파일의 목표
---------------------
src/agent/intent_classifier.py가 안전하게 동작하는지 검증합니다.

중요한 원칙
-----------
1. 테스트에서는 실제 OpenAI API를 호출하지 않습니다.
2. OpenAI 호출이 필요한 흐름은 monkeypatch로 가짜 함수로 대체합니다.
3. JSON payload 검증이 제대로 되는지 확인합니다.
4. OpenAI 실패 시 rule-based fallback이 동작하는지 확인합니다.
5. 지원하지 않는 intent가 들어오면 unknown으로 바뀌는지 확인합니다.

왜 실제 OpenAI API를 테스트에서 호출하지 않는가?
-----------------------------------------------
테스트는 항상 빠르고, 안정적이고, 반복 가능해야 합니다.

실제 API를 호출하면 다음 문제가 생깁니다.

- API key가 없는 환경에서 테스트 실패
- 네트워크 상태에 따라 테스트 실패
- 사용량 비용 발생
- OpenAI 응답 변화에 따라 테스트 결과 흔들림

따라서 단위 테스트에서는 외부 API를 직접 호출하지 않고,
"OpenAI 함수가 이런 값을 반환했다고 가정했을 때"
우리 코드가 올바르게 처리하는지를 검증합니다.
"""

from src.agent import intent_classifier
from src.agent.intent_classifier import (
    IntentClassificationResult,
    classify_intent,
    classify_intent_rule_based,
    validate_intent_payload,
)


def test_validate_intent_payload_accepts_valid_payload():
    """
    정상 payload가 들어오면 IntentClassificationResult로 변환되어야 합니다.

    이 테스트는 OpenAI가 올바른 JSON을 반환한 상황을 가정합니다.
    """

    payload = {
        "intent": "failure_prediction",
        "confidence": 0.92,
        "reason": "사용자가 설비 조건 기반 고장 위험 예측을 요청했습니다.",
    }

    result = validate_intent_payload(
        payload=payload,
        source="openai",
        raw_response='{"intent": "failure_prediction"}',
    )

    assert result.intent == "failure_prediction"
    assert result.confidence == 0.92
    assert result.reason == "사용자가 설비 조건 기반 고장 위험 예측을 요청했습니다."
    assert result.source == "openai"
    assert result.error is None
    assert result.raw_response == '{"intent": "failure_prediction"}'


def test_validate_intent_payload_rejects_unsupported_intent():
    """
    지원하지 않는 intent가 들어오면 unknown으로 바꿔야 합니다.

    LLM은 가끔 우리가 정의하지 않은 intent를 만들 수 있습니다.
    예를 들어 "maintenance_schedule" 같은 값을 반환할 수 있습니다.

    하지만 현재 Day 13에서 지원하는 intent는
    failure_prediction, dataset_schema_query, unknown뿐입니다.

    따라서 지원하지 않는 intent는 그대로 믿지 않고 unknown으로 바꿉니다.
    """

    payload = {
        "intent": "maintenance_schedule",
        "confidence": 0.88,
        "reason": "사용자가 정비 일정을 묻고 있습니다.",
    }

    result = validate_intent_payload(
        payload=payload,
        source="openai",
        raw_response='{"intent": "maintenance_schedule"}',
    )

    assert result.intent == "unknown"
    assert result.confidence == 0.0
    assert result.source == "openai"
    assert result.error == "unsupported_intent"
    assert "지원하지 않는 intent" in result.reason


def test_validate_intent_payload_normalizes_confidence_range():
    """
    confidence는 0.0 ~ 1.0 사이로 정규화되어야 합니다.

    LLM이 실수로 1.5 같은 값을 반환하더라도
    우리 코드에서는 1.0으로 제한해야 합니다.
    """

    payload = {
        "intent": "dataset_schema_query",
        "confidence": 1.5,
        "reason": "사용자가 데이터셋 컬럼을 질문했습니다.",
    }

    result = validate_intent_payload(
        payload=payload,
        source="openai",
    )

    assert result.intent == "dataset_schema_query"
    assert result.confidence == 1.0
    assert result.error is None


def test_validate_intent_payload_handles_string_confidence():
    """
    confidence가 문자열로 와도 float으로 변환할 수 있으면 허용합니다.

    예:
    "0.75" -> 0.75
    """

    payload = {
        "intent": "failure_prediction",
        "confidence": "0.75",
        "reason": "설비 위험 예측 질문입니다.",
    }

    result = validate_intent_payload(
        payload=payload,
        source="openai",
    )

    assert result.intent == "failure_prediction"
    assert result.confidence == 0.75
    assert result.error is None


def test_validate_intent_payload_handles_invalid_payload_type():
    """
    payload가 dict가 아니면 unknown으로 처리해야 합니다.

    LLM 응답 파싱 과정에서 예상과 다른 타입이 들어올 수 있으므로
    방어적으로 처리합니다.
    """

    # validate_intent_payload()는 원래 payload에 dict가 들어오길 기대합니다.
    #
    # 예를 들면 원래 정상 입력은 이런 형태입니다.
    #
    # payload = {
    #     "intent": "failure_prediction",
    #     "confidence": 0.92,
    #     "reason": "사용자가 고장 위험 예측을 요청했습니다.",
    # }
    #
    # 그런데 이 테스트에서는 일부러 dict가 아니라 list를 넣습니다.
    #
    # 이유:
    #   OpenAI 응답을 파싱하거나 외부 classifier 결과를 처리하다 보면
    #   우리가 기대한 dict가 아니라 list, str, None 같은 잘못된 타입이 들어올 수도 있습니다.
    #
    # 이때 함수가 에러로 죽으면 안 됩니다.
    # 대신 "잘못된 payload다"라고 판단하고
    # intent="unknown", error="invalid_payload_type" 같은 안전한 결과를 반환해야 합니다.
    result = validate_intent_payload(
        # payload는 intent 분류 결과 원본입니다.
        #
        # 원래는 dict가 들어와야 하지만,
        # 여기서는 방어 로직을 테스트하기 위해 일부러 list를 넣었습니다.
        payload=["not", "a", "dict"],

        # type: ignore[arg-type]는 타입 검사 도구에게
        # "여기서 일부러 잘못된 타입을 넣는 것이니 경고하지 말라"는 뜻입니다.
        #
        # validate_intent_payload() 함수 정의는 payload: dict[str, Any]처럼
        # dict 타입을 기대하고 있을 가능성이 높습니다.
        #
        # 그런데 테스트 목적상 list를 넣으면 mypy 같은 타입 검사 도구가
        # "dict가 들어가야 하는데 list를 넣었다"고 경고할 수 있습니다.
        #
        # 이 테스트는 그 잘못된 입력을 일부러 넣는 것이므로,
        # type: ignore[arg-type]로 타입 경고를 무시합니다.
        # payload=["not", "a", "dict"],  # type: ignore[arg-type]

        # source는 이 payload가 어디서 온 것인지 표시합니다.
        #
        # 여기서는 OpenAI 응답을 검증하는 상황을 가정하므로
        # source="openai"라고 넣습니다.
        source="openai",
    )

    assert result.intent == "unknown"
    assert result.confidence == 0.0
    assert result.error == "invalid_payload_type"


def test_classify_intent_returns_unknown_for_empty_question():
    """
    질문이 비어 있으면 OpenAI나 rule-based classifier를 호출할 필요 없이
    바로 unknown을 반환해야 합니다.
    """

    result = classify_intent("   ", use_openai=False)

    assert result.intent == "unknown"
    assert result.confidence == 0.0
    assert result.source == "validation"
    assert result.error == "empty_question"


def test_classify_intent_rule_based_detects_failure_prediction():
    """
    rule-based classifier가 고장 예측 질문을 failure_prediction으로 분류하는지 확인합니다.
    """

    result = classify_intent_rule_based(
        "Torque 62이고 Tool wear 220이면 고장 위험이 높아?"
    )

    assert result.intent == "failure_prediction"
    assert result.confidence > 0
    assert result.source == "rule_based"
    assert result.error is None


def test_classify_intent_rule_based_detects_dataset_schema_query():
    """
    rule-based classifier가 데이터셋 schema 질문을 dataset_schema_query로 분류하는지 확인합니다.
    """

    result = classify_intent_rule_based(
        "AI4I 데이터셋의 feature와 target 컬럼은 뭐야?"
    )

    assert result.intent == "dataset_schema_query"
    assert result.confidence > 0
    assert result.source == "rule_based"
    assert result.error is None


def test_classify_intent_rule_based_returns_unknown_for_unrelated_question():
    """
    현재 지원하지 않는 질문은 unknown으로 분류되어야 합니다.
    """

    result = classify_intent_rule_based(
        "오늘 점심 메뉴 추천해줘."
    )

    assert result.intent == "unknown"
    assert result.source == "rule_based"
    assert result.error is None


def test_classify_intent_uses_openai_result_when_openai_succeeds(monkeypatch):
    """
    OpenAI intent classifier가 성공하면 classify_intent()는 그 결과를 그대로 사용해야 합니다.

    여기서 중요한 점:
    실제 OpenAI API를 호출하지 않습니다.

    대신 src.agent.intent_classifier.classify_intent_with_openai 함수를
    fake 함수로 바꿉니다.

    monkeypatch는 "실제 실행 흐름에서 참조되는 위치"에 적용해야 합니다.
    classify_intent() 함수 내부에서는 같은 모듈의 classify_intent_with_openai를 호출하므로,
    intent_classifier.classify_intent_with_openai 위치를 바꿉니다.
    """

    def fake_classify_intent_with_openai(question: str) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent="failure_prediction",
            confidence=0.91,
            reason="OpenAI mock 응답입니다.",
            source="openai",
            raw_response='{"intent": "failure_prediction"}',
            error=None,
        )

    monkeypatch.setattr(
        intent_classifier,
        "classify_intent_with_openai",
        fake_classify_intent_with_openai,
    )

    result = classify_intent(
        "이 설비 조건이면 고장 위험이 높아?",
        use_openai=True,
    )

    assert result.intent == "failure_prediction"
    assert result.confidence == 0.91
    assert result.reason == "OpenAI mock 응답입니다."
    assert result.source == "openai"
    assert result.error is None


def test_classify_intent_falls_back_when_openai_fails(monkeypatch):
    """
    OpenAI intent classifier가 실패하면 rule-based fallback으로 넘어가야 합니다.

    예:
    - API key 없음
    - 네트워크 오류
    - JSON 파싱 실패
    - OpenAI 응답 형식 오류

    이 경우 Agent 전체가 죽으면 안 됩니다.
    최소한 rule-based 방식으로 intent를 분류하고,
    source는 fallback으로 남겨야 합니다.
    """

    # 1. 테스트용 가짜 OpenAI 함수 정의
    def fake_classify_intent_with_openai(question: str) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent="unknown",
            confidence=0.0,
            reason="OpenAI mock 실패입니다.",
            source="openai",
            raw_response=None,
            error="mock_openai_error",
        )

    # monkeypatch는 pytest에서 제공하는 테스트용 기능입니다.
    #
    # 역할:
    #   테스트가 실행되는 동안에만
    #   특정 함수, 변수, 객체를 임시로 다른 것으로 바꿔줍니다.
    #
    # 여기서는 실제 OpenAI API를 호출하는 함수인
    # classify_intent_with_openai()를
    # 테스트용 가짜 함수 fake_classify_intent_with_openai()로 바꿉니다.
    #
    # 왜 이렇게 하는가?
    #   테스트에서 진짜 OpenAI API를 호출하면 문제가 많습니다.
    #
    #   1. API key가 없는 환경에서는 테스트가 실패합니다.
    #   2. 인터넷 연결 상태에 따라 테스트가 실패할 수 있습니다.
    #   3. API 사용 비용이 발생할 수 있습니다.
    #   4. OpenAI 응답이 매번 조금씩 달라지면 테스트 결과가 흔들릴 수 있습니다.
    #
    # 그래서 단위 테스트에서는 외부 API를 직접 호출하지 않고,
    # "OpenAI가 이런 결과를 반환했다고 가정했을 때
    # 우리 코드가 제대로 동작하는가?"만 검증합니다.
    #
    # 즉, 테스트의 목적은 OpenAI 성능을 확인하는 것이 아니라,
    # 우리 코드의 처리 흐름을 확인하는 것입니다.

    # 2. 실제 OpenAI 호출 함수를 가짜 함수로 임시 교체
    monkeypatch.setattr(
        # intent_classifier 모듈 안에 있는 함수를 바꿉니다.
        intent_classifier,

        # 바꿀 대상 함수 이름입니다.
        # 원래는 classify_intent() 내부에서 이 함수가 호출됩니다.
        "classify_intent_with_openai",

        # 실제 OpenAI 호출 함수 대신 사용할 테스트용 가짜 함수입니다.
        fake_classify_intent_with_openai,
    )

    # 3. classify_intent() 실행
    result = classify_intent(
        "Torque 62이고 Tool wear 220이면 고장 위험이 높아?",
        use_openai=True,
    )

    # 4. classify_intent() 내부에서는 원래 classify_intent_with_openai()를 호출해야 하지만,
    #    테스트 중에는 monkeypatch 때문에 fake_classify_intent_with_openai()가 대신 호출됩니다.
    assert result.intent == "failure_prediction"
    assert result.confidence > 0
    assert result.source == "fallback"
    assert result.error == "mock_openai_error"
    assert "OpenAI intent 분류가 실패" in result.reason


def test_classify_intent_can_disable_openai(monkeypatch):
    """
    use_openai=False이면 OpenAI 함수를 호출하지 않고 바로 rule-based classifier를 사용해야 합니다.

    이 테스트는 실수로 OpenAI 호출이 발생하지 않는지 확인합니다.
    """

    def fake_classify_intent_with_openai(question: str) -> IntentClassificationResult:
        raise AssertionError("use_openai=False인데 OpenAI 함수가 호출되었습니다.")

    monkeypatch.setattr(
        intent_classifier,
        "classify_intent_with_openai",
        fake_classify_intent_with_openai,
    )

    result = classify_intent(
        "AI4I 데이터셋 feature 알려줘.",
        use_openai=False,
    )

    assert result.intent == "dataset_schema_query"
    assert result.source == "rule_based"


def test_intent_classification_result_to_dict():
    """
    IntentClassificationResult.to_dict()가 LangGraph state에 넣기 쉬운 dict를 반환하는지 확인합니다.
    """

    result = IntentClassificationResult(
        intent="failure_prediction",
        confidence=0.8,
        reason="테스트 reason",
        source="rule_based",
        raw_response=None,
        error=None,
    )

    result_dict = result.to_dict()

    assert result_dict["intent"] == "failure_prediction"
    assert result_dict["confidence"] == 0.8
    assert result_dict["reason"] == "테스트 reason"
    assert result_dict["source"] == "rule_based"
    assert result_dict["raw_response"] is None
    assert result_dict["error"] is None