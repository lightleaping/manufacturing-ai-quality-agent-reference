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
from src.agent.state import ChatMessage

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

    def fake_classify_intent_with_openai(
        question: str,
        *,
        # Day 15부터 실제 classify_intent_with_openai() 함수는
        # 현재 질문뿐 아니라 이전 대화 기록도 받을 수 있습니다.
        #
        # 따라서 테스트용 fake 함수도
        # 실제 함수와 같은 interface를 가져야 합니다.
        #
        # 현재 이 테스트는 OpenAI 성공 결과를 검증하는 것이 목적이므로
        # chat_history 값을 직접 사용하지는 않습니다.
        #
        # 하지만 운영 코드가 다음처럼 호출해도:
        #
        #     classify_intent_with_openai(
        #         question,
        #         chat_history=chat_history,
        #     )
        #
        # fake 함수가 같은 keyword argument를 받을 수 있어야
        # TypeError 없이 실제 호출 구조를 대신할 수 있습니다.
        chat_history: list[ChatMessage] | None = None,
    ) -> IntentClassificationResult:
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
    def fake_classify_intent_with_openai(
        question: str,
        *,
        # 실제 OpenAI 함수가 Day 15부터
        # chat_history keyword argument를 받으므로,
        # 실패 상황을 만드는 fake 함수도
        # 같은 호출 interface를 유지합니다.
        #
        # monkeypatch는 함수 이름만 바꾸는 것이 아니라
        # 실제 호출을 fake 함수가 대신 처리하게 만듭니다.
        #
        # 따라서 호출부가 전달하는 매개변수를
        # fake 함수도 받을 수 있어야 합니다.
        chat_history: list[ChatMessage] | None = None,
    ) -> IntentClassificationResult:
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

    def fake_classify_intent_with_openai(
        question: str,
        *,
        # 이 테스트에서는 use_openai=False이므로
        # fake 함수 자체가 호출되면 안 됩니다.
        #
        # 그래도 실제 함수의 Day 15 interface와 맞추기 위해
        # chat_history 매개변수를 동일하게 정의합니다.
        chat_history: list[ChatMessage] | None = None,
    ) -> IntentClassificationResult:
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

def test_classify_intent_passes_chat_history_to_openai_classifier(
    monkeypatch,
):
    """
    classify_intent()가 이전 chat_history를
    OpenAI classifier 함수까지 전달하는지 확인합니다.

    왜 이 테스트가 필요한가?
    -------------------------
    AgentState에 chat_history가 저장되어 있어도,
    classify_intent_with_openai()까지 전달하지 않으면
    실제 intent 분류에는 사용되지 않습니다.

    따라서 다음 연결을 직접 검증합니다.

        classify_intent()

        question
        +
        chat_history

                │

                ▼

        classify_intent_with_openai()
    """

    # fake OpenAI 함수가 실제로 받은 값을
    # 테스트 함수 밖에서도 확인하기 위한 dict입니다.
    #
    # 함수 내부의 지역 변수는
    # 함수 실행이 끝난 뒤 직접 확인하기 어렵기 때문에,
    # mutable dict에 값을 저장합니다.
    captured_arguments = {}

    def fake_classify_intent_with_openai(
        question: str,
        *,
        chat_history=None,
    ) -> IntentClassificationResult:
        # 실제 OpenAI API를 호출하지 않습니다.
        #
        # classify_intent()가 전달한
        # question과 chat_history만 저장합니다.
        captured_arguments["question"] = question
        captured_arguments["chat_history"] = chat_history

        return IntentClassificationResult(
            intent="failure_prediction",
            confidence=0.9,
            reason="multi-turn OpenAI mock 결과입니다.",
            source="openai",
            raw_response='{"intent": "failure_prediction"}',
            error=None,
        )

    monkeypatch.setattr(
        intent_classifier,
        "classify_intent_with_openai",
        fake_classify_intent_with_openai,
    )

    chat_history = [
        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?",
        },
        {
            "role": "assistant",
            "content": "현재 입력 조건에서는 고장 위험이 높게 예측되었습니다.",
        },
    ]

    result = classify_intent(
        "그건 왜 그래?",
        chat_history=chat_history,
        use_openai=True,
    )

    # OpenAI mock 결과가 정상적으로 사용되었는지 확인합니다.
    assert result.intent == "failure_prediction"
    assert result.source == "openai"

    # 현재 질문이 OpenAI classifier까지 전달되었는지 확인합니다.
    assert captured_arguments["question"] == "그건 왜 그래?"

    # 이전 대화 기록도 함께 전달되었는지 확인합니다.
    assert captured_arguments["chat_history"] == chat_history


def test_classify_intent_rule_based_uses_user_history_for_follow_up_question():
    """
    현재 질문만으로 intent를 알기 어려운 경우,
    rule-based classifier가 최근 user history를
    보조 문맥으로 사용하는지 확인합니다.

    이전 질문:
        "이 설비의 고장 위험을 예측해줘."

    현재 질문:
        "그건 왜 그래?"

    현재 질문만 보면
    고장, 위험, 예측 같은 keyword가 없습니다.

    하지만 이전 user 질문을 참고하면
    고장 예측 문맥의 후속 질문임을 알 수 있습니다.
    """

    chat_history = [
        {
            "role": "user",
            "content": "이 설비의 고장 위험을 예측해줘.",
        },
        {
            "role": "assistant",
            "content": "설비 입력값을 확인하겠습니다.",
        },
    ]

    result = classify_intent_rule_based(
        "그건 왜 그래?",
        chat_history=chat_history,
    )

    assert result.intent == "failure_prediction"

    # 현재 질문에서 직접 keyword를 찾은 것이 아니라
    # 이전 user history를 참고했으므로
    # 일반 rule-based confidence 0.65보다 낮은 0.55를 사용합니다.
    assert result.confidence == 0.55

    assert result.source == "rule_based"

    assert "최근 사용자 대화 문맥" in result.reason


def test_classify_intent_rule_based_prioritizes_current_question_over_history():
    """
    현재 질문에 명확한 intent가 있으면
    이전 chat_history보다 현재 질문을 우선해야 합니다.

    이전 질문:
        고장 위험 예측

    현재 질문:
        AI4I target 컬럼 질문

    현재 사용자가 실제로 묻는 것은
    dataset_schema_query입니다.

    history의 고장 예측 문맥 때문에
    failure_prediction으로 잘못 분류되면 안 됩니다.
    """

    chat_history = [
        {
            "role": "user",
            "content": "이 설비의 고장 위험을 예측해줘.",
        },
        {
            "role": "assistant",
            "content": "설비 입력값을 확인하겠습니다.",
        },
    ]

    result = classify_intent_rule_based(
        "AI4I target 컬럼은 뭐야?",
        chat_history=chat_history,
    )

    assert result.intent == "dataset_schema_query"

    # 현재 질문에서 직접 schema keyword를 찾았으므로
    # 기존 rule-based confidence를 유지합니다.
    assert result.confidence == 0.65

    assert result.source == "rule_based"


def test_build_user_prompt_separates_history_and_current_question():
    """
    OpenAI user prompt에서
    이전 chat_history와 현재 question이
    서로 다른 영역으로 구분되는지 확인합니다.

    왜 구분해야 하는가?
    -------------------
    이전 대화와 현재 질문을 단순히 이어 붙이면
    모델이 어디까지가 참고 문맥이고,
    무엇이 현재 분류 대상인지 혼동할 수 있습니다.

    Day 15에서는 다음 태그를 사용합니다.

        <chat_history>
        이전 대화
        </chat_history>

        <current_question>
        현재 질문
        </current_question>
    """

    chat_history = [
        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?",
        },
        {
            "role": "assistant",
            "content": "현재 입력 조건에서는 고장 위험이 높게 예측되었습니다.",
        },
    ]

    prompt = intent_classifier._build_user_prompt(
        "그건 왜 그래?",
        chat_history=chat_history,
    )

    # 이전 대화 영역이 존재해야 합니다.
    assert "<chat_history>" in prompt
    assert "</chat_history>" in prompt

    # 현재 질문 영역이 존재해야 합니다.
    assert "<current_question>" in prompt
    assert "</current_question>" in prompt

    # 이전 사용자 질문이 prompt에 포함되어야 합니다.
    assert "이 설비 조건이면 고장 위험이 높아?" in prompt

    # 이전 assistant 답변도 문맥 데이터로 포함되어야 합니다.
    assert (
        "현재 입력 조건에서는 고장 위험이 높게 예측되었습니다."
        in prompt
    )

    # 현재 질문도 별도의 영역에 포함되어야 합니다.
    assert "그건 왜 그래?" in prompt


def test_normalize_chat_history_keeps_only_recent_maximum_messages():
    """
    chat_history가 최대 개수를 초과하면
    최근 메시지만 유지하는지 확인합니다.

    MAX_CHAT_HISTORY_MESSAGES는 현재 6입니다.

    history가 8개라면:

        message 1
        message 2
        message 3
        message 4
        message 5
        message 6
        message 7
        message 8

    최근 6개인:

        message 3
        message 4
        message 5
        message 6
        message 7
        message 8

    만 유지해야 합니다.
    """

    chat_history = [
        {
            "role": "user",
            "content": f"message {index}",
        }
        for index in range(1, 9)
    ]

    normalized_history = (
        intent_classifier._normalize_chat_history(
            chat_history
        )
    )

    assert len(normalized_history) == 6

    # 가장 오래된 message 1, message 2는 제거됩니다.
    assert normalized_history[0]["content"] == "message 3"

    # 가장 최근 message 8은 유지됩니다.
    assert normalized_history[-1]["content"] == "message 8"


def test_normalize_chat_history_limits_each_message_content_length():
    """
    대화 메시지 한 개가 너무 길면
    MAX_CHAT_MESSAGE_CONTENT_LENGTH까지만 사용하는지 확인합니다.

    FastAPI schema에서도 길이를 제한할 예정이지만,
    classifier가 Python 코드에서 직접 호출될 수도 있으므로
    classifier 내부에서도 방어적으로 제한합니다.
    """

    long_content = (
        "A"
        * (
            intent_classifier.MAX_CHAT_MESSAGE_CONTENT_LENGTH
            + 100
        )
    )

    chat_history = [
        {
            "role": "user",
            "content": long_content,
        }
    ]

    normalized_history = (
        intent_classifier._normalize_chat_history(
            chat_history
        )
    )

    assert len(normalized_history) == 1

    assert (
        len(normalized_history[0]["content"])
        == intent_classifier.MAX_CHAT_MESSAGE_CONTENT_LENGTH
    )