"""
Day 13 - LangGraph Failure Agent workflow 테스트

이 테스트 파일의 역할
---------------------
src/agent/failure_agent_graph.py에서 만든 LangGraph workflow가
의도한 흐름대로 동작하는지 검증합니다.

중요한 테스트 원칙
------------------
1. 실제 OpenAI API를 호출하지 않습니다.
2. 실제 model artifact를 로드하지 않습니다.
3. 실제 SHAP 계산을 하지 않습니다.
4. 테스트에서는 monkeypatch로 classifier와 prediction service를 가짜 함수로 대체합니다.
5. 목표는 LangGraph workflow의 node 책임, 분기, fallback 처리를 검증하는 것입니다.

왜 실제 OpenAI와 모델을 호출하지 않는가?
----------------------------------------
단위 테스트는 빠르고 안정적이어야 합니다.

실제 API나 모델 artifact에 의존하면 다음 문제가 생깁니다.

- API key가 없는 환경에서 실패
- 네트워크 상태에 따라 실패
- 비용 발생
- artifact 파일 위치나 손상 여부에 따라 실패
- SHAP 계산으로 테스트가 느려짐

따라서 여기서는
"OpenAI가 이런 intent를 반환했다고 가정했을 때"
"prediction service가 이런 결과를 반환했다고 가정했을 때"
LangGraph가 올바르게 분기하는지를 검증합니다.
"""

from src.agent.intent_classifier import IntentClassificationResult
from src.agent.state import create_initial_agent_state
from src.agent import failure_agent_graph
from src.agent.failure_agent_graph import (
    build_dataset_schema_answer_node,
    build_fallback_answer_node,
    build_final_answer_node,
    call_failure_prediction_node,
    classify_intent_node,
    route_after_classification,
    route_after_prediction,
    route_after_validation,
    run_failure_agent_graph,
    validate_question_node,
    _get_sample_value,
)


def test_validate_question_node_strips_question():
    """
    validate_question_node는 question 앞뒤 공백을 제거해야 합니다.
    """

    state = create_initial_agent_state(
        question="   이 설비 조건이면 고장 위험이 높아?   "
    )

    result = validate_question_node(state)

    assert result["question"] == "이 설비 조건이면 고장 위험이 높아?"
    assert result["errors"] == []


def test_validate_question_node_adds_error_for_empty_question():
    """
    question이 비어 있으면 errors에 메시지를 추가해야 합니다.
    """

    state = create_initial_agent_state(question="   ")

    result = validate_question_node(state)

    assert len(result["errors"]) == 1
    assert "question이 비어" in result["errors"][0]


def test_classify_intent_node_stores_classification_result(monkeypatch):
    """
    classify_intent_node는 intent classifier 결과를 AgentState에 저장해야 합니다.

    실제 OpenAI API는 호출하지 않습니다.
    failure_agent_graph 모듈 안에서 참조되는 classify_intent를
    fake_classify_intent로 교체합니다.
    """

    def fake_classify_intent(
        question: str,
        *,
        # Day 15부터 실제 classify_intent() 함수는
        # 현재 질문뿐 아니라 이전 대화 기록도 받을 수 있습니다.
        #
        # 따라서 monkeypatch로 대신 실행되는 fake 함수도
        # 실제 함수와 같은 호출 interface를 가져야 합니다.
        #
        # 현재 테스트의 목적은
        # intent 분류 결과가 AgentState에 저장되는지 확인하는 것이므로,
        # chat_history 값을 직접 사용하지는 않습니다.
        chat_history=None,
    ) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent="failure_prediction",
            confidence=0.91,
            reason="테스트용 intent 분류 결과입니다.",
            source="openai",
            raw_response='{"intent": "failure_prediction"}',
            error=None,
        )

    monkeypatch.setattr(
        failure_agent_graph,
        "classify_intent",
        fake_classify_intent,
    )

    state = create_initial_agent_state(
        question="Torque 62이고 Tool wear 220이면 고장 위험이 높아?"
    )

    result = classify_intent_node(state)

    assert result["intent"] == "failure_prediction"
    assert result["confidence"] == 0.91
    assert result["intent_reason"] == "테스트용 intent 분류 결과입니다."
    assert result["intent_source"] == "openai"
    assert result["intent_raw_response"] == '{"intent": "failure_prediction"}'
    assert result["warnings"] == []


def test_classify_intent_node_adds_warning_when_classifier_has_error(monkeypatch):
    """
    intent classifier가 fallback 또는 오류를 반환하면 warnings에 기록해야 합니다.
    """

    def fake_classify_intent(
        question: str,
        *,
        # 실제 classify_intent()가 Day 15부터
        # chat_history keyword argument를 받을 예정이므로,
        # 오류 상황을 만드는 fake 함수도
        # 같은 매개변수를 받을 수 있어야 합니다.
        chat_history=None,
    ) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent="failure_prediction",
            confidence=0.65,
            reason="OpenAI 실패 후 rule-based fallback을 사용했습니다.",
            source="fallback",
            raw_response=None,
            error="mock_openai_error",
        )

    monkeypatch.setattr(
        failure_agent_graph,
        "classify_intent",
        fake_classify_intent,
    )

    state = create_initial_agent_state(
        question="Torque 62이고 Tool wear 220이면 고장 위험이 높아?"
    )

    result = classify_intent_node(state)

    assert result["intent"] == "failure_prediction"
    assert result["intent_source"] == "fallback"
    assert len(result["warnings"]) == 1
    assert "mock_openai_error" in result["warnings"][0]


def test_route_after_validation_goes_to_classify_when_no_errors():
    """
    validate_question_node 이후 errors가 없으면 classify 단계로 이동해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    assert route_after_validation(state) == "classify"


def test_route_after_validation_goes_to_fallback_when_errors_exist():
    """
    validate_question_node 이후 errors가 있으면 fallback 단계로 이동해야 합니다.
    """

    state = create_initial_agent_state(question="")
    state["errors"].append("question이 비어 있습니다.")

    assert route_after_validation(state) == "fallback"


def test_route_after_classification_goes_to_failure_prediction():
    """
    intent가 failure_prediction이면 prediction node로 이동해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )
    state["intent"] = "failure_prediction"

    assert route_after_classification(state) == "failure_prediction"


def test_route_after_classification_goes_to_dataset_schema():
    """
    intent가 dataset_schema_query이면 dataset schema answer node로 이동해야 합니다.
    """

    state = create_initial_agent_state(
        question="AI4I feature는 뭐야?"
    )
    state["intent"] = "dataset_schema_query"

    assert route_after_classification(state) == "dataset_schema"


def test_route_after_classification_goes_to_fallback_for_unknown():
    """
    intent가 unknown이면 fallback node로 이동해야 합니다.
    """

    state = create_initial_agent_state(
        question="오늘 점심 추천해줘."
    )
    state["intent"] = "unknown"

    assert route_after_classification(state) == "fallback"


def test_call_failure_prediction_node_adds_error_when_raw_sample_missing():
    """
    failure_prediction intent인데 raw_sample이 없으면 예측을 수행하지 않고 error를 추가해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )
    state["intent"] = "failure_prediction"

    result = call_failure_prediction_node(state)

    assert len(result["errors"]) == 1
    assert "raw_sample이 없어" in result["errors"][0]
    
    # chat_history는 현재 질문의 문맥을 이해하기 위한 데이터일 뿐,
    # 이전 설비 조건이나 raw_sample을
    # 새로운 prediction 입력으로 자동 재사용하지 않는다는
    # 안내가 error에 포함되어야 합니다.
    assert (
        "이전 대화의 설비 조건이나 raw_sample은 "
        "자동으로 재사용하지 않습니다"
        in result["errors"][0]
    )


def test_call_failure_prediction_node_stores_prediction_result(monkeypatch):
    """
    prediction service가 성공하면 결과를 AgentState에 저장해야 합니다.

    실제 Day 12 service는 호출하지 않습니다.
    _run_failure_prediction_service를 fake 함수로 교체합니다.
    """

    def fake_run_failure_prediction_service(
        raw_sample,
        include_shap=True,
        include_global_importance=True,
    ):
        """
        Day 14에서 실제 helper 함수에 옵션이 추가되었으므로
        fake 함수도 같은 매개변수를 받습니다.

        monkeypatch 대상 함수와 fake 함수의 인터페이스가 같아야
        실제 호출 방식도 테스트할 수 있습니다.
        """

        return {
            "prediction": 1,
            "probability": 0.9929,
            "threshold": 0.7,
            "risk_level": "HIGH",
            "recommended_action": (
                "고장 위험이 높습니다. 설비 점검을 권장합니다."
            ),
            "evidence": [
                {
                    "evidence_id": "prediction_summary_001",
                    "evidence_type": "prediction_summary",
                    "source": "model_prediction",
                    "title": "모델 예측 요약",
                    "summary": (
                        "모델은 고장 probability를 높게 예측했습니다."
                    ),
                    "severity": "HIGH",
                }
            ],
            "answer": "고장 위험이 높습니다.",
            "warnings": [
                "SHAP 계산은 테스트에서 생략되었습니다."
            ],
            "limitations": [
                "SHAP value는 실제 원인 단정이 아닙니다."
            ],
        }

    monkeypatch.setattr(
        failure_agent_graph,
        "_run_failure_prediction_service",
        fake_run_failure_prediction_service,
    )

    raw_sample = {
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
    }

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?",
        raw_sample=raw_sample,
    )
    state["intent"] = "failure_prediction"

    result = call_failure_prediction_node(state)

    assert result["prediction"] == 1
    assert result["probability"] == 0.9929
    assert result["threshold"] == 0.7
    assert result["risk_level"] == "HIGH"
    assert result["recommended_action"] == "고장 위험이 높습니다. 설비 점검을 권장합니다."
    assert result["answer"] == "고장 위험이 높습니다."
    assert len(result["evidence"]) == 1
    assert result["warnings"] == ["SHAP 계산은 테스트에서 생략되었습니다."]
    assert result["limitations"] == ["SHAP value는 실제 원인 단정이 아닙니다."]
    assert result["errors"] == []


def test_call_failure_prediction_node_adds_error_when_service_raises(monkeypatch):
    """
    prediction service 호출 중 예외가 발생하면 errors에 기록해야 합니다.
    """

    def fake_run_failure_prediction_service(
        raw_sample,
        include_shap=True,
        include_global_importance=True,
    ):
        raise RuntimeError("mock service error")

    monkeypatch.setattr(
        failure_agent_graph,
        "_run_failure_prediction_service",
        fake_run_failure_prediction_service,
    )

    raw_sample = {
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
    }

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?",
        raw_sample=raw_sample,
    )
    state["intent"] = "failure_prediction"

    result = call_failure_prediction_node(state)

    assert len(result["errors"]) == 1
    assert "mock service error" in result["errors"][0]


def test_route_after_prediction_goes_to_final_when_no_errors():
    """
    prediction 이후 errors가 없으면 final answer node로 이동해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    assert route_after_prediction(state) == "final"


def test_route_after_prediction_goes_to_fallback_when_errors_exist():
    """
    prediction 이후 errors가 있으면 fallback answer node로 이동해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )
    state["errors"].append("prediction service error")

    assert route_after_prediction(state) == "fallback"


def test_build_dataset_schema_answer_node_sets_answer_and_evidence():
    """
    dataset_schema_query intent에서는 정적 schema answer와 evidence를 만들어야 합니다.
    """

    state = create_initial_agent_state(
        question="AI4I feature와 target은 뭐야?"
    )
    state["intent"] = "dataset_schema_query"

    result = build_dataset_schema_answer_node(state)

    assert "AI4I 2020 Predictive Maintenance Dataset" in result["answer"]
    assert "Air temperature [K]" in result["answer"]
    assert "Machine failure" in result["answer"]
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["evidence_type"] == "dataset_schema"


def test_build_fallback_answer_node_uses_errors_when_present():
    """
    errors가 있으면
    fallback answer에 확인된 오류 내용을 포함해야 합니다.

    이 테스트의 목적
    ----------------
    이 테스트는 call_failure_prediction_node()의
    raw_sample 처리 정책을 검증하는 테스트가 아닙니다.

    이미 state["errors"]에 오류가 들어 있다고 가정한 뒤,
    build_fallback_answer_node()가 해당 오류를
    최종 answer에 포함하는지 확인합니다.

    이전 대화의 설비 조건이나 raw_sample을
    자동 재사용하지 않는 정책은
    전체 workflow 테스트에서 별도로 검증합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )

    # fallback node가 전달받았다고 가정할
    # 테스트용 오류 메시지를 직접 추가합니다.
    error_message = (
        "raw_sample이 없어 "
        "failure_prediction을 수행할 수 없습니다."
    )

    state["errors"].append(
        error_message
    )

    result = build_fallback_answer_node(state)

    # 오류가 있을 때 사용하는
    # fallback 안내 문구가 포함되어야 합니다.
    assert (
        "요청을 처리하는 중 문제가 발생했습니다."
        in result["answer"]
    )

    # state["errors"]에 넣은 실제 오류 내용이
    # 최종 answer에도 포함되어야 합니다.
    assert (
        error_message
        in result["answer"]
    )

    # 사용자가 다시 prediction을 요청하려면
    # 현재 요청에 새 raw_sample을 제공해야 한다는
    # 후속 안내가 포함되어야 합니다.
    assert (
        "현재 예측에 사용할 새 raw_sample"
        in result["answer"]
    )


def test_build_fallback_answer_node_handles_unknown_intent():
    """
    intent가 unknown이면 지원 가능한 질문 유형을 안내해야 합니다.
    """

    state = create_initial_agent_state(
        question="오늘 점심 추천해줘."
    )
    state["intent"] = "unknown"

    result = build_fallback_answer_node(state)

    assert "지원하는 작업으로 분류되지 않았습니다" in result["answer"]
    assert "설비 입력값 기반 고장 위험 예측" in result["answer"]


def test_build_final_answer_node_keeps_existing_answer():
    """
    이미 answer가 있으면 build_final_answer_node는 그대로 유지해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )
    state["answer"] = "이미 생성된 답변입니다."

    result = build_final_answer_node(state)

    assert result["answer"] == "이미 생성된 답변입니다."


def test_build_final_answer_node_creates_minimum_answer_when_missing():
    """
    answer가 비어 있으면 최소 fallback 답변을 생성해야 합니다.
    """

    state = create_initial_agent_state(
        question="이 설비 조건이면 고장 위험이 높아?"
    )
    state["risk_level"] = "HIGH"
    state["probability"] = 0.99
    state["recommended_action"] = "설비 점검을 권장합니다."

    result = build_final_answer_node(state)

    assert "risk_level=HIGH" in result["answer"]
    assert "probability=0.99" in result["answer"]
    assert "설비 점검을 권장합니다" in result["answer"]


def test_get_sample_value_prefers_api_key():
    """
    _get_sample_value는 api_key가 있으면 api_key 값을 우선 사용해야 합니다.
    """

    raw_sample = {
        "torque": 62.0,
        "Torque [Nm]": 40.0,
    }

    value = _get_sample_value(
        raw_sample,
        "torque",
        "Torque [Nm]",
    )

    assert value == 62.0


def test_get_sample_value_uses_feature_key_when_api_key_missing():
    """
    api_key가 없으면 AI4I feature key 값을 사용해야 합니다.
    """

    raw_sample = {
        "Torque [Nm]": 62.0,
    }

    value = _get_sample_value(
        raw_sample,
        "torque",
        "Torque [Nm]",
    )

    assert value == 62.0


def test_get_sample_value_raises_key_error_when_missing():
    """
    api_key와 feature_key가 모두 없으면 KeyError를 발생시켜야 합니다.

    이 오류는 call_failure_prediction_node에서 잡혀서
    state["errors"]에 저장됩니다.
    """

    raw_sample = {}

    try:
        _get_sample_value(
            raw_sample,
            "torque",
            "Torque [Nm]",
        )
    except KeyError as exc:
        assert "raw_sample에 필요한 값이 없습니다" in str(exc)
    else:
        raise AssertionError("KeyError가 발생해야 합니다.")


def test_run_failure_agent_graph_handles_dataset_schema_query(monkeypatch):
    """
    전체 LangGraph workflow가 dataset_schema_query 경로로 정상 실행되는지 확인합니다.

    실제 OpenAI API를 호출하지 않기 위해 classify_intent를 fake 함수로 교체합니다.
    """

    def fake_classify_intent(
        question: str,
        *,
        # LangGraph 전체 실행 중 classify_intent_node가
        # question과 chat_history를 함께 전달할 수 있도록
        # fake 함수도 동일한 interface를 유지합니다.
        chat_history=None,
    ) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent="dataset_schema_query",
            confidence=0.9,
            reason="사용자가 데이터셋 schema를 질문했습니다.",
            source="openai",
            raw_response=None,
            error=None,
        )

    monkeypatch.setattr(
        failure_agent_graph,
        "classify_intent",
        fake_classify_intent,
    )

    result = run_failure_agent_graph(
        question="AI4I 데이터셋 feature와 target은 뭐야?",
    )

    assert result["intent"] == "dataset_schema_query"
    assert "AI4I 2020 Predictive Maintenance Dataset" in result["answer"]
    assert len(result["evidence"]) == 1
    assert result["errors"] == []


def test_run_failure_agent_graph_handles_unknown_intent(monkeypatch):
    """
    전체 LangGraph workflow가 unknown intent를 fallback answer로 처리하는지 확인합니다.
    """

    def fake_classify_intent(
        question: str,
        *,
        # Day 15 multi-turn 연결 후에도
        # unknown intent workflow 테스트가
        # 새 함수 호출 방식과 호환되도록 합니다.
        chat_history=None,
    ) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent="unknown",
            confidence=0.3,
            reason="지원하지 않는 질문입니다.",
            source="openai",
            raw_response=None,
            error=None,
        )

    monkeypatch.setattr(
        failure_agent_graph,
        "classify_intent",
        fake_classify_intent,
    )

    result = run_failure_agent_graph(
        question="오늘 점심 메뉴 추천해줘.",
    )

    assert result["intent"] == "unknown"
    assert "지원하는 작업으로 분류되지 않았습니다" in result["answer"]
    assert result["errors"] == []


def test_run_failure_agent_graph_handles_failure_prediction(monkeypatch):
    """
    전체 LangGraph workflow가 failure_prediction 경로로 정상 실행되는지 확인합니다.

    실제 OpenAI API와 실제 Day 12 prediction service는 호출하지 않습니다.
    """

    def fake_classify_intent(
        question: str,
        *,
        # 실제 classifier와 fake classifier의
        # 함수 interface를 일치시킵니다.
        #
        # chat_history는 intent 문맥 이해용이며,
        # 아래 prediction service의 raw_sample과는 역할이 다릅니다.
        chat_history=None,
    ) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent="failure_prediction",
            confidence=0.95,
            reason="사용자가 고장 위험 예측을 요청했습니다.",
            source="openai",
            raw_response=None,
            error=None,
        )

    def fake_run_failure_prediction_service(
        raw_sample,
        include_shap=True,
        include_global_importance=True,
    ):
        """
        Day 14에서 실제 helper 함수에 옵션이 추가되었으므로
        fake 함수도 같은 매개변수를 받습니다.

        monkeypatch 대상 함수와 fake 함수의 인터페이스가 같아야
        실제 호출 방식도 테스트할 수 있습니다.
        """

        return {
            "prediction": 1,
            "probability": 0.9929,
            "threshold": 0.7,
            "risk_level": "HIGH",
            "recommended_action": (
                "고장 위험이 높습니다. 설비 점검을 권장합니다."
            ),
            "evidence": [],
            "answer": "고장 위험이 높습니다.",
            "warnings": [],
            "limitations": [],
        }

    monkeypatch.setattr(
        failure_agent_graph,
        "classify_intent",
        fake_classify_intent,
    )

    monkeypatch.setattr(
        failure_agent_graph,
        "_run_failure_prediction_service",
        fake_run_failure_prediction_service,
    )

    raw_sample = {
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
    }

    result = run_failure_agent_graph(
        question="이 설비 조건이면 고장 위험이 높아?",
        raw_sample=raw_sample,
        include_shap=True,
        include_global_importance=True,
    )

    assert result["intent"] == "failure_prediction"
    assert result["prediction"] == 1
    assert result["probability"] == 0.9929
    assert result["risk_level"] == "HIGH"
    assert result["answer"] == "고장 위험이 높습니다."
    assert result["errors"] == []


def test_run_failure_agent_graph_falls_back_when_failure_prediction_has_no_raw_sample(
    monkeypatch,
):
    """
    failure_prediction intent인데 raw_sample이 없으면 fallback answer로 끝나야 합니다.
    """

    def fake_classify_intent(
        question: str,
        *,
        # chat_history가 전달되더라도
        # 실제 prediction에 필요한 raw_sample을 대신할 수는 없습니다.
        #
        # 이 테스트는 여전히:
        #
        # failure_prediction intent
        # +
        # raw_sample 없음
        #
        # 상황에서 fallback으로 이동하는지 검증합니다.
        chat_history=None,
    ) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent="failure_prediction",
            confidence=0.95,
            reason="사용자가 고장 위험 예측을 요청했습니다.",
            source="openai",
            raw_response=None,
            error=None,
        )

    monkeypatch.setattr(
        failure_agent_graph,
        "classify_intent",
        fake_classify_intent,
    )

    result = run_failure_agent_graph(
        question="이 설비 조건이면 고장 위험이 높아?",
    )

    assert result["intent"] == "failure_prediction"
    assert len(result["errors"]) == 1
    assert "raw_sample이 없어" in result["errors"][0]
    assert "요청을 처리하는 중 문제가 발생했습니다" in result["answer"]

    # 최종 fallback answer에서도
    # 이전 대화의 설비 조건을 자동 재사용하지 않는다는
    # 현재 multi-turn 설계 원칙을 사용자에게 안내해야 합니다.
    assert (
        "이전 대화의 설비 조건이나 raw_sample은 "
        "자동으로 재사용하지 않습니다"
        in result["answer"]
    )

    # 사용자가 실제 prediction을 다시 요청하려면
    # 현재 예측에 사용할 새 raw_sample을
    # 요청에 포함해야 한다는 안내도 있어야 합니다.
    assert (
        "현재 예측에 사용할 새 raw_sample"
        in result["answer"]
    )

def test_classify_intent_node_passes_chat_history_to_classifier(
    monkeypatch,
):
    """
    classify_intent_node()가 AgentState에 저장된 chat_history를
    intent classifier까지 전달하는지 확인합니다.

    왜 이 테스트가 필요한가?
    -------------------------
    AgentState에 chat_history가 저장되어 있어도
    classify_intent() 호출에 전달하지 않으면
    실제 intent 분류에는 사용되지 않습니다.

    따라서 아래 연결을 직접 검증합니다.

        AgentState["chat_history"]

                │

                ▼

        classify_intent_node()

                │

                ▼

        classify_intent(
            question,
            chat_history=...
        )
    """

    # fake classifier가 실제로 받은 값을
    # 테스트 함수의 마지막에서 확인하기 위한 dict입니다.
    #
    # dict는 mutable 객체이므로
    # fake 함수 내부에서 값을 저장한 뒤
    # 바깥 테스트 코드에서 확인할 수 있습니다.
    captured_arguments = {}

    def fake_classify_intent(
        question: str,
        *,
        chat_history=None,
    ) -> IntentClassificationResult:
        """
        실제 OpenAI API를 호출하지 않는 테스트용 classifier입니다.

        이 테스트의 목적은 intent 분류 정확도가 아니라,
        classify_intent_node()가 question과 chat_history를
        올바르게 전달하는지 확인하는 것입니다.
        """

        # classify_intent_node()가 전달한
        # 현재 질문을 저장합니다.
        captured_arguments["question"] = question

        # classify_intent_node()가 전달한
        # 이전 대화 기록을 저장합니다.
        captured_arguments["chat_history"] = chat_history

        return IntentClassificationResult(
            intent="failure_prediction",
            confidence=0.9,
            reason="이전 고장 예측 대화의 후속 질문입니다.",
            source="openai",
            raw_response=None,
            error=None,
        )

    monkeypatch.setattr(
        failure_agent_graph,
        "classify_intent",
        fake_classify_intent,
    )

    chat_history = [
        {
            "role": "user",
            "content": "이 설비 조건이면 고장 위험이 높아?",
        },
        {
            "role": "assistant",
            "content": (
                "현재 입력 조건에서는 "
                "고장 위험이 높게 예측되었습니다."
            ),
        },
    ]

    state = create_initial_agent_state(
        question="그건 왜 그래?",
        chat_history=chat_history,
    )

    result = classify_intent_node(state)

    # 현재 질문이 classifier까지 전달되어야 합니다.
    assert captured_arguments["question"] == "그건 왜 그래?"

    # AgentState에 저장된 이전 대화 기록도
    # classifier까지 전달되어야 합니다.
    assert captured_arguments["chat_history"] == chat_history

    # fake classifier의 분류 결과가
    # AgentState에 정상 저장되어야 합니다.
    assert result["intent"] == "failure_prediction"
    assert result["confidence"] == 0.9
    assert result["intent_source"] == "openai"


def test_run_failure_agent_graph_passes_chat_history_through_workflow(
    monkeypatch,
):
    """
    공개 runner에 전달한 chat_history가
    LangGraph workflow 안의 classifier까지 전달되는지 확인합니다.

    이 테스트는 node 하나만 직접 호출하는 것이 아니라,
    실제 공개 runner부터 전체 graph를 실행합니다.

    검증 흐름:

        run_failure_agent_graph(
            question,
            chat_history,
        )

                │

                ▼

        create_initial_agent_state()

                │

                ▼

        AgentState["chat_history"]

                │

                ▼

        classify_intent_node()

                │

                ▼

        classify_intent()
    """

    captured_arguments = {}

    def fake_classify_intent(
        question: str,
        *,
        chat_history=None,
    ) -> IntentClassificationResult:
        """
        실제 OpenAI 호출 대신
        runner에서 전달된 question과 history를 기록합니다.
        """

        captured_arguments["question"] = question
        captured_arguments["chat_history"] = chat_history

        # 이번 테스트에서는 prediction service가 필요 없는
        # dataset_schema_query를 반환합니다.
        #
        # 이렇게 하면 실제 모델 artifact나 SHAP을 로드하지 않고
        # LangGraph 전체 흐름을 테스트할 수 있습니다.
        return IntentClassificationResult(
            intent="dataset_schema_query",
            confidence=0.88,
            reason="이전 데이터셋 대화의 후속 질문입니다.",
            source="openai",
            raw_response=None,
            error=None,
        )

    monkeypatch.setattr(
        failure_agent_graph,
        "classify_intent",
        fake_classify_intent,
    )

    chat_history = [
        {
            "role": "user",
            "content": "AI4I 데이터셋의 feature는 뭐야?",
        },
        {
            "role": "assistant",
            "content": (
                "현재 모델은 AI4I feature 6개를 사용합니다."
            ),
        },
    ]

    result = run_failure_agent_graph(
        question="그중 target은 뭐야?",
        chat_history=chat_history,
    )

    # 공개 runner에 전달한 현재 질문이
    # classifier까지 전달되어야 합니다.
    assert (
        captured_arguments["question"]
        == "그중 target은 뭐야?"
    )

    # 공개 runner에 전달한 chat_history가
    # 초기 AgentState와 LangGraph node를 거쳐
    # classifier까지 전달되어야 합니다.
    assert (
        captured_arguments["chat_history"]
        == chat_history
    )

    # 최종 state에도 chat_history가 유지되어야 합니다.
    assert result["chat_history"] == chat_history

    # fake classifier가 반환한 intent에 따라
    # dataset schema 경로가 실행되어야 합니다.
    assert result["intent"] == "dataset_schema_query"

    assert (
        "AI4I 2020 Predictive Maintenance Dataset"
        in result["answer"]
    )

    assert result["errors"] == []