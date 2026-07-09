from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_langgraph_query_api_returns_dataset_schema_answer(monkeypatch):
    """
    dataset_schema_query 질문을 보냈을 때
    LangGraph API endpoint가 schema 답변을 반환하는지 확인합니다.

    중요한 점:
    - 이 테스트는 실제 OpenAI API를 호출하지 않습니다.
    - run_failure_agent_graph()를 fake 함수로 바꿉니다.
    """

    def fake_run_failure_agent_graph(
        question,
        raw_sample=None,
        include_shap=True,
        include_global_importance=True,
    ):
        return {
            "question": question,
            "intent": "dataset_schema_query",
            "confidence": 0.95,
            "intent_source": "rule_based_fallback",
            "intent_reason": "데이터셋 feature와 target을 묻는 질문입니다.",
            "answer": "AI4I 데이터셋의 feature는 온도, 회전 속도, 토크, 공구 마모, Type이고 target은 Machine failure입니다.",
            "evidence": [],
            "warnings": [],
            "errors": [],
            "limitations": [],
        }

    monkeypatch.setattr(
        "src.api.langgraph_agent_api.run_failure_agent_graph",
        fake_run_failure_agent_graph,
    )

    response = client.post(
        "/agent/langgraph-query",
        json={
            "question": "AI4I 데이터셋 feature와 target은 뭐야?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "AI4I 데이터셋 feature와 target은 뭐야?"
    assert data["intent"] == "dataset_schema_query"
    assert data["prediction"] is None
    assert "feature" in data["answer"]
    assert data["errors"] == []


def test_langgraph_query_api_returns_unknown_fallback(monkeypatch):
    """
    제조 AI와 관련 없는 질문을 보냈을 때
    unknown intent와 fallback answer를 반환하는지 확인합니다.
    """

    def fake_run_failure_agent_graph(
        question,
        raw_sample=None,
        include_shap=True,
        include_global_importance=True,
    ):
        return {
            "question": question,
            "intent": "unknown",
            "confidence": 0.3,
            "intent_source": "rule_based_fallback",
            "intent_reason": "지원하지 않는 질문입니다.",
            "answer": "현재는 설비 고장 예측과 AI4I 데이터셋 관련 질문만 지원합니다.",
            "evidence": [],
            "warnings": [],
            "errors": [],
            "limitations": [
                "지원 intent는 failure_prediction, dataset_schema_query, unknown입니다."
            ],
        }

    monkeypatch.setattr(
        "src.api.langgraph_agent_api.run_failure_agent_graph",
        fake_run_failure_agent_graph,
    )

    response = client.post(
        "/agent/langgraph-query",
        json={
            "question": "오늘 점심 메뉴 추천해줘.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "unknown"
    assert data["prediction"] is None
    assert data["evidence"] == []
    assert data["errors"] == []
    assert len(data["limitations"]) >= 1


def test_langgraph_query_api_returns_failure_prediction(monkeypatch):
    """
    failure_prediction 질문과 raw_sample이 함께 들어왔을 때
    prediction, probability, risk_level, evidence를 반환하는지 확인합니다.
    """

    def fake_run_failure_agent_graph(
        question,
        raw_sample=None,
        include_shap=True,
        include_global_importance=True,
    ):
        assert raw_sample is not None
        assert raw_sample["air_temperature"] == 303.0
        assert include_shap is True
        assert include_global_importance is True

        return {
            "question": question,
            "intent": "failure_prediction",
            "confidence": 0.97,
            "intent_source": "openai",
            "intent_reason": "설비 조건에 대한 고장 위험 질문입니다.",
            "prediction": 1,
            "probability": 0.9929,
            "threshold": 0.7,
            "risk_level": "HIGH",
            "recommended_action": "고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다.",
            "answer": "현재 입력 조건에서는 고장 위험이 높게 예측됩니다.",
            "evidence": [
                {
                    "evidence_id": "prediction_summary_001",
                    "evidence_type": "prediction_summary",
                    "source": "model_prediction",
                    "title": "모델 예측 요약",
                    "summary": "probability=0.9929, threshold=0.7 기준 HIGH입니다.",
                    "severity": "HIGH",
                }
            ],
            "warnings": [],
            "errors": [],
            "limitations": [],
        }

    monkeypatch.setattr(
        "src.api.langgraph_agent_api.run_failure_agent_graph",
        fake_run_failure_agent_graph,
    )

    response = client.post(
        "/agent/langgraph-query",
        json={
            "question": "이 설비 조건이면 고장 위험이 높아?",
            "raw_sample": {
                "air_temperature": 303.0,
                "process_temperature": 312.5,
                "rotational_speed": 1380.0,
                "torque": 62.0,
                "tool_wear": 220.0,
                "type": "L",
            },
            "include_shap": True,
            "include_global_importance": True,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "failure_prediction"
    assert data["prediction"] == 1
    assert data["probability"] == 0.9929
    assert data["threshold"] == 0.7
    assert data["risk_level"] == "HIGH"
    assert len(data["evidence"]) == 1
    assert data["errors"] == []


def test_langgraph_query_api_does_not_force_prediction_without_raw_sample(monkeypatch):
    """
    failure_prediction intent인데 raw_sample이 없을 때
    prediction을 억지로 수행하지 않는지 확인합니다.

    이 상황은 API에서 막기보다 LangGraph workflow가 처리하는 것이 좋습니다.
    왜냐하면 question을 보고 failure_prediction인지 판단하는 책임은
    LangGraph workflow에 있기 때문입니다.
    """

    def fake_run_failure_agent_graph(
        question,
        raw_sample=None,
        include_shap=True,
        include_global_importance=True,
    ):
        assert raw_sample is None

        return {
            "question": question,
            "intent": "failure_prediction",
            "confidence": 0.93,
            "intent_source": "openai",
            "intent_reason": "고장 위험을 묻는 질문이지만 raw_sample이 없습니다.",
            "prediction": None,
            "probability": None,
            "threshold": None,
            "risk_level": "UNKNOWN",
            "recommended_action": "고장 위험 예측을 위해 설비 입력값을 함께 보내주세요.",
            "answer": "고장 위험 예측에는 air_temperature, process_temperature, rotational_speed, torque, tool_wear, type 값이 필요합니다.",
            "evidence": [],
            "warnings": [
                "failure_prediction intent지만 raw_sample이 없어 prediction을 수행하지 않았습니다."
            ],
            "errors": [],
            "limitations": [],
        }

    monkeypatch.setattr(
        "src.api.langgraph_agent_api.run_failure_agent_graph",
        fake_run_failure_agent_graph,
    )

    response = client.post(
        "/agent/langgraph-query",
        json={
            "question": "이 설비 고장 위험 예측해줘.",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["intent"] == "failure_prediction"
    assert data["prediction"] is None
    assert data["probability"] is None
    assert data["risk_level"] == "UNKNOWN"
    assert len(data["warnings"]) == 1
    assert data["errors"] == []


def test_existing_failure_prediction_endpoint_is_still_registered():
    """
    Day 14에서 새 endpoint를 추가해도
    기존 /agent/failure-prediction endpoint가 사라지지 않았는지 확인합니다.

    실제 prediction을 호출하지 않고,
    OpenAPI schema에 path가 등록되어 있는지만 확인합니다.
    """

    response = client.get("/openapi.json")

    assert response.status_code == 200

    paths = response.json()["paths"]

    assert "/agent/failure-prediction" in paths
    assert "/agent/langgraph-query" in paths