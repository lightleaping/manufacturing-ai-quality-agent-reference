"""
Day 13 - LangGraph Failure Agent demo script

이 파일의 역할
----------------
src/agent/failure_agent_graph.py에서 만든 LangGraph workflow를
터미널에서 직접 실행해보는 demo script입니다.

테스트 파일과 demo script의 차이
-------------------------------
tests/test_failure_agent_graph.py:
    - pytest가 자동으로 검증하는 테스트 코드
    - monkeypatch로 OpenAI API와 prediction service를 가짜로 대체
    - 목적: workflow 분기와 state 저장이 맞는지 검증

scripts/run_failure_agent_graph_demo.py:
    - 사람이 직접 실행해서 결과를 눈으로 확인하는 코드
    - 실제 intent classifier와 실제 failure prediction service를 사용할 수 있음
    - 목적: 포트폴리오/면접에서 보여줄 실행 흐름 확인

실행 명령
---------
프로젝트 루트에서 아래처럼 실행합니다.

    python -m scripts.run_failure_agent_graph_demo

주의
----
직접 파일 경로로 실행하지 말고, python -m 방식으로 실행합니다.

좋은 방식:
    python -m scripts.run_failure_agent_graph_demo

피하는 방식:
    python scripts/run_failure_agent_graph_demo.py

이유:
    python -m 방식은 프로젝트 루트를 기준으로 module import를 처리하므로
    src.agent.failure_agent_graph 같은 import가 안정적으로 동작합니다.
"""

from __future__ import annotations

import json
from typing import Any

from src.agent.failure_agent_graph import run_failure_agent_graph
from src.agent.state import AgentState, create_initial_agent_state


def main() -> None:
    """
    demo script의 시작 함수입니다.

    여기서는 3가지 대표 케이스를 실행합니다.

    1. dataset_schema_query
       - AI4I 데이터셋 feature와 target을 묻는 질문

    2. unknown intent
       - 현재 Agent가 지원하지 않는 질문

    3. failure_prediction + raw_sample
       - 설비 입력값을 함께 제공해서 실제 고장 예측 workflow 실행

    이 3가지를 확인하면 Day 13 LangGraph workflow의 핵심 분기를
    한 번에 볼 수 있습니다.
    """

    print_header("Day 13 LangGraph Failure Agent Demo")

    run_dataset_schema_demo()
    run_unknown_intent_demo()
    run_failure_prediction_demo()


def run_dataset_schema_demo() -> None:
    """
    Case 1. dataset_schema_query demo

    사용자가 AI4I 데이터셋의 feature와 target을 물어보는 상황입니다.

    예상 workflow:
        validate_question_node
        -> classify_intent_node
        -> build_dataset_schema_answer_node
        -> END

    이 케이스는 모델 artifact가 없어도 실행 가능해야 합니다.
    """

    print_header("Case 1 - Dataset schema query")

    state = create_initial_agent_state(
        question="AI4I 데이터셋 feature와 target은 뭐야?"
    )

    result = run_failure_agent_graph(state)

    print_state_summary(result)


def run_unknown_intent_demo() -> None:
    """
    Case 2. unknown intent demo

    사용자가 현재 Agent가 지원하지 않는 질문을 하는 상황입니다.

    예상 workflow:
        validate_question_node
        -> classify_intent_node
        -> build_fallback_answer_node
        -> END

    이 케이스에서는 억지로 모델 예측을 수행하지 않아야 합니다.
    """

    print_header("Case 2 - Unknown intent")

    state = create_initial_agent_state(
        question="오늘 점심 메뉴 추천해줘."
    )

    result = run_failure_agent_graph(state)

    print_state_summary(result)


def run_failure_prediction_demo() -> None:
    """
    Case 3. failure_prediction demo

    사용자가 설비 입력값을 제공하고 고장 위험 예측을 요청하는 상황입니다.

    예상 workflow:
        validate_question_node
        -> classify_intent_node
        -> call_failure_prediction_node
        -> build_final_answer_node
        -> END

    이 케이스는 Day 12에서 만든 failure_agent_service를 실제로 호출합니다.

    따라서 아래 artifact가 준비되어 있어야 합니다.

        models/failure_mlp/model.pt
        models/failure_mlp/scaler.joblib
        models/failure_mlp/metadata.json

    include_shap=True, include_global_importance=True로 service를 호출하므로
    아래 artifact가 있으면 SHAP/global importance evidence도 포함될 수 있습니다.

        models/failure_mlp/shap_background.pt
        models/failure_mlp/shap_reference_values.json
        models/failure_mlp/global_importance.json

    만약 SHAP artifact가 없어도 Day 12 fallback 구조가 정상이라면
    prediction은 수행되고 warnings에 실패 이유가 기록되어야 합니다.
    """

    print_header("Case 3 - Failure prediction with raw_sample")

    raw_sample = {
        # API 스타일 key를 사용합니다.
        #
        # failure_agent_graph.py의 _run_failure_prediction_service()는
        # API 스타일 key와 AI4I 원본 feature key를 모두 허용합니다.
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

    result = run_failure_agent_graph(state)

    print_state_summary(result)


def print_header(title: str) -> None:
    """
    출력 구분선을 보기 좋게 출력합니다.
    """

    print()
    print("=" * 80)
    print(f"[INFO] {title}")
    print("=" * 80)


def print_state_summary(state: AgentState) -> None:
    """
    LangGraph 실행 결과 state에서 중요한 항목만 보기 좋게 출력합니다.

    왜 전체 state를 그대로 출력하지 않는가?
    -------------------------------------
    전체 state에는 raw_response, evidence metadata 등 많은 정보가 들어갈 수 있습니다.

    demo에서는 사람이 흐름을 이해하는 것이 목적이므로
    아래 핵심 항목을 먼저 보여줍니다.

    - question
    - intent
    - confidence
    - intent_source
    - prediction
    - probability
    - risk_level
    - answer
    - warnings
    - errors
    - evidence 개수
    """

    summary = {
        "question": state.get("question"),
        "intent": state.get("intent"),
        "confidence": state.get("confidence"),
        "intent_source": state.get("intent_source"),
        "intent_reason": state.get("intent_reason"),
        "prediction": state.get("prediction"),
        "probability": state.get("probability"),
        "threshold": state.get("threshold"),
        "risk_level": state.get("risk_level"),
        "recommended_action": state.get("recommended_action"),
        "warnings": state.get("warnings", []),
        "errors": state.get("errors", []),
        "limitations": state.get("limitations", []),
        "evidence_count": len(state.get("evidence", [])),
    }

    print("[INFO] State summary")
    print_json(summary)

    answer = state.get("answer")
    if answer:
        print()
        print("[INFO] Answer")
        print(answer)

    evidence = state.get("evidence", [])
    if evidence:
        print()
        print("[INFO] Evidence preview")
        print_evidence_preview(evidence)


def print_evidence_preview(evidence: list[dict[str, Any]], *, max_items: int = 5) -> None:
    """
    evidence 목록 중 앞부분만 출력합니다.

    evidence가 많을 수 있으므로 기본적으로 앞 5개만 보여줍니다.
    """

    preview = evidence[:max_items]

    for index, item in enumerate(preview, start=1):
        print("-" * 80)
        print(f"[Evidence {index}]")
        print_json(
            {
                "evidence_id": item.get("evidence_id"),
                "evidence_type": item.get("evidence_type"),
                "source": item.get("source"),
                "title": item.get("title"),
                "feature": item.get("feature"),
                "value": item.get("value"),
                "direction": item.get("direction"),
                "contribution": item.get("contribution"),
                "importance": item.get("importance"),
                "severity": item.get("severity"),
                "summary": item.get("summary"),
            }
        )

    if len(evidence) > max_items:
        print("-" * 80)
        print(f"[INFO] Evidence가 {len(evidence)}개 있어 앞 {max_items}개만 출력했습니다.")


def print_json(data: Any) -> None:
    """
    dict/list 데이터를 한글이 깨지지 않게 JSON 형태로 출력합니다.
    """

    print(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()