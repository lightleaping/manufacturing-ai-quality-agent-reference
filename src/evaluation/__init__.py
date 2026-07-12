"""
Day 21 Agent Evaluation package입니다.

이 package의 책임
-----------------
기존 LangGraph Agent를 다시 구현하지 않고,
기존 Agent의 출력 품질과 안전 동작을 평가합니다.


현재 구성
---------
agent_evaluation_cases.py

    평가 질문

    기대 Intent

    기대 Fallback

    기대 Prediction

    기대 Answer 조건

    등을 정의합니다.


agent_evaluator.py

    기존 LangGraph Agent 실행

    기대값과 실제값 비교

    Check별 PASS / FAIL

    Case별 PASS / FAIL

    전체 Pass Rate 계산
"""

from src.evaluation.agent_evaluation_cases import (
    AgentEvaluationCase,
    build_day21_evaluation_cases,
)
from src.evaluation.agent_evaluator import (
    AgentEvaluationResult,
    AgentEvaluationSummary,
    EvaluationCheckResult,
    evaluate_agent_case,
    evaluate_agent_cases,
)


# package 외부에 공개할 이름입니다.
__all__ = [
    "AgentEvaluationCase",
    "AgentEvaluationResult",
    "AgentEvaluationSummary",
    "EvaluationCheckResult",
    "build_day21_evaluation_cases",
    "evaluate_agent_case",
    "evaluate_agent_cases",
]