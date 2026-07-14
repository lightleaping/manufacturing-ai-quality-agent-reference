"""
Day 24 Evidence 분석 Streamlit Page Test입니다.

Streamlit AppTest를 사용하여 실제 브라우저를 실행하지 않고
Evidence 화면의 빈 상태와 Backend Evidence 표시를 검증합니다.

중요:
- Evidence 전용 API를 호출하지 않습니다.
- 마지막 Failure Prediction Response의 evidence를 재사용합니다.
- SHAP Contribution과 Global Importance를 다시 계산하지 않습니다.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import (
    AppTest,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

EVIDENCE_ANALYSIS_PAGE_PATH = (
    PROJECT_ROOT
    / "src"
    / "dashboard"
    / "pages"
    / "evidence_analysis.py"
)


def build_prediction_result_with_evidence() -> dict:
    """
    Evidence Page 테스트에 사용할
    고정 Failure Prediction Response를 생성합니다.

    실제 FastAPI·PyTorch·SHAP는 실행하지 않습니다.
    """

    return {
        "prediction": 1,
        "probability": 0.9929,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "recommended_action": (
            "설비 점검을 권장합니다."
        ),
        "answer": (
            "고장 위험이 높게 예측되었습니다."
        ),
        "warnings": [],
        "limitations": [],
        "evidence": [
            {
                "evidence_id": (
                    "prediction_summary_001"
                ),
                "evidence_type": (
                    "prediction_summary"
                ),
                "source": "model_prediction",
                "title": "모델 예측 요약",
                "summary": (
                    "고장 확률 0.9929, "
                    "위험 등급 HIGH입니다."
                ),
                "feature": None,
                "value": None,
                "direction": None,
                "contribution": None,
                "importance": None,
                "severity": "HIGH",
                "metadata": {
                    "probability": 0.9929,
                },
            },
            {
                "evidence_id": (
                    "rule_based_001"
                ),
                "evidence_type": (
                    "rule_based"
                ),
                "source": "rule_engine",
                "title": "높은 Torque",
                "summary": (
                    "Torque가 위험 조건에 해당합니다."
                ),
                "feature": "Torque [Nm]",
                "value": 62.0,
                "direction": "increases_risk",
                "contribution": None,
                "importance": None,
                "severity": "HIGH",
                "metadata": {},
            },
            {
                "evidence_id": (
                    "shap_local_001"
                ),
                "evidence_type": (
                    "shap_local"
                ),
                "source": "shap",
                "title": "Torque SHAP",
                "summary": (
                    "Torque가 고장 위험을 증가시켰습니다."
                ),
                "feature": "Torque [Nm]",
                "value": 62.0,
                "direction": "increases_risk",
                "contribution": 1.2345,
                "importance": None,
                "severity": "HIGH",
                "metadata": {},
            },
            {
                "evidence_id": (
                    "global_importance_001"
                ),
                "evidence_type": (
                    "global_importance"
                ),
                "source": (
                    "permutation_importance"
                ),
                "title": "Torque 중요도",
                "summary": (
                    "Torque는 주요 전역 특성입니다."
                ),
                "feature": "Torque [Nm]",
                "value": None,
                "direction": None,
                "contribution": None,
                "importance": 0.3309,
                "severity": None,
                "metadata": {},
            },
        ],
    }


def test_evidence_page_displays_empty_state_without_prediction() -> None:
    """
    저장된 Prediction Response가 없으면
    먼저 고장 위험 분석을 실행하라는 안내를 표시하는지 검증합니다.
    """

    app = AppTest.from_file(
        EVIDENCE_ANALYSIS_PAGE_PATH,
    )

    app.run()

    assert not app.exception

    assert len(
        app.info
    ) >= 1

    assert (
        "설비 고장 위험 분석"
        in app.info[0].value
    )

    assert len(
        app.dataframe
    ) == 0


def test_evidence_page_displays_backend_evidence() -> None:
    """
    Session State의 Failure Prediction Response에서
    기존 Backend Evidence를 읽어 화면에 표시하는지 검증합니다.
    """

    app = AppTest.from_file(
        EVIDENCE_ANALYSIS_PAGE_PATH,
    )

    app.session_state[
        "failure_prediction_result"
    ] = (
        build_prediction_result_with_evidence()
    )

    app.run()

    assert not app.exception

    assert len(
        app.dataframe
    ) >= 1

    assert len(
        app.expander
    ) >= 4

    subheader_values = [
        item.value
        for item in app.subheader
    ]

    assert (
        "Prediction Summary"
        in subheader_values
    )

    assert (
        "Rule-based Evidence"
        in subheader_values
    )

    assert (
        "SHAP Local Evidence"
        in subheader_values
    )

    assert (
        "Global Importance"
        in subheader_values
    )
