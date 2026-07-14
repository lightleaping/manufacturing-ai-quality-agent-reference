# tests/test_dashboard_failure_prediction_explanation.py

"""
Failure Prediction Page의
선택형 OpenAI 운영 해설 UI Test입니다.

실제 FastAPI와 OpenAI API는 호출하지 않습니다.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import (
    MagicMock,
    patch,
)

from streamlit.testing.v1 import (
    AppTest,
)

from src.dashboard.api_client import (
    DashboardApiClientError,
)


PAGE_PATH = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "src"
    / "dashboard"
    / "pages"
    / "failure_prediction.py"
)


def build_prediction_result() -> dict:
    """
    Page Session State에 저장할
    기존 Prediction 결과입니다.
    """

    return {
        "prediction": 1,
        "probability": 0.9929,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "recommended_action": (
            "설비 점검을 권장합니다."
        ),
        "evidence": [],
        "answer": (
            "고장 위험이 높게 예측되었습니다."
        ),
        "warnings": [],
        "limitations": [],
    }


def build_explanation_response() -> dict:
    """
    Mock FastAPI 운영 해설 Response입니다.
    """

    return {
        "summary": (
            "현재 고장 위험이 높게 예측되었습니다."
        ),
        "key_signals": [
            "공구 마모 신호를 확인하세요.",
            "토크 신호를 함께 확인하세요.",
        ],
        "recommended_checks": [
            "공구 상태를 우선 점검하세요.",
            "운전 조건을 확인하세요.",
        ],
        "caution": (
            "SHAP는 실제 고장의 "
            "물리적 원인을 확정하지 않습니다."
        ),
        "source": "openai",
        "model": "test-model",
        "error": None,
    }


def test_operational_explanation_button_calls_fastapi_client() -> None:
    """
    AI 운영 해설 버튼이
    기존 Prediction 결과를 FastAPI Client에 전달하고,
    Response를 Session State에 저장하는지 검증합니다.
    """

    prediction_result = (
        build_prediction_result()
    )

    explanation_response = (
        build_explanation_response()
    )

    app = AppTest.from_file(
        PAGE_PATH,
    )

    app.session_state[
        "failure_prediction_result"
    ] = prediction_result

    app.run()

    mock_context_client = (
        MagicMock()
    )

    mock_api_client = (
        MagicMock()
    )

    mock_context_client.__enter__.return_value = (
        mock_api_client
    )

    (
        mock_api_client
        .generate_failure_prediction_explanation
        .return_value
    ) = explanation_response

    app.button[1].click()

    with patch(
        (
            "src.dashboard.api_client."
            "DashboardApiClient"
        ),
        return_value=mock_context_client,
    ):
        app.run()

    assert not app.exception

    (
        mock_api_client
        .generate_failure_prediction_explanation
        .assert_called_once_with(
            {
                "prediction_result": (
                    prediction_result
                )
            }
        )
    )

    assert (
        app.session_state[
            "failure_prediction_explanation"
        ]
        == explanation_response
    )

    assert all(
        "테스트 API 오류"
        not in item.value
        for item in app.markdown
    )
def test_operational_explanation_displays_api_error() -> None:
    """
    운영 해설 FastAPI 호출 실패가
    Page 전체 예외가 아니라 오류 안내로 표시되는지 검증합니다.
    """

    app = AppTest.from_file(
        PAGE_PATH,
    )

    app.session_state[
        "failure_prediction_result"
    ] = build_prediction_result()

    app.run()

    mock_context_client = (
        MagicMock()
    )

    mock_api_client = (
        MagicMock()
    )

    mock_context_client.__enter__.return_value = (
        mock_api_client
    )

    (
        mock_api_client
        .generate_failure_prediction_explanation
        .side_effect
    ) = DashboardApiClientError(
        "테스트 API 오류"
    )

    app.button[1].click()

    with patch(
        (
            "src.dashboard.api_client."
            "DashboardApiClient"
        ),
        return_value=mock_context_client,
    ):
        app.run()

    assert not app.exception

    assert (
        any(
            "테스트 API 오류"
            in item.value
            for item in app.markdown
        )
    )


