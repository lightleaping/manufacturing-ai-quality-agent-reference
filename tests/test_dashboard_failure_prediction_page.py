"""
Day 24 설비 고장 위험 분석 Streamlit Page Test입니다.

Streamlit AppTest를 사용하여 실제 브라우저를 실행하지 않고
Page의 기본 Widget 구조와 초기값을 검증합니다.

이 테스트에서는 분석 버튼을 클릭하지 않습니다.

따라서 다음 기능은 실행되지 않습니다.

- 실제 FastAPI 요청
- 실제 PyTorch Prediction
- 실제 SHAP 계산
- 실제 Global Importance 조회
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import (
    AppTest,
)


# 현재 테스트 파일 위치:
#
#     project_root/
#         tests/
#             test_dashboard_failure_prediction_page.py
#
# Path(__file__).resolve().parents[1]:
#
#     project_root/
#
# AppTest.from_file()에 단순 상대 경로를 전달하면
# tests 폴더를 기준으로 경로가 해석될 수 있습니다.
#
# 따라서 프로젝트 루트에서 시작하는 절대 Path를 만들어
# 실제 Streamlit Page 위치를 명확하게 지정합니다.
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

FAILURE_PREDICTION_PAGE_PATH = (
    PROJECT_ROOT
    / "src"
    / "dashboard"
    / "pages"
    / "failure_prediction.py"
)


def test_failure_prediction_page_renders_expected_widgets() -> None:
    """
    설비 고장 위험 분석 Page가
    필요한 입력 Widget과 분석 버튼을 표시하는지 검증합니다.
    """

    app = AppTest.from_file(
        FAILURE_PREDICTION_PAGE_PATH,
    )

    app.run()

    assert not app.exception

    assert len(
        app.number_input
    ) == 5

    assert len(
        app.selectbox
    ) == 1

    assert len(
        app.checkbox
    ) == 2

    assert len(
        app.button
    ) == 1


def test_failure_prediction_page_uses_expected_default_values() -> None:
    """
    Day 17·18 검증에 사용한 고위험 예제값과
    Evidence 옵션 기본값이 화면에 반영되는지 검증합니다.
    """

    app = AppTest.from_file(
        FAILURE_PREDICTION_PAGE_PATH,
    )

    app.run()

    assert not app.exception

    assert (
        app.number_input[0].value
        == 303.0
    )

    assert (
        app.number_input[1].value
        == 312.5
    )

    assert (
        app.number_input[2].value
        == 1380.0
    )

    assert (
        app.number_input[3].value
        == 62.0
    )

    assert (
        app.number_input[4].value
        == 220.0
    )

    assert (
        app.selectbox[0].value
        == "L"
    )

    assert (
        app.checkbox[0].value
        is True
    )

    assert (
        app.checkbox[1].value
        is True
    )

def test_failure_prediction_page_displays_success_response() -> None:
    """
    분석 버튼 클릭 후 Mock FastAPI Response가
    Session State와 결과 Metric에 표시되는지 검증합니다.

    실제 FastAPI 서버·PyTorch·SHAP는 실행하지 않습니다.
    """

    from unittest.mock import (
        MagicMock,
        patch,
    )

    expected_response = {
        "prediction": 1,
        "probability": 0.9929,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "recommended_action": (
            "설비 점검을 권장합니다."
        ),
        "evidence": [
            {
                "evidence_type": (
                    "prediction_summary"
                ),
            }
        ],
        "answer": (
            "고장 위험이 높게 예측되었습니다."
        ),
        "warnings": [],
        "limitations": [],
    }

    app = AppTest.from_file(
        FAILURE_PREDICTION_PAGE_PATH,
    )

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

    mock_api_client.predict_failure.return_value = (
        expected_response
    )

    app.button[0].click()

    with patch(
        "src.dashboard.api_client."
        "DashboardApiClient",
        return_value=mock_context_client,
    ):
        app.run()

    assert not app.exception

    assert len(
        app.success
    ) == 1

    assert (
        "완료"
        in app.success[0].value
    )

    assert len(
        app.metric
    ) == 4

    assert (
        app.metric[0].value
        == "\uace0\uc7a5 \uc704\ud5d8 \uc788\uc74c"
    )

    assert (
        app.metric[1].value
        == "99.29%"
    )

    assert (
        app.metric[2].value
        == "70.00%"
    )

    assert (
        app.metric[3].value
        == "\ub192\uc74c (HIGH)"
    )

    assert (
        app.session_state[
            "failure_prediction_result"
        ]
        == expected_response
    )

    mock_api_client.predict_failure.assert_called_once()

def test_failure_prediction_page_displays_api_error() -> None:
    """
    FastAPI 연결 오류가 발생하면
    Dashboard 전용 오류 메시지를 표시하는지 검증합니다.

    실패한 요청은 성공 결과로 저장하지 않습니다.
    """

    from unittest.mock import (
        MagicMock,
        patch,
    )

    from src.dashboard.api_client import (
        DashboardApiConnectionError,
    )

    app = AppTest.from_file(
        FAILURE_PREDICTION_PAGE_PATH,
    )

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

    mock_api_client.predict_failure.side_effect = (
        DashboardApiConnectionError(
            "FastAPI 서버에 연결할 수 없습니다."
        )
    )

    app.button[0].click()

    with patch(
        "src.dashboard.api_client."
        "DashboardApiClient",
        return_value=mock_context_client,
    ):
        app.run()

    assert not app.exception

    assert len(
        app.error
    ) == 1

    assert (
        "FastAPI 서버에 연결할 수 없습니다."
        in app.error[0].value
    )

    assert len(
        app.success
    ) == 0

    assert (
        app.session_state[
            "failure_prediction_result"
        ]
        is None
    )

    mock_api_client.predict_failure.assert_called_once()



