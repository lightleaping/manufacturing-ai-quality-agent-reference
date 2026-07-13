# tests/test_dashboard_api_client.py

"""
Day 23 Dashboard FastAPI Client Unit Test입니다.

이 테스트는 실제 FastAPI 서버를 실행하지 않습니다.

httpx.MockTransport를 사용하여 다음 상황을 재현합니다.

- 정상 JSON 응답
- HTTP 4xx·5xx
- 연결 실패
- Timeout
- 잘못된 JSON
- 예상과 다른 JSON 구조

따라서 테스트 중에는 다음이 실행되지 않습니다.

- 실제 OpenAI API
- 실제 LangGraph
- 실제 PyTorch 모델
- 실제 SHAP 계산
- 실제 SQLite 조회
"""

from __future__ import annotations

import json

import httpx2 as httpx
import pytest

from src.dashboard.api_client import (
    DashboardApiClient,
    DashboardApiConnectionError,
    DashboardApiHttpError,
    DashboardApiInvalidResponseError,
    DashboardApiTimeoutError,
)
from src.dashboard.config import (
    DEFAULT_DASHBOARD_API_BASE_URL,
    DEFAULT_DASHBOARD_API_TIMEOUT_SECONDS,
    DashboardApiConfig,
    load_dashboard_api_config,
)


def build_test_config() -> DashboardApiConfig:
    """
    Dashboard API Client 테스트에서 사용할 고정 설정을 생성합니다.

    실제 .env 또는 OS 환경 변수에 의존하지 않도록
    테스트 전용 Base URL과 Timeout을 사용합니다.
    """

    return DashboardApiConfig(
        base_url="http://testserver",
        timeout_seconds=5.0,
    )


def build_failure_prediction_payload() -> dict:
    """
    POST /agent/failure-prediction 요청에 사용할
    고정 설비 입력값을 생성합니다.

    주의:
    실제 API JSON 필드명은 machine_type이 아니라 type입니다.
    """

    return {
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
        "include_shap": True,
        "include_global_importance": True,
    }


def test_predict_failure_returns_response_json() -> None:
    """
    고장 예측 Client가 올바른 Endpoint와 JSON body를 사용하고,
    FastAPI JSON 응답을 dict로 반환하는지 검증합니다.
    """

    expected_response = {
        "prediction": 1,
        "probability": 0.9929,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "recommended_action": "설비 점검을 권장합니다.",
        "evidence": [],
        "answer": "고장 위험이 높게 예측되었습니다.",
        "warnings": [],
        "limitations": [],
    }

    request_payload = (
        build_failure_prediction_payload()
    )

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "POST"

        assert (
            request.url.path
            == "/agent/failure-prediction"
        )

        actual_request_json = json.loads(
            request.content.decode(
                "utf-8",
            )
        )

        assert (
            actual_request_json
            == request_payload
        )

        return httpx.Response(
            status_code=200,
            json=expected_response,
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        actual_response = (
            api_client.predict_failure(
                request_payload,
            )
        )

    assert actual_response == expected_response


def test_query_langgraph_agent_returns_response_json() -> None:
    """
    LangGraph Agent Client가 요청 JSON을 전달하고,
    FastAPI 응답을 dict로 반환하는지 검증합니다.
    """

    request_payload = {
        "question": (
            "이 설비 조건의 고장 위험을 예측해줘."
        ),
        "chat_history": [],
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
    }

    expected_response = {
        "question": request_payload[
            "question"
        ],
        "intent": "failure_prediction",
        "confidence": 0.95,
        "intent_source": "openai",
        "intent_reason": (
            "고장 위험 예측 요청입니다."
        ),
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
        "evidence": [],
        "warnings": [],
        "errors": [],
        "limitations": [],
        "trace_id": "trace-001",
        "trace_status": "success",
        "fallback_occurred": False,
        "trace_events": [],
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "POST"

        assert (
            request.url.path
            == "/agent/langgraph-query"
        )

        actual_request_json = json.loads(
            request.content.decode(
                "utf-8",
            )
        )

        assert (
            actual_request_json
            == request_payload
        )

        return httpx.Response(
            status_code=200,
            json=expected_response,
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        actual_response = (
            api_client.query_langgraph_agent(
                request_payload,
            )
        )

    assert actual_response == expected_response


def test_get_agent_executions_sends_limit_query_parameter() -> None:
    """
    실행 이력 목록 Client가 limit Query Parameter를 전달하고,
    JSON object 목록을 반환하는지 검증합니다.
    """

    expected_response = [
        {
            "id": 1,
            "trace_id": "trace-001",
            "question": "고장 위험을 예측해줘.",
            "intent": "failure_prediction",
            "risk_level": "HIGH",
            "trace_status": "success",
            "fallback_occurred": False,
            "warning_count": 0,
            "error_count": 0,
            "created_at": (
                "2026-07-13T00:00:00+00:00"
            ),
        }
    ]

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "GET"

        assert (
            request.url.path
            == "/agent/executions"
        )

        assert (
            request.url.params["limit"]
            == "5"
        )

        return httpx.Response(
            status_code=200,
            json=expected_response,
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        actual_response = (
            api_client.get_agent_executions(
                limit=5,
            )
        )

    assert actual_response == expected_response


def test_get_agent_execution_detail_encodes_trace_id() -> None:
    """
    trace_id의 공백과 슬래시가 URL Path에서
    안전하게 Percent Encoding되는지 검증합니다.
    """

    expected_response = {
        "id": 1,
        "trace_id": "trace id/001",
        "question": "고장 위험을 예측해줘.",
        "created_at": (
            "2026-07-13T00:00:00+00:00"
        ),
        "evidence": [],
        "trace_events": [],
        "warnings": [],
        "errors": [],
        "limitations": [],
    }

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        assert request.method == "GET"

        assert request.url.raw_path == (
            b"/agent/executions/"
            b"trace%20id%2F001"
        )

        return httpx.Response(
            status_code=200,
            json=expected_response,
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        actual_response = (
            api_client.get_agent_execution_detail(
                trace_id="  trace id/001  ",
            )
        )

    assert actual_response == expected_response


def test_get_agent_execution_detail_rejects_empty_trace_id() -> None:
    """
    공백뿐인 trace_id를 HTTP 요청 전에 거부하는지 검증합니다.
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        pytest.fail(
            "빈 trace_id에서는 HTTP 요청이 실행되면 안 됩니다."
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            ValueError,
            match="trace_id는 비어 있을 수 없습니다",
        ):
            api_client.get_agent_execution_detail(
                trace_id="   ",
            )


def test_request_timeout_is_converted_to_dashboard_error() -> None:
    """
    httpx Timeout 오류가
    DashboardApiTimeoutError로 변환되는지 검증합니다.
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ReadTimeout(
            "test timeout",
            request=request,
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiTimeoutError,
            match="5.0초",
        ):
            api_client.get_agent_executions()


def test_connection_failure_is_converted_to_dashboard_error() -> None:
    """
    FastAPI 연결 실패가
    DashboardApiConnectionError로 변환되는지 검증합니다.
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ConnectError(
            "test connection failure",
            request=request,
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiConnectionError,
            match="FastAPI 서버에 연결할 수 없습니다",
        ):
            api_client.get_agent_executions()


def test_other_request_error_is_converted_to_connection_error() -> None:
    """
    Timeout과 ConnectError 이외의 httpx RequestError도
    Dashboard 전용 연결 오류로 변환되는지 검증합니다.
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.NetworkError(
            "test network failure",
            request=request,
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiConnectionError,
            match="네트워크 오류",
        ):
            api_client.get_agent_executions()


def test_http_422_error_includes_safe_validation_detail() -> None:
    """
    FastAPI 422 Validation 오류의 detail이
    Dashboard HTTP 오류에 포함되는지 검증합니다.
    """

    validation_detail = [
        {
            "type": "missing",
            "loc": [
                "body",
                "question",
            ],
            "msg": "Field required",
        }
    ]

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=422,
            json={
                "detail": validation_detail,
            },
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiHttpError,
        ) as exc_info:
            api_client.query_langgraph_agent(
                {},
            )

    error = exc_info.value

    assert error.status_code == 422

    assert error.detail is not None

    assert "Field required" in error.detail

    assert "HTTP 422" in str(
        error,
    )


def test_http_500_error_does_not_expose_server_response_body() -> None:
    """
    5xx 응답 본문의 내부 예외 문자열이
    Dashboard 오류 메시지에 노출되지 않는지 검증합니다.
    """

    internal_secret = (
        "database_password=do-not-expose"
    )

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=500,
            json={
                "detail": internal_secret,
            },
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiHttpError,
        ) as exc_info:
            api_client.get_agent_executions()

    error = exc_info.value

    assert error.status_code == 500

    assert error.detail is None

    assert internal_secret not in str(
        error,
    )

    assert "서버 내부 오류" in str(
        error,
    )


def test_non_json_success_response_raises_invalid_response_error() -> None:
    """
    HTTP 200이어도 Response가 JSON이 아니면
    InvalidResponseError가 발생하는지 검증합니다.
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            text="<html>not json</html>",
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiInvalidResponseError,
            match="JSON으로 해석할 수 없습니다",
        ):
            api_client.get_agent_executions()


def test_object_response_rejects_list_top_level_json() -> None:
    """
    단일 객체를 기대하는 API가 List를 반환하면
    InvalidResponseError가 발생하는지 검증합니다.
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=[],
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiInvalidResponseError,
            match="예상 형식: dict",
        ):
            api_client.predict_failure(
                build_failure_prediction_payload(),
            )


def test_list_response_rejects_dict_top_level_json() -> None:
    """
    실행 이력 목록 API가 Dict를 반환하면
    InvalidResponseError가 발생하는지 검증합니다.
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "trace_id": "trace-001",
            },
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiInvalidResponseError,
            match="예상 형식: list",
        ):
            api_client.get_agent_executions()


def test_execution_list_rejects_non_object_item() -> None:
    """
    실행 이력 List 내부에 JSON object가 아닌 값이 있으면
    InvalidResponseError가 발생하는지 검증합니다.
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json=[
                {
                    "trace_id": "trace-001",
                },
                "invalid-item",
            ],
        )

    transport = httpx.MockTransport(
        handler,
    )

    with DashboardApiClient(
        config=build_test_config(),
        transport=transport,
    ) as api_client:
        with pytest.raises(
            DashboardApiInvalidResponseError,
            match="JSON object가 아닌 항목",
        ):
            api_client.get_agent_executions()


def test_load_dashboard_api_config_uses_default_values(
    monkeypatch,
) -> None:
    """
    Dashboard 환경 변수가 없으면
    로컬 개발 기본값을 사용하는지 검증합니다.
    """

    monkeypatch.setattr(
        (
            "src.dashboard.config."
            "_load_dotenv_safely"
        ),
        lambda: None,
    )

    monkeypatch.delenv(
        "DASHBOARD_API_BASE_URL",
        raising=False,
    )

    monkeypatch.delenv(
        "DASHBOARD_API_TIMEOUT_SECONDS",
        raising=False,
    )

    config = load_dashboard_api_config()

    assert (
        config.base_url
        == DEFAULT_DASHBOARD_API_BASE_URL
    )

    assert (
        config.timeout_seconds
        == DEFAULT_DASHBOARD_API_TIMEOUT_SECONDS
    )


def test_load_dashboard_api_config_normalizes_environment_values(
    monkeypatch,
) -> None:
    """
    Base URL의 앞뒤 공백과 마지막 슬래시를 제거하고,
    Timeout 문자열을 float로 변환하는지 검증합니다.
    """

    monkeypatch.setattr(
        (
            "src.dashboard.config."
            "_load_dotenv_safely"
        ),
        lambda: None,
    )

    monkeypatch.setenv(
        "DASHBOARD_API_BASE_URL",
        "  https://api.example.com/  ",
    )

    monkeypatch.setenv(
        "DASHBOARD_API_TIMEOUT_SECONDS",
        "45.5",
    )

    config = load_dashboard_api_config()

    assert (
        config.base_url
        == "https://api.example.com"
    )

    assert config.timeout_seconds == 45.5


@pytest.mark.parametrize(
    "invalid_base_url",
    [
        "localhost:8000",
        "ftp://example.com",
        "http://",
    ],
)
def test_load_dashboard_api_config_rejects_invalid_base_url(
    monkeypatch,
    invalid_base_url,
) -> None:
    """
    http 또는 https 형식이 아닌 Base URL을 거부하는지 검증합니다.
    """

    monkeypatch.setattr(
        (
            "src.dashboard.config."
            "_load_dotenv_safely"
        ),
        lambda: None,
    )

    monkeypatch.setenv(
        "DASHBOARD_API_BASE_URL",
        invalid_base_url,
    )

    monkeypatch.delenv(
        "DASHBOARD_API_TIMEOUT_SECONDS",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="올바른 URL",
    ):
        load_dashboard_api_config()


@pytest.mark.parametrize(
    "invalid_timeout",
    [
        "abc",
        "0",
        "-1",
        "NaN",
        "Infinity",
        "-Infinity",
    ],
)
def test_load_dashboard_api_config_rejects_invalid_timeout(
    monkeypatch,
    invalid_timeout,
) -> None:
    """
    숫자가 아니거나 0 이하이거나
    NaN·Infinity인 Timeout을 거부하는지 검증합니다.
    """

    monkeypatch.setattr(
        (
            "src.dashboard.config."
            "_load_dotenv_safely"
        ),
        lambda: None,
    )

    monkeypatch.delenv(
        "DASHBOARD_API_BASE_URL",
        raising=False,
    )

    monkeypatch.setenv(
        "DASHBOARD_API_TIMEOUT_SECONDS",
        invalid_timeout,
    )

    with pytest.raises(
        ValueError,
    ):
        load_dashboard_api_config()
