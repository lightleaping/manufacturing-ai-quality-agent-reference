# src/dashboard/api_client.py

"""
Streamlit Dashboard에서 기존 FastAPI를 호출하기 위한 HTTP Client입니다.

이 모듈의 역할:
- 설비 고장 위험 예측 API 호출
- LangGraph Agent Query API 호출
- Agent 실행 이력 목록 API 호출
- Agent 실행 이력 상세 API 호출
- HTTP 오류를 Dashboard 전용 예외로 변환
- FastAPI Response JSON의 기본 구조 검증

중요:
- PyTorch 모델을 직접 로드하지 않습니다.
- LangGraph workflow를 직접 실행하지 않습니다.
- SQLite를 직접 조회하지 않습니다.
- Risk Level, Evidence, Answer를 다시 계산하지 않습니다.
- 기존 FastAPI가 반환한 검증된 결과를 그대로 사용합니다.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import httpx2 as httpx

from src.dashboard.config import (
    DashboardApiConfig,
    load_dashboard_api_config,
)


# Dashboard에서 호출할 FastAPI Endpoint를 상수로 관리합니다.
#
# Endpoint 문자열을 여러 함수에 직접 반복해서 작성하지 않으면
# 경로 변경 시 한 곳만 수정할 수 있고 오타도 줄일 수 있습니다.
FAILURE_PREDICTION_ENDPOINT = (
    "/agent/failure-prediction"
)

LANGGRAPH_AGENT_QUERY_ENDPOINT = (
    "/agent/langgraph-query"
)

AGENT_EXECUTIONS_ENDPOINT = (
    "/agent/executions"
)


class DashboardApiClientError(Exception):
    """
    Dashboard FastAPI Client 오류의 최상위 예외입니다.

    Streamlit UI에서는 필요하면
    세부 예외를 각각 처리할 수도 있고,
    이 부모 예외 하나로 공통 처리할 수도 있습니다.

    예:

        try:
            result = api_client.predict_failure(
                payload,
            )
        except DashboardApiClientError as exc:
            st.error(str(exc))
    """


class DashboardApiConnectionError(
    DashboardApiClientError,
):
    """
    FastAPI 서버에 연결하지 못했을 때 발생합니다.

    가능한 원인:
    - FastAPI 서버가 실행 중이 아님
    - Base URL이 잘못됨
    - 네트워크 연결 문제
    """


class DashboardApiTimeoutError(
    DashboardApiClientError,
):
    """
    설정된 Timeout 안에 API 요청이 끝나지 않았을 때 발생합니다.
    """


class DashboardApiHttpError(
    DashboardApiClientError,
):
    """
    FastAPI가 4xx 또는 5xx HTTP 상태 코드를 반환했을 때 발생합니다.

    Attributes
    ----------
    status_code:
        FastAPI가 반환한 HTTP 상태 코드입니다.

    detail:
        사용자에게 보여줄 수 있는 안전한 오류 상세입니다.

        4xx 응답에서는 FastAPI validation detail을
        가능한 범위에서 포함할 수 있습니다.

        5xx 응답에서는 내부 구현 정보 노출을 줄이기 위해
        서버 응답 본문을 그대로 표시하지 않습니다.
    """

    def __init__(
        self,
        *,
        status_code: int,
        detail: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.detail = detail

        if detail:
            message = (
                "FastAPI 요청이 실패했습니다. "
                f"HTTP {status_code}: {detail}"
            )
        elif status_code >= 500:
            message = (
                "FastAPI 서버 내부 오류가 발생했습니다. "
                f"HTTP {status_code}"
            )
        else:
            message = (
                "FastAPI 요청이 실패했습니다. "
                f"HTTP {status_code}"
            )

        super().__init__(
            message,
        )


class DashboardApiInvalidResponseError(
    DashboardApiClientError,
):
    """
    FastAPI 응답을 정상적인 JSON으로 해석할 수 없거나,
    예상한 JSON 구조와 다를 때 발생합니다.
    """


def _extract_safe_http_error_detail(
    response: httpx.Response,
) -> str | None:
    """
    4xx Response에서 사용자에게 보여줄 안전한 detail을 추출합니다.

    FastAPI validation 오류는 보통 다음 구조입니다.

        {
            "detail": [
                {
                    "type": "missing",
                    "loc": [
                        "body",
                        "question"
                    ],
                    "msg": "Field required"
                }
            ]
        }

    detail이 문자열이면 그대로 사용하고,
    list 또는 dict이면 JSON 문자열로 변환합니다.

    5xx 응답은 내부 예외 문자열이 포함될 가능성이 있으므로
    이 함수의 호출 대상에서 제외합니다.
    """

    try:
        payload = response.json()
    except ValueError:
        # JSON이 아닌 4xx 응답이면
        # 지나치게 긴 본문을 그대로 노출하지 않고
        # 앞부분만 제한적으로 사용합니다.
        response_text = response.text.strip()

        if not response_text:
            return None

        return response_text[:500]

    if not isinstance(
        payload,
        dict,
    ):
        return None

    detail = payload.get(
        "detail",
    )

    if detail is None:
        return None

    if isinstance(
        detail,
        str,
    ):
        return detail[:500]

    try:
        serialized_detail = json.dumps(
            detail,
            ensure_ascii=False,
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    return serialized_detail[:500]


class DashboardApiClient:
    """
    Streamlit Dashboard에서 기존 FastAPI를 호출하는 동기 HTTP Client입니다.

    왜 동기 Client를 사용하는가?
    ----------------------------
    현재 Dashboard는 Streamlit 기반입니다.

    Streamlit의 일반적인 Widget 실행 흐름에서는
    동기 HTTP 요청 구조가 이해하고 사용하기 단순합니다.

    Parameters
    ----------
    config:
        Base URL과 Timeout 설정입니다.

        전달하지 않으면
        load_dashboard_api_config()를 호출하여
        환경 변수 또는 기본값으로 생성합니다.

    transport:
        httpx 전송 계층입니다.

        운영 코드에서는 전달하지 않습니다.

        Unit Test에서는 httpx.MockTransport를 전달하여
        실제 FastAPI 서버 없이 HTTP 응답을 재현할 수 있습니다.
    """

    def __init__(
        self,
        *,
        config: DashboardApiConfig | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else load_dashboard_api_config()
        )

        self._client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
            transport=transport,
            headers={
                "Accept": "application/json",
            },
        )

    def __enter__(
        self,
    ) -> DashboardApiClient:
        """
        with 문에서 Client를 사용할 수 있게 합니다.

        예:

            with DashboardApiClient() as api_client:
                result = api_client.get_agent_executions()
        """

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """
        with 문이 끝날 때 HTTP 연결 자원을 정리합니다.
        """

        self.close()

    def close(
        self,
    ) -> None:
        """
        내부 httpx.Client의 연결 자원을 정리합니다.

        Streamlit에서 Client를 장시간 재사용하는 경우에는
        애플리케이션 생명주기에 맞춰 호출할 수 있습니다.
        """

        self._client.close()

    def _request_json(
        self,
        *,
        method: str,
        endpoint: str,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        expected_type: type[dict] | type[list],
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        FastAPI에 HTTP 요청을 보내고 JSON Response를 반환합니다.

        공통 처리:
        1. HTTP 요청
        2. Timeout 처리
        3. 연결 실패 처리
        4. HTTP 상태 코드 오류 처리
        5. JSON 변환
        6. 예상 Response 구조 검증

        expected_type:
            단일 객체 Response:
                dict

            실행 이력 목록 Response:
                list
        """

        try:
            response = self._client.request(
                method=method,
                url=endpoint,
                json=json_body,
                params=params,
            )

        except httpx.TimeoutException as exc:
            raise DashboardApiTimeoutError(
                (
                    "FastAPI 요청 시간이 초과되었습니다. "
                    f"현재 Timeout은 "
                    f"{self.config.timeout_seconds:.1f}초입니다."
                )
            ) from exc

        except httpx.ConnectError as exc:
            raise DashboardApiConnectionError(
                (
                    "FastAPI 서버에 연결할 수 없습니다. "
                    "서버 실행 상태와 Base URL을 확인해주세요. "
                    f"현재 Base URL: {self.config.base_url}"
                )
            ) from exc

        except httpx.RequestError as exc:
            # TimeoutException과 ConnectError를 제외한
            # 기타 네트워크·전송 계층 오류입니다.
            raise DashboardApiConnectionError(
                (
                    "FastAPI 요청 중 네트워크 오류가 발생했습니다. "
                    f"현재 Base URL: {self.config.base_url}"
                )
            ) from exc

        if response.is_error:
            safe_detail = None

            # 4xx는 잘못된 입력값 등
            # 사용자가 수정할 수 있는 정보일 수 있으므로
            # FastAPI detail을 제한적으로 전달합니다.
            if 400 <= response.status_code < 500:
                safe_detail = (
                    _extract_safe_http_error_detail(
                        response,
                    )
                )

            raise DashboardApiHttpError(
                status_code=response.status_code,
                detail=safe_detail,
            )

        try:
            response_payload = response.json()

        except ValueError as exc:
            raise DashboardApiInvalidResponseError(
                (
                    "FastAPI 응답을 JSON으로 해석할 수 없습니다. "
                    "서버 응답 형식을 확인해주세요."
                )
            ) from exc

        if not isinstance(
            response_payload,
            expected_type,
        ):
            raise DashboardApiInvalidResponseError(
                (
                    "FastAPI 응답의 최상위 JSON 구조가 "
                    "예상한 형식과 다릅니다. "
                    f"예상 형식: {expected_type.__name__}, "
                    f"실제 형식: "
                    f"{type(response_payload).__name__}"
                )
            )

        # 실행 이력 목록은
        # list 안의 각 항목도 JSON object여야 합니다.
        if expected_type is list:
            invalid_item_exists = any(
                not isinstance(
                    item,
                    dict,
                )
                for item in response_payload
            )

            if invalid_item_exists:
                raise DashboardApiInvalidResponseError(
                    (
                        "FastAPI 실행 이력 응답에 "
                        "JSON object가 아닌 항목이 포함되어 있습니다."
                    )
                )

        return response_payload

    def predict_failure(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        설비 고장 위험 예측 API를 호출합니다.

        Endpoint:

            POST /agent/failure-prediction

        예상 Request 예:

            {
                "air_temperature": 303.0,
                "process_temperature": 312.5,
                "rotational_speed": 1380.0,
                "torque": 62.0,
                "tool_wear": 220.0,
                "type": "L",
                "include_shap": True,
                "include_global_importance": True
            }

        중요:
        - API JSON 필드명은 type입니다.
        - machine_type은 FastAPI 내부 Python 필드명입니다.
        """

        response_payload = self._request_json(
            method="POST",
            endpoint=FAILURE_PREDICTION_ENDPOINT,
            json_body=payload,
            expected_type=dict,
        )

        return response_payload

    def query_langgraph_agent(
        self,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        LangGraph Agent Query API를 호출합니다.

        Endpoint:

            POST /agent/langgraph-query

        chat_history는 질문 문맥 이해용입니다.

        이전 요청의 raw_sample은 자동 재사용하지 않습니다.
        고장 예측이 필요하면 현재 payload의 raw_sample에
        설비 입력값을 다시 포함해야 합니다.
        """

        response_payload = self._request_json(
            method="POST",
            endpoint=LANGGRAPH_AGENT_QUERY_ENDPOINT,
            json_body=payload,
            expected_type=dict,
        )

        return response_payload

    def get_agent_executions(
        self,
        *,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """
        최근 Agent 실행 이력 목록을 조회합니다.

        Endpoint:

            GET /agent/executions?limit=20

        반환값은 실행 이력 JSON object의 list입니다.
        """

        response_payload = self._request_json(
            method="GET",
            endpoint=AGENT_EXECUTIONS_ENDPOINT,
            params={
                "limit": limit,
            },
            expected_type=list,
        )

        return response_payload

    def get_agent_execution_detail(
        self,
        *,
        trace_id: str,
    ) -> dict[str, Any]:
        """
        trace_id에 해당하는 Agent 실행 이력 상세를 조회합니다.

        Endpoint:

            GET /agent/executions/{trace_id}

        trace_id는 URL 경로에 들어가므로
        공백을 제거하고 URL에 안전한 문자열로 변환합니다.
        """

        normalized_trace_id = trace_id.strip()

        if not normalized_trace_id:
            raise ValueError(
                "trace_id는 비어 있을 수 없습니다."
            )

        encoded_trace_id = quote(
            normalized_trace_id,
            safe="",
        )

        endpoint = (
            f"{AGENT_EXECUTIONS_ENDPOINT}/"
            f"{encoded_trace_id}"
        )

        response_payload = self._request_json(
            method="GET",
            endpoint=endpoint,
            expected_type=dict,
        )

        return response_payload
