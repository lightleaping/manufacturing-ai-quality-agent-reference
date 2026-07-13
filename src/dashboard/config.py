# src/dashboard/config.py

"""
Dashboard FastAPI Client에서 사용할 설정을 관리합니다.

이 모듈은 다음 설정을 담당합니다.

- FastAPI Base URL
- HTTP 요청 Timeout

환경 변수가 설정되어 있으면 해당 값을 사용하고,
설정되어 있지 않으면 로컬 개발용 기본값을 사용합니다.

중요:
- OpenAI API Key와 같은 비밀정보를 다루지 않습니다.
- PyTorch 모델, LangGraph, SQLite에 직접 접근하지 않습니다.
- HTTP 요청 자체는 api_client.py가 담당합니다.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from urllib.parse import urlparse


# 환경 변수 이름을 상수로 관리합니다.
#
# 문자열을 여러 파일에 직접 반복해서 쓰지 않으면
# 오타를 줄이고 설정 이름을 한 곳에서 확인할 수 있습니다.
DASHBOARD_API_BASE_URL_ENV = "DASHBOARD_API_BASE_URL"
DASHBOARD_API_TIMEOUT_SECONDS_ENV = (
    "DASHBOARD_API_TIMEOUT_SECONDS"
)


# 환경 변수가 없을 때 사용할 로컬 개발 기본값입니다.
DEFAULT_DASHBOARD_API_BASE_URL = (
    "http://127.0.0.1:8000"
)

# Day 18 실측 요청 시간이 약 2~5초였으므로,
# 첫 Artifact 로딩, OpenAI 응답 지연, SHAP 계산 시간을 고려해
# 기본 Timeout은 30초로 설정합니다.
DEFAULT_DASHBOARD_API_TIMEOUT_SECONDS = 30.0


@dataclass(
    frozen=True,
    slots=True,
)
class DashboardApiConfig:
    """
    Dashboard FastAPI Client가 사용할 설정입니다.

    frozen=True
    -----------
    객체를 생성한 뒤 설정값이 실수로 변경되는 것을 막습니다.

    slots=True
    ----------
    이 클래스에 정의되지 않은 속성이
    실수로 추가되는 것을 막습니다.

    Attributes
    ----------
    base_url:
        Dashboard가 호출할 FastAPI 서버의 기본 주소입니다.

    timeout_seconds:
        HTTP 요청이 완료되기를 기다릴 최대 시간입니다.
        단위는 초입니다.
    """

    base_url: str

    timeout_seconds: float


def _load_dotenv_safely() -> None:
    """
    프로젝트 루트의 .env 파일을 가능한 경우 로드합니다.

    python-dotenv import 또는 .env 로딩에 실패해도
    Dashboard 설정 전체를 즉시 중단하지 않습니다.

    운영 환경에서는 .env 대신
    OS 환경 변수나 배포 환경의 Secret 설정을
    사용할 수도 있기 때문입니다.
    """

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        # .env 로딩 실패가 곧 환경 변수 부재를 의미하지는 않습니다.
        #
        # os.getenv()는 이미 등록된 OS 환경 변수를
        # 계속 확인할 수 있습니다.
        pass


def _normalize_base_url(
    raw_base_url: str | None,
) -> str:
    """
    환경 변수의 Base URL을 정리하고 검증합니다.

    처리 순서:
    1. 값이 없거나 공백뿐이면 기본 URL 사용
    2. 앞뒤 공백 제거
    3. 마지막 슬래시 제거
    4. http 또는 https URL인지 검증

    마지막 슬래시를 제거하는 이유:
    ------------------------------
    Base URL:

        http://127.0.0.1:8000/

    Endpoint:

        /agent/failure-prediction

    을 단순 결합할 때 이중 슬래시가 생기는 것을 줄이기 위함입니다.
    """

    if raw_base_url is None:
        return DEFAULT_DASHBOARD_API_BASE_URL

    normalized_base_url = (
        raw_base_url
        .strip()
        .rstrip("/")
    )

    if not normalized_base_url:
        return DEFAULT_DASHBOARD_API_BASE_URL

    parsed_url = urlparse(normalized_base_url)

    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
    ):
        raise ValueError(
            (
                f"{DASHBOARD_API_BASE_URL_ENV}는 "
                "http 또는 https 형식의 올바른 URL이어야 합니다."
            )
        )

    return normalized_base_url


def _parse_timeout_seconds(
    raw_timeout_seconds: str | None,
) -> float:
    """
    환경 변수의 Timeout 값을 양의 유한한 float로 변환합니다.

    유효한 예:

        10
        30
        45.5

    잘못된 예:

        abc
        0
        -1
        NaN
        Infinity
    """

    if raw_timeout_seconds is None:
        return DEFAULT_DASHBOARD_API_TIMEOUT_SECONDS

    normalized_timeout = raw_timeout_seconds.strip()

    if not normalized_timeout:
        return DEFAULT_DASHBOARD_API_TIMEOUT_SECONDS

    try:
        timeout_seconds = float(
            normalized_timeout,
        )
    except ValueError as exc:
        raise ValueError(
            (
                f"{DASHBOARD_API_TIMEOUT_SECONDS_ENV}는 "
                "초 단위 숫자여야 합니다."
            )
        ) from exc

    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0.0
    ):
        raise ValueError(
            (
                f"{DASHBOARD_API_TIMEOUT_SECONDS_ENV}는 "
                "0보다 큰 유한한 숫자여야 합니다."
            )
        )

    return timeout_seconds


def load_dashboard_api_config() -> DashboardApiConfig:
    """
    환경 변수와 기본값을 이용해 Dashboard API 설정을 생성합니다.

    요청 흐름:

        .env 또는 OS 환경 변수
            ↓
        값 정리·검증
            ↓
        DashboardApiConfig
            ↓
        DashboardApiClient
    """

    _load_dotenv_safely()

    base_url = _normalize_base_url(
        os.getenv(
            DASHBOARD_API_BASE_URL_ENV,
        ),
    )

    timeout_seconds = _parse_timeout_seconds(
        os.getenv(
            DASHBOARD_API_TIMEOUT_SECONDS_ENV,
        ),
    )

    return DashboardApiConfig(
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
