# src/mcp_server/server.py

"""
Day 20 - Manufacturing AI MCP Server.

이 파일의 역할
----------------
현재 제조 AI 프로젝트의 기능을
Model Context Protocol Server로 제공합니다.

현재 최소 구현 Tool:

    get_dataset_schema


전체 연결 구조
--------------
MCP Host

    ↓

MCP Client

    ↓

Manufacturing AI MCP Server

    ↓

get_dataset_schema Tool

    ↓

get_ai4i_dataset_schema()

    ↓

src/data/schemas.py

    ↓

구조화된 AI4I Dataset 정보 반환


중요한 설계 원칙
----------------
MCP Tool 안에 Dataset 정보를
다시 직접 작성하지 않습니다.

잘못된 예:

    @mcp.tool()
    def get_dataset_schema():

        return {
            "features": [
                "Air temperature [K]",
                ...
            ]
        }


이렇게 작성하면 LangGraph와 MCP에
동일한 정보가 중복됩니다.


현재 구현:

    MCP Tool

        ↓

    기존 Dataset Schema Service

        ↓

    기존 AI4I 상수


따라서 LangGraph와 MCP는
동일한 Dataset 기준을 사용합니다.


FastAPI와 MCP의 관계
--------------------
FastAPI:

    일반 HTTP API 제공


MCP Server:

    AI Host와 Agent가 사용할
    표준 Tool interface 제공


두 인터페이스는 서로 대체하지 않습니다.

둘 다 기존 Application Service를
재사용하는 병행 구조입니다.
"""

from __future__ import annotations

# Any
# ---
# MCP Tool은 Dataset 이름, feature list,
# target, mapping, 설명 문자열 등
# 여러 종류의 JSON 호환 값을 반환합니다.
#
# 따라서 반환 dict의 value type을
# Any로 표현합니다.
from typing import Any

# FastMCP
# -------
# 공식 MCP Python SDK가 제공하는
# 고수준 MCP Server class입니다.
#
# FastMCP는 다음 역할을 담당합니다.
#
# MCP protocol 처리
#
# Tool 등록
#
# Tool schema 생성
#
# Client 요청 routing
#
# transport 실행
from mcp.server.fastmcp import FastMCP

# 기존 Dataset Schema Application Service입니다.
#
# MCP Tool 안에 AI4I feature와 target을
# 다시 작성하지 않고,
# 이미 테스트한 기존 Service를 호출합니다.
from src.services.dataset_schema_service import (
    get_ai4i_dataset_schema,
)


# MCP Server 이름입니다.
#
# MCP Client가 Server 정보를 조회할 때
# 어떤 Server에 연결했는지 식별하는 데 사용합니다.
MCP_SERVER_NAME = (
    "manufacturing-ai-quality-agent"
)


# MCP Server 설명입니다.
#
# Host 또는 Client가 이 Server의 목적을
# 이해할 수 있도록 간단한 지침을 제공합니다.
MCP_SERVER_INSTRUCTIONS = (
    "AI4I 2020 기반 제조 AI 프로젝트의 "
    "Dataset Schema와 제조 분석 기능을 "
    "MCP Tool로 제공합니다."
)


# FastMCP Server 객체를 생성합니다.
#
# 이 객체가 현재 MCP Server의 중심입니다.
#
# 이후:
#
# @mcp.tool()
#
# decorator를 사용하면
# Python 함수를 MCP Tool로 등록할 수 있습니다.
mcp = FastMCP(
    name=MCP_SERVER_NAME,
    instructions=MCP_SERVER_INSTRUCTIONS,
)


@mcp.tool(
    name="get_dataset_schema",
    title="Get AI4I Dataset Schema",
    description=(
        "현재 제조 AI 프로젝트에서 사용하는 "
        "AI4I 2020 Dataset의 feature, target, "
        "제외 column과 범주형 mapping 정보를 반환합니다."
    ),
    structured_output=True,
)
def get_dataset_schema() -> dict[str, Any]:
    """
    현재 프로젝트의 AI4I Dataset Schema를 반환합니다.

    Parameters
    ----------
    없음

    현재 프로젝트에서 사용할 Dataset은
    AI4I 2020으로 정해져 있으므로
    별도의 Dataset 이름을 입력받지 않습니다.


    Returns
    -------
    dict[str, Any]

    반환 정보:

        dataset

        features

        numeric_features

        categorical_features

        target

        excluded_columns

        categorical_mappings

        current_encoding_note

        improvement_note


    MCP Tool의 역할
    ----------------
    이 함수는 Dataset 정보를
    직접 생성하지 않습니다.

    기존 Application Service와
    MCP protocol을 연결하는
    Adapter 역할만 담당합니다.


    실행 흐름
    ----------
    MCP Client

        ↓

    tools/call

        ↓

    get_dataset_schema

        ↓

    get_ai4i_dataset_schema

        ↓

    기존 AI4I 상수

        ↓

    구조화 결과 반환


    왜 LangGraph를 호출하지 않는가?
    -------------------------------
    Dataset Schema 자체를 조회하는 데는
    Intent 분류나 Agent workflow가 필요하지 않습니다.

    따라서:

        MCP Tool

            ↓

        LangGraph

            ↓

        Dataset Service

    처럼 불필요하게 우회하지 않습니다.


    현재 구조:

        MCP Tool

            ↓

        Dataset Service


    이렇게 하면 MCP Tool은
    필요한 Application Logic만
    직접 재사용할 수 있습니다.
    """

    # 기존 Dataset Schema Service를 호출합니다.
    #
    # Service는:
    #
    # src/data/schemas.py
    #
    # 에 정의된 기존 상수를 사용합니다.
    schema = get_ai4i_dataset_schema()

    # 일반 dict 형태로 반환합니다.
    #
    # get_ai4i_dataset_schema()는 이미
    # JSON으로 표현 가능한 값만 포함합니다.
    #
    # 예:
    #
    # str
    #
    # list[str]
    #
    # dict[str, int]
    return dict(schema)


def main() -> None:
    """
    Manufacturing AI MCP Server를 실행합니다.

    Parameters
    ----------
    없음


    Returns
    -------
    None

    Server 실행이 종료될 때까지
    현재 process가 MCP 요청을 처리합니다.


    Transport
    ---------
    현재 Day 20에서는 stdio를 사용합니다.

    stdio:

        standard input

        +

        standard output


    MCP Client가 Server process를 실행한 뒤
    표준 입력과 표준 출력을 통해
    MCP JSON-RPC 메시지를 주고받습니다.


    stdio를 먼저 사용하는 이유
    --------------------------
    별도 port가 필요하지 않습니다.

    외부 network가 필요하지 않습니다.

    별도 HTTP Server를 실행하지 않아도 됩니다.

    MCP Host·Client·Server 구조를
    가장 작은 범위에서 검증할 수 있습니다.


    향후 확장
    ---------
    원격 MCP Server가 필요하면:

        streamable-http

    transport를 검토할 수 있습니다.
    """

    # transport를 명시적으로 stdio로 지정합니다.
    #
    # 현재 SDK의 기본값도 stdio이지만,
    # 코드만 읽어도 현재 전송 방식을
    # 알 수 있도록 명시합니다.
    mcp.run(
        transport="stdio",
    )


# 이 파일을 module로 import할 때는
# Server를 자동 실행하지 않습니다.
#
# 예:
#
# from src.mcp_server.server import mcp
#
# 이 경우:
#
# Tool 목록 검사
#
# 단위 테스트
#
# Client 연결 준비
#
# 만 수행할 수 있습니다.
#
#
# 다음처럼 직접 실행할 때만:
#
# python -m src.mcp_server.server
#
# main()을 호출하여
# stdio MCP Server를 시작합니다.
if __name__ == "__main__":
    main()