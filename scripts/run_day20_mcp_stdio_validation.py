# scripts/run_day20_mcp_stdio_validation.py

"""
Day 20 - Real MCP stdio Local E2E Validation.

이 스크립트의 목적
-----------------
실제 MCP Client가
별도 MCP Server process를 실행한 뒤,

stdio transport를 통해:

    initialize

    tools/list

    tools/call

요청을 수행하는 전체 연결을 검증합니다.


전체 실행 구조
--------------
현재 validation process

    ↓

stdio_client()

    ↓

별도 Python subprocess 실행

    python -m src.mcp_server.server

    ↓

Manufacturing AI MCP Server

    ↓

get_dataset_schema Tool

    ↓

기존 Dataset Schema Service

    ↓

기존 AI4I schema 상수

    ↓

MCP Client로 결과 반환


이번 검증과 단위 테스트의 차이
------------------------------
기존 tests/test_mcp_server.py:

    같은 Python process 안에서

        mcp.list_tools()

        mcp.call_tool()

    호출


현재 실제 Local E2E:

    MCP Client process

        ↓

    stdio transport

        ↓

    별도 MCP Server subprocess

        ↓

    MCP protocol 요청·응답


따라서 이번 단계에서는
실제 Host·Client·Server 연결에 더 가까운
로컬 MCP 통신 경로를 확인합니다.


안전 정책
---------
실제 OpenAI API를 호출하지 않습니다.

외부 MCP network를 사용하지 않습니다.

PyTorch 모델을 실행하지 않습니다.

운영 SQLite DB를 사용하지 않습니다.

환경 변수 전체를 출력하지 않습니다.

OPENAI_API_KEY를 출력하지 않습니다.
"""

from __future__ import annotations

# json
# ----
# MCP Tool의 TextContent 안에는
# Dataset Schema가 JSON 문자열로 들어 있습니다.
#
# 이를 Python dict로 변환하여
# structuredContent와 같은지 검증합니다.
import json

# sys
# ---
# 현재 validation script를 실행 중인
# Python interpreter의 정확한 경로를 가져옵니다.
#
# sys.executable 예:
#
# C:\...\manufacturing-ai-quality-agent-reference\
# .venv\Scripts\python.exe
#
# 단순히 command="python"을 사용하지 않고
# 현재 가상환경의 Python을 명시하기 위해 사용합니다.
import sys

# Path
# ----
# 현재 script 위치를 기준으로
# 프로젝트 root 경로를 계산합니다.
#
# scripts/
# └─ run_day20_mcp_stdio_validation.py
#
# 현재 파일의 부모:
#
# scripts
#
# 부모의 부모:
#
# 프로젝트 root
from pathlib import Path

# Any
# ---
# MCP SDK의 model_dump() 결과에는
# 문자열, list, dict 등
# 여러 JSON 호환 값이 들어 있습니다.
from typing import Any

# anyio
# -----
# MCP ClientSession과 stdio 연결은
# 비동기 방식으로 동작합니다.
#
# 현재 script의 async main 함수를
# 실행하기 위해 사용합니다.
import anyio

# ClientSession
# -------------
# MCP Client와 MCP Server 사이의
# protocol session을 관리합니다.
#
# initialize()
#
# list_tools()
#
# call_tool()
#
# 등의 MCP 요청을 제공합니다.
#
#
# StdioServerParameters
# ---------------------
# stdio Client가 어떤 command로
# MCP Server process를 실행할지 정의합니다.
from mcp import (
    ClientSession,
    StdioServerParameters,
)

# stdio_client
# ------------
# 별도 MCP Server subprocess를 실행하고,
# Client와 Server 사이의 표준 입력·출력 stream을
# 연결하는 async context manager입니다.
from mcp.client.stdio import stdio_client


# 현재 script 파일:
#
# project_root/
# └─ scripts/
#    └─ run_day20_mcp_stdio_validation.py
#
# parents[0]:
#
# scripts
#
# parents[1]:
#
# project_root
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# 실제 연결할 MCP Server 이름입니다.
#
# src/mcp_server/server.py의:
#
# MCP_SERVER_NAME
#
# 과 같아야 합니다.
EXPECTED_SERVER_NAME = (
    "manufacturing-ai-quality-agent"
)


# 실제 MCP Server에서 조회할
# Day 20 최소 Tool 이름입니다.
EXPECTED_TOOL_NAME = (
    "get_dataset_schema"
)


# Dataset Schema 결과의
# 핵심 예상값입니다.
EXPECTED_DATASET_NAME = (
    "AI4I 2020 Predictive Maintenance Dataset"
)

EXPECTED_TARGET = (
    "Machine failure"
)


def print_section(
    title: str,
) -> None:
    """
    Validation 출력 구역을 보기 좋게 구분합니다.

    Parameters
    ----------
    title:
        출력할 section 제목입니다.


    Returns
    -------
    None
    """

    print()
    print(
        "=" * 100
    )

    print(title)

    print(
        "=" * 100
    )


def to_alias_dict(
    value: Any,
) -> dict[str, Any]:
    """
    MCP Pydantic 결과 객체를
    JSON alias 이름 기준 dict로 변환합니다.

    Parameters
    ----------
    value:
        MCP SDK가 반환한 Pydantic model입니다.


    Returns
    -------
    dict[str, Any]

    예:

        InitializeResult

            ↓

        {
            "serverInfo": {
                ...
            }
        }


    왜 by_alias=True를 사용하는가?
    --------------------------------
    MCP protocol field 이름은 다음처럼
    camelCase인 경우가 있습니다.

        serverInfo

        structuredContent

        isError


    Python 내부 field 이름과
    protocol field 이름이 다를 수 있으므로,
    실제 MCP message에 가까운 alias 이름을 사용합니다.
    """

    if not hasattr(
        value,
        "model_dump",
    ):
        raise TypeError(
            "MCP result must support model_dump(). "
            f"actual_type={type(value).__name__}"
        )

    result = value.model_dump(
        by_alias=True,
        exclude_none=True,
    )

    if not isinstance(
        result,
        dict,
    ):
        raise TypeError(
            "model_dump() result must be dict. "
            f"actual_type={type(result).__name__}"
        )

    return result


def find_tool_by_name(
    tools: list[Any],
    tool_name: str,
) -> Any:
    """
    MCP Tool 목록에서
    이름이 일치하는 Tool을 찾습니다.

    Parameters
    ----------
    tools:
        session.list_tools()가 반환한
        Tool 객체 목록입니다.

    tool_name:
        찾을 MCP Tool 이름입니다.


    Returns
    -------
    Any

    이름이 일치하는 MCP Tool 객체입니다.


    Raises
    ------
    AssertionError

    요청한 Tool이 등록되어 있지 않으면
    validation을 실패시킵니다.
    """

    for tool in tools:
        if getattr(
            tool,
            "name",
            None,
        ) == tool_name:
            return tool

    available_tool_names = [
        getattr(
            tool,
            "name",
            None,
        )
        for tool in tools
    ]

    raise AssertionError(
        f"Expected MCP Tool was not found. "
        f"expected={tool_name}, "
        f"available={available_tool_names}"
    )


def extract_text_output(
    call_result_data: dict[str, Any],
) -> dict[str, Any]:
    """
    CallToolResult의 TextContent에서
    JSON Dataset Schema를 추출합니다.

    Parameters
    ----------
    call_result_data:
        CallToolResult를
        alias 기반 dict로 변환한 결과입니다.


    Returns
    -------
    dict[str, Any]

    TextContent.text 안의 JSON 문자열을
    Python dict로 변환한 결과입니다.
    """

    content_blocks = (
        call_result_data.get(
            "content",
            [],
        )
    )

    assert isinstance(
        content_blocks,
        list,
    ), (
        "CallToolResult.content must be list. "
        f"actual_type={type(content_blocks).__name__}"
    )

    assert content_blocks, (
        "CallToolResult.content must not be empty."
    )

    first_content = (
        content_blocks[0]
    )

    assert isinstance(
        first_content,
        dict,
    ), (
        "First content block must be dict "
        "after model_dump()."
    )

    assert (
        first_content.get(
            "type"
        )
        ==
        "text"
    ), (
        "First MCP content block must be "
        "TextContent."
    )

    text = first_content.get(
        "text"
    )

    assert isinstance(
        text,
        str,
    ), (
        "TextContent.text must be str."
    )

    parsed_output = json.loads(
        text
    )

    assert isinstance(
        parsed_output,
        dict,
    ), (
        "Parsed TextContent JSON "
        "must be dict."
    )

    return parsed_output


async def run_validation() -> None:
    """
    실제 MCP stdio Local E2E를 실행합니다.

    실행 순서
    ----------
    1.

    현재 가상환경 Python 경로 확인


    2.

    MCP Server subprocess 실행 정보 생성


    3.

    stdio Client 연결


    4.

    ClientSession 생성


    5.

    MCP initialize


    6.

    Tool 목록 조회


    7.

    get_dataset_schema Tool 확인


    8.

    Tool 실제 호출


    9.

    TextContent 확인


    10.

    structuredContent 확인


    11.

    Dataset Schema 핵심값 검증


    12.

    Client·Server context 정상 종료
    """

    print_section(
        "DAY 20 - REAL MCP STDIO LOCAL E2E VALIDATION"
    )

    print(
        "[PROJECT]"
    )

    print(
        f"project_root      : {PROJECT_ROOT}"
    )

    print(
        f"python_executable : {sys.executable}"
    )

    print(
        f"server_module     : "
        f"src.mcp_server.server"
    )

    print(
        f"transport         : stdio"
    )


    # 현재 가상환경 Python으로
    # MCP Server module을 실행합니다.
    #
    # 실제 subprocess 명령:
    #
    # <현재 .venv Python>
    #
    #     -m
    #
    #     src.mcp_server.server
    server_parameters = (
        StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "src.mcp_server.server",
            ],
            cwd=str(
                PROJECT_ROOT
            ),
            encoding="utf-8",
            encoding_error_handler=(
                "strict"
            ),
        )
    )


    print_section(
        "1. MCP STDIO SERVER CONNECTION"
    )

    print(
        "[INFO] Starting MCP Server "
        "as a local subprocess."
    )

    print(
        "[INFO] Connecting through "
        "stdin/stdout streams."
    )


    # stdio_client()는 async 함수가 아니라
    # async context manager를 반환합니다.
    #
    # context 진입:
    #
    # Server subprocess 시작
    #
    # stdio stream 연결
    #
    #
    # context 종료:
    #
    # stream 정리
    #
    # Server subprocess 종료
    async with stdio_client(
        server_parameters
    ) as (
        read_stream,
        write_stream,
    ):

        print(
            "[PASS] MCP stdio streams connected"
        )


        # ClientSession은
        # MCP protocol 요청·응답을 관리합니다.
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:

            print(
                "[PASS] MCP ClientSession created"
            )


            print_section(
                "2. MCP INITIALIZE"
            )


            # initialize는 MCP Client와 Server가
            # protocol version과 capability,
            # Server 정보를 교환하는 초기 단계입니다.
            initialize_result = (
                await session.initialize()
            )

            initialize_data = (
                to_alias_dict(
                    initialize_result
                )
            )

            server_info = (
                initialize_data.get(
                    "serverInfo",
                    {},
                )
            )

            assert isinstance(
                server_info,
                dict,
            )

            server_name = (
                server_info.get(
                    "name"
                )
            )

            print(
                f"server_name       : "
                f"{server_name}"
            )

            print(
                "protocol_version   : "
                f"{initialize_data.get('protocolVersion')}"
            )

            print(
                "server_capabilities: "
                f"{initialize_data.get('capabilities')}"
            )

            assert (
                server_name
                ==
                EXPECTED_SERVER_NAME
            ), (
                "Unexpected MCP Server name. "
                f"expected={EXPECTED_SERVER_NAME}, "
                f"actual={server_name}"
            )

            print(
                "[PASS] MCP initialize completed"
            )


            print_section(
                "3. MCP TOOLS/LIST"
            )


            # 실제 MCP protocol을 통해
            # Server가 공개하는 Tool 목록을 조회합니다.
            list_tools_result = (
                await session.list_tools()
            )

            tools = (
                list_tools_result.tools
            )

            print(
                f"tool_count        : "
                f"{len(tools)}"
            )

            # Tool 이름 목록을 먼저 일반 Python list로 만듭니다.
            #
            # Python 3.11에서는 f-string의 중괄호 안에
            # 여러 줄 list comprehension을 직접 작성하면
            # SyntaxError가 발생할 수 있습니다.
            #
            # 따라서 계산과 문자열 출력을 분리합니다.
            tool_names = [
                tool.name
                for tool in tools
            ]

            print(
                "tool_names        : "
                f"{tool_names}"
            )


            # 예상 Tool을 이름으로 찾습니다.
            #
            # 앞으로 Tool이 추가되더라도
            # 무조건 Tool 수가 1이라고 가정하지 않고
            # 필요한 Tool이 존재하는지 확인합니다.
            dataset_schema_tool = (
                find_tool_by_name(
                    tools,
                    EXPECTED_TOOL_NAME,
                )
            )

            tool_data = (
                dataset_schema_tool.model_dump(
                    by_alias=True,
                    exclude_none=True,
                )
            )

            print(
                f"selected_tool     : "
                f"{dataset_schema_tool.name}"
            )

            print(
                f"tool_title        : "
                f"{dataset_schema_tool.title}"
            )

            print(
                f"input_schema      : "
                f"{tool_data.get('inputSchema')}"
            )

            print(
                f"output_schema     : "
                f"{tool_data.get('outputSchema')}"
            )


            input_schema = (
                tool_data[
                    "inputSchema"
                ]
            )

            assert (
                input_schema.get(
                    "type"
                )
                ==
                "object"
            )

            assert (
                input_schema.get(
                    "properties"
                )
                ==
                {}
            )

            print(
                "[PASS] get_dataset_schema "
                "Tool discovered"
            )


            print_section(
                "4. MCP TOOLS/CALL"
            )


            # 실제 MCP ClientSession을 통해
            # Tool을 호출합니다.
            #
            # 현재 Tool은 입력 parameter가 없으므로
            # 빈 arguments dict를 전달합니다.
            call_result = (
                await session.call_tool(
                    EXPECTED_TOOL_NAME,
                    arguments={},
                )
            )

            call_result_data = (
                to_alias_dict(
                    call_result
                )
            )


            # MCP Tool 실행 중 오류가 발생하면
            # isError가 True일 수 있습니다.
            is_error = (
                call_result_data.get(
                    "isError",
                    False,
                )
            )

            print(
                f"is_error          : "
                f"{is_error}"
            )

            assert (
                is_error
                is False
            ), (
                "MCP Tool returned an error. "
                f"result={call_result_data}"
            )


            # 구조화된 Tool 결과입니다.
            structured_output = (
                call_result_data.get(
                    "structuredContent"
                )
            )

            assert isinstance(
                structured_output,
                dict,
            ), (
                "structuredContent must be dict. "
                f"actual_type="
                f"{type(structured_output).__name__}"
            )


            # TextContent 안의 JSON 결과도
            # 별도로 추출합니다.
            text_output = (
                extract_text_output(
                    call_result_data
                )
            )


            print(
                "dataset           : "
                f"{structured_output.get('dataset')}"
            )

            # 구조화 결과에서 출력할 값을 먼저 꺼냅니다.
            #
            # 계산을 f-string 내부에 여러 줄로 작성하지 않고,
            # 변수 생성과 출력을 분리합니다.
            features = structured_output.get(
                "features",
                [],
            )

            target = structured_output.get(
                "target"
            )

            excluded_columns = (
                structured_output.get(
                    "excluded_columns"
                )
            )

            feature_count = len(
                features
            )

            print(
                "feature_count     : "
                f"{feature_count}"
            )

            print(
                "features          : "
                f"{features}"
            )

            print(
                "target            : "
                f"{target}"
            )

            print(
                "excluded_columns  : "
                f"{excluded_columns}"
            )


            # TextContent와 structuredContent는
            # 같은 Dataset 결과를 표현해야 합니다.
            assert (
                text_output
                ==
                structured_output
            ), (
                "TextContent JSON and "
                "structuredContent differ."
            )


            # 핵심 Dataset 정보 검증입니다.
            assert (
                structured_output.get(
                    "dataset"
                )
                ==
                EXPECTED_DATASET_NAME
            )

            assert (
                structured_output.get(
                    "target"
                )
                ==
                EXPECTED_TARGET
            )

            features = (
                structured_output.get(
                    "features"
                )
            )

            assert isinstance(
                features,
                list,
            )

            assert (
                len(features)
                ==
                6
            )

            assert (
                "Type"
                in features
            )

            assert (
                structured_output.get(
                    "excluded_columns"
                )
                ==
                [
                    "UDI",
                    "Product ID",
                ]
            )

            print(
                "[PASS] get_dataset_schema "
                "Tool call completed"
            )

            print(
                "[PASS] TextContent and "
                "structuredContent matched"
            )

            print(
                "[PASS] Dataset Schema "
                "values validated"
            )


    print_section(
        "VALIDATION RESULT"
    )

    print(
        "[PASS] MCP Server subprocess started"
    )

    print(
        "[PASS] MCP stdio transport connected"
    )

    print(
        "[PASS] MCP initialize completed"
    )

    print(
        "[PASS] tools/list completed"
    )

    print(
        "[PASS] get_dataset_schema discovered"
    )

    print(
        "[PASS] tools/call completed"
    )

    print(
        "[PASS] Existing Dataset Schema "
        "Service result returned"
    )

    print()

    print(
        "DAY 20 REAL MCP STDIO "
        "LOCAL E2E VALIDATION PASSED"
    )


def main() -> None:
    """
    비동기 MCP validation을 실행합니다.

    anyio.run()
    -----------
    일반 동기 Python 진입점에서
    async 함수인 run_validation()을 실행합니다.
    """

    anyio.run(
        run_validation
    )


if __name__ == "__main__":
    main()