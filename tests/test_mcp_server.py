# tests/test_mcp_server.py

"""
Day 20 - Manufacturing AI MCP Server unit tests.

이 테스트 파일의 목적
---------------------
현재 제조 AI 프로젝트에 추가한
MCP Server와 첫 MCP Tool을 검증합니다.

현재 MCP 구조:

MCP Host

    ↓

MCP Client

    ↓

Manufacturing AI MCP Server

    ↓

get_dataset_schema Tool

    ↓

Dataset Schema Service

    ↓

기존 AI4I schema 상수


현재 테스트 범위
---------------
이 파일에서는 별도 MCP Server process를
실행하지 않습니다.

또한 stdio transport를 통해
실제 Client를 연결하지 않습니다.

현재는 같은 Python process 안에서
FastMCP Server 객체를 사용해 다음을 검증합니다.

1.

Server metadata


2.

MCP Tool Adapter


3.

기존 Service 재사용


4.

Tool 등록


5.

Tool input schema


6.

Tool output schema


7.

FastMCP call_tool 실행


8.

ContentBlock과 구조화 결과


실제 stdio 연결 검증은
이후 Local MCP E2E 단계에서 별도로 수행합니다.


기본 테스트 안전 정책
--------------------
실제 OpenAI API 호출 금지

외부 MCP network 호출 금지

PyTorch 모델 실행 금지

운영 SQLite DB 사용 금지

별도 MCP Server process 실행 금지
"""

from __future__ import annotations

# json
# ----
# FastMCP call_tool() 결과의 첫 번째 값은
# 현재 TextContent 목록입니다.
#
# TextContent.text에는 Dataset Schema가
# JSON 문자열로 들어 있습니다.
#
# 이 문자열을 Python dict로 변환하여
# structured output과 같은지 검증합니다.
import json

# Any
# ---
# 테스트용 가짜 Dataset Schema는
# 문자열, list, dict 등
# 여러 JSON 호환 값을 포함합니다.
from typing import Any

# anyio
# -----
# 현재 FastMCP의:
#
#     list_tools()
#
#     call_tool()
#
# 메서드는 async 함수입니다.
#
# 현재 테스트 함수는 일반 동기 pytest 함수로 작성하고,
# anyio.run()을 사용해 async helper를 실행합니다.
#
# 이렇게 하면 별도 MCP Server process나
# 외부 network 없이 FastMCP 객체를 검증할 수 있습니다.
import anyio

# TextContent
# -----------
# MCP Tool 호출 결과에 포함되는
# MCP content block 타입입니다.
#
# 현재 get_dataset_schema Tool은
# JSON 문자열을 TextContent로 반환합니다.
from mcp.types import TextContent

# module 자체를 import합니다.
#
# 왜 함수만 import하지 않는가?
# -----------------------------
# monkeypatch를 사용해:
#
# server_module.get_ai4i_dataset_schema
#
# 를 테스트용 함수로 교체하기 위해서입니다.
#
# 이렇게 하면 MCP Tool Adapter가
# 기존 Service를 실제로 호출하는지 검증할 수 있습니다.
import src.mcp_server.server as server_module

# 실제 Dataset Schema Service입니다.
#
# MCP Tool 결과가 기존 Service 결과와
# 같은 기준을 사용하는지 비교합니다.
from src.services.dataset_schema_service import (
    get_ai4i_dataset_schema,
)


async def _list_registered_tools() -> list[Any]:
    """
    현재 FastMCP Server에 등록된 Tool 목록을 반환합니다.

    Parameters
    ----------
    없음


    Returns
    -------
    list[Any]

    현재 SDK runtime에서는:

        list[mcp.types.Tool]

    형태를 반환합니다.


    왜 helper 함수가 필요한가?
    -------------------------
    mcp.list_tools()는 async 함수입니다.

    테스트 본문은 일반 def로 유지하고,
    다음처럼 실행하기 위해 분리합니다.

        tools = anyio.run(
            _list_registered_tools
        )
    """

    return await server_module.mcp.list_tools()


async def _call_dataset_schema_tool() -> Any:
    """
    FastMCP Server를 통해
    get_dataset_schema Tool을 호출합니다.

    Parameters
    ----------
    없음


    Returns
    -------
    Any

    현재 mcp 1.28.1 runtime 결과:

        (
            list[ContentBlock],
            dict[str, Any],
        )


    Tool argument가 빈 dict인 이유
    -----------------------------
    get_dataset_schema Tool에는
    입력 parameter가 없습니다.

    MCP의 tools/call 요청에서는
    Tool argument 객체를 전달하므로
    빈 dict를 사용합니다.
    """

    return await server_module.mcp.call_tool(
        "get_dataset_schema",
        {},
    )


def test_mcp_server_has_expected_metadata():
    """
    MCP Server 이름과 설명이
    예상값으로 등록됐는지 검증합니다.

    Server 이름의 역할
    ------------------
    MCP Client 또는 Host가
    어떤 MCP Server에 연결했는지
    식별할 수 있게 합니다.
    """

    assert (
        server_module.mcp.name
        ==
        server_module.MCP_SERVER_NAME
    )

    assert (
        server_module.MCP_SERVER_NAME
        ==
        "manufacturing-ai-quality-agent"
    )

    # Server instructions가
    # 비어 있지 않아야 합니다.
    assert isinstance(
        server_module.MCP_SERVER_INSTRUCTIONS,
        str,
    )

    assert (
        server_module.MCP_SERVER_INSTRUCTIONS
    )


def test_get_dataset_schema_tool_calls_existing_service(
    monkeypatch,
):
    """
    MCP Tool Adapter가
    기존 Dataset Schema Service를 호출하는지 검증합니다.

    이 테스트가 중요한 이유
    ----------------------
    MCP Tool 안에 Dataset 정보를
    다시 직접 작성하면 안 됩니다.

    올바른 구조:

        MCP Tool

            ↓

        기존 Dataset Schema Service


    테스트 방법
    -----------
    기존 Service 함수를
    테스트용 가짜 함수로 교체합니다.

    MCP Tool을 호출했을 때
    가짜 함수의 결과가 반환된다면:

        MCP Tool

            ↓

        Service 함수

    연결이 실제로 존재한다는 뜻입니다.
    """

    expected_schema: dict[
        str,
        Any,
    ] = {
        "dataset": (
            "Test AI4I Dataset"
        ),
        "features": [
            "Test feature",
        ],
        "target": (
            "Test target"
        ),
    }

    # Service가 호출됐는지 확인하기 위한
    # mutable 상태입니다.
    call_state = {
        "called": False,
    }

    def fake_get_ai4i_dataset_schema(
    ) -> dict[str, Any]:
        """
        실제 Service 대신 사용하는
        테스트용 가짜 함수입니다.
        """

        call_state["called"] = True

        return expected_schema

    # server.py가 참조하는 Service 함수를
    # 현재 테스트 동안만 교체합니다.
    monkeypatch.setattr(
        server_module,
        "get_ai4i_dataset_schema",
        fake_get_ai4i_dataset_schema,
    )

    # MCP Tool Adapter Python 함수를
    # 직접 호출합니다.
    result = (
        server_module.get_dataset_schema()
    )

    # 기존 Service 위치가
    # 실제로 호출돼야 합니다.
    assert (
        call_state["called"]
        is True
    )

    # Tool은 Service 결과를
    # 구조화 dict로 반환해야 합니다.
    assert result == expected_schema

    # Tool은 새로운 dict를 반환합니다.
    #
    # server.py에서:
    #
    #     return dict(schema)
    #
    # 를 사용하기 때문입니다.
    assert result is not expected_schema


def test_get_dataset_schema_tool_returns_service_result():
    """
    실제 MCP Tool Adapter 반환값이
    실제 Dataset Schema Service 결과와
    일치하는지 검증합니다.

    검증 의미
    ---------
    LangGraph와 MCP가
    같은 Dataset 기준을 사용하는지 확인합니다.
    """

    service_result = (
        get_ai4i_dataset_schema()
    )

    tool_result = (
        server_module.get_dataset_schema()
    )

    assert (
        tool_result
        ==
        service_result
    )

    assert (
        tool_result["dataset"]
        ==
        (
            "AI4I 2020 "
            "Predictive Maintenance Dataset"
        )
    )

    assert (
        tool_result["target"]
        ==
        "Machine failure"
    )

    assert (
        len(
            tool_result["features"]
        )
        ==
        6
    )


def test_mcp_server_registers_dataset_schema_tool():
    """
    @mcp.tool() decorator가
    get_dataset_schema Tool을
    FastMCP Server에 등록했는지 검증합니다.

    직접 Python 함수 호출과의 차이
    -----------------------------
    다음 호출은:

        get_dataset_schema()

    일반 Python 함수 호출입니다.


    반면:

        await mcp.list_tools()

    는 FastMCP Server가 실제로 공개하는
    Tool metadata를 조회합니다.

    따라서 MCP 등록 여부는
    list_tools()로 별도 검증합니다.
    """

    tools = anyio.run(
        _list_registered_tools
    )

    # Day 20 최소 구현에서는
    # Tool 하나만 등록합니다.
    assert len(tools) == 1

    tool = tools[0]

    assert (
        tool.name
        ==
        "get_dataset_schema"
    )

    assert (
        tool.title
        ==
        "Get AI4I Dataset Schema"
    )

    assert (
        tool.description
        is not None
    )

    assert (
        "AI4I 2020"
        in tool.description
    )

    assert (
        "feature"
        in tool.description
    )

    assert (
        "target"
        in tool.description
    )


def test_mcp_dataset_schema_tool_has_expected_schemas():
    """
    MCP Tool의 inputSchema와 outputSchema를 검증합니다.

    현재 Tool 입력
    ---------------
    입력 parameter 없음

    따라서 inputSchema는:

        type: object

        properties: {}


    현재 Tool 출력
    ---------------
    구조화된 Dataset Schema dict

    따라서 outputSchema는:

        type: object
    """

    tools = anyio.run(
        _list_registered_tools
    )

    tool = tools[0]

    # MCP Tool 객체를
    # dict 형태로 변환합니다.
    #
    # 현재 SDK runtime 결과에는:
    #
    # inputSchema
    #
    # outputSchema
    #
    # key가 포함됩니다.
    tool_data = tool.model_dump()

    input_schema = (
        tool_data["inputSchema"]
    )

    output_schema = (
        tool_data["outputSchema"]
    )

    # 입력은 JSON object 형식입니다.
    assert (
        input_schema["type"]
        ==
        "object"
    )

    # 입력 parameter가 없으므로
    # properties는 빈 dict입니다.
    assert (
        input_schema["properties"]
        ==
        {}
    )

    # 필수 입력도 없어야 합니다.
    assert (
        input_schema.get(
            "required",
            [],
        )
        ==
        []
    )

    # 출력은 구조화 object입니다.
    assert (
        output_schema["type"]
        ==
        "object"
    )

    # 현재 반환 타입:
    #
    # dict[str, Any]
    #
    # 이므로 임의의 추가 key를
    # 허용하는 object schema입니다.
    assert (
        output_schema[
            "additionalProperties"
        ]
        is True
    )


def test_mcp_server_call_tool_returns_content_and_structured_output():
    """
    FastMCP Server의 call_tool()을 통해
    get_dataset_schema Tool을 실행합니다.

    현재 runtime 반환 구조
    ----------------------
    mcp 1.28.1 현재 환경에서는:

        (
            content_blocks,
            structured_output,
        )

    tuple을 반환합니다.


    첫 번째 값
    ----------
    list[ContentBlock]

    현재 첫 item:

        TextContent

    TextContent.text에는
    JSON 문자열이 들어 있습니다.


    두 번째 값
    ----------
    구조화된 Dataset Schema dict


    검증 목표
    ---------
    TextContent의 JSON 결과

    ==

    structured output

    ==

    기존 Dataset Schema Service 결과
    """

    result = anyio.run(
        _call_dataset_schema_tool
    )

    # 현재 실제 SDK runtime 결과가
    # tuple인지 확인합니다.
    assert isinstance(
        result,
        tuple,
    )

    assert len(result) == 2

    content_blocks = result[0]

    structured_output = result[1]

    # 첫 번째 값은
    # MCP content block 목록입니다.
    assert isinstance(
        content_blocks,
        list,
    )

    assert len(
        content_blocks
    ) == 1

    # 현재 Dataset Schema 결과는
    # JSON 문자열 TextContent입니다.
    first_content = (
        content_blocks[0]
    )

    assert isinstance(
        first_content,
        TextContent,
    )

    assert (
        first_content.type
        ==
        "text"
    )

    # TextContent 안의 JSON 문자열을
    # Python dict로 역직렬화합니다.
    text_output = json.loads(
        first_content.text
    )

    # 두 번째 값은 구조화 dict입니다.
    assert isinstance(
        structured_output,
        dict,
    )

    # 기존 공통 Service 결과입니다.
    expected_output = (
        get_ai4i_dataset_schema()
    )

    # TextContent의 JSON 결과,
    # structured output,
    # 기존 Service 결과가
    # 모두 같아야 합니다.
    assert (
        text_output
        ==
        structured_output
    )

    assert (
        structured_output
        ==
        expected_output
    )

    # 핵심 Dataset 정보도 확인합니다.
    assert (
        structured_output["dataset"]
        ==
        (
            "AI4I 2020 "
            "Predictive Maintenance Dataset"
        )
    )

    assert (
        structured_output["target"]
        ==
        "Machine failure"
    )

    assert (
        "Type"
        in structured_output[
            "features"
        ]
    )