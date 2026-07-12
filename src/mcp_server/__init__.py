"""
Manufacturing AI MCP Server package.

이 package의 역할
-----------------
현재 제조 AI 프로젝트의 기능을
Model Context Protocol 방식으로 제공합니다.

현재 구조:

MCP Host

    ↓

MCP Client

    ↓

MCP Server

    ↓

MCP Tool

    ↓

기존 Application Service

    ↓

Data / Inference / Persistence


Day 20 최소 구현
----------------
현재는 다음 Tool 하나부터 구현합니다.

    get_dataset_schema


이 Tool은 Dataset 정보를
자체적으로 다시 작성하지 않습니다.

다음 기존 Service를 호출합니다.

    src.services.dataset_schema_service


향후 확장 후보
--------------
predict_machine_failure

get_execution_history

get_execution_detail


중요
----
FastAPI를 MCP로 대체하지 않습니다.

현재 프로젝트는 앞으로:

일반 HTTP Client

    ↓

FastAPI


그리고:

MCP 지원 AI Host

    ↓

MCP Server


두 인터페이스를 병행할 수 있습니다.
"""