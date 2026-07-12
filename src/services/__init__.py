"""
Application service package.

이 package의 역할
-----------------
FastAPI, LangGraph, MCP 같은 외부 인터페이스와
실제 프로젝트 데이터를 연결하는 공통 application logic을 관리합니다.

예:

FastAPI
    ↓

LangGraph
    ↓

Application Service
    ↓

Data / Inference / Persistence


또는:

MCP Client
    ↓

MCP Server
    ↓

MCP Tool
    ↓

Application Service
    ↓

Data / Inference / Persistence


왜 Service를 별도 package로 분리하는가?
--------------------------------------
FastAPI endpoint나 MCP Tool 안에
실제 업무 로직을 직접 반복해서 작성하면
같은 기능이 여러 위치에 중복될 수 있습니다.

예:

LangGraph node:
    AI4I feature 목록 직접 작성

MCP Tool:
    AI4I feature 목록 다시 작성


이 경우 feature가 변경되면
두 파일을 모두 수정해야 합니다.

따라서 공통 Service가 실제 정보를 한 번만 만들고,
LangGraph와 MCP가 그 Service를 함께 호출하도록 구성합니다.
"""