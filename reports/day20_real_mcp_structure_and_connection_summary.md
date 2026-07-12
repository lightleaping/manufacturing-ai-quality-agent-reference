# Day 20 — Real MCP Structure and Connection Summary

## 1. 목표

Day 20의 목표는 기존 Python 함수에 단순히 `mcp`라는 이름을 붙이는 것이 아니라, Model Context Protocol의 실제 Host·Client·Server 구조를 이해하고 현재 제조 AI Agent 프로젝트에 연결하는 것이었다.

현재 프로젝트는 Day 19까지 다음 구조를 구현했다.

```text
사용자 자연어 질문

↓

FastAPI

↓

LangGraph Agent

↓

OpenAI Intent Classification

↓

Conditional Routing

↓

PyTorch Failure Prediction

↓

Rule-based Evidence

↓

SHAP Local Explanation

↓

Permutation Importance

↓

구조화 Agent Answer

↓

Trace / Observability

↓

SQLite Execution History Persistence
```

Day 20에서는 기존 FastAPI·LangGraph·PyTorch 구조를 삭제하거나 MCP 내부에 다시 구현하지 않았다.

대신 MCP Server와 MCP Tool을 별도 protocol 계층으로 추가하고, MCP Tool이 기존 Application Service를 재사용하도록 구성했다.

최종 Day 20 구조:

```text
일반 HTTP Client

↓

FastAPI

↓

LangGraph 또는 기존 Service
```

```text
MCP Host

↓

MCP Client

↓

MCP Server

↓

MCP Tool

↓

기존 Application Service
```

FastAPI와 MCP는 서로 대체하지 않고 병행한다.

---

# 2. MCP란 무엇인가

MCP는 Model Context Protocol의 약자다.

AI Host나 Agent가 외부 Tool, Resource, Prompt와 연결될 때 공통 구조를 사용할 수 있도록 정의한 protocol이다.

기존에는 AI 애플리케이션마다 외부 기능을 연결하는 방식이 달랐다.

예:

```text
AI Application A

→ 자체 Tool 연결 코드


AI Application B

→ 별도 API 연결 코드


AI Application C

→ 다른 Plugin 구조
```

MCP를 사용하면 다음과 같은 공통 구조를 사용할 수 있다.

```text
MCP Host

↓

MCP Client

↓

MCP Server

↓

Tool / Resource / Prompt
```

---

# 3. MCP Host·Client·Server 구조

## 3.1 MCP Host

MCP Host는 MCP 연결을 사용하는 상위 AI 애플리케이션이다.

Host는 하나 이상의 MCP Client를 관리하고, AI 모델이나 Agent가 어떤 MCP Server 기능을 사용할지 연결한다.

현재 프로젝트 관점의 개념 구조:

```text
사용자

↓

AI Host 또는 Agent

↓

MCP Client

↓

Manufacturing AI MCP Server
```

Day 20에서는 별도 상용 Host를 구현하지 않았다.

대신 공식 MCP Python SDK의 `ClientSession`을 사용한 Local Validation Client가 Host·Client 역할 일부를 수행했다.

---

## 3.2 MCP Client

MCP Client는 MCP Server와 실제 protocol session을 생성한다.

현재 실제 검증에서는 다음 SDK 객체를 사용했다.

```python
ClientSession
```

주요 실행 순서:

```text
ClientSession 생성

↓

initialize

↓

list_tools

↓

call_tool
```

실제 검증:

```text
initialize

PASS


tools/list

PASS


tools/call

PASS
```

---

## 3.3 MCP Server

MCP Server는 Tool·Resource·Prompt 같은 기능을 MCP protocol로 제공한다.

Day 20에서는 공식 Python SDK의 다음 class를 사용했다.

```python
from mcp.server.fastmcp import FastMCP
```

Server 이름:

```text
manufacturing-ai-quality-agent
```

Server 생성:

```python
mcp = FastMCP(
    name=MCP_SERVER_NAME,
    instructions=MCP_SERVER_INSTRUCTIONS,
)
```

현재 Server는 첫 MCP Tool 하나를 제공한다.

```text
get_dataset_schema
```

---

# 4. Tool·Resource·Prompt 차이

## 4.1 Tool

Tool은 AI Agent가 실행할 수 있는 기능이다.

예:

```text
Dataset Schema 조회

설비 고장 예측

실행 이력 조회
```

Day 20 구현:

```text
get_dataset_schema
```

호출 구조:

```text
MCP Client

↓

tools/call

↓

get_dataset_schema

↓

Dataset Schema Service

↓

AI4I Schema 반환
```

---

## 4.2 Resource

Resource는 AI가 읽을 수 있는 데이터 또는 문서다.

예:

```text
Dataset 설명 문서

모델 metadata

실행 보고서

설비 매뉴얼
```

Day 20에서는 구현하지 않았다.

향후 AI4I Dataset 설명이나 모델 metadata를 읽기 전용 Resource로 제공할 수 있다.

---

## 4.3 Prompt

Prompt는 재사용 가능한 Prompt Template이다.

예:

```text
설비 위험 분석 Prompt

고장 원인 분석 Prompt

실행 이력 요약 Prompt
```

Day 20에서는 구현하지 않았다.

현재 프로젝트는 OpenAI를 Intent 분류에 사용하고 있으므로 MCP Prompt를 바로 추가하지 않고, Tool 구조 학습을 우선했다.

---

# 5. Transport

MCP Client와 MCP Server가 메시지를 주고받는 연결 방식이다.

주요 Transport:

```text
stdio

Streamable HTTP
```

---

## 5.1 stdio

stdio는 standard input과 standard output을 사용한다.

구조:

```text
MCP Client

↓

Server subprocess 실행

↓

stdin

↓

MCP Server

↓

stdout

↓

MCP Client
```

Day 20에서는 stdio를 사용했다.

선택 이유:

```text
별도 port 불필요

외부 network 불필요

별도 HTTP Server 불필요

Local MCP 구조 학습에 적합

Client·Server subprocess 연결 확인 가능
```

실제 실행 command:

```powershell
python -m src.mcp_server.server
```

실제 Validation Client는 현재 가상환경 Python 경로를 사용했다.

```python
command=sys.executable
```

---

## 5.2 Streamable HTTP

Streamable HTTP는 원격 MCP Server 연결에 사용할 수 있는 Transport다.

현재 Day 20에서는 구현하지 않았다.

향후 여러 사용자 또는 원격 AI Host가 동일한 Manufacturing MCP Server를 사용할 필요가 있을 때 검토할 수 있다.

---

# 6. JSON-RPC

MCP Client와 MCP Server는 MCP protocol 메시지를 주고받는다.

실제 Day 20 Local E2E에서는 SDK가 내부 protocol 처리를 담당했다.

검증한 순서:

```text
initialize

↓

tools/list

↓

tools/call
```

직접 JSON 문자열을 생성하지 않고 공식 SDK를 사용했다.

---

# 7. MCP Server와 일반 FastAPI API 차이

## FastAPI

FastAPI는 일반 HTTP Client에 REST API를 제공한다.

현재 endpoint:

```http
POST /agent/failure-prediction
```

```http
POST /agent/langgraph-query
```

```http
GET /agent/executions
```

```http
GET /agent/executions/{trace_id}
```

FastAPI 구조:

```text
HTTP Client

↓

HTTP Request

↓

FastAPI Endpoint

↓

Service

↓

HTTP Response
```

---

## MCP Server

MCP Server는 MCP 지원 Host와 Agent가 Tool을 발견하고 호출할 수 있게 한다.

MCP 구조:

```text
MCP Host

↓

MCP Client

↓

MCP Protocol

↓

MCP Server

↓

MCP Tool

↓

Service
```

---

## 병행 이유

FastAPI는 일반 웹·앱·Swagger·HTTP Client에 적합하다.

MCP는 AI Host와 Agent가 Tool을 표준 구조로 발견하고 호출하는 데 적합하다.

따라서 기존 FastAPI를 삭제하지 않고 MCP를 추가했다.

---

# 8. MCP Tool과 일반 Python 함수 차이

일반 Python 함수:

```python
result = get_dataset_schema()
```

같은 Python process 안에서 직접 호출한다.

MCP Tool:

```text
MCP Client

↓

tools/list

↓

Tool Metadata 확인

↓

tools/call

↓

MCP Server

↓

Python 함수 실행
```

MCP Tool에는 다음 정보가 포함된다.

```text
Tool 이름

Title

Description

Input Schema

Output Schema
```

현재 Tool:

```text
name

get_dataset_schema


title

Get AI4I Dataset Schema
```

---

# 9. MCP Tool과 LangGraph Node 차이

## MCP Tool

MCP Tool은 MCP Client 요청을 기존 Application 기능에 연결한다.

현재:

```text
get_dataset_schema Tool

↓

Dataset Schema Service
```

---

## LangGraph Node

LangGraph Node는 Agent workflow의 한 단계다.

현재:

```text
dataset_schema_query

↓

build_dataset_schema_answer_node

↓

Dataset Schema Service

↓

AgentState 저장
```

---

## 차이

```text
MCP Tool

→ Protocol Adapter


LangGraph Node

→ Agent Workflow Step


Application Service

→ 실제 공통 Logic
```

LangGraph Node 안에 MCP Server 로직을 넣지 않았다.

MCP Tool 안에 LangGraph workflow 전체를 불필요하게 호출하지 않았다.

---

# 10. MCP와 OpenAI Function Calling 차이

OpenAI Function Calling은 OpenAI 모델이 정의된 함수 또는 Tool 호출 구조를 선택하는 기능이다.

MCP는 특정 모델 제공자에 종속되지 않고 Host·Client·Server 구조로 외부 기능을 연결하는 protocol이다.

현재 프로젝트:

```text
OpenAI

→ Intent Classification
```

```text
MCP

→ Dataset Schema Tool 제공
```

둘은 서로 대체 관계가 아니다.

향후 OpenAI 기반 Host가 MCP Tool을 호출하는 구조도 가능하다.

---

# 11. 현재 프로젝트에서 MCP가 필요한 이유

기존 프로젝트는 FastAPI를 통해 기능을 제공했다.

Day 20에서는 같은 기능을 AI Host나 Agent가 MCP Tool로 사용할 수 있는 구조를 학습했다.

현재 구조:

```text
FastAPI

↓

기존 Application Service
```

```text
MCP Tool

↓

동일한 Application Service
```

장점:

```text
기존 로직 중복 방지

FastAPI 기능 유지

AI Host 연결 확장

Tool metadata 제공

Tool input·output 구조화

Client의 Tool 발견 가능

Transport 분리
```

---

# 12. 추가 파일

Day 20 추가 파일:

```text
src/
├─ services/
│  ├─ __init__.py
│  └─ dataset_schema_service.py
│
└─ mcp_server/
   ├─ __init__.py
   └─ server.py
```

```text
tests/
├─ test_dataset_schema_service.py
└─ test_mcp_server.py
```

```text
scripts/
└─ run_day20_mcp_stdio_validation.py
```

```text
reports/
└─ day20_real_mcp_structure_and_connection_summary.md
```

수정 파일:

```text
requirements.txt
```

추가 dependency:

```text
mcp==1.28.1
```

수정 파일:

```text
src/agent/failure_agent_graph.py
```

변경:

```text
Dataset Schema 직접 작성

↓

Dataset Schema Service 재사용
```

---

# 13. 기존 Service 재사용 구조

기존 데이터 기준:

```text
src/data/schemas.py
```

주요 상수:

```text
AI4I_FEATURE_COLUMNS

AI4I_CATEGORICAL_COLUMNS

AI4I_TARGET_COLUMN

AI4I_DROP_COLUMNS

AI4I_TYPE_MAPPING
```

공통 Service:

```text
src/services/dataset_schema_service.py
```

호출 구조:

```text
src/data/schemas.py

↓

Dataset Schema Service

↓

┌──────────────────────┐
│                      │
LangGraph Node      MCP Tool
```

MCP Tool 내부에 feature·target을 다시 작성하지 않았다.

---

# 14. 구현한 MCP Tool

Tool:

```text
get_dataset_schema
```

입력:

```text
없음
```

Input Schema:

```json
{
  "properties": {},
  "type": "object"
}
```

출력:

```text
dataset

features

numeric_features

categorical_features

target

excluded_columns

categorical_mappings

current_encoding_note

improvement_note
```

주요 결과:

```text
Dataset

AI4I 2020 Predictive Maintenance Dataset


Feature 수

6


Target

Machine failure


제외 column

UDI

Product ID
```

---

# 15. Tool 등록 결과

등록 Tool 수:

```text
1
```

Tool 이름:

```text
get_dataset_schema
```

Tool Title:

```text
Get AI4I Dataset Schema
```

Output Schema:

```json
{
  "additionalProperties": true,
  "type": "object"
}
```

---

# 16. 실제 MCP 연결 결과

Transport:

```text
stdio
```

Server:

```text
manufacturing-ai-quality-agent
```

Protocol version:

```text
2025-11-25
```

실제 연결 순서:

```text
MCP Server subprocess 시작

↓

stdio stream 연결

↓

ClientSession 생성

↓

initialize

↓

tools/list

↓

get_dataset_schema 발견

↓

tools/call

↓

Dataset Schema 반환
```

실제 결과:

```text
[PASS] MCP Server subprocess started

[PASS] MCP stdio transport connected

[PASS] MCP initialize completed

[PASS] tools/list completed

[PASS] get_dataset_schema discovered

[PASS] tools/call completed

[PASS] Existing Dataset Schema Service result returned
```

최종:

```text
DAY 20 REAL MCP STDIO LOCAL E2E VALIDATION PASSED
```

---

# 17. TextContent와 Structured Content

FastMCP Tool 호출 결과는 다음 두 데이터를 반환했다.

```text
TextContent

+

structuredContent
```

TextContent에는 Dataset Schema JSON 문자열이 포함됐다.

structuredContent에는 Python dict 형태의 구조화 결과가 포함됐다.

검증:

```text
TextContent JSON

==

structuredContent

==

기존 Dataset Schema Service 결과
```

통과했다.

---

# 18. 테스트

## Dataset Schema Service

```text
6 passed
```

검증:

```text
기본 구조

기존 AI4I 상수 재사용

원본 상수 보호

자연어 Answer

Evidence 계약

LangGraph 통합 결과
```

---

## MCP Server

```text
6 passed
```

검증:

```text
Server metadata

기존 Service 호출

Tool 결과

Tool 등록

Input·Output Schema

FastMCP Tool 호출
```

---

## 관련 회귀 테스트

```text
89 passed

0 failed
```

포함:

```text
Dataset Schema Service

MCP Server

LangGraph

Failure Prediction API

LangGraph API

SQLite Execution History
```

---

## 전체 회귀 테스트

Day 19:

```text
194 passed
```

Day 20:

```text
206 passed
```

신규 테스트:

```text
12개
```

구성:

```text
Dataset Schema Service

6개


MCP Server

6개
```

최종:

```text
206 passed

0 failed
```

---

# 19. 기존 기능 보존 결과

다음 기능을 유지했다.

```text
POST /agent/failure-prediction

POST /agent/langgraph-query

GET /agent/executions

GET /agent/executions/{trace_id}

OpenAI Intent Classification

LangGraph Routing

PyTorch Prediction

Evidence

SHAP

Permutation Importance

Trace

SQLite Persistence
```

MCP 추가 후 전체 회귀 테스트가 통과했다.

---

# 20. 오류 처리

현재 `get_dataset_schema` Tool은 외부 API, 모델, DB를 사용하지 않는다.

따라서 초기 Tool은 오류 가능성이 낮다.

현재 MCP SDK는 Tool 호출 결과에 다음 상태를 제공한다.

```text
isError
```

실제 Local E2E:

```text
is_error

False
```

향후 Prediction Tool 구현 시에는 다음을 설계해야 한다.

```text
입력 validation

모델 artifact 오류

Prediction 오류

SHAP 선택 실패

구조화 MCP 오류 반환
```

---

# 21. 현재 한계

현재 MCP Tool은 Dataset Schema 조회 하나만 구현했다.

구현하지 않은 Tool:

```text
predict_machine_failure

get_execution_history

get_execution_detail

get_model_metrics
```

현재 Transport:

```text
stdio
```

구현하지 않은 Transport:

```text
Streamable HTTP
```

현재 별도 인증과 권한 관리는 없다.

Dataset Schema는 현재 코드 상수를 기반으로 한다.

Dataset metadata artifact 또는 RAG 기반 동적 조회는 구현하지 않았다.

---

# 22. 향후 개선

우선순위:

```text
1.

predict_machine_failure Tool


2.

Prediction 입력 Pydantic Schema


3.

기존 Failure Prediction Service의
FastAPI 결합도 완화


4.

get_execution_history Tool


5.

get_execution_detail Tool


6.

Streamable HTTP


7.

인증·권한


8.

MCP Resource


9.

MCP Prompt
```

Prediction Tool 구현 시 MCP Tool 내부에 다음을 복사하지 않는다.

```text
모델 load

전처리

Prediction

SHAP

Evidence
```

기존 Service를 재사용한다.

---

# 23. 면접 답변

## MCP 구조

“Model Context Protocol은 AI Host가 외부 Tool과 데이터를 표준 구조로 연결하기 위한 protocol입니다.

Host가 MCP Client를 관리하고, Client가 MCP Server와 연결한 뒤 Server가 Tool·Resource·Prompt를 제공합니다.

이번 프로젝트에서는 공식 MCP Python SDK의 FastMCP로 Manufacturing MCP Server를 구현했습니다.”

---

## MCP Tool 역할

“첫 Tool은 `get_dataset_schema`로 구현했습니다.

MCP Tool 안에 AI4I feature와 target을 다시 작성하지 않고, 기존 Dataset Schema Application Service를 호출하는 Adapter 구조로 만들었습니다.”

---

## 기존 Service 재사용

“기존 LangGraph node에 직접 작성된 Dataset Schema 로직을 공통 Service로 분리했습니다.

이후 LangGraph node와 MCP Tool이 동일한 Service를 호출하도록 구성해 중복 구현과 응답 불일치 가능성을 줄였습니다.”

---

## FastAPI와 MCP 병행 이유

“FastAPI는 웹·Swagger·일반 HTTP Client를 위한 인터페이스로 유지했습니다.

MCP는 AI Host와 Agent가 Tool을 발견하고 호출할 수 있는 별도 protocol 계층으로 추가했습니다.

두 인터페이스는 서로 대체하지 않고 동일한 Application Service를 재사용합니다.”

---

## Tool Input·Output

“현재 Tool은 고정된 AI4I Dataset Schema를 반환하므로 입력 parameter가 없습니다.

FastMCP가 빈 object Input Schema와 구조화 object Output Schema를 생성하는 것을 확인했습니다.”

---

## 테스트

“MCP Server 객체 내부의 Tool 등록과 호출을 단위 테스트했습니다.

또한 공식 ClientSession과 stdio transport를 사용해 별도 MCP Server subprocess에 실제 연결했습니다.

initialize, tools/list, tools/call을 순서대로 검증했습니다.”

---

## 실제 연결 검증

“실제 stdio Local E2E에서 Manufacturing MCP Server에 연결하고 `get_dataset_schema` Tool을 발견한 뒤 호출했습니다.

TextContent JSON과 structuredContent가 기존 Dataset Schema Service 결과와 같은 것도 검증했습니다.”

---

## 현재 한계

“현재는 Dataset Schema Tool 하나와 stdio Transport만 구현했습니다.

향후 Prediction Service의 FastAPI 결합도를 완화한 뒤 `predict_machine_failure` Tool을 추가하고, 원격 사용을 위한 Streamable HTTP와 인증을 검토할 수 있습니다.”

---

# 24. AI 활용 설명

“AI를 개발 보조 도구로 활용해 코드 초안을 빠르게 구성했고, 이후 제가 직접 실행·검증·수정하면서 코드 구조와 처리 흐름을 제 것으로 만들었습니다.

단순히 코드를 생성하는 데서 끝내지 않고 MCP Host·Client·Server 구조, 기존 Service 연결, Tool 입출력, 실제 stdio 연결, 테스트 결과와 한계점까지 확인하며 설명할 수 있도록 정리했습니다.”

---

# 25. Day 20 완료 기준

```text
[✓] 현재 공식 MCP 구조 확인

[✓] MCP Host·Client·Server 이해

[✓] Tool·Resource·Prompt 차이 이해

[✓] Transport 이해

[✓] 현재 프로젝트 MCP 연결 설계

[✓] MCP package·version 확인

[✓] MCP 1.28.1 설치

[✓] requirements.txt 기록

[✓] MCP Server 최소 구현

[✓] 기존 Service 재사용

[✓] get_dataset_schema Tool 구현

[✓] Tool input 검증

[✓] Tool output 검증

[✓] MCP 단위 테스트

[✓] 기존 FastAPI 기능 유지

[✓] 관련 회귀 테스트 89개 통과

[✓] 전체 테스트 206개 통과

[✓] 실제 MCP Client 연결

[✓] MCP initialize

[✓] Tool 목록 조회

[✓] 실제 Tool 호출

[✓] 구조화 결과 검증

[✓] 실제 stdio Local E2E 통과

[✓] Day 20 보고서

[✓] 면접 답변
```

---

# 26. 최종 결과

```text
Day 20

REAL MCP STRUCTURE AND CONNECTION

COMPLETED
```

```text
MCP SDK

1.28.1
```

```text
Transport

stdio
```

```text
MCP Tool

get_dataset_schema
```

```text
Related Regression

89 passed
```

```text
Full Regression

206 passed

0 failed
```

```text
Real MCP stdio Local E2E

PASSED
```
