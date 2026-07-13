# Day 23 - Dashboard Architecture and FastAPI Client Summary

## 1. Day 23 목표

Day 23의 목표는 기존 Backend 비즈니스 로직을 다시 구현하지 않고, 향후 Streamlit Dashboard가 기존 FastAPI Endpoint를 안전하게 호출할 수 있도록 Dashboard Architecture와 전용 FastAPI Client Layer를 설계·구현하는 것입니다.

Day 23에서는 실제 Streamlit 전체 화면을 구현하지 않습니다.

오늘의 핵심 범위:

- 현재 FastAPI Endpoint와 Request·Response Schema 재확인
- Dashboard 요구사항 정의
- Dashboard 전체 Architecture 설계
- 화면별 역할 정의
- 화면별 API 연결 관계 정의
- Dashboard 전용 FastAPI Client 구현
- Base URL·Timeout 설정 분리
- HTTP 오류·연결 실패·Timeout 처리
- FastAPI Client Unit Test 작성
- 전체 회귀 테스트

Day 24에서는 실제 Streamlit 화면을 구현합니다.

Day 25에서는 Dashboard Test, Screenshot, README 보완, 포트폴리오 최종 정리를 진행할 예정입니다.

---

## 2. Day 22와 Day 23의 차이

### Day 22

Day 22는 기존 Backend의 품질과 안정성을 검토하는 단계였습니다.

핵심 질문:

> 현재 구현된 Agent Backend가 안전하고 일관되게 동작하는가?

주요 작업:

- 전체 코드 리뷰
- 방어 로직 강화
- `NaN`·`Infinity` 처리
- Warning·Error 누적 개선
- Evidence 정규화
- OpenAI 내부 예외 정보 비노출
- 불필요한 코드 제거
- 전체 회귀 테스트
- README
- Architecture
- 포트폴리오·면접 설명

최종 결과:

```text
232 passed
```

### Day 23

Day 23은 안정화된 Backend 앞에 Dashboard 연결 계층을 추가하는 단계입니다.

핵심 질문:

> Streamlit Dashboard가 기존 FastAPI를 안전하고 일관되게 호출하려면 어떤 구조가 필요한가?

주요 작업:

- Dashboard 요구사항 정의
- 화면별 API 연결 설계
- API Data Contract 확인
- Dashboard 전용 FastAPI Client 구현
- HTTP 오류 처리
- Client Unit Test

---

## 3. Dashboard Architecture

Day 23에서 확정한 전체 구조:

```text
User
    ↓
Streamlit Dashboard
    ↓
Dashboard FastAPI Client
    ↓
Existing FastAPI Endpoint
    ↓
Pydantic Validation
    ↓
Service / LangGraph
    ↓
PyTorch / Evidence / Answer
    ↓
Trace / SQLite Persistence
```

Dashboard는 Presentation Layer입니다.

Dashboard가 담당하는 역할:

- 사용자 입력 Widget
- API Request
- Loading State
- Success State
- Warning State
- Error State
- API Response 시각화

Dashboard가 담당하지 않는 역할:

- PyTorch 모델 직접 로드
- PyTorch Prediction 직접 실행
- LangGraph 직접 실행
- SQLite 직접 조회
- Risk Level 재계산
- Evidence 재계산
- Answer 재생성

잘못된 구조:

```text
Streamlit
    ↓
PyTorch Model 직접 Load
```

```text
Streamlit
    ↓
LangGraph 직접 실행
```

```text
Streamlit
    ↓
SQLite 직접 조회
```

Day 23에서 선택한 구조:

```text
Streamlit UI
    ↓
DashboardApiClient
    ↓
Existing FastAPI
    ↓
Existing Backend Business Logic
```

---

## 4. Dashboard 예정 화면

### 4.1 설비 고장 위험 분석

입력:

```text
Air temperature

Process temperature

Rotational speed

Torque

Tool wear

Type

include_shap

include_global_importance
```

표시:

```text
Prediction

Probability

Threshold

Risk Level

Recommended Action

Answer

Warnings

Limitations
```

연결 API:

```http
POST /agent/failure-prediction
```

---

### 4.2 Evidence 분석

표시:

```text
Prediction Summary

Rule-based Evidence

SHAP Local Evidence

Global Importance
```

예정 UI:

```text
Metric

Table

Bar Chart

Expandable Detail
```

Evidence 전용 API는 새로 만들지 않습니다.

설비 고장 예측 Response의 `evidence`를 재사용합니다.

```text
POST /agent/failure-prediction
        ↓
Prediction Result
        +
Evidence
        ↓
설비 고장 위험 분석 화면
        +
Evidence 분석 화면
```

---

### 4.3 LangGraph Agent Chat

입력:

```text
question

chat_history

optional raw_sample

include_shap

include_global_importance
```

표시:

```text
Answer

Intent

Confidence

Intent Source

Intent Reason

Prediction

Risk Level

Warnings

Errors
```

연결 API:

```http
POST /agent/langgraph-query
```

중요 정책:

```text
chat_history
    =
질문 문맥 이해용

raw_sample
    =
현재 PyTorch 예측 입력

이전 요청의 raw_sample
    =
자동 재사용하지 않음
```

고장 예측이 필요하면 현재 요청의 `raw_sample`에 설비 입력값을 다시 포함해야 합니다.

---

### 4.4 Trace·Execution History

목록 연결 API:

```http
GET /agent/executions
```

목록 표시 후보:

```text
created_at

trace_id

question

intent

intent_source

selected_route

prediction

risk_level

trace_status

fallback_occurred

trace_duration_ms
```

상세 연결 API:

```http
GET /agent/executions/{trace_id}
```

상세 표시:

```text
Answer

Evidence

Trace Events

Warnings

Errors

Limitations

Raw Sample
```

---

## 5. FastAPI Data Contract 확인

### 5.1 Direct Failure Prediction

Request Schema:

```text
FailurePredictionRequest
```

Response Schema:

```text
FailurePredictionResponse
```

연결:

```text
FailurePredictionRequest
        ↓
POST /agent/failure-prediction
        ↓
FailurePredictionResponse
```

실제 API JSON Request:

```json
{
  "air_temperature": 303.0,
  "process_temperature": 312.5,
  "rotational_speed": 1380.0,
  "torque": 62.0,
  "tool_wear": 220.0,
  "type": "L",
  "include_shap": true,
  "include_global_importance": true
}
```

주의:

Python 내부 필드명:

```text
machine_type
```

실제 API JSON 필드명:

```text
type
```

Pydantic Alias 흐름:

```text
Dashboard JSON

type
    ↓
Pydantic Alias
    ↓
Python Object

machine_type
```

---

### 5.2 LangGraph Agent Query

Request Schema:

```text
LangGraphAgentQueryRequest
```

Response Schema:

```text
LangGraphAgentQueryResponse
```

연결:

```text
LangGraphAgentQueryRequest
        ↓
POST /agent/langgraph-query
        ↓
LangGraphAgentQueryResponse
```

확인된 제약:

```text
question
- 필수

chat_history
- 선택
- 최대 6개

chat_history.role
- user 또는 assistant

chat_history.content
- 최소 1자
- 최대 1000자

raw_sample
- 선택

include_shap
- 기본값 True

include_global_importance
- 기본값 True
```

---

### 5.3 Execution History

목록:

```text
GET /agent/executions
        ↓
list[AgentExecutionSummaryResponse]
```

상세:

```text
GET /agent/executions/{trace_id}
        ↓
AgentExecutionDetailResponse
```

---

## 6. 신규 Dashboard Package

추가 구조:

```text
src/dashboard/
├── __init__.py
├── config.py
└── api_client.py
```

테스트:

```text
tests/test_dashboard_api_client.py
```

---

## 7. `src/dashboard/__init__.py`

역할:

- Dashboard Package 설명
- Presentation Layer 책임 정의
- Backend 계층과의 책임 경계 정의

명시한 원칙:

- PyTorch 모델 직접 실행 금지
- LangGraph 직접 실행 금지
- SQLite 직접 조회 금지
- Risk Level 재계산 금지
- Evidence 재계산 금지
- Answer 재생성 금지
- 기존 FastAPI Endpoint 재사용

---

## 8. `src/dashboard/config.py`

역할:

- FastAPI Base URL 관리
- HTTP Timeout 관리
- 환경 변수 또는 기본값 사용
- 설정값 정규화·검증

환경 변수:

```text
DASHBOARD_API_BASE_URL
```

기본값:

```text
http://127.0.0.1:8000
```

환경 변수:

```text
DASHBOARD_API_TIMEOUT_SECONDS
```

기본값:

```text
30.0
```

Day 18 실측 요청 시간:

```text
약 2.08초 ~ 약 4.97초
```

기본 Timeout을 30초로 설정한 이유:

- 첫 모델·Artifact 로딩
- OpenAI API 응답 지연
- SHAP 계산 지연
- 사용자 PC 부하
- 네트워크 지연

현재 실측 최대 시간보다 충분한 여유를 두되, 서버 응답이 멈춘 경우 무제한 대기하지 않도록 설정했습니다.

### 8.1 Base URL 검증

처리:

```text
환경 변수 없음
    ↓
기본값 사용
```

```text
앞뒤 공백 제거

마지막 / 제거

http 또는 https 확인

Host 존재 여부 확인
```

잘못된 예:

```text
localhost:8000

ftp://example.com

http://
```

### 8.2 Timeout 검증

유효한 예:

```text
10

30

45.5
```

거부:

```text
abc

0

-1

NaN

Infinity

-Infinity
```

Timeout은 다음 조건을 만족해야 합니다.

```text
0보다 큼

유한한 숫자
```

---

## 9. `src/dashboard/api_client.py`

역할:

- 기존 FastAPI Endpoint 호출
- 공통 Base URL 사용
- 공통 Timeout 사용
- Response JSON 반환
- HTTP 오류 변환
- Response 기본 구조 검증

구현 메서드:

```python
predict_failure()
```

연결:

```http
POST /agent/failure-prediction
```

```python
query_langgraph_agent()
```

연결:

```http
POST /agent/langgraph-query
```

```python
get_agent_executions()
```

연결:

```http
GET /agent/executions
```

```python
get_agent_execution_detail()
```

연결:

```http
GET /agent/executions/{trace_id}
```

---

## 10. Dashboard 전용 예외

추가한 예외 계층:

```text
DashboardApiClientError
    ├── DashboardApiConnectionError
    ├── DashboardApiTimeoutError
    ├── DashboardApiHttpError
    └── DashboardApiInvalidResponseError
```

UI가 `httpx2` 내부 예외를 직접 처리하지 않도록 Dashboard 전용 예외로 변환합니다.

예상 Day 24 UI 코드:

```python
try:
    result = api_client.predict_failure(
        payload,
    )
except DashboardApiClientError as exc:
    st.error(
        str(exc),
    )
```

---

## 11. HTTP 오류 흐름

### 11.1 Timeout

```text
httpx2 TimeoutException
        ↓
DashboardApiTimeoutError
        ↓
Streamlit Error State
```

사용자 메시지에 현재 Timeout 값을 포함합니다.

### 11.2 연결 실패

```text
httpx2 ConnectError
        ↓
DashboardApiConnectionError
        ↓
FastAPI 실행 상태와 Base URL 확인 안내
```

### 11.3 기타 네트워크 오류

```text
httpx2 RequestError
        ↓
DashboardApiConnectionError
```

### 11.4 HTTP 4xx

```text
FastAPI 4xx
        ↓
안전한 detail 추출
        ↓
DashboardApiHttpError
```

FastAPI Validation 오류는 사용자가 입력값을 수정할 수 있도록 제한적으로 전달합니다.

### 11.5 HTTP 5xx

```text
FastAPI 5xx
        ↓
서버 Response Body 비노출
        ↓
안전한 서버 내부 오류 메시지
```

내부 Exception 문자열, DB 정보, 환경 정보, Secret 등이 Dashboard 오류 메시지에 노출되는 것을 줄입니다.

### 11.6 Invalid JSON

```text
HTTP 200
        ↓
JSON 변환 실패
        ↓
DashboardApiInvalidResponseError
```

HTTP Status가 성공이어도 Response 형식이 올바르지 않으면 정상 결과로 처리하지 않습니다.

### 11.7 Unexpected Response

단일 객체 API:

```text
Expected

dict
```

실행 이력 목록 API:

```text
Expected

list[dict]
```

예상과 다른 최상위 JSON 구조 또는 목록 내부에 JSON Object가 아닌 항목이 있으면 `DashboardApiInvalidResponseError`를 발생시킵니다.

---

## 12. `httpx2` 사용 이유

현재 설치된 Starlette TestClient는 실제로 `httpx2`를 우선 사용합니다.

확인 결과:

```text
starlette_testclient_http_module=httpx2

module_version=2.5.0
```

Starlette 내부 Import:

```python
import httpx2 as httpx
```

Day 23 Dashboard Client에 필요한 기능도 `httpx2`가 모두 제공했습니다.

확인 기능:

```text
Client

MockTransport

BaseTransport

TimeoutException

ReadTimeout

ConnectError

RequestError

NetworkError

Request

Response
```

따라서 프로젝트가 직접 관리하는 HTTP 계층을 `httpx2`로 통일했습니다.

```text
기존 FastAPI TestClient
        ↓
httpx2

DashboardApiClient
        ↓
httpx2
```

Day 23 코드:

```python
import httpx2 as httpx
```

이는 `httpx2`를 Import하고, 현재 Python 파일 안에서만 `httpx`라는 별칭으로 사용하는 것입니다.

일반 `httpx` 패키지는 OpenAI SDK, MCP, LangGraph SDK, LangSmith 등의 간접 의존성으로 가상환경에 함께 설치되어 있습니다.

```text
Project Direct HTTP Code
        ↓
httpx2
```

```text
OpenAI / MCP / LangGraph Internal Dependency
        ↓
httpx
```

---

## 13. Dashboard API Client Unit Test

추가 파일:

```text
tests/test_dashboard_api_client.py
```

실제 FastAPI 서버를 실행하지 않습니다.

테스트 구조:

```text
DashboardApiClient
        ↓
httpx2.MockTransport
        ↓
Fake HTTP Response
        ↓
Client Result or Exception 검증
```

실행하지 않는 항목:

- 실제 OpenAI API
- 실제 LangGraph
- 실제 PyTorch Model
- 실제 SHAP
- 실제 SQLite

### 13.1 정상 요청 테스트

검증:

- 고장 예측 Endpoint
- 고장 예측 JSON Body
- LangGraph Query Endpoint
- LangGraph Query JSON Body
- 실행 이력 목록
- `limit` Query Parameter
- 실행 이력 상세
- `trace_id` URL Encoding

### 13.2 오류 테스트

검증:

- Timeout
- FastAPI 연결 실패
- 기타 Network Error
- HTTP 422 Validation Error
- HTTP 500 내부 정보 비노출
- JSON이 아닌 성공 Response
- 예상과 다른 Dict·List 구조
- 실행 이력 목록 내부 잘못된 항목

### 13.3 입력·설정 테스트

검증:

- 공백 `trace_id` 거부
- 기본 Base URL
- 기본 Timeout
- 환경 변수 값 적용
- Base URL 정규화
- 잘못된 Base URL 거부
- 문자열 Timeout 거부
- 0 이하 Timeout 거부
- `NaN` 거부
- `Infinity` 거부

---

## 14. Day 23 Test 결과

Dashboard API Client Test:

```text
25 passed
```

기존 FastAPI API Test와 Dashboard API Client Test 집중 검증:

```text
51 passed
```

전체 회귀 테스트:

```text
257 passed
```

기존 기준:

```text
232 passed
```

Day 23 신규 테스트:

```text
25 passed
```

전체:

```text
232 + 25 = 257 passed
```

기존 기능을 유지하면서 새 Dashboard Client 테스트가 모두 추가되었습니다.

---

## 15. Git Diff 검사

실행:

```powershell
git diff --check
```

최종 결과:

```text
출력 없음
```

의미:

- 불필요한 줄 끝 공백 없음
- 잘못된 공백 오류 없음
- 파일 끝 불필요한 빈 줄 없음

---

## 16. 현재 한계

Day 23에서는 Client Layer와 Data Contract를 우선 구현했습니다.

아직 구현하지 않은 항목:

- 실제 Streamlit Application
- Sidebar Navigation
- 설비 입력 Widget
- Prediction Metric
- Probability Visualization
- Evidence Table
- SHAP Bar Chart
- Global Importance Chart
- LangGraph Chat UI
- Session State
- Execution History Table
- Trace Detail UI
- Dashboard Screenshot

---

## 17. Day 24 계획

Day 24 핵심:

```text
실제 Streamlit Dashboard 구현
```

예정 화면:

```text
1. 설비 고장 위험 분석

2. Evidence 분석

3. LangGraph Agent Chat

4. Trace·Execution History
```

예정 흐름:

```text
Streamlit Widget
        ↓
DashboardApiClient
        ↓
FastAPI
        ↓
Loading / Success / Warning / Error
        ↓
Result Visualization
```

---

## 18. 포트폴리오 설명

### 프로젝트 한 문장

> 기존 제조 AI Agent Backend의 비즈니스 로직을 중복 구현하지 않고, Streamlit Dashboard가 FastAPI를 통해 예측·Evidence·Agent·Trace 결과를 재사용할 수 있도록 전용 HTTP Client Layer와 오류 처리 구조를 설계했습니다.

### 문제 상황

Dashboard가 PyTorch 모델, LangGraph, SQLite에 직접 접근하면 Backend 비즈니스 로직이 UI에 중복될 수 있습니다.

이 경우 다음 문제가 발생합니다.

- 모델 로딩 코드 중복
- Risk 정책 중복
- Evidence 생성 코드 중복
- Backend와 UI 결과 불일치
- 유지보수 비용 증가
- 테스트 범위 복잡화

### 해결

Dashboard를 Presentation Layer로 제한했습니다.

```text
Streamlit UI
        ↓
DashboardApiClient
        ↓
Existing FastAPI
        ↓
Existing Backend
```

또한 Base URL과 Timeout을 설정 계층으로 분리하고, HTTP 4xx·5xx, 연결 실패, Timeout, Invalid JSON, Unexpected Response를 Dashboard 전용 예외로 변환했습니다.

### 효과·의미

- 기존 Backend 로직 재사용
- UI와 비즈니스 로직 분리
- API 계약 중심 연결
- HTTP 오류 처리 일관성 확보
- 실제 서버 없는 Client Unit Test 가능
- 향후 Streamlit 화면 구현 단순화

---

## 19. 면접 답변

> Dashboard를 만들 때 Streamlit에서 PyTorch 모델이나 LangGraph, SQLite를 직접 호출하지 않았습니다. 기존 FastAPI를 Backend 경계로 유지하고, Dashboard 전용 HTTP Client를 통해 API를 호출하도록 구성했습니다. Base URL과 Timeout은 설정 계층으로 분리했고, HTTP 오류·연결 실패·Timeout·잘못된 JSON을 Dashboard 전용 예외로 변환했습니다. 또한 `MockTransport` 기반 Unit Test를 작성해 실제 서버 없이 정상 요청과 오류 흐름을 검증했습니다.

---

## 20. Day 23 완료 기준

완료:

- [x] 현재 FastAPI Endpoint 재확인
- [x] Request·Response Schema 재확인
- [x] Dashboard 요구사항 정의
- [x] Dashboard Architecture 설계
- [x] 화면별 역할 정의
- [x] 화면별 API 연결 정의
- [x] Dashboard Package 추가
- [x] Dashboard Config 구현
- [x] Dashboard FastAPI Client 구현
- [x] HTTP 오류 처리
- [x] 연결 실패 처리
- [x] Timeout 처리
- [x] Invalid JSON 처리
- [x] Unexpected Response 처리
- [x] FastAPI Client Unit Test 작성
- [x] 신규 테스트 25개 통과
- [x] 집중 테스트 51개 통과
- [x] 전체 회귀 테스트 257개 통과
- [x] `git diff --check` 통과
- [x] Day 23 보고서 작성

---

## 21. 최종 결론

Day 23에서는 실제 Streamlit 화면을 서둘러 구현하지 않고, 먼저 Dashboard 요구사항, 화면별 API 연결, Data Contract, Presentation Layer Architecture를 정리했습니다.

또한 Dashboard가 PyTorch, LangGraph, SQLite를 직접 호출하지 않고 기존 FastAPI Endpoint를 재사용하도록 `DashboardApiClient`를 구현했습니다.

HTTP 오류, 연결 실패, Timeout, Invalid JSON, Unexpected Response를 Dashboard 전용 예외로 변환했으며, `httpx2.MockTransport` 기반 Unit Test를 통해 실제 FastAPI 서버 없이 Client 동작을 검증했습니다.

최종 전체 테스트:

```text
257 passed
```

Day 24에서는 이번에 구현한 Client Layer를 사용해 실제 Streamlit Dashboard 화면을 구현합니다.
