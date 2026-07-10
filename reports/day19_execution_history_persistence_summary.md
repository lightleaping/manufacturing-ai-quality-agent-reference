# Day 19 - Agent Execution History Persistence

## 1. 목표

Day 19의 목표는 LangGraph Agent의 실행 결과를 SQLite에 영구 저장하고, 저장된 실행 이력을 FastAPI를 통해 목록 및 상세 형태로 조회할 수 있도록 구현하는 것이다.

Day 16에서는 Agent 실행 과정에 구조화된 trace를 추가했고, Day 17~18에서는 실제 OpenAI·PyTorch 경로의 E2E 동작과 신뢰성·성능을 검증했다.

Day 19에서는 한 번의 API 응답으로 끝나던 Agent 실행 결과를 데이터베이스에 저장하여 이후에도 다음 정보를 조회할 수 있도록 확장했다.

* 사용자 질문
* Intent 분류 결과
* Intent 분류 출처와 신뢰도
* 모델 prediction
* 고장 probability
* 운영 threshold
* 위험 등급
* 권장 조치
* 최종 Agent 답변
* 설비 입력값
* Evidence
* LangGraph trace
* Warning
* Error
* Limitation
* 실행시간
* Fallback 발생 여부

---

## 2. 주요 구현 파일

```text
src/
└─ persistence/
   ├─ __init__.py
   └─ execution_history.py

src/
└─ api/
   ├─ langgraph_agent_api.py
   └─ schemas.py

tests/
├─ test_execution_history.py
└─ test_api_langgraph_agent.py

data/
└─ runtime/
   └─ agent_execution_history.db
```

---

## 3. Persistence 계층 분리

SQLite 저장 로직은 LangGraph node 내부에 직접 구현하지 않았다.

각 계층의 책임은 다음과 같이 분리했다.

```text
LangGraph

→ Agent workflow 실행

→ Intent 분류

→ Prediction

→ Evidence 생성

→ Trace 생성


Persistence

→ SQLite table 생성

→ Agent 실행 결과 저장

→ 최근 실행 목록 조회

→ trace_id 상세 조회


FastAPI

→ LangGraph 실행

→ 최종 AgentState 수신

→ Persistence 저장 호출

→ HTTP response 반환
```

이 구조를 통해 Agent workflow와 데이터 저장 기술을 분리했다.

향후 SQLite를 PostgreSQL이나 다른 저장소로 교체하더라도 LangGraph workflow 자체의 변경을 줄일 수 있다.

---

## 4. SQLite Table

실행 이력은 `agent_executions` table에 저장한다.

주요 column:

```text
id

trace_id

question

intent

intent_source

confidence

intent_reason

selected_route

prediction

probability

threshold

risk_level

recommended_action

answer

trace_status

trace_started_at

trace_finished_at

fallback_occurred

trace_duration_ms

warning_count

error_count

raw_sample_json

evidence_json

trace_events_json

warnings_json

errors_json

limitations_json

created_at
```

`id`는 SQLite 내부 식별자다.

```sql
id INTEGER PRIMARY KEY AUTOINCREMENT
```

`trace_id`는 Agent 실행 한 건을 외부에서 조회할 때 사용하는 고유 식별자다.

```sql
trace_id TEXT NOT NULL UNIQUE
```

최근 실행 목록의 정렬 성능을 위해 `created_at` index도 추가했다.

```sql
CREATE INDEX IF NOT EXISTS
idx_agent_executions_created_at
ON agent_executions(created_at);
```

---

## 5. JSON 저장 정책

SQLite는 Python의 중첩 list·dict 구조를 직접 저장하지 않으므로 다음 필드는 JSON 문자열로 직렬화하여 TEXT column에 저장했다.

```text
raw_sample

evidence

trace_events

warnings

errors

limitations
```

저장:

```text
Python dict 또는 list

→ json.dumps()

→ SQLite TEXT
```

조회:

```text
SQLite TEXT

→ json.loads()

→ Python dict 또는 list
```

한글을 그대로 유지하기 위해 다음 옵션을 사용했다.

```python
json.dumps(
    value,
    ensure_ascii=False,
)
```

---

## 6. 초기 저장 제외 항목

Day 19 초기 구현에서는 다음 항목을 저장하지 않았다.

```text
chat_history

intent_raw_response

include_shap

include_global_importance
```

이유:

### chat_history

대화 이력은 사용자 데이터와 개인정보를 포함할 수 있고 저장량도 빠르게 증가할 수 있다.

향후 저장하려면 다음 정책이 필요하다.

* 개인정보 마스킹
* 접근 권한
* 보존 기간
* 삭제 정책

### intent_raw_response

OpenAI 원본 응답 전체를 그대로 저장하지 않았다.

현재는 구조화·검증된 다음 결과만 저장한다.

```text
intent

confidence

intent_reason

intent_source
```

### include_shap·include_global_importance

현재는 실행 제어용 request option이므로 초기 실행 이력 table에서는 제외했다.

---

## 7. Persistence 주요 함수

### initialize_database()

SQLite database directory와 table 및 index를 생성한다.

### build_execution_record()

최종 AgentState를 SQLite INSERT에 사용할 평탄한 record로 변환한다.

### insert_execution()

Agent 실행 결과를 SQLite에 저장하고 생성된 내부 `id`를 반환한다.

### get_execution_by_trace_id()

`trace_id`를 기준으로 실행 이력 한 건을 상세 조회한다.

저장된 JSON TEXT는 다시 Python dict·list로 역직렬화한다.

### list_recent_executions()

최근 실행 이력을 최신순으로 조회한다.

```sql
ORDER BY created_at DESC, id DESC
```

목록 조회에서는 큰 JSON 필드를 제외하고 핵심 요약 정보만 반환한다.

---

## 8. SQLite Connection 관리

SQLite connection은 각 작업마다 열고 닫는다.

```text
함수 호출

→ sqlite3.connect()

→ SQL 실행

→ commit 또는 rollback

→ cursor.close()

→ connection.close()
```

INSERT 성공:

```text
commit
```

INSERT 실패:

```text
rollback
```

모든 경로:

```text
cursor.close()

connection.close()
```

Connection을 장시간 전역으로 유지하지 않아 단순하고 명확한 초기 Persistence 구조를 사용했다.

---

## 9. FastAPI 저장 연결

기존 endpoint:

```http
POST /agent/langgraph-query
```

처리 흐름:

```text
HTTP request

→ Request schema 검증

→ LangGraph workflow 실행

→ final AgentState 생성

→ SQLite 실행 이력 저장

→ API response 생성

→ HTTP 200
```

최종 AgentState가 완성된 뒤 저장한다.

이 시점에는 일반적으로 다음 데이터가 모두 존재한다.

```text
trace_id

intent

prediction

probability

evidence

answer

trace_status

trace_duration_ms

trace_events
```

---

## 10. 저장 실패 정책

Agent 실행 성공과 실행 이력 저장 성공은 서로 다른 결과로 처리했다.

예:

```text
OpenAI Intent 분류 성공

PyTorch Prediction 성공

Evidence 생성 성공

Agent 답변 생성 성공

SQLite 파일 저장 실패
```

이 경우 이미 성공한 Agent 결과까지 HTTP 500으로 바꾸지 않는다.

정책:

```text
SQLite 저장 성공

→ 기존 Agent response 반환


SQLite 저장 실패

→ 서버 exception log 기록

→ AgentState warnings에 저장 실패 안내 추가

→ 기존 prediction·answer 유지

→ HTTP response 반환
```

이를 통해 Persistence 장애가 Agent 핵심 기능 전체의 장애로 확대되지 않도록 했다.

---

## 11. 실행 이력 조회 API

### 최근 실행 목록

```http
GET /agent/executions
```

Query parameter:

```text
limit
```

기본값:

```text
20
```

허용 범위:

```text
1 ~ 100
```

예:

```http
GET /agent/executions?limit=5
```

목록 응답은 다음 핵심 정보만 포함한다.

```text
id

trace_id

question

intent

intent_source

confidence

selected_route

prediction

probability

threshold

risk_level

trace_status

fallback_occurred

trace_duration_ms

warning_count

error_count

created_at
```

큰 데이터는 제외한다.

```text
raw_sample

evidence

trace_events

warnings 전체

errors 전체
```

---

### 특정 실행 상세

```http
GET /agent/executions/{trace_id}
```

상세 응답에는 다음 정보가 포함된다.

```text
기본 실행 정보

Intent 결과

Prediction 결과

권장 조치

Agent answer

raw_sample

evidence

Trace 요약

trace_events

warnings

errors

limitations

created_at
```

존재하지 않는 `trace_id`:

```text
HTTP 404 Not Found
```

---

## 12. Pydantic Response Schema

목록과 상세 조회의 목적이 다르므로 Response schema를 분리했다.

```text
AgentExecutionSummaryResponse

→ 최근 실행 목록용


AgentExecutionDetailResponse

→ 특정 실행 상세용
```

상세 schema는 Summary schema를 상속한다.

```text
AgentExecutionSummaryResponse

              ↓

AgentExecutionDetailResponse
```

이를 통해 공통 필드의 중복을 줄이고 목록·상세 응답의 역할을 구분했다.

---

## 13. 테스트

### Persistence 단위 테스트

파일:

```text
tests/test_execution_history.py
```

검증:

```text
SQLite table 생성

Index 생성

실행 이력 INSERT

필수 field 저장

JSON 직렬화

JSON 역직렬화

None ↔ SQL NULL

bool ↔ SQLite INTEGER

trace_id 상세 조회

없는 trace_id → None

trace_id UNIQUE 검증

최근 실행 최신순 정렬

limit 적용

잘못된 limit 거부

빈 DB 목록 조회

Connection 종료

실제 운영 DB 미사용
```

결과:

```text
18 passed
```

---

### FastAPI 테스트

파일:

```text
tests/test_api_langgraph_agent.py
```

검증:

```text
POST 이후 final AgentState 저장 호출

저장 실패 시 기존 Agent 결과 유지

저장 실패 warning

GET /agent/executions

limit 전달

GET /agent/executions/{trace_id}

없는 trace_id → HTTP 404

limit=0 → HTTP 422

limit=101 → HTTP 422

OpenAPI endpoint 등록

실제 OpenAI 미호출

실제 운영 SQLite DB 미사용
```

기존 Day 14~16 테스트 포함 결과:

```text
23 passed
```

---

### 전체 회귀 테스트

```powershell
pytest -v
```

결과:

```text
194 passed
```

기존 Day 1~18 기능과 Day 19 신규 기능이 모두 통과했다.

---

## 14. 실제 E2E 검증

실제 FastAPI 서버를 실행하고 다음 흐름을 검증했다.

```text
POST /agent/langgraph-query

→ 실제 OpenAI Intent 분류

→ 실제 PyTorch Prediction

→ Evidence 생성

→ LangGraph Trace 생성

→ SQLite 저장

→ GET /agent/executions

→ GET /agent/executions/{trace_id}
```

실제 결과:

```text
intent

failure_prediction


intent_source

openai


confidence

0.95


prediction

1


probability

0.9929707646369934


threshold

0.7


risk_level

HIGH


trace_status

success


fallback_occurred

false


warning_count

0


error_count

0
```

SQLite 내부 실행 이력:

```text
id

1
```

실제 `trace_id`를 사용해 상세 조회했고 다음 데이터가 정상 복원됐다.

```text
raw_sample

evidence

trace_events

warnings

errors

limitations
```

---

## 15. 실제 Evidence 저장 결과

총 Evidence:

```text
11개
```

구성:

```text
prediction_summary

1개


rule_based

2개


shap_local

5개


global_importance

3개
```

JSON 직렬화·저장·조회 후에도 중첩 metadata를 포함한 Evidence 구조가 유지됐다.

---

## 16. 실제 Trace 저장 결과

총 Trace event:

```text
7개
```

실행 순서:

```text
1. validate_question

2. route_after_validation

3. classify_intent

4. route_after_classification

5. call_failure_prediction

6. route_after_prediction

7. build_final_answer
```

Node와 Route event의 다음 정보도 정상 저장·복원됐다.

```text
sequence

event_type

event_name

status

started_at

finished_at

duration_ms

metadata
```

---

## 17. 실제 성능 결과

전체 LangGraph 실행시간:

```text
13886.961 ms

약 13.89초
```

OpenAI Intent 분류:

```text
13377.348 ms

약 13.38초
```

Prediction·SHAP·Evidence 생성:

```text
463.728 ms

약 0.46초
```

현재 실행 지연의 대부분은 OpenAI Intent 분류 구간에서 발생했다.

대략적인 비율:

```text
OpenAI Intent 분류

약 96%


Prediction·SHAP·Evidence

약 3%
```

따라서 현재 E2E 응답속도 개선의 우선 대상은 PyTorch inference보다 OpenAI Intent 분류 요청이다.

향후 검토 항목:

```text
OpenAI request timeout

재시도 정책

Fallback 전환 시간

Prompt 길이 축소

Intent 결과 cache

Rule 기반 사전 routing

성능 추적 누적 통계
```

---

## 18. 현재 한계

현재 구현은 학습용 초기 Persistence 구조다.

한계:

```text
SQLite 단일 파일 사용

다중 서버 환경 미고려

사용자 인증 없음

실행 이력 접근 권한 없음

질문 개인정보 마스킹 없음

보존 기간 정책 없음

삭제 API 없음

Pagination 미구현

복합 검색 조건 없음

Database migration 도구 없음

Connection pool 미사용

비동기 DB 미사용
```

실제 운영 환경에서는 추가 설계가 필요하다.

---

## 19. 향후 확장

### Database

```text
SQLite

→ PostgreSQL
```

### ORM

```text
SQLAlchemy
```

### Migration

```text
Alembic
```

### 조회

```text
Pagination

Intent filter

Risk level filter

Trace status filter

기간 검색
```

### 보안

```text
사용자 인증

실행 이력 접근 권한

질문 개인정보 마스킹

데이터 보존 기간

삭제 정책
```

### 관측성

```text
평균 실행시간

P95 latency

Intent별 실행 건수

Fallback 비율

Error 비율

Risk level 분포
```

---

## 20. 핵심 학습 내용

### Persistence

프로그램 실행이 끝나도 데이터를 유지하기 위한 저장 계층이다.

### Serialization

Python의 dict·list 구조를 JSON 문자열로 변환하여 SQLite TEXT에 저장했다.

### Deserialization

SQLite JSON 문자열을 다시 Python dict·list 구조로 복원했다.

### Primary Key

SQLite 내부 record를 구분하기 위해 자동 증가 `id`를 사용했다.

### Unique Key

Agent 실행 한 건을 외부에서 조회하기 위해 `trace_id`에 UNIQUE 제약을 적용했다.

### Index

최근 실행 정렬 조회 성능을 위해 `created_at` index를 추가했다.

### Transaction

INSERT 성공 시 commit하고 실패 시 rollback했다.

### Layer Separation

LangGraph workflow, Persistence, FastAPI의 책임을 분리했다.

### Failure Isolation

Persistence 실패가 성공한 Agent 결과 전체의 실패로 확대되지 않도록 했다.

---

## 21. 면접 설명

> Day 19에서는 LangGraph Agent의 최종 실행 결과를 SQLite에 저장하는 Persistence 계층을 추가했습니다. `trace_id`를 고유 조회 key로 사용했고, 설비 입력값·Evidence·Trace event 같은 중첩 구조는 JSON TEXT로 직렬화해 저장한 뒤 상세 조회 시 다시 복원했습니다. 최근 실행 목록 API와 trace_id 상세 조회 API를 분리했으며, 저장 실패가 이미 성공한 모델 예측과 Agent 응답을 HTTP 500으로 바꾸지 않도록 Persistence 오류를 격리했습니다. 단위 테스트와 API 테스트를 추가했고 전체 회귀 테스트 194개를 통과했으며, 실제 OpenAI·PyTorch·FastAPI·SQLite E2E 저장 및 조회 흐름도 검증했습니다.

---

## 22. Day 19 완료 기준

```text
[PASS] SQLite Persistence 계층

[PASS] agent_executions table

[PASS] created_at index

[PASS] AgentState record 변환

[PASS] JSON 직렬화·역직렬화

[PASS] 실행 이력 INSERT

[PASS] trace_id 상세 조회

[PASS] 최근 실행 목록

[PASS] POST 실행 후 자동 저장

[PASS] 저장 실패 격리

[PASS] GET /agent/executions

[PASS] GET /agent/executions/{trace_id}

[PASS] HTTP 404

[PASS] Query validation

[PASS] OpenAPI 문서

[PASS] Persistence 테스트 18개

[PASS] API 테스트 23개

[PASS] 전체 회귀 테스트 194개

[PASS] 실제 OpenAI E2E

[PASS] 실제 PyTorch Prediction

[PASS] 실제 SQLite 저장

[PASS] 실제 목록 조회

[PASS] 실제 상세 조회

[PASS] Evidence 복원

[PASS] Trace 복원
```

Day 19 완료.
