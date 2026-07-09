# Day 16 - LangGraph Trace / Observability 구현 및 검증 보고서

## 1. Day 16 목표

Day 16의 목표는 LangGraph Agent가 최종 답변만 반환하는 구조를 넘어, **내부에서 어떤 node와 route가 어떤 순서로 실행되었는지 관찰할 수 있는 구조화 trace 기능을 구현하는 것**이다.

Day 15까지는 다음과 같은 최종 결과를 확인할 수 있었다.

* intent
* confidence
* prediction
* probability
* threshold
* risk_level
* recommended_action
* answer
* evidence
* warnings
* errors
* limitations

그러나 최종 결과만으로는 다음 질문에 답하기 어려웠다.

* 어떤 node가 실행되었는가?
* node는 어떤 순서로 실행되었는가?
* 어떤 conditional route가 선택되었는가?
* 어느 처리 단계에서 시간이 오래 걸렸는가?
* warning 또는 error는 어느 node에서 추가되었는가?
* 실제 LangGraph fallback 경로가 실행되었는가?
* 전체 workflow는 `success`, `fallback`, `error` 중 어떤 상태로 종료되었는가?

Day 16에서는 위 문제를 해결하기 위해 외부 관측성 플랫폼을 바로 도입하지 않고, 먼저 프로젝트 내부에 구조화 trace 계층을 구현했다.

---

# 2. 핵심 개념

## 2.1 Trace

Trace는 사용자 요청 하나가 Agent 내부에서 처리된 전체 실행 흐름이다.

예:

```text
사용자 질문

→ question 검증

→ intent 분류

→ route 선택

→ prediction 실행

→ 최종 답변 생성

→ 응답 반환
```

요청마다 새로운 `trace_id`를 생성하여 서로 다른 Agent 실행을 구분한다.

예:

```text
trace_id

5d0507ebae9a490db99ac61ab2477dea
```

---

## 2.2 Trace Event

Trace event는 전체 trace 안에서 발생한 개별 실행 기록이다.

현재 Day 16에서는 두 종류를 사용한다.

### node event

LangGraph node가 실행된 기록이다.

예:

```text
validate_question

classify_intent

call_failure_prediction

build_final_answer

build_fallback_answer
```

### route event

conditional routing 결과를 기록한다.

예:

```text
route_after_validation

route_after_classification

route_after_prediction
```

---

## 2.3 Observability

Observability는 시스템 외부에서 내부 상태를 이해할 수 있는 능력이다.

이번 프로젝트에서는 다음 정보를 구조화하여 Agent 내부 처리 흐름을 관찰한다.

```text
어떤 단계가 실행됐는가?

어떤 순서로 실행됐는가?

얼마나 오래 걸렸는가?

어떤 경로가 선택됐는가?

warning 또는 error가 발생했는가?

fallback이 실행됐는가?
```

Day 16의 구조는 로그 문자열을 단순 출력하는 방식이 아니라, Python dict와 Pydantic schema를 사용하는 구조화된 관측성이다.

---

# 3. 외부 관측성 도구보다 내부 Trace를 먼저 구현한 이유

LangSmith, OpenTelemetry와 같은 외부 관측성 도구를 바로 연결할 수도 있다.

그러나 이번 학습 프로젝트에서는 먼저 내부 trace 구조를 구현했다.

이유는 다음과 같다.

1. Agent workflow의 실행 구조를 직접 이해할 수 있다.

2. 외부 서비스가 없어도 로컬 환경에서 실행 흐름을 확인할 수 있다.

3. trace에 어떤 정보를 저장할지 직접 설계할 수 있다.

4. node와 route의 책임을 코드 수준에서 이해할 수 있다.

5. 이후 LangSmith 또는 OpenTelemetry를 연결할 때 현재 구조를 기반으로 확장할 수 있다.

현재 구조는 외부 관측성 도구를 대체하기 위한 최종 구조가 아니라, 관측성의 기본 원리를 학습하고 프로젝트 내부 실행을 직접 제어하기 위한 기반 구조이다.

---

# 4. Day 16 전체 실행 구조

```text
FastAPI Request

        │

        ▼

LangGraphAgentQueryRequest

        │

        ▼

run_failure_agent_graph()

        │

        ├─ 전체 실행 시간 측정 시작
        │
        ▼

create_initial_agent_state()

        │

        ├─ trace_id 생성
        ├─ trace_status = running
        ├─ trace_started_at 생성
        ├─ fallback_occurred = False
        └─ trace_events = []

        │

        ▼

Compiled LangGraph

        │

        ▼

Traced Business Node

        │

        ├─ 기존 business node 실행
        ├─ 시작 시각 기록
        ├─ 종료 시각 기록
        ├─ 실행 시간 계산
        ├─ warning 증가량 계산
        ├─ error 증가량 계산
        └─ node trace event 추가

        │

        ▼

Route Trace Node

        │

        ├─ 기존 route 함수 실행
        ├─ selected_route 계산
        ├─ route trace event 추가
        └─ selected_route를 AgentState에 저장

        │

        ▼

Conditional Edge

        │

        └─ 저장된 selected_route를 읽어 다음 node 선택

        │

        ▼

finalize_trace()

        │

        ├─ trace_finished_at 저장
        ├─ trace_duration_ms 계산
        └─ trace_status 결정

        │

        ▼

LangGraph AgentState

        │

        ▼

_state_to_response()

        │

        ▼

LangGraphAgentQueryResponse

        │

        ▼

FastAPI JSON / Swagger
```

---

# 5. 주요 구현 파일

## 5.1 `src/agent/state.py`

기존 `AgentState`에 trace 관련 상태를 추가했다.

추가한 주요 타입:

```python
TraceEventType

TraceEventStatus

TraceStatus

TraceEvent
```

추가한 주요 상태:

```python
trace_id

trace_status

trace_started_at

trace_finished_at

trace_duration_ms

fallback_occurred

trace_events

selected_route
```

초기 AgentState를 생성할 때 요청마다 새로운 trace 정보를 만든다.

```python
trace_id = uuid4().hex

trace_started_at = datetime.now(
    timezone.utc
).isoformat()
```

초기 상태:

```python
{
    "trace_status": "running",
    "trace_finished_at": None,
    "trace_duration_ms": None,
    "fallback_occurred": False,
    "trace_events": [],
}
```

---

## 5.2 `src/agent/trace.py`

Day 16 trace 기능을 공통 helper로 분리했다.

주요 함수:

### `utc_now_iso()`

UTC timezone 정보가 포함된 ISO 8601 문자열을 생성한다.

예:

```text
2026-07-09T23:13:48.321149+00:00
```

---

### `calculate_duration_ms()`

`time.perf_counter()`를 사용하여 실행 시간을 millisecond 단위로 계산한다.

```text
초

×

1000

=

millisecond
```

---

### `ensure_trace_state()`

일부 필드만 가진 부분 AgentState에서도 trace helper가 안전하게 동작하도록 누락된 trace 기본값을 추가한다.

---

### `carry_trace_context()`

node가 기존 state를 직접 수정하지 않고 새로운 dict를 반환해도 기존 trace 정보를 이어서 사용할 수 있게 한다.

---

### `append_trace_event()`

node 또는 route event를 `trace_events`에 추가한다.

event 예:

```json
{
  "sequence": 3,
  "event_type": "node",
  "event_name": "classify_intent",
  "status": "success",
  "started_at": "2026-07-09T23:13:48.337222+00:00",
  "finished_at": "2026-07-09T23:13:54.699674+00:00",
  "duration_ms": 6362.452,
  "metadata": {
    "intent": "dataset_schema_query",
    "intent_source": "openai",
    "confidence": 0.95
  }
}
```

---

### `run_traced_node()`

기존 business node를 실행하면서 공통 trace 정보를 기록한다.

처리 흐름:

```text
node 시작 시각 저장

→ perf_counter 시작값 저장

→ 기존 node 실행

→ 종료 시각 저장

→ 실행 시간 계산

→ warning 증가량 계산

→ error 증가량 계산

→ event status 결정

→ metadata 생성

→ node trace event 추가
```

event status 결정 기준:

```text
fallback node

→ fallback


현재 node에서 error 추가

→ error


현재 node에서 warning 추가

→ warning


그 외 정상 실행

→ success
```

---

### `run_traced_route()`

기존 route 함수를 실행하면서 선택된 경로와 실행 시간을 기록한다.

예:

```json
{
  "event_type": "route",
  "event_name": "route_after_classification",
  "status": "success",
  "metadata": {
    "intent": "failure_prediction",
    "selected_route": "failure_prediction",
    "has_errors": false
  }
}
```

---

### `finalize_trace()`

전체 LangGraph 실행이 끝난 뒤 최종 trace 정보를 저장한다.

```python
trace_finished_at

trace_duration_ms

trace_status
```

---

# 6. 기존 Business Logic과 Trace Logic 분리

기존 node 함수는 유지했다.

예:

```python
validate_question_node

classify_intent_node

call_failure_prediction_node

build_dataset_schema_answer_node

build_fallback_answer_node

build_final_answer_node
```

기존 business node 안에 시간 측정 코드를 직접 반복해서 추가하지 않았다.

대신 별도의 traced wrapper를 만들었다.

```python
traced_validate_question_node

traced_classify_intent_node

traced_call_failure_prediction_node

traced_build_dataset_schema_answer_node

traced_build_fallback_answer_node

traced_build_final_answer_node
```

구조:

```text
Traced Wrapper

        │

        ▼

기존 Business Node

        │

        ▼

Trace Event 기록
```

이 구조의 장점:

1. 기존 business logic을 유지할 수 있다.

2. trace 코드를 여러 node에 반복하지 않는다.

3. trace 기능을 수정해도 prediction logic에 미치는 영향을 줄일 수 있다.

4. node의 핵심 책임과 관측성 책임을 구분할 수 있다.

---

# 7. Route Trace 설계

기존 구조:

```text
Business Node

        │

        ▼

Conditional Router

        │

        ▼

다음 Node
```

Day 16 구조:

```text
Traced Business Node

        │

        ▼

Route Trace Node

        │

        ├─ 기존 route 함수 실행
        ├─ trace event 추가
        └─ selected_route 저장

        │

        ▼

route_by_selected_route()

        │

        ▼

다음 Node
```

route 판단과 trace 기록을 일반 LangGraph node에서 수행한 이유는 conditional router의 경로 선택 책임과 state 변경 책임을 분리하기 위해서이다.

한 번 계산한 route는 `selected_route`에 저장한다.

이후 conditional edge는 route를 다시 계산하지 않고 저장된 값을 읽는다.

따라서 다음 두 값이 같은 결과를 사용한다.

```text
실제 LangGraph 이동 경로

=

trace에 기록된 selected_route
```

---

# 8. Trace Event 구조

각 trace event는 다음 필드를 가진다.

| 필드            | 의미                                        |
| ------------- | ----------------------------------------- |
| `sequence`    | 요청 안에서 event가 실행된 순서                      |
| `event_type`  | `node` 또는 `route`                         |
| `event_name`  | 실행된 node 또는 route 이름                      |
| `status`      | `success`, `warning`, `error`, `fallback` |
| `started_at`  | UTC ISO 8601 시작 시각                        |
| `finished_at` | UTC ISO 8601 종료 시각                        |
| `duration_ms` | 실행 시간, 단위 ms                              |
| `metadata`    | 단계별 구조화 요약 정보                             |

---

# 9. Node별 Metadata

## 9.1 Question validation

저장 정보:

```text
question_valid

question_length

error_count
```

사용자 질문 원문 전체는 기본 trace에 저장하지 않는다.

---

## 9.2 Intent classification

저장 정보:

```text
intent

intent_source

confidence

warning_count
```

OpenAI 원본 응답 전체는 기본 trace metadata에 저장하지 않는다.

---

## 9.3 Failure prediction

저장 정보:

```text
raw_sample_provided

prediction_succeeded

prediction

risk_level

evidence_count

warning_count

error_count
```

전체 raw sample 값은 기본 trace에 저장하지 않는다.

---

## 9.4 Route

저장 정보:

```text
selected_route

has_errors
```

route 종류에 따라 아래 정보도 포함한다.

```text
intent

prediction
```

---

## 9.5 Answer

저장 정보:

```text
answer_created

risk_level

evidence_count

error_count
```

사용자에게 반환한 전체 answer 원문은 trace metadata에 중복 저장하지 않는다.

---

# 10. `intent_source=fallback`과 `fallback_occurred=True`의 차이

두 값은 의미가 다르다.

## 10.1 `intent_source == "fallback"`

OpenAI intent classification이 실패하여 rule-based intent classifier를 사용했다는 의미이다.

예:

```text
OpenAI API 오류

        │

        ▼

Rule-based Intent Classifier

        │

        ▼

intent_source = fallback
```

이후 정상 Agent 경로를 실행했다면:

```text
trace_status = success

fallback_occurred = false
```

일 수 있다.

---

## 10.2 `fallback_occurred == True`

LangGraph가 실제 fallback route 또는 fallback answer node를 실행했다는 의미이다.

예:

```text
failure_prediction intent

+

raw_sample 없음

        │

        ▼

prediction 수행 불가

        │

        ▼

fallback route

        │

        ▼

build_fallback_answer

        │

        ▼

fallback_occurred = true
```

---

# 11. FastAPI Response 확장

## 11.1 `src/api/schemas.py`

추가한 response model:

```python
TraceEventResponse
```

추가한 LangGraph API response 필드:

```python
trace_id

trace_status

trace_started_at

trace_finished_at

trace_duration_ms

fallback_occurred

trace_events
```

---

## 11.2 `src/api/langgraph_agent_api.py`

`_state_to_response()`에서 AgentState의 trace 값을 FastAPI response model에 연결했다.

```text
AgentState

        │

        ▼

_state_to_response()

        │

        ▼

LangGraphAgentQueryResponse

        │

        ▼

JSON / Swagger
```

Pydantic은 `trace_events` 안의 일반 dict를 `TraceEventResponse` schema에 맞게 자동 검증한다.

---

# 12. 테스트 구성

## 12.1 `tests/test_agent_state.py`

검증 내용:

* trace 기본값 생성
* 요청별 독립 trace ID
* 32자리 UUID hex 형식
* UTC ISO trace 시작 시각
* 요청별 독립 trace event list

확인된 실행 결과:

```text
19 passed
```

---

## 12.2 `tests/test_agent_trace.py`

검증 내용:

* UTC ISO 시각 생성
* 실행 시간 계산
* 누락된 trace 기본값 보완
* 기존 trace 값 유지
* trace context 전달
* event sequence
* metadata 복사
* node success
* node warning
* node error
* node fallback
* node 예외 기록 후 재발생
* route 선택 기록
* fallback route
* route 예외 기록 후 재발생
* 전체 trace 상태 결정
* 전체 trace 종료 처리

확인된 실행 결과:

```text
18 passed
```

---

## 12.3 현재 확인된 Trace 단위 테스트

```text
tests/test_agent_state.py

19 passed


tests/test_agent_trace.py

18 passed


합계

37 passed
```

---

## 12.4 LangGraph Trace 통합 테스트

`tests/test_failure_agent_graph.py`에 다음 경로의 trace 통합 검증을 추가했다.

* dataset schema 정상 경로
* failure prediction 정상 경로
* unknown intent fallback
* raw sample 누락 fallback
* 빈 question fallback
* event 순서
* event sequence
* route metadata
* intent metadata
* prediction metadata
* trace status
* fallback 여부

---

## 12.5 FastAPI Trace 테스트

`tests/test_api_langgraph_agent.py`에 다음 검증을 추가했다.

* 정상 trace JSON 응답
* fallback trace JSON 응답
* trace가 없는 기존 fake state와의 호환성
* OpenAPI trace field 등록
* `trace_events` 배열 schema
* `TraceEventResponse` 참조

---

## 12.6 프로젝트 전체 회귀 테스트

Day 16 구현 이후 프로젝트 전체 테스트를 실행했다.

실행 명령:

```powershell
pytest -v

---

# 13. 실제 Trace Demo

실행 명령:

```powershell
python -m scripts.run_day16_trace_demo --scenario all
```

실행 시나리오:

```text
1. Dataset schema success path

2. Failure prediction success path

3. Missing raw_sample fallback path
```

실제 결과:

```text
completed scenarios : 3/3

result              : SUCCESS
```

---

# 14. 실제 실행 결과 분석

## 14.1 Dataset schema 정상 경로

최종 결과:

```text
trace_status        : success

fallback_occurred   : false

intent              : dataset_schema_query

intent_source       : openai

confidence          : 0.95

error_count         : 0
```

event 실행 순서:

```text
1. validate_question

2. route_after_validation

3. classify_intent

4. route_after_classification

5. build_dataset_schema_answer
```

실제 시간:

| 구간                            |       실행 시간 |
| ----------------------------- | ----------: |
| 전체 workflow                   | 6384.030 ms |
| `validate_question`           |    0.020 ms |
| `route_after_validation`      |    0.006 ms |
| `classify_intent`             | 6362.452 ms |
| `route_after_classification`  |    0.006 ms |
| `build_dataset_schema_answer` |    0.018 ms |

관찰 결과:

`classify_intent`가 전체 실행 시간의 약 99.7%를 차지했다.

즉 이번 실행에서는 LangGraph routing이나 정적 schema answer 생성보다 OpenAI intent classification이 주요 응답 지연 구간이었다.

---

## 14.2 Failure prediction 정상 경로

최종 결과:

```text
trace_status        : success

fallback_occurred   : false

intent              : failure_prediction

intent_source       : openai

confidence          : 0.95

prediction          : 1

probability         : 0.9929707646369934

risk_level          : HIGH

error_count         : 0
```

event 실행 순서:

```text
1. validate_question

2. route_after_validation

3. classify_intent

4. route_after_classification

5. call_failure_prediction

6. route_after_prediction

7. build_final_answer
```

실제 시간:

| 구간                           |        실행 시간 |
| ---------------------------- | -----------: |
| 전체 workflow                  | 10696.419 ms |
| `validate_question`          |     0.031 ms |
| `route_after_validation`     |     0.033 ms |
| `classify_intent`            |  2764.799 ms |
| `route_after_classification` |     0.006 ms |
| `call_failure_prediction`    |  7919.816 ms |
| `route_after_prediction`     |     0.006 ms |
| `build_final_answer`         |     0.017 ms |

관찰 결과:

```text
OpenAI intent classification

약 2.765초


Failure prediction service

약 7.920초
```

두 구간이 전체 실행 시간의 대부분을 차지했다.

반면 LangGraph route 자체는 약 `0.006 ms` 수준이었다.

따라서 이번 단일 실행에서는 LangGraph routing보다 OpenAI API 호출과 prediction service 구간이 주요 지연 구간으로 관찰되었다.

---

## 14.3 raw sample 누락 fallback 경로

최종 결과:

```text
trace_status        : fallback

fallback_occurred   : true

intent              : failure_prediction

intent_source       : openai

prediction          : null

risk_level          : UNKNOWN

error_count         : 1
```

event 실행 순서:

```text
1. validate_question

2. route_after_validation

3. classify_intent

4. route_after_classification

5. call_failure_prediction

6. route_after_prediction

7. build_fallback_answer
```

상태 변화:

```text
call_failure_prediction

status = error

errors_added = 1
```

```text
route_after_prediction

status = fallback

selected_route = fallback
```

```text
build_fallback_answer

status = fallback
```

최종:

```text
trace_status = fallback

fallback_occurred = true
```

이 결과는 서버가 예상하지 못한 예외로 종료된 것이 아니다.

Agent가 다음 상황을 감지했다.

```text
failure_prediction intent

+

raw_sample 없음
```

그 후:

```text
prediction 수행 중단

→ error 기록

→ fallback route

→ 사용자 안내 답변 생성

→ 정상적인 Agent 응답 종료
```

흐름으로 처리했다.

---

# 15. 관측 결과 해석 시 주의점

이번 시간 측정값은 로컬 환경에서 수행한 단일 실행 결과이다.

따라서 다음과 같은 성능 벤치마크로 단정하면 안 된다.

```text
OpenAI는 항상 6초가 걸린다.

prediction service는 항상 8초가 걸린다.

LangGraph는 항상 0.006 ms가 걸린다.
```

실제 시간은 다음 요인에 따라 달라질 수 있다.

* 네트워크 상태
* OpenAI API 응답 시간
* 첫 실행 cold start
* Python module import
* 모델 artifact 로딩
* 모델 초기화
* 운영체제 캐시
* CPU 성능
* 동시 요청 수
* evidence 생성 옵션

현재 trace 결과는 최종 성능 평가가 아니라, **어느 구간을 추가로 관찰하고 최적화해야 하는지 찾기 위한 실행 근거**이다.

---

# 16. 새로 발견한 개선 후보

정상 failure prediction 실행에서 `call_failure_prediction` node가 약 7.9초 걸렸다.

현재 trace는 prediction node 전체 실행 시간만 측정한다.

따라서 node 내부의 어느 단계가 시간을 차지했는지는 아직 구분할 수 없다.

향후 다음처럼 세부 관측을 추가할 수 있다.

```text
call_failure_prediction

        │

        ├─ request 변환
        │
        ├─ artifact load
        │
        ├─ preprocessing
        │
        ├─ model inference
        │
        ├─ rule evidence
        │
        ├─ SHAP evidence
        │
        ├─ global importance evidence
        │
        └─ answer build
```

개선 후보:

1. 모델 artifact를 요청마다 다시 로드하는지 확인

2. 모델과 scaler를 application startup 시 한 번만 로드하는 구조 검토

3. prediction service 내부 단계별 duration 측정

4. 첫 요청과 두 번째 이후 요청의 cold start 차이 측정

5. SHAP 활성화·비활성화 실행 시간 비교

6. global importance 활성화·비활성화 실행 시간 비교

7. OpenAI intent 분류 timeout과 retry 정책 검토

8. 반복 실행 평균, 중앙값, 최소값, 최대값 측정

---

# 17. 개인정보·운영 정보 보호 설계

기본 trace에는 전체 원문을 무조건 저장하지 않는다.

저장하지 않는 정보:

```text
전체 사용자 question

전체 chat_history

전체 raw_sample

OpenAI API key

OpenAI raw response 전체
```

대신 요약 정보를 저장한다.

예:

```json
{
  "question_valid": true,
  "question_length": 20
}
```

```json
{
  "raw_sample_provided": true,
  "prediction_succeeded": true,
  "prediction": 1,
  "risk_level": "HIGH"
}
```

이 구조는 관측에 필요한 정보를 남기면서 불필요한 원문 저장을 줄인다.

실제 운영 환경에서는 다음도 추가로 검토해야 한다.

* trace 보관 기간
* 사용자별 trace 접근 권한
* 민감 정보 masking
* raw log 암호화
* trace 삭제 정책
* 운영 환경과 개발 환경 로그 수준 분리

---

# 18. 현재 한계

1. trace는 현재 Python 메모리와 API response에만 존재한다.

2. 서버 재시작 후 이전 trace를 조회할 수 없다.

3. trace 검색 API가 없다.

4. trace를 DB에 저장하지 않는다.

5. 여러 서비스에 걸친 distributed trace는 지원하지 않는다.

6. OpenTelemetry trace ID 형식을 사용하지 않는다.

7. 외부 시각화 dashboard가 없다.

8. node 내부 세부 처리 시간은 아직 구분하지 않는다.

9. 단일 실행 결과만으로 성능을 일반화할 수 없다.

10. 현재 trace event metadata는 프로젝트 내부 기준으로 직접 설계한 구조이다.

---

# 19. 이후 확장 방향

## 단기

```text
Day 17

실제 OpenAI 경로 E2E 검증

실제 API request 검증

fallback 경로 검증

응답 schema 검증
```

## 중기

```text
Day 19

Agent 실행 이력 DB 저장

trace_id 기반 조회

실행 결과 검색
```

## 장기

```text
LangSmith 연결

OpenTelemetry 연결

분산 trace

Dashboard 시각화

node별 latency chart

error·fallback 비율

intent별 요청 통계
```

---

# 20. 면접 답변

## 질문 1

### Agent 실행 과정을 어떻게 추적했나요?

답변:

> LangGraph가 최종 답변만 반환하면 어떤 node와 route가 실행됐는지 확인하기 어려웠습니다. 그래서 요청마다 UUID 기반 trace ID를 생성하고, node와 route 실행을 구조화 event로 기록했습니다. 각 event에는 실행 순서, 시작·종료 시각, 실행 시간, 상태, 주요 metadata를 저장했습니다. 또한 전체 workflow 종료 시 success, fallback, error 상태와 전체 실행 시간을 계산했습니다.

---

## 질문 2

### 왜 LangSmith를 바로 사용하지 않았나요?

답변:

> 외부 관측성 도구를 연결하기 전에 trace가 어떤 정보로 구성되고 node와 route가 어떻게 기록되는지 직접 이해하기 위해 내부 구조화 trace를 먼저 구현했습니다. 현재 구조는 외부 서비스 없이도 로컬에서 실행 흐름을 검증할 수 있고, 이후 LangSmith나 OpenTelemetry를 연결할 때 기반 구조로 확장할 수 있습니다.

---

## 질문 3

### 기존 node 코드를 직접 수정했나요?

답변:

> 기존 business node 안에 시간 측정 코드를 반복해서 넣지 않고 traced wrapper를 만들었습니다. wrapper가 기존 node를 실행한 뒤 duration, warning 증가량, error 증가량, metadata를 기록하도록 구성했습니다. 이를 통해 business logic과 observability 책임을 분리했습니다.

---

## 질문 4

### Route는 어떻게 추적했나요?

답변:

> 기존 route 함수는 다음 경로를 문자열로 반환하는 책임을 유지했습니다. 별도의 route trace node가 기존 route 함수를 실행하고 선택 결과를 trace event와 AgentState의 selected_route에 저장합니다. 이후 conditional edge는 저장된 selected_route만 읽도록 구성하여 실제 이동 경로와 trace 기록이 같은 값을 사용하게 했습니다.

---

## 질문 5

### `intent_source=fallback`과 `fallback_occurred=True`는 어떤 차이인가요?

답변:

> intent_source의 fallback은 OpenAI intent 분류 실패 후 rule-based classifier를 사용했다는 의미입니다. 반면 fallback_occurred는 LangGraph가 실제 fallback route 또는 fallback answer node를 실행했다는 의미입니다. 두 개념을 분리하여 분류기의 fallback과 workflow의 fallback을 구분했습니다.

---

## 질문 6

### Trace를 적용해서 무엇을 발견했나요?

답변:

> 실제 실행 결과 LangGraph routing 자체의 실행 시간은 매우 작았고, 응답 지연은 OpenAI intent classification과 failure prediction service 구간에 집중되어 있었습니다. 이를 통해 추측이 아니라 실행 근거를 바탕으로 다음 최적화 대상을 정할 수 있었습니다.

---

## 질문 7

### Trace에 전체 입력값을 저장하나요?

답변:

> 기본 trace에는 전체 question, chat history, raw sample, OpenAI 원본 응답을 저장하지 않았습니다. 대신 question length, raw sample 제공 여부, prediction 성공 여부, risk level과 같은 요약 metadata를 저장했습니다. 관측성을 확보하면서 불필요한 원문과 민감 정보 저장을 줄이기 위한 설계입니다.

---

# 21. Day 16 핵심 성과

```text
최종 결과만 확인

        ↓

node·route 실행 과정 관찰 가능
```

```text
응답이 느리다는 추측

        ↓

실제 node별 duration 확인
```

```text
fallback 여부를 답변으로 추정

        ↓

fallback_occurred와 trace_status로 구조화
```

```text
문자열 로그 중심

        ↓

TypedDict + Pydantic 기반 구조화 trace
```

```text
LangGraph 내부 상태

        ↓

FastAPI JSON과 Swagger에서 확인 가능
```

---

# 22. Day 16 완료 체크리스트

## Trace State

* [x] `trace_id`
* [x] `trace_status`
* [x] `trace_started_at`
* [x] `trace_finished_at`
* [x] `trace_duration_ms`
* [x] `fallback_occurred`
* [x] `trace_events`

## Trace Helper

* [x] UTC ISO 시각
* [x] millisecond 실행 시간
* [x] event sequence
* [x] metadata
* [x] node trace
* [x] route trace
* [x] warning 상태
* [x] error 상태
* [x] fallback 상태
* [x] 최종 trace 상태

## LangGraph Integration

* [x] traced node wrapper
* [x] route trace node
* [x] `selected_route`
* [x] 실제 event 순서
* [x] 정상 prediction 경로
* [x] dataset schema 경로
* [x] unknown fallback 경로
* [x] raw sample 누락 fallback 경로

## FastAPI

* [x] `TraceEventResponse`
* [x] LangGraph response trace field
* [x] AgentState → API response 연결
* [x] OpenAPI / Swagger schema

## Validation

* [x] AgentState·Trace helper 테스트 37개 통과
* [x] 실제 OpenAI intent classification 성공
* [x] 실제 failure prediction 성공
* [x] 실제 fallback 경로 성공
* [x] Demo 시나리오 3/3 성공
* [x] 프로젝트 전체 pytest 최종 회귀 결과 기록

---

# 23. 최종 정리

Day 16에서는 LangGraph Agent에 내부 구조화 trace와 observability 기능을 구현했다.

요청마다 고유한 `trace_id`를 생성하고, node와 route 실행을 순서대로 기록했다.

각 event에는 다음 정보를 저장했다.

```text
실행 순서

event 종류

node·route 이름

실행 상태

시작 시각

종료 시각

실행 시간

구조화 metadata
```

전체 workflow 종료 후에는 다음 정보를 계산했다.

```text
trace_status

trace_finished_at

trace_duration_ms

fallback_occurred
```

실제 실행에서는 세 시나리오를 검증했다.

```text
Dataset schema 정상 경로

→ success


Failure prediction 정상 경로

→ success


raw sample 누락 경로

→ fallback
```

실제 trace를 통해 LangGraph routing보다 OpenAI intent classification과 failure prediction service가 주요 지연 구간임을 관찰했다.

다만 이번 측정은 단일 로컬 실행 결과이므로 성능 벤치마크로 일반화하지 않고, 이후 세부 관측과 최적화 대상을 찾기 위한 근거로 사용한다.

Day 16을 통해 Agent가 단순히 결과를 반환하는 구조에서 벗어나, 내부 처리 흐름을 설명하고 문제 발생 위치와 성능 병목 후보를 추적할 수 있는 구조로 확장되었다.
