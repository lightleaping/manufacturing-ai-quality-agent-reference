# Day 18 - 실제 OpenAI E2E 반복 안정성 및 응답시간 Benchmark

## 1. Day 18 목표

Day 18의 목표는 Day 17에서 검증한 실제 OpenAI E2E 경로를 반복 실행하고, 단순 성공 여부뿐 아니라 반복 안정성과 응답시간을 구조화된 지표로 측정하는 것이었다.

Day 17에서는 각 시나리오가 실제 환경에서 한 번 이상 정상적으로 동작하는지를 확인했다.

Day 18에서는 동일한 시나리오를 여러 번 실행하여 다음 항목을 측정했다.

```text
전체 실행 횟수

성공 횟수

실패 횟수

성공률

최소 응답시간

최대 응답시간

평균 응답시간

중앙값

p95 참고값

intent 일치율

intent_source=openai 비율

LangGraph route 일치율

Trace 상태 일치율

fallback 발생률

fallback 기대값 일치율
```

실제 OpenAI API 반복 호출은 네트워크, API 비용, 외부 서비스 상태의 영향을 받기 때문에 기본 `pytest`에는 포함하지 않고 사용자가 명시적으로 실행하는 별도 Benchmark 스크립트로 구성했다.

---

## 2. Day 17 E2E와 Day 18 Benchmark의 차이

### Day 17

Day 17의 목적은 실제 시스템 경로가 정상적으로 연결되어 있는지 검증하는 것이었다.

```text
질문 입력

→ 실제 OpenAI intent 분류

→ LangGraph route 선택

→ 필요하면 실제 PyTorch 추론

→ Evidence·Answer 생성

→ Trace 생성

→ FastAPI 응답 검증
```

각 시나리오가 한 번 이상 정상 동작하는지를 확인했다.

```text
동작하는가?
```

를 검증한 단계였다.

### Day 18

Day 18의 목적은 동일한 경로를 반복 실행했을 때 결과가 얼마나 안정적으로 유지되는지를 확인하는 것이었다.

```text
같은 질문을 반복 실행

→ intent가 계속 일치하는가?

→ intent_source가 계속 openai인가?

→ 같은 LangGraph route를 선택하는가?

→ Trace가 예상 상태로 끝나는가?

→ fallback 발생 여부가 기대와 일치하는가?

→ 응답시간 분포는 어떤가?
```

즉, Day 18은 다음 질문에 답하기 위한 단계였다.

```text
한 번 동작하는가?

↓

반복 실행해도 안정적으로 동작하는가?
```

---

## 3. 추가 및 수정 파일

### 신규 파일

```text
scripts/run_day18_e2e_benchmark.py
```

역할:

```text
Day 17 실제 E2E 시나리오 반복 실행

응답시간 측정

구조화 안정성 지표 계산

시나리오별 통계 계산

전체 통계 계산

JSON Artifact 저장

Process exit code 반환
```

### 생성 Artifact

```text
reports/artifacts/day18_e2e_benchmark.json
```

역할:

```text
실행 설정 저장

실행별 원본 결과 저장

시나리오별 통계 저장

전체 요약 저장

Benchmark 한계와 주의사항 저장
```

### 수정 파일

```text
scripts/run_day17_e2e_openai_validation.py
```

기존 Day 17 동작을 유지하면서 다음 선택 옵션을 추가했다.

```python
run_schema_scenario(
    return_state=True,
)

run_prediction_scenario(
    return_state=True,
)

run_api_prediction_scenario(
    return_response=True,
)
```

기본값은 모두 `False`이므로 기존 Day 17 실행 방식은 변경되지 않는다.

```python
run_schema_scenario()

run_prediction_scenario()

run_api_prediction_scenario()
```

위와 같이 기존 방식으로 호출하면 성공 시 기존과 동일하게 `True`를 반환한다.

Day 18에서 구조화 결과가 필요할 때만 최종 `AgentState` 또는 FastAPI `response_json`을 반환한다.

---

## 4. 기본 실행 방법

### 프로젝트 경로 이동 및 가상환경 실행

```powershell
cd C:\Users\kflow\Downloads\manufacturing-ai-quality-agent-reference

.\.venv\Scripts\Activate.ps1
```

### 기본 Benchmark 실행

```powershell
python -m scripts.run_day18_e2e_benchmark
```

기본 설정은 다음과 같다.

```text
scenario : core

repeat   : 3
```

실질적으로 다음 명령과 같다.

```powershell
python -m scripts.run_day18_e2e_benchmark `
    --scenario core `
    --repeat 3
```

### 단일 시나리오 실행

```powershell
python -m scripts.run_day18_e2e_benchmark `
    --scenario prediction `
    --repeat 3
```

### 전체 Day 17 시나리오 실행

```powershell
python -m scripts.run_day18_e2e_benchmark `
    --scenario all `
    --repeat 3
```

### 출력 경로 직접 지정

```powershell
python -m scripts.run_day18_e2e_benchmark `
    --scenario core `
    --repeat 3 `
    --output reports/artifacts/custom_day18_benchmark.json
```

---

## 5. Benchmark 시나리오

### core 시나리오

Day 18의 기본 대표 Benchmark는 다음 세 시나리오로 구성했다.

```text
schema

prediction

api_prediction
```

### schema

실행 범위:

```text
실제 OpenAI intent 분류

→ LangGraph dataset_schema route

→ Dataset schema answer

→ Trace
```

기대값:

```text
intent              : dataset_schema_query

intent_source       : openai

selected_route      : dataset_schema

trace_status        : success

fallback_occurred   : false
```

### prediction

실행 범위:

```text
실제 OpenAI intent 분류

→ LangGraph failure prediction route

→ 실제 PyTorch MLP 추론

→ prediction·probability·threshold

→ risk level

→ recommended action

→ final answer

→ Trace
```

기대값:

```text
intent              : failure_prediction

intent_source       : openai

selected_route      : final

trace_status        : success

fallback_occurred   : false
```

실제 모델 결과:

```text
prediction          : 1

probability         : 0.9929707646369934

threshold           : 0.7

risk_level          : HIGH
```

### api_prediction

실행 범위:

```text
FastAPI TestClient

→ POST /agent/langgraph-query

→ Pydantic request validation

→ 실제 OpenAI

→ LangGraph

→ 실제 PyTorch MLP

→ Pydantic response validation

→ HTTP JSON response

→ Trace 검증
```

기대값:

```text
HTTP status         : 200

intent              : failure_prediction

intent_source       : openai

route               : final

trace_status        : success

fallback_occurred   : false
```

---

## 6. CLI 설계

Day 18 스크립트는 다음 옵션을 제공한다.

### `--scenario`

지원값:

```text
schema

prediction

multi_turn

missing_sample

unknown

api_prediction

core

all
```

`core`:

```text
schema

+

prediction

+

api_prediction
```

`all`:

```text
Day 17 전체 6개 시나리오
```

### `--repeat`

시나리오별 반복 횟수를 지정한다.

기본값:

```text
3
```

잘못된 입력:

```powershell
python -m scripts.run_day18_e2e_benchmark `
    --scenario schema `
    --repeat 0
```

결과:

```text
[ERROR] --repeat must be at least 1.
```

Process exit code:

```text
2
```

### `--output`

JSON Artifact 저장 경로를 지정한다.

기본값:

```text
reports/artifacts/day18_e2e_benchmark.json
```

---

## 7. 응답시간 측정 방식

응답시간은 Python의 다음 함수를 사용해 측정했다.

```python
time.perf_counter()
```

측정 방식:

```python
started_at = time.perf_counter()

result = scenario_runner()

duration_ms = (
    time.perf_counter() - started_at
) * 1000.0
```

`time.perf_counter()`는 현재 날짜와 시각을 구하기 위한 함수가 아니라 두 시점 사이의 경과시간을 정밀하게 측정하기 위한 고해상도 타이머다.

측정 범위에는 다음이 포함된다.

```text
Day 17 시나리오 함수 시작

→ 실제 OpenAI 호출

→ LangGraph 실행

→ 필요하면 PyTorch 추론

→ Day 17 검증

→ 구조화 결과 추출

→ Day 18 안정성 검증

→ 시나리오 함수 종료
```

---

## 8. 응답시간 통계

시나리오별로 다음 통계를 계산한다.

```text
min_duration_ms

max_duration_ms

mean_duration_ms

median_duration_ms

p95_duration_ms
```

### 최소값

반복 실행 중 가장 빠른 응답시간이다.

### 최대값

반복 실행 중 가장 느린 응답시간이다.

### 평균

모든 성공 실행시간의 합을 성공 실행 횟수로 나눈 값이다.

극단적으로 느린 실행 하나의 영향을 크게 받을 수 있다.

### 중앙값

실행시간을 정렬했을 때 가운데에 위치한 값이다.

극단적인 값의 영향을 평균보다 적게 받는다.

### p95

전체 실행의 약 95%가 해당 시간 이하에서 완료된다는 의미의 백분위수다.

다만 Day 18의 기본 반복 횟수는 3회이므로 현재 p95는 운영 환경의 SLA를 의미하지 않는다.

```text
작은 표본에서 계산한 참고값
```

으로만 사용한다.

---

## 9. 성공한 실행만 성능 통계에 포함한 이유

실패한 요청은 인증 오류나 네트워크 연결 실패 때문에 정상 요청보다 매우 빠르게 종료될 수 있다.

예:

```text
정상 실행:

4.2초


API 인증 실패:

0.1초
```

이 둘을 같은 평균에 포함하면 시스템이 실제보다 빠른 것처럼 보일 수 있다.

따라서 다음처럼 분리했다.

```text
안정성 통계:

모든 실행을 대상으로 성공률 계산


성능 통계:

성공한 실행시간만 대상으로 계산
```

성공한 실행이 없으면 응답시간 통계는 다음처럼 저장한다.

```json
{
  "min_duration_ms": null,
  "max_duration_ms": null,
  "mean_duration_ms": null,
  "median_duration_ms": null,
  "p95_duration_ms": null
}
```

---

## 10. 구조화 안정성 지표

실행별로 다음 필드를 저장한다.

```text
expected_intent

actual_intent

intent_match

intent_source

intent_source_is_openai

expected_route

actual_route

route_match

expected_trace_status

actual_trace_status

trace_status_match

expected_fallback_occurred

actual_fallback_occurred

fallback_match
```

예:

```json
{
  "scenario": "prediction",
  "iteration": 1,
  "success": true,
  "duration_ms": 2829.06,

  "expected_intent": "failure_prediction",
  "actual_intent": "failure_prediction",
  "intent_match": true,

  "intent_source": "openai",
  "intent_source_is_openai": true,

  "expected_route": "final",
  "actual_route": "final",
  "route_match": true,

  "expected_trace_status": "success",
  "actual_trace_status": "success",
  "trace_status_match": true,

  "expected_fallback_occurred": false,
  "actual_fallback_occurred": false,
  "fallback_match": true,

  "error_type": null,
  "error_message": null
}
```

---

## 11. 최종 성공 판정 기준

Day 18의 실행 성공 여부는 단순히 예외가 발생하지 않았는지만으로 결정하지 않는다.

핵심 시나리오는 다음 조건을 모두 만족해야 한다.

```text
Day 17 E2E 검증 성공

intent 기대값 일치

intent_source=openai

route 기대값 일치

trace_status 기대값 일치

fallback 기대값 일치
```

코드 개념:

```python
success = (
    day17_success
    and structured_metrics_success
)
```

즉 프로그램이 종료되지 않았더라도 route나 intent가 예상과 다르면 실패로 처리한다.

---

## 12. FastAPI route 추출 문제와 해결

### 문제

Agent를 직접 실행하는 `schema`와 `prediction` 시나리오는 최종 `AgentState`에 다음 값이 포함되어 있었다.

```python
state["selected_route"]
```

하지만 FastAPI의 `response_json`에는 `selected_route`가 직접 포함되지 않았다.

그 결과 최초 Day 18 API 구조화 검증에서는 다음처럼 잘못 판정됐다.

```text
expected_route : final

actual_route   : None

route_match    : False

result         : FAILURE
```

실제 FastAPI E2E 검증은 모두 성공했고 HTTP status도 200이었으므로 API 기능 문제가 아니라 Day 18의 route 추출 방식 문제였다.

### 해결

FastAPI 응답에 직접 `selected_route`를 추가하는 대신 기존 `trace_events`를 사용해 실제 최종 route를 판정했다.

Prediction 정상 경로:

```text
validate_question

route_after_validation

classify_intent

route_after_classification

call_failure_prediction

route_after_prediction

build_final_answer
```

위 Trace 순서가 확인되면 다음으로 판정한다.

```text
actual_route : final
```

Fallback 경로:

```text
validate_question

route_after_validation

classify_intent

route_after_classification

call_failure_prediction

route_after_prediction

build_fallback_answer
```

위 순서가 확인되면 다음으로 판정한다.

```text
actual_route : fallback
```

### 해결 결과

```text
expected_route : final

actual_route   : final

route_match    : True

result         : SUCCESS
```

이 방식은 기대값을 그대로 복사하지 않고 실제 Trace 결과를 이용해 route를 판정한다.

---

## 13. JSON Artifact 구조

최상위 구조:

```json
{
  "benchmark_name": "day18_e2e_reliability_performance",
  "generated_at": "...",
  "configuration": {},
  "execution": {},
  "environment": {},
  "scenario_expectations": {},
  "scenario_summaries": [],
  "runs": [],
  "overall_summary": {},
  "current_limitations": [],
  "disclaimer": "..."
}
```

### configuration

```text
요청한 시나리오

실제 실행된 시나리오

반복 횟수

실제 OpenAI 사용 여부

pytest 포함 여부

구조화 지표 지원 시나리오
```

### execution

```text
Benchmark 시작 시각

Benchmark 종료 시각

전체 실행시간
```

### scenario_summaries

시나리오별 다음 값을 저장한다.

```text
성공률

응답시간 통계

intent 일치율

OpenAI source 비율

route 일치율

Trace 상태 일치율

fallback 발생률

fallback 기대값 일치율
```

### runs

각 반복 실행의 원본 결과를 저장한다.

### overall_summary

전체 시나리오를 합친 성공률과 안정성 비율을 저장한다.

---

## 14. 최종 core Benchmark 결과

실행 명령:

```powershell
python -m scripts.run_day18_e2e_benchmark `
    --scenario core `
    --repeat 3
```

총 실행 횟수:

```text
schema         × 3

prediction     × 3

api_prediction × 3

총 9회
```

최종 결과:

```text
total_runs                  : 9

successful_runs             : 9

failed_runs                 : 0

success_rate                : 1.0000

intent_match_rate           : 1.0

openai_source_rate          : 1.0

route_match_rate            : 1.0

trace_status_match_rate     : 1.0

fallback_rate               : 0.0

fallback_match_rate         : 1.0

total_duration_ms           : 24907.73

result                      : SUCCESS
```

---

## 15. 시나리오별 응답시간 결과

### schema

```text
total_runs                  : 3

successful_runs             : 3

failed_runs                 : 0

success_rate                : 1.0000

min_duration_ms             : 2079.49

max_duration_ms             : 4973.48

mean_duration_ms            : 3276.33

median_duration_ms          : 2776.02

p95_duration_ms             : 4753.73
```

구조화 안정성:

```text
intent_match_rate           : 1.0

openai_source_rate          : 1.0

route_match_rate            : 1.0

trace_status_match_rate     : 1.0

fallback_rate               : 0.0

fallback_match_rate         : 1.0
```

### prediction

```text
total_runs                  : 3

successful_runs             : 3

failed_runs                 : 0

success_rate                : 1.0000

min_duration_ms             : 2101.45

max_duration_ms             : 2829.06

mean_duration_ms            : 2425.81

median_duration_ms          : 2346.93

p95_duration_ms             : 2780.85
```

구조화 안정성:

```text
intent_match_rate           : 1.0

openai_source_rate          : 1.0

route_match_rate            : 1.0

trace_status_match_rate     : 1.0

fallback_rate               : 0.0

fallback_match_rate         : 1.0
```

### api_prediction

```text
total_runs                  : 3

successful_runs             : 3

failed_runs                 : 0

success_rate                : 1.0000

min_duration_ms             : 2390.35

max_duration_ms             : 2895.49

mean_duration_ms            : 2584.33

median_duration_ms          : 2467.14

p95_duration_ms             : 2852.65
```

구조화 안정성:

```text
intent_match_rate           : 1.0

openai_source_rate          : 1.0

route_match_rate            : 1.0

trace_status_match_rate     : 1.0

fallback_rate               : 0.0

fallback_match_rate         : 1.0
```

---

## 16. 결과 해석

### 반복 안정성

총 9회의 실제 OpenAI E2E 실행에서 실패가 발생하지 않았다.

```text
9 / 9 성공
```

다음 결과가 모든 실행에서 기대값과 일치했다.

```text
intent

intent_source

LangGraph route

Trace 상태

fallback 발생 여부
```

### OpenAI 자연어 출력 변화

`intent_reason` 문장은 실행마다 일부 표현이 달라질 수 있었다.

예:

```text
데이터셋 설명 관련 질문으로 판단됨.

데이터셋 설명에 대한 질문으로 분류됨.
```

하지만 시스템이 의존하는 핵심 구조화 필드는 동일하게 유지됐다.

```text
intent              : dataset_schema_query

intent_source       : openai

selected_route      : dataset_schema
```

이는 LLM 자연어 문장은 변할 수 있지만 구조화 출력과 실행 경로가 안정적으로 유지되는지를 별도로 검증해야 하는 이유다.

### 응답시간 변동

일부 첫 실행은 이후 실행보다 느렸다.

가능한 원인:

```text
첫 네트워크 연결

TLS 연결 설정

OpenAI client 초기화

외부 서비스의 일시적 응답 지연

운영체제와 Python 런타임 상태
```

하지만 현재 표본만으로 특정 원인이라고 단정하지 않았다.

### 평균과 중앙값

`schema` 결과:

```text
평균   : 3276.33ms

중앙값 : 2776.02ms
```

첫 실행의 큰 값이 평균을 끌어올렸기 때문에 중앙값이 일반적인 실행시간을 더 잘 나타냈다.

### FastAPI TestClient 주의

`api_prediction`은 별도 `uvicorn` 서버를 실행한 실제 네트워크 HTTP Benchmark가 아니다.

```text
FastAPI TestClient

→ 같은 프로세스 내부 실행

→ Pydantic 및 FastAPI route 검증 포함

→ 실제 배포 서버 네트워크 지연 미포함
```

따라서 이 결과를 실제 운영 서버 HTTP 응답시간으로 해석하면 안 된다.

---

## 17. Process exit code

Day 18 스크립트는 다음 exit code를 사용한다.

```text
0:

모든 반복 실행 성공
```

```text
1:

하나 이상의 반복 실행 실패
```

```text
2:

CLI 입력 또는 OpenAI 환경 설정 문제
```

예:

```text
OPENAI_API_KEY 없음

repeat가 1 미만

Artifact 저장 실패
```

---

## 18. 기본 pytest와 분리한 이유

Day 18 Benchmark는 다음 명령에 포함되지 않는다.

```powershell
pytest -v
```

이유:

```text
실제 OpenAI API key 필요

실제 네트워크 필요

API 비용 발생 가능

외부 서비스 응답시간 변동

외부 장애로 인한 비결정적 실패 가능

기본 회귀 테스트 실행시간 증가
```

기본 pytest는 monkeypatch 기반으로 빠르고 안정적인 회귀 검증을 수행한다.

실제 OpenAI Benchmark는 사용자가 명시적으로 실행한다.

```powershell
python -m scripts.run_day18_e2e_benchmark `
    --scenario core `
    --repeat 3
```

---

## 19. 전체 회귀 테스트

실행 명령:

```powershell
pytest -v
```

결과:

```text
168 passed
```

Day 18 구현 후 기존 Day 1~17 기능의 회귀 오류는 발견되지 않았다.

특히 Day 17 시나리오 함수에 구조화 결과 선택 반환 옵션을 추가했지만 기본값을 `False`로 유지했기 때문에 기존 Day 17 실행과 테스트에 영향을 주지 않았다.

---

## 20. 개인정보 및 보안

다음 정보는 콘솔이나 JSON Artifact에 저장하지 않는다.

```text
OpenAI API key 값

OpenAI raw response 전체

전체 question 원문을 포함한 불필요한 민감정보

전체 chat_history 원문

전체 raw_sample의 불필요한 복제
```

Artifact에는 다음 정보만 기록한다.

```text
API key 설정 여부

API key 값 기록 여부

시나리오명

실행 결과

응답시간

구조화 안정성 지표

오류 유형과 메시지
```

---

## 21. 현재 한계

### 작은 표본

기본 반복 횟수는 3회이므로 통계적으로 충분한 성능 표본이 아니다.

```text
p95는 참고값
```

으로만 사용한다.

### 실제 운영 부하 테스트가 아님

동시 사용자, 대량 요청, 장시간 실행을 검증하지 않았다.

```text
Load Test

Stress Test

Soak Test
```

와는 목적이 다르다.

### 외부 서비스 영향

OpenAI 응답시간과 네트워크 상태의 영향을 받는다.

### FastAPI TestClient

실제 배포 서버와 네트워크 구간의 latency를 포함하지 않는다.

### 구조화 지표 지원 범위

현재 구조화 안정성 지표는 다음 핵심 시나리오에 적용했다.

```text
schema

prediction

api_prediction
```

다음 시나리오는 기존 `bool` 기반 성공률과 응답시간 측정 구조를 유지한다.

```text
multi_turn

missing_sample

unknown
```

필요하면 같은 반환 옵션 패턴으로 확장할 수 있다.

---

## 22. Day 18에서 배운 핵심 개념

### E2E와 Benchmark의 차이

```text
E2E:

전체 시스템이 한 번 정상 동작하는지 확인


Benchmark:

반복 실행 결과를 수집하고 안정성·응답시간을 비교
```

### 성공률과 응답시간을 분리해야 하는 이유

빠르게 실패한 요청을 정상 성능에 포함하면 결과가 왜곡될 수 있다.

### 평균과 중앙값을 함께 봐야 하는 이유

평균은 극단값의 영향을 받고 중앙값은 일반적인 응답시간을 더 잘 나타낼 수 있다.

### p95를 과장하면 안 되는 이유

표본이 적은 p95는 운영 환경의 SLA를 의미하지 않는다.

### 자연어 출력과 구조화 결과를 구분해야 하는 이유

LLM의 설명 문장은 달라질 수 있지만 시스템이 의존하는 intent와 route는 안정적으로 유지되어야 한다.

### Trace를 route 근거로 사용할 수 있는 이유

API 응답에 route가 직접 노출되지 않더라도 실제 노드 실행 순서를 이용해 최종 경로를 검증할 수 있다.

---

## 23. 면접 답변

### Day 17과 Day 18의 차이는 무엇인가요?

> Day 17에서는 실제 OpenAI, LangGraph, PyTorch, FastAPI가 연결된 E2E 경로가 정상 동작하는지를 검증했습니다. Day 18에서는 같은 시나리오를 반복 실행하여 성공률, 응답시간, Intent 일치율, OpenAI 분류 비율, LangGraph Route 일치율, Trace 상태 일치율, Fallback 발생률을 측정하는 Benchmark를 구현했습니다.

### 한 번의 실행시간만으로 성능을 판단하면 안 되는 이유는 무엇인가요?

> 실제 OpenAI 호출은 네트워크 상태, 외부 서비스 부하, 초기 연결 비용 등의 영향을 받기 때문에 실행마다 응답시간이 달라질 수 있습니다. 그래서 최소값, 최대값, 평균, 중앙값, p95를 함께 기록했고, 작은 표본 결과를 운영 SLA로 과장하지 않았습니다.

### 평균과 중앙값의 차이는 무엇인가요?

> 평균은 모든 값을 더해 개수로 나누기 때문에 하나의 매우 느린 실행에도 크게 영향을 받습니다. 중앙값은 정렬했을 때 가운데 값이라 극단값의 영향을 덜 받습니다. API Benchmark에서는 두 값을 함께 봐야 일반적인 응답시간과 지연 변동을 모두 이해할 수 있습니다.

### 성공률과 Intent·Route 일치율을 따로 측정한 이유는 무엇인가요?

> 프로그램이 오류 없이 종료됐다고 해서 올바른 Agent 결과라는 보장은 없습니다. 예를 들어 HTTP 200을 반환해도 intent가 틀리거나 fallback 경로를 잘못 선택할 수 있습니다. 그래서 전체 성공률과 별도로 Intent, Intent Source, Route, Trace, Fallback 일치율을 측정했습니다.

### FastAPI 응답에 selected_route가 없을 때 어떻게 해결했나요?

> FastAPI response model에는 selected_route가 직접 포함되지 않아 처음에는 route가 None으로 측정됐습니다. API 스키마를 Benchmark 때문에 변경하는 대신 이미 응답에 포함된 trace_events를 사용했습니다. call_failure_prediction, route_after_prediction, build_final_answer가 순서대로 실행된 경우 final route로 판정하도록 구현했습니다.

### 실제 OpenAI Benchmark를 pytest에 넣지 않은 이유는 무엇인가요?

> 실제 OpenAI 호출은 네트워크, API key, 비용, 외부 서비스 상태에 의존하기 때문에 기본 회귀 테스트가 비결정적으로 실패할 수 있습니다. 기본 pytest는 monkeypatch 기반으로 빠르고 안정적으로 유지하고, 실제 OpenAI Benchmark는 사용자가 명시적으로 실행하도록 분리했습니다.

### AI를 어떻게 활용했나요?

> AI를 개발 보조 도구로 활용해 Benchmark 구조와 코드 초안을 빠르게 구성했고, 이후 제가 직접 실제 OpenAI 호출을 반복 실행하면서 응답시간 변동, FastAPI route 누락 문제, Trace 기반 route 추출, JSON Artifact 결과를 검증하고 수정했습니다. 단순히 코드를 생성하는 데서 끝내지 않고 실행 결과와 회귀 테스트까지 확인해 전체 흐름을 제 것으로 만들었습니다.

---

## 24. Day 18 최종 완료 상태

```text
실제 OpenAI 반복 Benchmark 구현

응답시간 자동 측정

성공률 자동 계산

min·max·mean·median·p95 계산

Intent 안정성 측정

OpenAI source 비율 측정

LangGraph Route 안정성 측정

Trace 상태 안정성 측정

Fallback 발생률 측정

JSON Artifact 저장

FastAPI Trace 기반 Route 추출

core 9회 실행 성공

전체 pytest 168개 통과

Day 18 보고서 작성
```

최종 결과:

```text
core Benchmark:

9 / 9 성공


전체 성공률:

100%


intent 일치율:

100%


intent_source=openai 비율:

100%


route 일치율:

100%


trace_status 일치율:

100%


fallback 발생률:

0%


fallback 기대값 일치율:

100%


pytest:

168 passed
```

Day 18 완료.
