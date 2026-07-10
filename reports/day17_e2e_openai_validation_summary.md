# Day 17 — E2E 검증 및 실제 OpenAI 경로 검증

## 1. Day 17 목표

Day 17의 목표는 기존 unit test와 monkeypatch 기반 integration test를 넘어 실제 외부 환경을 포함한 Agent 전체 실행 흐름을 검증하는 것이었다.

검증 대상은 다음과 같다.

```text
사용자 question

+

선택적 chat_history

+

선택적 raw_sample

↓

실제 OpenAI gpt-4o-mini

↓

intent JSON

↓

JSON parsing 및 validation

↓

LangGraph routing

↓

dataset schema answer

또는

실제 PyTorch failure prediction

또는

안전한 fallback answer

↓

Trace / Observability

↓

FastAPI response

↓

최종 JSON 검증
```

Day 17에서는 새로운 business 기능을 추가하기보다 Day 1~16에서 구현한 기능이 실제 외부 OpenAI 연결을 포함한 전체 경로에서도 정상적으로 연결되는지 검증했다.

---

# 2. Unit Test, Integration Test, E2E Test 구분

## 2.1 Unit Test

Unit test는 함수나 클래스처럼 작은 코드 단위를 독립적으로 검증한다.

예:

```text
validate_intent_payload()

calculate_risk_level()

route_after_classification()

append_trace_event()
```

특징:

```text
실행 속도가 빠르다.

외부 네트워크가 필요하지 않다.

OpenAI API 비용이 발생하지 않는다.

실패 원인을 작은 단위에서 찾기 쉽다.

같은 입력에 대해 안정적으로 재현할 수 있다.
```

현재 프로젝트의 기본 pytest에는 개별 Agent node, intent classifier, prediction, evidence, trace, FastAPI schema 등에 대한 unit test가 포함되어 있다.

---

## 2.2 Integration Test

Integration test는 여러 내부 컴포넌트가 올바르게 연결되는지 검증한다.

예:

```text
FastAPI

+

LangGraph
```

```text
LangGraph

+

failure prediction service
```

현재 프로젝트의 기존 테스트에서는 OpenAI 호출 결과를 monkeypatch하여 외부 네트워크에 의존하지 않으면서 내부 연결을 검증한다.

예:

```text
고정된 intent 결과

↓

LangGraph route

↓

prediction service

↓

AgentState

↓

FastAPI response
```

---

## 2.3 E2E Test

E2E는 End-to-End의 약자이다.

실제 사용자 입력부터 최종 출력까지 전체 경로를 검증한다.

Day 17 E2E:

```text
실제 question

↓

실제 OpenAI API

↓

실제 intent validation

↓

실제 LangGraph

↓

실제 PyTorch model

↓

실제 answer builder

↓

실제 Trace

↓

실제 FastAPI response
```

Day 17에서는 mock이나 monkeypatch를 사용하지 않고 실제 OpenAI `gpt-4o-mini`를 호출했다.

---

# 3. Day 16 Demo와 Day 17 E2E의 차이

## Day 16

Day 16의 주목적은 LangGraph 내부 실행 흐름을 관찰하는 것이었다.

주요 확인 항목:

```text
어떤 node가 실행되었는가?

어떤 route가 선택되었는가?

실행 순서는 무엇인가?

각 단계는 얼마나 걸렸는가?

fallback이 발생했는가?
```

Day 16의 핵심:

```text
실행 흐름 관찰
```

---

## Day 17

Day 17의 주목적은 실제 실행 결과가 사전에 정의한 기대 조건을 만족하는지 자동으로 판정하는 것이었다.

예:

```text
[PASS] intent == failure_prediction

[PASS] intent_source == openai

[PASS] prediction is 0 or 1

[PASS] probability is between 0.0 and 1.0

[PASS] trace_status == success

[PASS] trace event order matches prediction route
```

Day 17의 핵심:

```text
실제 전체 경로 실행

+

기대 조건 자동 검증

+

PASS / FAIL 판정

+

process exit code 반환
```

---

# 4. 실제 OpenAI E2E를 기본 pytest와 분리한 이유

실제 OpenAI 호출은 다음 외부 조건에 의존한다.

```text
네트워크 연결

OPENAI_API_KEY

API 사용 비용

OpenAI 서비스 상태

외부 응답 시간

LLM 응답의 일부 비결정성
```

따라서 실제 OpenAI E2E를 기본 pytest에 항상 포함하면 CI와 로컬 회귀 테스트가 불안정해질 수 있다.

예:

```text
프로젝트 코드는 정상

하지만 네트워크 오류

↓

pytest 실패
```

또는:

```text
GitHub Actions에 API key 없음

↓

pytest 실패
```

따라서 다음 구조를 사용했다.

## 기본 회귀 테스트

```powershell
pytest -v
```

특징:

```text
monkeypatch 기반

외부 OpenAI 호출 없음

빠른 실행

안정적인 반복 검증

CI에 적합
```

---

## 실제 OpenAI E2E

```powershell
python -m scripts.run_day17_e2e_openai_validation `
    --scenario all
```

특징:

```text
사용자가 명시적으로 실행

실제 OpenAI API 사용

실제 네트워크 사용

실제 LangGraph 실행

실제 PyTorch 모델 실행

실제 FastAPI 응답 검증
```

---

# 5. 추가 파일

Day 17에서 다음 파일을 추가했다.

```text
scripts/run_day17_e2e_openai_validation.py
```

역할:

```text
OpenAI 환경 검사

↓

실제 E2E 시나리오 실행

↓

결과 출력

↓

기대 조건 검증

↓

PASS / FAIL 판정

↓

최종 요약

↓

process exit code 반환
```

---

# 6. 기존 production code 수정 여부

Day 17에서는 기존 Agent business logic을 새로 구현하지 않았다.

기존 경로를 그대로 재사용했다.

주요 재사용 함수:

```python
run_failure_agent_graph(
    question=...,
    raw_sample=...,
    include_shap=...,
    include_global_importance=...,
    chat_history=...,
)
```

이 함수 내부에서 다음 흐름이 실행된다.

```text
AgentState 생성

↓

LangGraph compile

↓

graph.invoke()

↓

OpenAI intent classification

↓

LangGraph route

↓

prediction 또는 schema 또는 fallback

↓

Trace 종료

↓

최종 AgentState 반환
```

Day 17 스크립트에서 OpenAI SDK를 직접 다시 호출하지 않았다.

이유:

```text
OpenAI만 독립적으로 호출하면

기존 Agent 전체 경로를 검증할 수 없기 때문
```

---

# 7. E2E Scenario 목록

Day 17에서는 총 6개 시나리오를 구현했다.

---

## Scenario 1 — Dataset Schema 실제 OpenAI

질문:

```text
AI4I 데이터셋의 feature와 target은 뭐야?
```

기대:

```text
intent:

dataset_schema_query


intent_source:

openai


selected_route:

dataset_schema


prediction:

수행하지 않음


trace_status:

success


fallback_occurred:

false
```

실제 결과:

```text
intent:

dataset_schema_query


intent_source:

openai


confidence:

0.95


selected_route:

dataset_schema


trace_status:

success


fallback_occurred:

false


error_count:

0
```

실제 Trace:

```text
1. validate_question

2. route_after_validation

3. classify_intent

4. route_after_classification

5. build_dataset_schema_answer
```

결과:

```text
SUCCESS
```

---

## Scenario 2 — 실제 OpenAI + 실제 PyTorch Failure Prediction

질문:

```text
이 설비 조건이면 고장 위험이 높아?
```

입력:

```json
{
  "air_temperature": 303.0,
  "process_temperature": 312.5,
  "rotational_speed": 1380.0,
  "torque": 62.0,
  "tool_wear": 220.0,
  "type": "L"
}
```

실제 실행 경로:

```text
question

+

raw_sample

↓

실제 OpenAI

↓

failure_prediction

↓

LangGraph

↓

실제 PyTorch MLP

↓

prediction

↓

risk level

↓

answer

↓

Trace
```

실제 결과:

```text
intent:

failure_prediction


intent_source:

openai


confidence:

0.95


prediction:

1


probability:

0.9929707646369934


threshold:

0.7


risk_level:

HIGH


recommended_action:

고장 위험이 높습니다.
설비 점검 및 생산 조건 확인을 권장합니다.


selected_route:

final


trace_status:

success


fallback_occurred:

false


error_count:

0
```

실제 Trace:

```text
1. validate_question

2. route_after_validation

3. classify_intent

4. route_after_classification

5. call_failure_prediction

6. route_after_prediction

7. build_final_answer
```

검증:

```text
prediction은 0 또는 1

probability는 0.0~1.0

threshold는 0.0~1.0

prediction과 probability·threshold 관계 일치

risk_level은 LOW·MEDIUM·HIGH 중 하나

answer 존재

Trace 순서 일치
```

결과:

```text
SUCCESS
```

---

## Scenario 3 — 실제 Multi-turn Context

이전 대화:

```text
user:

AI4I 데이터셋의 feature는 뭐야?


assistant:

현재 모델은 AI4I feature 6개를 사용합니다.
```

현재 질문:

```text
그중 target은 뭐야?
```

실제 결과:

```text
intent:

dataset_schema_query


intent_source:

openai


confidence:

0.95


selected_route:

dataset_schema


prediction:

null


trace_status:

success


fallback_occurred:

false
```

실제 `intent_reason`:

```text
현재 질문은 AI4I 데이터셋의 target 컬럼에 대한 설명을 요청하고 있으며,
이전 대화에서 데이터셋의 feature에 대한 질문이 있었기 때문에 관련성이 높습니다.
```

이 결과를 통해 OpenAI가 현재 질문뿐 아니라 이전 대화 문맥도 참고했다는 것을 확인했다.

검증:

```text
현재 question 유지

chat_history list 유지

chat_history 메시지 수 2개

chat_history 내용 유지

dataset_schema_query

intent_source=openai

prediction 미수행

Trace 정상
```

결과:

```text
SUCCESS
```

---

## Scenario 4 — raw_sample 누락 Fallback

이전 대화:

```text
user:

공기 온도 303.0K,
공정 온도 312.5K,
회전 속도 1380rpm,
토크 62.0Nm,
공구 마모 220분,
Type L 조건이면 고장 위험이 높아?


assistant:

해당 설비 조건으로 고장 위험 예측을 요청하셨습니다.
```

현재 질문:

```text
그 조건으로 고장 위험을 다시 예측해줘.
```

현재 `raw_sample`:

```text
없음
```

실제 OpenAI 결과:

```text
intent:

failure_prediction


intent_source:

openai
```

하지만 현재 요청에 `raw_sample`이 없으므로 이전 대화의 자연어 설비 조건을 실제 PyTorch 입력으로 자동 재사용하지 않았다.

실제 결과:

```text
prediction:

null


probability:

null


risk_level:

UNKNOWN


selected_route:

fallback


trace_status:

fallback


fallback_occurred:

true


error_count:

1
```

권장 조치:

```text
현재 요청에는 raw_sample이 없어 고장 예측을 수행할 수 없습니다.

이전 대화의 설비 조건이나 raw_sample은 자동으로 재사용하지 않으므로,
현재 예측에 사용할 설비 입력값을 요청에 함께 보내주세요.
```

실제 Trace:

```text
1. validate_question

2. route_after_validation

3. classify_intent

4. route_after_classification

5. call_failure_prediction

6. route_after_prediction

7. build_fallback_answer
```

결과:

```text
SUCCESS
```

---

## Scenario 5 — 지원하지 않는 질문

질문:

```text
오늘 점심 메뉴 추천해줘.
```

실제 결과:

```text
intent:

unknown


intent_source:

openai


confidence:

0.9


prediction:

null


probability:

null


selected_route:

fallback


trace_status:

fallback


fallback_occurred:

true


error_count:

0
```

지원하지 않는 질문은 시스템 장애가 아니므로 오류를 강제로 추가하지 않고 안전한 fallback answer를 반환했다.

실제 Trace:

```text
1. validate_question

2. route_after_validation

3. classify_intent

4. route_after_classification

5. build_fallback_answer
```

결과:

```text
SUCCESS
```

---

## Scenario 6 — FastAPI 실제 OpenAI + PyTorch E2E

HTTP 요청:

```text
POST /agent/langgraph-query
```

Request:

```json
{
  "question": "이 설비 조건이면 고장 위험이 높아?",
  "raw_sample": {
    "air_temperature": 303.0,
    "process_temperature": 312.5,
    "rotational_speed": 1380.0,
    "torque": 62.0,
    "tool_wear": 220.0,
    "type": "L"
  },
  "include_shap": false,
  "include_global_importance": false,
  "chat_history": []
}
```

실제 흐름:

```text
FastAPI TestClient

↓

POST /agent/langgraph-query

↓

LangGraphAgentQueryRequest

↓

Pydantic request validation

↓

실제 OpenAI

↓

LangGraph

↓

실제 PyTorch MLP

↓

AgentState

↓

LangGraphAgentQueryResponse

↓

Pydantic response validation

↓

HTTP JSON response
```

실제 결과:

```text
status_code:

200


intent:

failure_prediction


intent_source:

openai


prediction:

1


probability:

0.9929707646369934


risk_level:

HIGH


trace_status:

success


fallback_occurred:

false
```

검증:

```text
HTTP status code = 200

response JSON object

intent_source=openai

prediction 정상

probability 정상

threshold 정상

prediction과 threshold 관계 일치

risk_level 정상

answer 존재

trace_id 존재

trace_events list

각 trace event는 JSON object

TraceEvent 순서 정상

TraceEvent sequence 정상
```

결과:

```text
SUCCESS
```

---

# 8. 최종 전체 E2E 실행

실행 명령:

```powershell
cd C:\Users\kflow\Downloads\manufacturing-ai-quality-agent-reference

.\.venv\Scripts\Activate.ps1

python -m scripts.run_day17_e2e_openai_validation `
    --scenario all
```

최종 결과:

```text
completed scenarios : 6

passed scenarios    : 6

failed scenarios    : 0

result              : SUCCESS
```

Process exit code:

```powershell
$LASTEXITCODE
```

결과:

```text
0
```

---

# 9. 최종 실행 시간 기록

최종 `--scenario all` 실행에서 기록한 개별 Trace 시간:

| Scenario           | trace_duration_ms |
| ------------------ | ----------------: |
| Dataset schema     |       6547.264 ms |
| Failure prediction |       2486.380 ms |
| Multi-turn         |       2650.174 ms |
| Missing raw_sample |       3545.403 ms |
| Unknown            |       2107.768 ms |

FastAPI scenario도 정상 수행되었다.

주의:

```text
이 값은 단 한 번의 실제 실행 결과이다.

네트워크 상태

OpenAI 응답 시간

로컬 시스템 상태

모델 artifact 로딩 상태

등의 영향을 받을 수 있다.
```

따라서 일반적인 성능 벤치마크나 SLA로 단정하지 않는다.

현재 측정값은 실제 연결이 동작했음을 확인하고 병목 후보를 관찰하기 위한 실행 기록이다.

---

# 10. 기본 pytest 최종 회귀 테스트

실행:

```powershell
pytest -v
```

결과:

```text
168 passed in 15.35s
```

확인 내용:

```text
Day 1~16 기존 기능 회귀 오류 없음

Day 16 Trace 기능 정상

Day 17 E2E 스크립트 추가 후 기존 테스트 정상

실제 OpenAI E2E는 기본 pytest에 자동 포함되지 않음
```

---

# 11. 최종 검증 구조

```text
기본 pytest

→ unit test

→ integration test

→ monkeypatch

→ 빠르고 안정적인 회귀 테스트

→ CI에 적합
```

```text
Day 17 E2E script

→ 실제 OpenAI

→ 실제 네트워크

→ 실제 LangGraph

→ 실제 PyTorch

→ 실제 Trace

→ 실제 FastAPI response

→ 명시적으로 실행
```

---

# 12. Day 17에서 검증한 핵심 정책

## OpenAI 역할

```text
OpenAI:

질문의 intent 분류


OpenAI가 하지 않는 일:

고장 probability 계산

prediction 생성

risk level 계산
```

실제 prediction은 기존 PyTorch MLP가 담당한다.

---

## Multi-turn 정책

```text
chat_history:

현재 질문의 문맥 이해용


raw_sample:

실제 PyTorch prediction 입력용
```

이전 대화에 설비 조건이 있어도 현재 `raw_sample`로 자동 재사용하지 않는다.

---

## Fallback 구분

```text
intent_source = fallback

의미:

OpenAI intent 분류 실패

→ rule-based classifier 사용
```

```text
fallback_occurred = true

의미:

LangGraph의 실제 fallback route

또는

build_fallback_answer node 실행
```

두 개념은 서로 다르다.

---

## Trace 개인정보 보호

Trace에는 다음 원문 전체를 기본 저장하지 않는다.

```text
전체 question

전체 chat_history

전체 raw_sample

OpenAI API key

OpenAI raw response 전체
```

대신 다음 요약값을 기록한다.

```text
question_length

raw_sample_provided

intent

intent_source

confidence

prediction_succeeded

risk_level

selected_route
```

---

# 13. 구현 중 발견한 오류와 해결

Scenario 2 구현 후 다음 오류가 발생했다.

```text
NameError:

name 'validate_prediction_state' is not defined
```

원인:

```text
run_prediction_scenario()에서
validate_prediction_state()를 호출했지만

해당 함수가 module 전역에 정의되지 않았거나
올바른 위치에 추가되지 않음
```

해결:

```text
validate_prediction_state()를

run_prediction_scenario()보다 먼저

module 전역에 정의
```

검증:

```powershell
python -m py_compile `
    scripts\run_day17_e2e_openai_validation.py
```

```powershell
python -c "from scripts.run_day17_e2e_openai_validation import validate_prediction_state; print(validate_prediction_state.__name__)"
```

그 후 Scenario 2를 다시 실행하여 정상 통과했다.

---

# 14. Day 17 최종 결과

```text
실제 OpenAI E2E:

6 / 6 SUCCESS


Process exit code:

0


기본 pytest:

168 passed


회귀 오류:

없음
```

---

# 15. Day 17 완료 체크리스트

* [x] 실제 OpenAI API key 환경 검사

* [x] API key 값을 콘솔에 출력하지 않음

* [x] 실제 `gpt-4o-mini` intent classification

* [x] 실제 OpenAI intent JSON validation

* [x] `intent_source=openai` 검증

* [x] confidence 범위 검증

* [x] intent reason 존재 검증

* [x] Dataset schema 실제 OpenAI 경로 검증

* [x] 실제 OpenAI + PyTorch prediction 검증

* [x] prediction 0·1 범위 검증

* [x] probability 범위 검증

* [x] threshold 범위 검증

* [x] prediction·probability·threshold 일관성 검증

* [x] risk level 검증

* [x] recommended action 검증

* [x] Agent answer 검증

* [x] 실제 multi-turn context 검증

* [x] chat history 전달 검증

* [x] 현재 question 유지 검증

* [x] 이전 raw sample 자동 재사용 방지 검증

* [x] raw sample 누락 fallback 검증

* [x] unknown 질문 fallback 검증

* [x] selected route 검증

* [x] trace status 검증

* [x] fallback occurred 검증

* [x] trace ID 검증

* [x] trace duration 검증

* [x] trace event 순서 검증

* [x] trace sequence 연속성 검증

* [x] FastAPI HTTP 200 검증

* [x] FastAPI request validation 경로 검증

* [x] Pydantic response validation 경로 검증

* [x] FastAPI JSON response 검증

* [x] TraceEvent API JSON 구조 검증

* [x] 실제 OpenAI E2E와 기본 pytest 분리

* [x] 전체 E2E 6개 성공

* [x] process exit code 0

* [x] 프로젝트 전체 pytest 168개 통과

---

# 16. 면접 답변

## Q. 실제 OpenAI 연결도 검증했나요?

네.

기존 unit test와 integration test에서는 OpenAI 결과를 monkeypatch하여 빠르고 안정적인 회귀 테스트를 유지했습니다.

별도로 실제 OpenAI E2E 검증 스크립트를 구현하여 `gpt-4o-mini` intent 분류부터 JSON validation, LangGraph routing, PyTorch 고장 예측, Trace, FastAPI 응답까지 전체 경로를 검증했습니다.

Dataset schema, failure prediction, multi-turn, raw sample 누락, unknown 질문, FastAPI prediction 등 총 6개 시나리오를 실행했고 모두 성공했습니다.

---

## Q. 실제 OpenAI 테스트를 왜 기본 pytest에 포함하지 않았나요?

실제 OpenAI 호출은 네트워크, API key, 비용, 외부 서비스 상태, 응답 시간에 의존합니다.

이를 기본 pytest에 항상 포함하면 프로젝트 코드가 정상이어도 외부 문제로 CI가 실패할 수 있습니다.

따라서 기본 pytest는 monkeypatch 기반으로 빠르고 안정적으로 유지하고, 실제 OpenAI E2E는 사용자가 명시적인 명령으로만 실행하도록 분리했습니다.

---

## Q. OpenAI가 고장 예측도 하나요?

아닙니다.

OpenAI는 현재 질문과 chat history를 참고하여 intent JSON만 반환합니다.

실제 고장 probability와 prediction은 기존 PyTorch MLP prediction service가 계산합니다.

즉 역할을 다음과 같이 분리했습니다.

```text
OpenAI:

자연어 intent 분류


LangGraph:

실행 경로 제어


PyTorch:

고장 확률과 prediction 계산


Evidence·Answer:

모델 결과 기반 설명


Trace:

실행 흐름 기록
```

---

## Q. Multi-turn에서 이전 설비 조건도 자동으로 재사용하나요?

아닙니다.

`chat_history`는 현재 질문의 문맥을 이해하는 용도로만 사용합니다.

실제 PyTorch prediction은 현재 요청의 구조화된 `raw_sample`만 사용합니다.

이전 대화에 설비 조건이 있어도 자동으로 재사용하지 않습니다.

이는 이전 조건을 잘못 추출하거나 오래된 조건으로 예측하는 위험을 줄이기 위한 안전 정책입니다.

실제 E2E에서도 이전 대화에는 설비 조건이 있지만 현재 `raw_sample`이 없는 상황을 검증했고, prediction을 수행하지 않고 안전한 fallback answer를 반환했습니다.

---

## Q. 실제 실행 결과는 어떻게 검증했나요?

최종 intent만 확인하지 않았다.

다음 항목을 함께 검증했다.

```text
intent

intent source

confidence

intent reason

prediction

probability

threshold

prediction과 threshold 관계

risk level

recommended action

answer

selected route

trace status

fallback 여부

trace ID

trace duration

TraceEvent 순서

TraceEvent sequence

HTTP status code

FastAPI JSON response
```

또한 성공 여부를 process exit code로 반환하도록 구현했다.

```text
0:

모든 검증 성공


1:

E2E 조건 실패


2:

OpenAI 환경 설정 실패
```

---

# 17. Day 17 최종 한 줄 요약

> 실제 OpenAI `gpt-4o-mini` intent 분류부터 LangGraph routing, PyTorch 고장 예측, multi-turn 문맥, fallback, Trace, FastAPI JSON 응답까지 6개 E2E 시나리오를 검증했으며, 전체 시나리오 성공과 기존 pytest 168개 통과를 확인했다.
