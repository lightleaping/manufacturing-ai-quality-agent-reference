# Day 14 - LangGraph Agent FastAPI Endpoint 연결

## 1. Day 14 목표

Day 14의 목표는 Day 13에서 구현한 LangGraph Agent workflow를 FastAPI endpoint와 연결하는 것이다.

Day 10~12에서는 정형화된 설비 입력값을 직접 받아 고장 예측을 수행하는 FastAPI endpoint를 구현했다.

기존 endpoint:

```text
POST /agent/failure-prediction
```

Day 13에서는 자연어 질문을 받아 intent를 분류하고, LangGraph `AgentState` 기반 workflow를 통해 적절한 node로 이동하는 구조를 구현했다.

Day 14에서는 두 구조를 연결하여 사용자가 자연어 질문과 선택적 설비 입력값을 API로 전달하면 다음 흐름이 실행되도록 구성했다.

```text
자연어 질문
→ FastAPI endpoint
→ LangGraph workflow
→ intent 분류
→ intent별 node 실행
→ 필요한 경우 prediction service 호출
→ prediction/evidence/answer 반환
```

새 endpoint:

```text
POST /agent/langgraph-query
```

---

# 2. Day 14에서 새로 만든 파일

```text
src/api/langgraph_agent_api.py

tests/test_api_langgraph_agent.py

reports/day14_langgraph_api_endpoint_summary.md
```

---

# 3. Day 14에서 수정한 파일

```text
src/api/schemas.py

src/api/main.py

src/agent/state.py

src/agent/failure_agent_graph.py

tests/test_failure_agent_graph.py
```

---

# 4. 기존 endpoint 유지

Day 14에서는 기존 prediction endpoint를 삭제하거나 대체하지 않았다.

기존 endpoint:

```text
POST /agent/failure-prediction
```

기존 endpoint는 사용자가 정형화된 설비 입력값을 직접 전달하면 prediction service를 호출한다.

처리 흐름:

```text
설비 입력값
→ FastAPI endpoint
→ Day 12 prediction service
→ prediction
→ evidence
→ answer
```

새 endpoint:

```text
POST /agent/langgraph-query
```

새 endpoint는 자연어 질문을 먼저 받고 LangGraph workflow를 실행한다.

처리 흐름:

```text
자연어 question
→ FastAPI endpoint
→ LangGraph workflow
→ intent 분류
→ intent별 처리
→ 필요한 경우 prediction service 호출
→ answer/evidence 반환
```

두 endpoint는 목적과 입력 방식이 다르므로 기존 endpoint를 유지하면서 자연어 Agent용 endpoint를 별도로 추가했다.

---

# 5. 기존 endpoint와 새 endpoint 비교

## 기존 `/agent/failure-prediction`

입력 중심:

```text
정형화된 설비 데이터
```

예:

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

처리 흐름:

```text
raw sample
→ FastAPI
→ prediction service
→ 모델 예측
→ evidence
→ answer
```

특징:

```text
사용자가 이미 수행할 기능을 알고 있음

고장 예측 전용 endpoint

intent 분류 없음

LangGraph routing 없음
```

---

## 새 `/agent/langgraph-query`

입력 중심:

```text
자연어 question
+
선택적 raw_sample
```

예:

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
  "include_shap": true,
  "include_global_importance": true
}
```

처리 흐름:

```text
자연어 question
→ FastAPI
→ LangGraph
→ intent 분류
→ conditional routing
→ 필요한 service 호출
→ answer/evidence 반환
```

특징:

```text
자연어 질문 중심

LLM intent classifier 사용

OpenAI 실패 시 rule-based fallback

LangGraph AgentState 사용

intent별 node 분기

prediction이 필요할 때만 모델 service 호출
```

---

# 6. LangGraph Agent API request schema

새 endpoint는 다음 형태의 request를 받는다.

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
  "include_shap": true,
  "include_global_importance": true
}
```

각 필드의 역할은 다음과 같다.

---

## 6.1 `question`

사용자의 자연어 질문이다.

LangGraph workflow는 `question`을 검증한 뒤 intent classifier에 전달한다.

예:

```text
AI4I 데이터셋 feature와 target은 뭐야?
```

```text
이 설비 조건이면 고장 위험이 높아?
```

```text
오늘 점심 메뉴 추천해줘.
```

질문의 의미에 따라 다음 intent 중 하나로 분류된다.

```python
SUPPORTED_INTENTS = {
    "failure_prediction",
    "dataset_schema_query",
    "unknown",
}
```

---

## 6.2 `raw_sample`

모델 예측에 사용할 설비 입력값이다.

다음 질문에서는 필요하다.

```text
이 설비 조건이면 고장 위험이 높아?
```

모델이 고장 probability를 계산하려면 실제 설비 feature 값이 필요하기 때문이다.

하지만 다음 질문에서는 필요하지 않다.

```text
AI4I 데이터셋 feature와 target은 뭐야?
```

dataset schema 질문은 모델 예측을 수행하지 않기 때문이다.

따라서 `raw_sample`은 필수 입력이 아니라 optional 입력으로 설계했다.

```python
raw_sample: LangGraphRawSampleRequest | None = None
```

---

## 6.3 `include_shap`

SHAP local explanation을 계산할지 결정한다.

```json
{
  "include_shap": true
}
```

SHAP local evidence를 생성한다.

```json
{
  "include_shap": false
}
```

SHAP 계산을 생략한다.

SHAP 계산은 모델 prediction보다 상대적으로 비용이 큰 설명 단계이므로 사용자가 필요 여부를 선택할 수 있도록 구성했다.

---

## 6.4 `include_global_importance`

permutation importance 기반 global importance evidence를 포함할지 결정한다.

```json
{
  "include_global_importance": true
}
```

global importance artifact를 읽어 evidence에 포함한다.

```json
{
  "include_global_importance": false
}
```

global importance evidence 생성을 생략한다.

---

# 7. LangGraph Agent API response schema

응답에는 다음과 같은 값이 포함된다.

```json
{
  "question": "이 설비 조건이면 고장 위험이 높아?",
  "intent": "failure_prediction",
  "confidence": 0.95,
  "intent_source": "openai",
  "intent_reason": "설비 고장 위험 예측을 요청한 질문입니다.",
  "prediction": 1,
  "probability": 0.9929,
  "threshold": 0.7,
  "risk_level": "HIGH",
  "recommended_action": "고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다.",
  "answer": "현재 입력 조건에서는 고장 위험이 높게 예측되었습니다.",
  "evidence": [],
  "warnings": [],
  "errors": [],
  "limitations": []
}
```

주요 필드:

```text
question
→ 사용자가 입력한 자연어 질문

intent
→ 분류된 질문 의도

confidence
→ intent classifier의 신뢰도

intent_source
→ openai 또는 rule_based_fallback

intent_reason
→ 해당 intent로 분류한 이유

prediction
→ threshold 비교 후 최종 예측값

probability
→ 모델이 계산한 고장 확률

threshold
→ 운영 예측 기준값

risk_level
→ LOW / MEDIUM / HIGH / UNKNOWN

recommended_action
→ 예측 결과에 따른 권장 조치

answer
→ 사용자에게 반환할 최종 설명

evidence
→ prediction, rule, SHAP, importance 근거

warnings
→ 핵심 기능은 가능하지만 일부 부가 기능에 문제가 있는 경우

errors
→ 핵심 요청을 정상 수행할 수 없는 경우

limitations
→ 결과 해석 시 주의할 한계
```

---

# 8. prediction 관련 필드를 optional로 만든 이유

모든 intent가 prediction을 수행하는 것은 아니다.

예:

```text
dataset_schema_query
```

이 intent는 AI4I 데이터셋의 feature, target, 제외 컬럼 등을 설명한다.

따라서 다음 값은 생성되지 않는다.

```text
prediction

probability

threshold

risk_level

recommended_action
```

또한:

```text
unknown
```

intent도 지원하지 않는 질문에 대한 fallback answer만 반환하므로 모델 prediction을 수행하지 않는다.

따라서 prediction 관련 response 필드를 optional로 설계했다.

prediction을 수행하지 않은 경우 JSON 응답에서는 `null`로 반환될 수 있다.

---

# 9. 전체 API 처리 흐름

```text
사용자
  │
  │ POST /agent/langgraph-query
  ▼
FastAPI
src/api/langgraph_agent_api.py
  │
  │ request schema 검증
  ▼
run_failure_agent_graph()
  │
  ▼
초기 AgentState 생성
  │
  ▼
LangGraph workflow
  │
  ├─ validate_question_node
  │
  ├─ classify_intent_node
  │
  └─ intent별 conditional routing
       │
       ├─ failure_prediction
       │      │
       │      ├─ raw_sample 있음
       │      │      │
       │      │      ▼
       │      │   Day 12 prediction service
       │      │      │
       │      │      ▼
       │      │   prediction
       │      │   evidence
       │      │   answer
       │      │
       │      └─ raw_sample 없음
       │             │
       │             ▼
       │          error 기록
       │             │
       │             ▼
       │          fallback answer
       │
       ├─ dataset_schema_query
       │      │
       │      ▼
       │   dataset schema answer
       │
       └─ unknown
              │
              ▼
           fallback answer
  │
  ▼
최종 AgentState
  │
  ▼
API response schema 변환
  │
  ▼
JSON response
```

---

# 10. API endpoint를 별도 파일로 분리한 이유

새 endpoint는 다음 파일에 구현했다.

```text
src/api/langgraph_agent_api.py
```

기존 endpoint 파일:

```text
src/api/failure_agent_api.py
```

역할:

```text
정형화된 설비 입력값
→ prediction service
```

새 endpoint 파일:

```text
src/api/langgraph_agent_api.py
```

역할:

```text
자연어 question
→ LangGraph Agent workflow
```

두 API는 입력 방식과 처리 흐름이 다르다.

한 파일에 모든 endpoint와 변환 로직을 작성하면 파일의 책임이 커지고 코드 흐름을 이해하기 어려워질 수 있다.

따라서 별도의 `APIRouter` 파일로 분리했다.

---

# 11. API endpoint를 얇게 유지한 이유

새 endpoint는 intent를 직접 분류하지 않는다.

또한 다음 작업을 직접 수행하지 않는다.

```text
OpenAI API 호출

모델 artifact 로드

feature scaling

PyTorch inference

SHAP 계산

global importance 로드

evidence 생성

answer 생성
```

endpoint의 역할:

```text
HTTP request 받기

request schema 검증

LangGraph runner 호출

최종 state를 response schema로 변환

JSON response 반환
```

실제 역할은 각 계층에 분리되어 있다.

```text
FastAPI endpoint
→ HTTP request/response

LangGraph
→ workflow와 상태 관리

intent classifier
→ 자연어 질문 intent 분류

prediction service
→ 모델 artifact와 prediction 처리

evidence builder
→ 근거 통합

answer builder
→ 사용자 답변 생성
```

이 구조는 endpoint의 책임을 줄이고 기존 service를 재사용할 수 있게 한다.

---

# 12. LangGraph runner 인터페이스 확장

Day 13에서는 LangGraph workflow 자체를 구현했다.

Day 14에서는 FastAPI endpoint에서 쉽게 호출할 수 있도록 runner의 공개 인터페이스를 확장했다.

```python
def run_failure_agent_graph(
    question: str,
    raw_sample: dict[str, Any] | None = None,
    include_shap: bool = True,
    include_global_importance: bool = True,
) -> AgentState:
```

각 값의 역할:

```text
question
→ intent 분류에 사용할 자연어 질문

raw_sample
→ failure prediction 모델 입력값

include_shap
→ SHAP local evidence 포함 여부

include_global_importance
→ global importance evidence 포함 여부
```

FastAPI endpoint는 AgentState 내부 구조를 직접 만들지 않고 필요한 입력값만 runner에 전달한다.

---

# 13. AgentState 초기화

runner 내부에서 초기 AgentState를 생성한다.

```python
initial_state = create_initial_agent_state(
    question=question,
    raw_sample=raw_sample,
)
```

이후 FastAPI request에서 전달된 옵션을 state에 저장한다.

```python
initial_state["include_shap"] = include_shap

initial_state["include_global_importance"] = (
    include_global_importance
)
```

생성된 AgentState는 LangGraph node 사이를 이동한다.

```text
초기 state

question
raw_sample
include_shap
include_global_importance
warnings
errors
limitations
```

workflow가 실행되면서 다음 값이 추가된다.

```text
intent

confidence

intent_reason

intent_source

prediction

probability

threshold

risk_level

recommended_action

evidence

answer
```

---

# 14. raw_sample이 없을 때 처리

failure prediction 질문이라도 raw sample이 없으면 prediction을 수행하지 않는다.

잘못된 처리:

```text
자연어 질문만 입력

→ LLM이 고장 probability 추측

→ 임의 위험도 반환
```

현재 프로젝트의 올바른 처리:

```text
failure_prediction intent

→ raw_sample 확인

→ raw_sample 없음

→ prediction service 호출하지 않음

→ errors 기록

→ fallback answer 반환
```

오류 메시지:

```text
failure_prediction intent이지만 raw_sample이 없어 prediction을 수행할 수 없습니다.
```

고장 probability는 PyTorch 모델이 실제 설비 feature를 사용해 계산해야 한다.

LLM은 고장 probability를 직접 생성하거나 예측하지 않는다.

---

# 15. warning과 error 구분

## warning

핵심 prediction은 성공했지만 일부 부가 기능을 정상 수행하지 못한 경우다.

예:

```text
SHAP artifact 로드 실패

→ 모델 prediction 가능

→ warnings 기록

→ SHAP evidence만 생략
```

예:

```text
global importance artifact 로드 실패

→ 모델 prediction 가능

→ warnings 기록

→ global importance evidence만 생략
```

---

## error

핵심 요청을 수행할 수 없는 경우다.

예:

```text
raw_sample 없음

→ 모델 입력 없음

→ prediction 불가능

→ errors 기록
```

예:

```text
model artifact 로드 실패

→ PyTorch prediction 불가능

→ errors 기록
```

---

# 16. prediction service 예외 처리

prediction service 실행 중 예외가 발생해도 예외를 FastAPI 밖으로 그대로 전달하지 않도록 처리했다.

예외 처리가 없는 경우:

```text
prediction service 예외

→ RuntimeError

→ LangGraph 실행 중단

→ FastAPI 500
```

개선 구조:

```text
prediction service 예외

→ try/except

→ AgentState errors에 기록

→ conditional routing

→ fallback answer node 이동

→ 구조화된 Agent response 반환
```

예:

```python
try:
    prediction_result = _run_failure_prediction_service(
        raw_sample=raw_sample,
        include_shap=state.get(
            "include_shap",
            True,
        ),
        include_global_importance=state.get(
            "include_global_importance",
            True,
        ),
    )

except Exception as exc:
    state.setdefault(
        "errors",
        [],
    ).append(
        "failure prediction service 실행 중 "
        f"오류가 발생했습니다: {exc}"
    )

    return state
```

---

# 17. Day 14 구현 중 발생한 Swagger 500 오류

Swagger에서 실제 LangGraph Agent API를 실행했을 때 처음에는 다음 응답이 발생했다.

```text
500 Internal Server Error
```

Uvicorn traceback을 확인한 결과 실제 오류는 다음과 같았다.

```text
TypeError:

_run_failure_prediction_service()
got an unexpected keyword argument 'include_shap'
```

LangGraph task:

```text
call_failure_prediction
```

---

# 18. Swagger 500 오류 원인

호출부는 다음 값을 전달하고 있었다.

```python
prediction_result = _run_failure_prediction_service(
    raw_sample=raw_sample,
    include_shap=state.get(
        "include_shap",
        True,
    ),
    include_global_importance=state.get(
        "include_global_importance",
        True,
    ),
)
```

전달 값:

```text
raw_sample

include_shap

include_global_importance
```

하지만 기존 helper 함수는 `raw_sample`만 받을 수 있었다.

기존:

```python
def _run_failure_prediction_service(
    raw_sample,
):
```

따라서 Python은 함수 정의에 없는 `include_shap` keyword argument를 받아 `TypeError`를 발생시켰다.

---

# 19. helper 함수 인터페이스 수정

helper 함수가 Day 14 API 옵션을 받을 수 있도록 인터페이스를 확장했다.

수정:

```python
def _run_failure_prediction_service(
    raw_sample,
    include_shap: bool = True,
    include_global_importance: bool = True,
):
```

이후 값을 Day 12 prediction service request까지 전달하도록 수정했다.

전체 전달 흐름:

```text
Swagger request

include_shap
include_global_importance
        │
        ▼
FastAPI endpoint
        │
        ▼
run_failure_agent_graph()
        │
        ▼
AgentState
        │
        ▼
call_failure_prediction_node()
        │
        ▼
_run_failure_prediction_service()
        │
        ▼
Day 12 prediction service
```

---

# 20. 기존 Day 13 테스트 실패

production 함수 인터페이스를 확장한 뒤 기존 Day 13 테스트 일부가 실패했다.

원인은 테스트의 fake 함수가 기존 인터페이스를 유지하고 있었기 때문이다.

기존 fake 함수:

```python
def fake_run_failure_prediction_service(
    raw_sample,
):
```

production 호출:

```python
_run_failure_prediction_service(
    raw_sample=raw_sample,
    include_shap=True,
    include_global_importance=True,
)
```

결과:

```text
fake 함수가 include_shap을 받을 수 없음

→ TypeError
```

---

# 21. monkeypatch fake 함수 인터페이스 수정

monkeypatch 대상 함수와 fake 함수의 호출 인터페이스를 맞췄다.

수정:

```python
def fake_run_failure_prediction_service(
    raw_sample,
    include_shap=True,
    include_global_importance=True,
):
```

monkeypatch는 실제 함수 대신 fake 함수를 실행한다.

따라서 production 코드가 전달하는 argument를 fake 함수도 받을 수 있어야 한다.

---

# 22. 테스트 fake response 수정

prediction 성공 테스트에서는 다음 값을 검증했다.

```text
prediction

probability

threshold

risk_level

recommended_action

answer

evidence

warnings

limitations
```

따라서 fake service response에도 실제 테스트에서 검증하는 값을 포함했다.

예:

```python
return {
    "prediction": 1,
    "probability": 0.9929,
    "threshold": 0.7,
    "risk_level": "HIGH",
    "recommended_action": (
        "고장 위험이 높습니다. "
        "설비 점검을 권장합니다."
    ),
    "evidence": [
        {
            "evidence_id": (
                "prediction_summary_001"
            ),
            "evidence_type": (
                "prediction_summary"
            ),
            "source": "model_prediction",
            "title": "모델 예측 요약",
            "summary": (
                "모델은 고장 probability를 "
                "높게 예측했습니다."
            ),
            "severity": "HIGH",
        }
    ],
    "answer": "고장 위험이 높습니다.",
    "warnings": [
        "SHAP 계산은 테스트에서 생략되었습니다."
    ],
    "limitations": [
        "SHAP value는 실제 원인 단정이 아닙니다."
    ],
}
```

---

# 23. runner 테스트 인터페이스 수정

Day 13 테스트에서는 초기 AgentState를 직접 생성하고 runner에 전달했다.

기존:

```python
state = create_initial_agent_state(
    question=(
        "AI4I 데이터셋 feature와 "
        "target은 뭐야?"
    )
)

result = run_failure_agent_graph(
    state
)
```

하지만 Day 14 runner는 다음 입력을 직접 받도록 변경되었다.

```python
run_failure_agent_graph(
    question,
    raw_sample,
    include_shap,
    include_global_importance,
)
```

따라서 테스트도 실제 공개 인터페이스를 사용하도록 수정했다.

dataset schema:

```python
result = run_failure_agent_graph(
    question=(
        "AI4I 데이터셋 feature와 "
        "target은 뭐야?"
    ),
)
```

unknown:

```python
result = run_failure_agent_graph(
    question="오늘 점심 메뉴 추천해줘.",
)
```

failure prediction:

```python
result = run_failure_agent_graph(
    question=(
        "이 설비 조건이면 "
        "고장 위험이 높아?"
    ),
    raw_sample=raw_sample,
    include_shap=True,
    include_global_importance=True,
)
```

---

# 24. 기존 runner 호출에서 KeyError가 발생한 이유

기존 테스트:

```python
result = run_failure_agent_graph(
    state
)
```

새 runner:

```python
def run_failure_agent_graph(
    question: str,
    ...
):
```

Python은 첫 번째 positional argument인 `state` 전체를 `question` 값으로 해석했다.

의도한 값:

```python
question = "오늘 점심 메뉴 추천해줘."
```

실제 전달된 값:

```python
question = {
    "question": "오늘 점심 메뉴 추천해줘.",
    "warnings": [],
    "errors": [],
    "limitations": [],
}
```

즉, 문자열 질문이 들어가야 할 자리에 AgentState dict 전체가 들어갔다.

이 때문에 정상적인 intent 처리 결과가 만들어지지 않았고 다음 코드에서 오류가 발생했다.

```python
result["intent"]
```

오류:

```text
KeyError: 'intent'
```

테스트 호출을 새 runner 인터페이스에 맞게 수정한 뒤 해결했다.

---

# 25. 새 LangGraph API 테스트

테스트 파일:

```text
tests/test_api_langgraph_agent.py
```

테스트 항목:

```text
test_langgraph_query_api_returns_dataset_schema_answer

test_langgraph_query_api_returns_unknown_fallback

test_langgraph_query_api_returns_failure_prediction

test_langgraph_query_api_does_not_force_prediction_without_raw_sample

test_existing_failure_prediction_endpoint_is_still_registered
```

총 테스트:

```text
5개
```

---

# 26. dataset schema API 테스트

질문:

```text
AI4I 데이터셋 feature와 target은 뭐야?
```

확인 내용:

```text
HTTP response 성공

intent
→ dataset_schema_query

prediction
→ None

dataset schema answer 포함

errors
→ 빈 리스트
```

---

# 27. unknown API 테스트

질문:

```text
오늘 점심 메뉴 추천해줘.
```

확인 내용:

```text
intent
→ unknown

prediction 수행 안 함

fallback answer 반환

evidence
→ 빈 리스트

errors
→ 빈 리스트

limitations 포함
```

---

# 28. failure prediction API 테스트

질문:

```text
이 설비 조건이면 고장 위험이 높아?
```

설비 입력:

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

확인 내용:

```text
intent
→ failure_prediction

prediction

probability

threshold

risk_level

recommended_action

answer

evidence

errors
```

---

# 29. raw_sample 없는 failure prediction 테스트

질문:

```text
이 설비 고장 위험 예측해줘.
```

하지만 `raw_sample`은 전달하지 않았다.

확인 내용:

```text
failure_prediction intent

prediction
→ None

probability
→ None

risk_level
→ UNKNOWN

모델 호출 강행 안 함

설비 입력값 필요 안내
```

---

# 30. 기존 endpoint 유지 테스트

Day 14에서 새 endpoint를 추가하면서 기존 endpoint가 사라지지 않았는지 확인했다.

OpenAPI schema 확인:

```python
response = client.get(
    "/openapi.json"
)
```

검증:

```python
assert (
    "/agent/failure-prediction"
    in paths
)

assert (
    "/agent/langgraph-query"
    in paths
)
```

확인 endpoint:

```text
POST /agent/failure-prediction

POST /agent/langgraph-query
```

두 endpoint가 모두 유지되었다.

---

# 31. 실제 OpenAI API를 단위 테스트에서 호출하지 않은 이유

API 단위 테스트에서는 실제 OpenAI API를 호출하지 않았다.

이유:

```text
API key 필요

네트워크 필요

비용 발생 가능

응답 변동 가능

외부 서비스 장애 영향

테스트 속도 저하
```

API endpoint 테스트의 목적은 LLM 분류 성능 검증이 아니다.

확인 대상:

```text
request 전달

→ LangGraph runner 호출

→ state 반환

→ response schema 변환

→ HTTP response 반환
```

따라서 `run_failure_agent_graph()`를 monkeypatch했다.

```python
monkeypatch.setattr(
    (
        "src.api.langgraph_agent_api."
        "run_failure_agent_graph"
    ),
    fake_run_failure_agent_graph,
)
```

---

# 32. 실제 모델 artifact를 단위 테스트에서 사용하지 않은 이유

API endpoint 단위 테스트에서는 실제 모델 artifact도 로드하지 않았다.

모델 artifact를 사용하면 테스트가 다음 환경에 의존할 수 있다.

```text
model.pt 존재 여부

scaler.joblib 존재 여부

metadata.json 존재 여부

shap_background.pt 존재 여부

shap_reference_values.json 존재 여부

global_importance.json 존재 여부
```

Day 14 API 테스트의 목적은 모델 성능 검증이 아니라 API 연결 구조 검증이다.

모델, inference, evidence, service 기능은 이전 Day 테스트에서 별도로 검증했다.

---

# 33. Day 14 전체 핵심 테스트 실행

실행 명령:

```powershell
pytest `
tests/test_evidence_builder.py `
tests/test_answer_builder.py `
tests/test_api_failure_agent.py `
tests/test_intent_classifier.py `
tests/test_agent_state.py `
tests/test_failure_agent_graph.py `
tests/test_api_langgraph_agent.py `
-v
```

실행 환경:

```text
Python 3.11.9

pytest 9.1.1

pluggy 1.6.0

anyio 4.14.1

langsmith 0.10.0
```

수집된 테스트:

```text
67 items
```

최종 결과:

```text
67 passed in 7.87s
```

---

# 34. 테스트 파일별 결과

```text
tests/test_evidence_builder.py

5 passed
```

검증 내용:

```text
prediction summary evidence

rule-based evidence 변환

SHAP local evidence 변환

evidence source 분리

evidence type grouping
```

---

```text
tests/test_answer_builder.py

2 passed
```

검증 내용:

```text
prediction summary answer 포함

rule evidence와 SHAP evidence 분리
```

---

```text
tests/test_api_failure_agent.py

3 passed
```

검증 내용:

```text
기존 prediction API response 구조

SHAP artifact 실패 warning

include_shap=false 동작
```

---

```text
tests/test_intent_classifier.py

13 passed
```

검증 내용:

```text
LLM JSON payload 검증

지원하지 않는 intent 처리

confidence 범위 정규화

문자열 confidence 처리

잘못된 payload 처리

빈 질문 처리

rule-based failure prediction 분류

rule-based dataset schema 분류

unknown 질문 처리

OpenAI 성공 결과 사용

OpenAI 실패 fallback

OpenAI 비활성화

결과 dict 변환
```

---

```text
tests/test_agent_state.py

13 passed
```

검증 내용:

```text
초기 AgentState

raw_sample 포함

raw_sample 생략

chat_history 포함

warning 추가

error 추가

error 존재 확인

raw_sample 존재 확인

빈 raw_sample 처리
```

---

```text
tests/test_failure_agent_graph.py

26 passed
```

검증 내용:

```text
질문 검증

intent 분류 node

validation routing

intent routing

raw_sample 없음 처리

prediction 결과 저장

prediction service 예외 처리

prediction 이후 routing

dataset schema answer

fallback answer

final answer

raw sample key 변환

dataset schema 전체 workflow

unknown 전체 workflow

failure prediction 전체 workflow

raw_sample 없는 전체 workflow
```

---

```text
tests/test_api_langgraph_agent.py

5 passed
```

검증 내용:

```text
dataset schema API

unknown fallback API

failure prediction API

raw_sample 없는 prediction API

기존 endpoint 유지
```

---

# 35. Swagger 실제 실행

서버 실행:

```powershell
python -m uvicorn src.api.main:app --reload
```

Swagger 접속:

```text
http://127.0.0.1:8000/docs
```

확인 endpoint:

```text
POST /agent/langgraph-query
```

실제 Swagger 요청 결과:

```text
HTTP 200 OK
```

기존에 발생했던:

```text
500 Internal Server Error
```

문제가 해결되었다.

---

# 36. Swagger 확인 결과

확인한 처리 흐름:

```text
자연어 question

→ FastAPI request 검증

→ LangGraph runner

→ AgentState 생성

→ intent 분류

→ conditional routing

→ failure prediction node

→ Day 12 service

→ prediction/evidence/answer

→ FastAPI response

→ HTTP 200 OK
```

Swagger 실제 실행을 통해 단위 테스트의 monkeypatch 결과뿐 아니라 실제 endpoint와 LangGraph workflow 연결도 정상 동작하는 것을 확인했다.

---

# 37. 기존 Manufacturing MCP Agent와 비교

기존 프로젝트:

```text
사용자 질문

→ rule-based intent 분류

→ tool 선택

→ tool 호출

→ answer 반환
```

기존 구조는 keyword와 조건문을 사용해 intent를 분류했다.

예:

```text
불량률

센서 이상

라인 성능

품질 원인 후보
```

---

Day 14 구조:

```text
사용자 자연어 질문

→ FastAPI

→ LangGraph workflow

→ OpenAI intent classifier

→ 실패 시 rule-based fallback

→ AgentState 기반 conditional routing

→ prediction service

→ evidence

→ answer

→ 구조화된 API response
```

---

# 38. 기존 프로젝트 대비 개선점

```text
기존

규칙 기반 intent
```

```text
Day 14

OpenAI intent
+
rule-based fallback
```

---

```text
기존

고정된 함수 호출 흐름
```

```text
Day 14

LangGraph node
+
conditional edge
```

---

```text
기존

endpoint와 처리 흐름 결합
```

```text
Day 14

API
workflow
service
evidence
answer 계층 분리
```

---

```text
기존

주로 정형 입력과 규칙 중심
```

```text
Day 14

자연어 question 중심 Agent API
```

---

```text
기존

결과 중심 응답
```

```text
Day 14

prediction

probability

threshold

risk_level

recommended_action

answer

evidence

warnings

errors

limitations
```

---

# 39. Day 14 핵심 학습 내용

## FastAPI endpoint

HTTP request를 받고 response를 반환하는 API 입구다.

---

## APIRouter

관련 endpoint를 별도 파일에 분리하고 FastAPI app에 등록한다.

---

## request schema

사용자가 보낸 JSON의 구조와 타입을 검증한다.

---

## response schema

API가 반환할 JSON 구조를 명확히 정의한다.

---

## LangGraph runner

FastAPI 외부 입력과 LangGraph 내부 AgentState 사이를 연결한다.

---

## AgentState

LangGraph node 사이에서 데이터를 전달하는 공유 상태다.

---

## conditional routing

intent와 errors 상태에 따라 다음 node를 선택한다.

---

## optional input

질문 종류에 따라 필요한 입력값이 다를 때 사용한다.

---

## service layer

모델 prediction과 artifact 처리를 endpoint 밖으로 분리한다.

---

## monkeypatch

테스트에서 실제 외부 API나 무거운 service 대신 fake 함수를 사용한다.

---

## regression test

새 기능 추가 후 기존 기능이 망가지지 않았는지 확인한다.

---

## function interface

함수를 호출하는 코드와 호출받는 함수의 parameter 구조가 일치해야 한다.

---

# 40. Day 14에서 배운 오류 해결 과정

이번 구현에서는 단순히 새 endpoint만 추가한 것이 아니라 실제 오류를 추적하고 수정했다.

```text
Swagger 실행

→ 500 Internal Server Error

→ Uvicorn traceback 확인

→ LangGraph call_failure_prediction task 확인

→ unexpected keyword argument 확인

→ helper 함수 interface 불일치 발견

→ include_shap parameter 추가

→ include_global_importance parameter 추가

→ Day 12 service까지 옵션 전달

→ 기존 Day 13 테스트 실패

→ monkeypatch fake 함수 interface 수정

→ runner 테스트 호출 방식 수정

→ 예외 처리 복구

→ 테스트 데이터 보완

→ LangGraph workflow 26개 통과

→ 전체 핵심 테스트 67개 통과

→ Swagger HTTP 200 OK 확인
```

---

# 41. 면접 답변

기존 Manufacturing MCP Agent에서는 사용자의 질문을 규칙 기반으로 intent 분류한 뒤 해당 tool을 직접 호출했습니다.

이번 학습 프로젝트에서는 이 구조를 확장하여 FastAPI endpoint와 LangGraph workflow를 연결했습니다.

새로 추가한 `/agent/langgraph-query` endpoint는 자연어 question과 선택적 raw sample을 입력받으며, endpoint 내부에서 직접 모델을 실행하지 않고 LangGraph runner에 요청을 전달합니다.

LangGraph workflow는 question을 기반으로 intent를 분류하고, `failure_prediction` intent일 때만 기존 prediction service layer를 호출합니다.

dataset schema 질문이나 지원하지 않는 질문은 모델을 호출하지 않고 각각 schema answer 또는 fallback answer를 반환하도록 구성했습니다.

또한 failure prediction 질문인데 raw sample이 없는 경우에는 자연어만으로 probability를 임의 생성하지 않고 error를 기록한 뒤 필요한 설비 입력값을 안내하도록 했습니다.

LLM은 고장 probability를 직접 계산하지 않으며 사용자의 자연어 질문을 intent로 분류하는 역할만 담당합니다.

실제 고장 prediction은 PyTorch MLP 모델과 기존 prediction service가 담당합니다.

테스트에서는 실제 OpenAI API와 모델 artifact에 의존하지 않도록 LangGraph runner와 prediction service를 monkeypatch했습니다.

Day 14 구현 과정에서는 FastAPI Swagger 실행 시 helper 함수가 새 `include_shap` option을 받지 못해 500 오류가 발생했습니다.

Uvicorn traceback을 확인하여 호출부와 helper 함수의 interface 불일치를 찾았고, helper parameter와 기존 monkeypatch 테스트를 함께 수정했습니다.

최종적으로 evidence, answer, 기존 prediction API, intent classifier, AgentState, LangGraph workflow, 새 LangGraph API를 포함한 핵심 테스트 67개가 모두 통과했으며 Swagger 실제 요청에서도 HTTP 200 응답을 확인했습니다.

이를 통해 API, workflow, prediction service의 역할을 분리하고 기존 rule-based endpoint보다 확장 가능한 제조 AI Agent API 구조를 구현했습니다.

---

# 42. Day 14 최종 구조

```text
사용자
↓
POST /agent/langgraph-query
↓
FastAPI request schema
↓
LangGraph runner
↓
AgentState 생성
↓
질문 검증
↓
OpenAI intent classifier
↓
OpenAI 실패 시 rule-based fallback
↓
LangGraph conditional routing
↓
failure prediction
/
dataset schema
/
fallback
↓
Day 12 prediction service
↓
PyTorch model prediction
↓
prediction
↓
evidence
↓
answer
↓
FastAPI response schema
↓
HTTP 200
↓
사용자
```

---

# 43. Day 14 완료 결과

```text
[완료] LangGraph API endpoint 추가

[완료] 기존 prediction endpoint 유지

[완료] 자연어 question 입력 지원

[완료] optional raw_sample 지원

[완료] dataset_schema_query 처리

[완료] unknown fallback 처리

[완료] failure_prediction 처리

[완료] raw_sample 없음 처리

[완료] SHAP option 전달

[완료] global importance option 전달

[완료] prediction service 예외 처리

[완료] request schema 추가

[완료] response schema 추가

[완료] API router 등록

[완료] LangGraph runner 확장

[완료] 기존 LangGraph 테스트 수정

[완료] LangGraph API 테스트 작성

[완료] LangGraph workflow 테스트 26개 통과

[완료] LangGraph API 테스트 5개 통과

[완료] 전체 핵심 테스트 67개 통과

[완료] 전체 테스트 실행 시간 7.87초

[완료] Swagger 실제 요청 확인

[완료] Swagger HTTP 200 OK 확인

[확인 필요] Git status 확인

[확인 필요] Day 14 Git commit

[확인 필요] Git push
```

---

# 44. Day 14 최종 테스트 결과

```text
==============================
67 passed in 7.87s
==============================
```

Swagger:

```text
POST /agent/langgraph-query

HTTP 200 OK
```

Day 14의 구현, 테스트, 실제 API 실행 확인이 완료되었다.
