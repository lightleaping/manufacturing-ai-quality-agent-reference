# Day 13 - LangGraph Agent Workflow 연결 정리

## 1. Day 13 목표

Day 13의 목표는 지금까지 만든 제조 고장 예측 API/service/evidence 구조를 LangGraph Agent workflow로 연결하는 것이다.

기존 `manufacturing-mcp-agent`에서는 사용자의 질문을 규칙 기반으로 intent에 매핑하고, intent에 따라 정해진 Tool을 호출하는 구조였다.

이번 `manufacturing-ai-quality-agent-reference` 프로젝트에서는 그 구조를 다음처럼 개선했다.

```text
기존 구조:
사용자 질문
→ rule-based intent routing
→ tool 호출
→ answer/evidence 반환

Day 13 개선 구조:
사용자 질문
→ OpenAI gpt-4o-mini intent classifier
→ JSON 검증
→ 실패 시 rule-based fallback
→ LangGraph AgentState 기반 workflow
→ Day 12 failure_agent_service 재사용
→ answer/evidence 반환
```

핵심은 LLM이 고장 예측을 직접 수행하지 않는다는 점이다.

LLM은 사용자의 자연어 질문을 보고 어떤 workflow로 보낼지 intent만 분류한다.
실제 고장 예측은 Day 12에서 만든 `failure_agent_service.py`가 담당한다.

---

## 2. Day 13에서 만든 파일

```text
src/agent/intent_classifier.py
src/agent/state.py
src/agent/failure_agent_graph.py

tests/test_intent_classifier.py
tests/test_agent_state.py
tests/test_failure_agent_graph.py

scripts/run_failure_agent_graph_demo.py
reports/day13_langgraph_agent_workflow_summary.md
```

각 파일의 역할은 다음과 같다.

```text
src/agent/intent_classifier.py
→ OpenAI gpt-4o-mini 기반 intent 분류
→ JSON 결과 검증
→ 실패 시 rule-based fallback

src/agent/state.py
→ LangGraph workflow에서 node들이 공유할 AgentState 정의
→ question, intent, raw_sample, prediction, evidence, answer, warnings, errors 관리

src/agent/failure_agent_graph.py
→ LangGraph StateGraph 구성
→ validate, classify, prediction, fallback, final answer node 연결

tests/test_intent_classifier.py
→ intent classifier 검증
→ OpenAI API 실제 호출 없이 monkeypatch로 흐름 테스트

tests/test_agent_state.py
→ AgentState helper 함수 검증
→ warning/error/raw_sample 처리 테스트

tests/test_failure_agent_graph.py
→ LangGraph workflow 분기 테스트
→ OpenAI API와 실제 모델 artifact를 호출하지 않고 graph 흐름 검증

scripts/run_failure_agent_graph_demo.py
→ 실제 LangGraph workflow 실행 demo
→ dataset_schema_query, unknown, failure_prediction 케이스 확인

reports/day13_langgraph_agent_workflow_summary.md
→ Day 13 학습 내용, 실행 명령, 테스트 결과, 면접 답변 정리
```

---

## 3. 가상환경 실행 명령

이 프로젝트는 Windows PowerShell 기준으로 작업한다.

프로젝트 폴더로 이동한다.

```powershell
cd C:\Users\kflow\Downloads\manufacturing-ai-quality-agent-reference
```

가상환경을 실행한다.

```powershell
.\.venv\Scripts\Activate.ps1
```

가상환경이 실행되면 터미널 앞에 보통 아래처럼 표시된다.

```text
(.venv) PS C:\Users\kflow\Downloads\manufacturing-ai-quality-agent-reference>
```

PowerShell에서 실행 정책 문제로 막히면 아래 명령을 한 번 실행한 뒤 다시 가상환경을 실행한다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

그다음 다시 실행한다.

```powershell
.\.venv\Scripts\Activate.ps1
```

CMD를 사용할 경우에는 아래 명령을 사용한다.

```cmd
.venv\Scripts\activate.bat
```

가상환경을 끄고 싶을 때는 아래 명령을 사용한다.

```bash
deactivate
```

---

## 4. 패키지 설치 명령

Day 13에서 필요한 주요 패키지는 다음과 같다.

```text
openai
python-dotenv
langgraph
```

설치 명령은 다음과 같다.

```bash
python -m pip install openai python-dotenv langgraph
```

또는 `requirements.txt`에 아래 항목을 추가한 뒤 설치한다.

```text
openai
python-dotenv
langgraph
```

```bash
python -m pip install -r requirements.txt
```

---

## 5. `.env` 설정

OpenAI API key는 코드에 직접 쓰지 않고 `.env` 파일에 저장한다.

프로젝트 루트에 `.env` 파일을 만든다.

```text
OPENAI_API_KEY=너의_API_KEY
OPENAI_MODEL=gpt-4o-mini
```

`.gitignore`에는 반드시 아래 항목을 추가한다.

```text
.env
```

이유는 API key가 GitHub에 올라가면 안 되기 때문이다.

`intent_classifier.py`에서는 `python-dotenv`의 `load_dotenv()`를 사용해서 `.env` 파일을 읽고, OpenAI SDK는 `OPENAI_API_KEY` 값을 사용한다.

---

## 6. Day 13 전체 실행 순서

가상환경 실행:

```powershell
cd C:\Users\kflow\Downloads\manufacturing-ai-quality-agent-reference
.\.venv\Scripts\Activate.ps1
```

패키지 설치:

```bash
python -m pip install -r requirements.txt
```

Day 13 테스트 실행:

```bash
pytest tests/test_intent_classifier.py -v
pytest tests/test_agent_state.py -v
pytest tests/test_failure_agent_graph.py -v
```

Day 12까지 포함한 핵심 테스트 실행:

```bash
pytest tests/test_evidence_builder.py -v
pytest tests/test_answer_builder.py -v
pytest tests/test_api_failure_agent.py -v
pytest tests/test_intent_classifier.py -v
pytest tests/test_agent_state.py -v
pytest tests/test_failure_agent_graph.py -v
```

LangGraph demo 실행:

```bash
python -m scripts.run_failure_agent_graph_demo
```

주의할 점은 `python scripts/run_failure_agent_graph_demo.py`처럼 직접 파일 경로로 실행하지 않는 것이다.

권장 방식은 다음과 같다.

```bash
python -m scripts.run_failure_agent_graph_demo
```

이 방식은 프로젝트 루트를 기준으로 module import를 처리하므로 `src.agent.failure_agent_graph` 같은 import가 더 안정적으로 동작한다.

---

## 7. `intent_classifier.py` 핵심 정리

`intent_classifier.py`는 사용자의 자연어 질문을 intent로 분류하는 파일이다.

지원 intent는 우선 3개로 시작했다.

```python
SUPPORTED_INTENTS = {
    "failure_prediction",
    "dataset_schema_query",
    "unknown",
}
```

각 intent의 의미는 다음과 같다.

```text
failure_prediction
→ 설비 입력값 기반 고장 위험 예측 요청

dataset_schema_query
→ AI4I 데이터셋 feature, target, schema 질문

unknown
→ 현재 Agent가 지원하지 않는 질문 또는 판단하기 어려운 질문
```

처리 흐름은 다음과 같다.

```text
question
→ OpenAI gpt-4o-mini 호출
→ intent/confidence/reason JSON 반환
→ validate_intent_payload()로 검증
→ 지원하지 않는 intent면 unknown 처리
→ OpenAI 실패 시 rule-based fallback
```

중요한 점은 OpenAI가 최종 고장 예측을 하지 않는다는 것이다.

OpenAI는 단지 질문을 어떤 workflow로 보낼지 분류한다.

---

## 8. `intent_classifier.py`에서 중요한 방어 로직

LLM 출력은 항상 신뢰할 수 없기 때문에 검증이 필요하다.

검증하는 항목은 다음과 같다.

```text
1. payload가 dict인지 확인
2. intent key가 있는지 확인
3. intent가 지원 목록에 있는지 확인
4. confidence를 float으로 변환
5. confidence를 0.0 ~ 1.0 사이로 제한
6. reason이 비어 있으면 기본 reason 추가
```

예를 들어 OpenAI가 아래처럼 지원하지 않는 intent를 반환할 수 있다.

```json
{
  "intent": "maintenance_schedule",
  "confidence": 0.88,
  "reason": "사용자가 정비 일정을 묻고 있습니다."
}
```

하지만 현재 Day 13에서 지원하는 intent는 다음뿐이다.

```text
failure_prediction
dataset_schema_query
unknown
```

따라서 지원하지 않는 intent는 그대로 믿지 않고 `unknown`으로 처리한다.

---

## 9. rule-based fallback 의미

OpenAI API 호출은 실패할 수 있다.

예를 들면 다음과 같은 상황이다.

```text
1. OPENAI_API_KEY가 없음
2. 네트워크 오류 발생
3. API 응답이 비어 있음
4. JSON 파싱 실패
5. 지원하지 않는 intent 반환
```

이때 Agent 전체가 죽으면 안 된다.

그래서 OpenAI intent 분류가 실패하면 기존 `manufacturing-mcp-agent`에서 사용했던 방식처럼 rule-based fallback을 사용한다.

```text
OpenAI intent classification 실패
→ rule-based keyword matching
→ 최소 intent 분류 수행
→ source="fallback"으로 기록
→ warnings에 실패 이유 기록
```

이 구조 덕분에 LLM이 실패해도 workflow는 최소한의 방식으로 계속 동작할 수 있다.

---

## 10. `state.py` 핵심 정리

`state.py`는 LangGraph workflow 안에서 node들이 공유하는 상태 구조를 정의한다.

LangGraph workflow는 여러 node가 순서대로 실행된다.

```text
validate_question_node
→ classify_intent_node
→ call_failure_prediction_node
→ build_final_answer_node
```

각 node는 같은 state를 주고받는다.

그래서 `AgentState`는 workflow가 들고 다니는 상태 상자라고 볼 수 있다.

초기 state는 보통 다음처럼 시작한다.

```python
{
    "question": "이 설비 조건이면 고장 위험이 높아?",
    "warnings": [],
    "errors": [],
    "limitations": [],
}
```

이후 node를 지나면서 값이 추가된다.

```python
{
    "question": "이 설비 조건이면 고장 위험이 높아?",
    "intent": "failure_prediction",
    "confidence": 0.91,
    "intent_reason": "사용자가 고장 위험 예측을 요청했습니다.",
    "raw_sample": {...},
    "prediction": 1,
    "probability": 0.9929,
    "risk_level": "HIGH",
    "evidence": [...],
    "answer": "고장 위험이 높습니다.",
    "warnings": [],
    "errors": [],
    "limitations": [],
}
```

---

## 11. `TypedDict(total=False)`를 사용한 이유

`AgentState`는 `TypedDict(total=False)`로 정의했다.

이유는 모든 필드가 처음부터 존재하지 않기 때문이다.

처음에는 `question`만 있고, workflow를 지나면서 `intent`, `prediction`, `evidence`, `answer` 등이 점점 채워진다.

```text
처음:
question

intent 분류 후:
question
intent
confidence
intent_reason

prediction 후:
question
intent
prediction
probability
risk_level
evidence
answer
```

따라서 모든 필드를 처음부터 강제하면 오히려 불편하다.

그래서 `question`만 Required로 두고, 나머지는 NotRequired로 처리했다.

---

## 12. warnings와 errors의 차이

Day 13에서도 Day 12와 마찬가지로 warnings와 errors를 구분했다.

```text
warnings
→ 핵심 기능은 성공했지만 부가 기능 일부가 실패한 경우

errors
→ workflow 수행 자체에 문제가 있는 경우
```

예시는 다음과 같다.

```text
warnings 예시:
- OpenAI intent 분류가 실패해 rule-based fallback을 사용했습니다.
- SHAP artifact가 없어 shap_local evidence를 생략했습니다.
- global importance artifact가 없어 global_importance evidence를 생략했습니다.

errors 예시:
- question이 비어 있습니다.
- failure_prediction intent인데 raw_sample이 없습니다.
- prediction service 호출 중 오류가 발생했습니다.
```

이 구분이 중요한 이유는 사용자가 결과를 해석할 때 다르게 받아들여야 하기 때문이다.

warnings는 결과는 나왔지만 일부 설명 기능이 빠졌다는 뜻이다.
errors는 요청을 정상 처리하지 못했다는 뜻이다.

---

## 13. `failure_agent_graph.py` workflow 구조

Day 13 LangGraph workflow는 다음 node로 구성했다.

```text
validate_question_node
classify_intent_node
call_failure_prediction_node
build_dataset_schema_answer_node
build_fallback_answer_node
build_final_answer_node
```

전체 흐름은 다음과 같다.

```text
START
→ validate_question_node
→ route_after_validation

route_after_validation:
  errors 있음 → build_fallback_answer_node
  errors 없음 → classify_intent_node

classify_intent_node
→ route_after_classification

route_after_classification:
  failure_prediction → call_failure_prediction_node
  dataset_schema_query → build_dataset_schema_answer_node
  unknown → build_fallback_answer_node

call_failure_prediction_node
→ route_after_prediction

route_after_prediction:
  errors 있음 → build_fallback_answer_node
  errors 없음 → build_final_answer_node

build_dataset_schema_answer_node → END
build_fallback_answer_node → END
build_final_answer_node → END
```

도식으로 보면 다음과 같다.

```text
사용자 질문
   │
   ▼
validate_question_node
   │
   ├─ question 비어 있음 → build_fallback_answer_node → END
   │
   ▼
classify_intent_node
   │
   ├─ failure_prediction → call_failure_prediction_node
   │                          │
   │                          ├─ raw_sample 없음/오류 → build_fallback_answer_node → END
   │                          └─ 성공 → build_final_answer_node → END
   │
   ├─ dataset_schema_query → build_dataset_schema_answer_node → END
   │
   └─ unknown → build_fallback_answer_node → END
```

---

## 14. 각 node의 책임

### 14.1 `validate_question_node`

사용자 질문이 비어 있는지 확인한다.

```text
question이 정상
→ 앞뒤 공백 제거 후 다음 node로 이동

question이 비어 있음
→ errors에 메시지 추가
→ fallback answer로 이동
```

---

### 14.2 `classify_intent_node`

사용자 질문을 intent로 분류한다.

내부적으로 `classify_intent()`를 호출한다.

```text
OpenAI gpt-4o-mini intent 분류
→ JSON 검증
→ 실패 시 rule-based fallback
```

분류 결과는 state에 저장한다.

```python
state["intent"] = result.intent
state["confidence"] = result.confidence
state["intent_reason"] = result.reason
state["intent_source"] = result.source
state["intent_raw_response"] = result.raw_response
```

---

### 14.3 `call_failure_prediction_node`

`failure_prediction` intent일 때 실제 고장 예측 service를 호출한다.

중요한 점은 이 node가 직접 모델을 로드하지 않는다는 것이다.

이 node는 Day 12에서 만든 service layer를 호출한다.

```text
LangGraph node
→ _run_failure_prediction_service()
→ Day 12 run_failure_prediction_agent()
→ artifact cache
→ prediction
→ evidence
→ answer
```

이렇게 분리한 이유는 node의 책임을 workflow orchestration으로 제한하기 위해서다.

모델 로딩, SHAP 계산, global importance 처리, fallback warning은 Day 12 service layer가 담당한다.

---

### 14.4 `build_dataset_schema_answer_node`

AI4I 데이터셋 feature와 target에 대한 질문에 답한다.

현재는 정적 답변으로 처리한다.

```text
feature:
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Type

target:
- Machine failure

excluded columns:
- UDI
- Product ID
```

이후에는 docs 기반 RAG나 dataset metadata artifact 로딩 방식으로 확장할 수 있다.

---

### 14.5 `build_fallback_answer_node`

unknown intent 또는 오류 상황에서 fallback answer를 만든다.

fallback이 필요한 경우는 다음과 같다.

```text
1. question이 비어 있음
2. intent가 unknown
3. failure_prediction intent인데 raw_sample이 없음
4. prediction service 호출 실패
```

이 node는 억지로 예측하지 않고 사용자에게 필요한 입력값이나 지원 가능한 질문 유형을 안내한다.

---

### 14.6 `build_final_answer_node`

prediction service에서 이미 answer를 만들었다면 그대로 유지한다.

만약 answer가 비어 있다면 최소 답변을 만든다.

```text
risk_level
probability
recommended_action
```

을 사용해 최소한의 답변을 구성한다.

---

## 15. 테스트 전략

Day 13 테스트에서는 실제 OpenAI API를 호출하지 않았다.

또한 실제 model artifact나 SHAP 계산도 수행하지 않았다.

이유는 단위 테스트가 빠르고 안정적이어야 하기 때문이다.

실제 API나 artifact에 의존하면 다음 문제가 생긴다.

```text
1. API key가 없는 환경에서 실패
2. 네트워크 상태에 따라 실패
3. 비용 발생
4. artifact 파일 위치나 손상 여부에 따라 실패
5. SHAP 계산 때문에 테스트가 느려짐
6. 외부 응답 변화로 테스트 결과가 흔들림
```

따라서 테스트에서는 `monkeypatch`를 사용해 외부 의존성을 fake 함수로 대체했다.

```text
실제 OpenAI API 호출
→ fake_classify_intent()

실제 prediction service 호출
→ fake_run_failure_prediction_service()
```

테스트의 목적은 OpenAI 성능이나 모델 성능을 검증하는 것이 아니다.

테스트의 목적은 LangGraph workflow가 의도한 경로로 분기되는지 확인하는 것이다.

---

## 16. 테스트 실행 결과

Day 13 테스트 명령:

```bash
pytest tests/test_intent_classifier.py -v
pytest tests/test_agent_state.py -v
pytest tests/test_failure_agent_graph.py -v
```

확인해야 할 결과:

```text
tests/test_intent_classifier.py 통과
tests/test_agent_state.py 통과
tests/test_failure_agent_graph.py 통과
```

전체 핵심 테스트 명령:

```bash
pytest tests/test_evidence_builder.py -v
pytest tests/test_answer_builder.py -v
pytest tests/test_api_failure_agent.py -v
pytest tests/test_intent_classifier.py -v
pytest tests/test_agent_state.py -v
pytest tests/test_failure_agent_graph.py -v
```

확인해야 할 결과:

```text
Day 9 evidence builder 테스트 통과
Day 9 answer builder 테스트 통과
Day 12 API service 테스트 통과
Day 13 intent classifier 테스트 통과
Day 13 AgentState 테스트 통과
Day 13 LangGraph workflow 테스트 통과
```

---

## 17. demo script 실행

실행 전 가상환경을 켠다.

```powershell
cd C:\Users\kflow\Downloads\manufacturing-ai-quality-agent-reference
.\.venv\Scripts\Activate.ps1
```

demo script 실행:

```bash
python -m scripts.run_failure_agent_graph_demo
```

demo script에서는 3가지 케이스를 확인한다.

```text
Case 1 - Dataset schema query
Case 2 - Unknown intent
Case 3 - Failure prediction with raw_sample
```

---

## 18. demo script 기대 결과

### Case 1 - Dataset schema query

질문:

```text
AI4I 데이터셋 feature와 target은 뭐야?
```

예상 workflow:

```text
validate_question_node
→ classify_intent_node
→ build_dataset_schema_answer_node
→ END
```

예상 결과:

```text
intent: dataset_schema_query
answer: AI4I 2020 Predictive Maintenance Dataset 설명
evidence_count: 1
errors: []
```

---

### Case 2 - Unknown intent

질문:

```text
오늘 점심 메뉴 추천해줘.
```

예상 workflow:

```text
validate_question_node
→ classify_intent_node
→ build_fallback_answer_node
→ END
```

예상 결과:

```text
intent: unknown
answer: 현재 Agent가 지원하는 작업으로 분류되지 않았다는 안내
errors: []
```

---

### Case 3 - Failure prediction with raw_sample

질문:

```text
이 설비 조건이면 고장 위험이 높아?
```

raw_sample:

```python
{
    "air_temperature": 303.0,
    "process_temperature": 312.5,
    "rotational_speed": 1380.0,
    "torque": 62.0,
    "tool_wear": 220.0,
    "type": "L",
}
```

예상 workflow:

```text
validate_question_node
→ classify_intent_node
→ call_failure_prediction_node
→ build_final_answer_node
→ END
```

모델 artifact가 준비되어 있으면 예상 결과는 다음과 같다.

```text
intent: failure_prediction
prediction: 1
probability: 0.99...
risk_level: HIGH
answer: 고장 위험이 높습니다...
```

SHAP artifact가 없더라도 Day 12 fallback 구조가 정상이라면 prediction은 수행되고, SHAP 관련 실패는 warnings에 기록되어야 한다.

```text
warnings:
- SHAP artifact가 없어 shap_local evidence를 생략했습니다...
```

---

## 19. 기존 manufacturing-mcp-agent와 비교

기존 `manufacturing-mcp-agent`의 핵심 구조는 다음과 같았다.

```text
사용자 질문
→ rule-based intent classification
→ intent에 맞는 tool_name 선택
→ tool 함수 호출
→ answer/evidence 반환
```

장점은 구조가 단순하고 예측 가능하다는 점이다.

하지만 한계도 있었다.

```text
1. 표현이 조금만 달라져도 intent 분류가 어려움
2. 자연어 이해가 제한적임
3. workflow 상태 관리가 단순함
4. LLM 기반 routing이 없음
5. fallback과 error/warning 구분이 약함
```

Day 13 개선 구조는 다음과 같다.

```text
사용자 질문
→ OpenAI gpt-4o-mini intent classifier
→ JSON 검증
→ 실패 시 rule-based fallback
→ LangGraph AgentState 기반 workflow
→ Day 12 service layer 호출
→ answer/evidence 반환
```

개선점은 다음과 같다.

```text
1. 자연어 질문을 LLM으로 intent 분류할 수 있음
2. LLM 출력은 JSON schema와 검증 함수로 제한함
3. LLM 실패 시 rule-based fallback으로 안정성 확보
4. LangGraph AgentState로 workflow 상태를 명확히 관리함
5. prediction service와 workflow orchestration을 분리함
6. warning과 error를 구분해 운영 안정성을 높임
```

---

## 20. 오늘 배운 개념이 코드에서 쓰인 위치

### 20.1 LLM intent classification

개념:

```text
LLM을 최종 답변 생성기가 아니라 intent router로 사용한다.
```

코드 위치:

```text
src/agent/intent_classifier.py
```

---

### 20.2 JSON validation

개념:

```text
LLM 출력은 검증 없이 믿지 않는다.
```

코드 위치:

```text
validate_intent_payload()
```

---

### 20.3 rule-based fallback

개념:

```text
LLM 실패 시 기존 rule-based 방식으로 최소 기능을 유지한다.
```

코드 위치:

```text
classify_intent()
classify_intent_rule_based()
```

---

### 20.4 AgentState

개념:

```text
LangGraph node들이 공유하는 상태 상자다.
```

코드 위치:

```text
src/agent/state.py
AgentState
create_initial_agent_state()
append_warning()
append_error()
```

---

### 20.5 LangGraph node

개념:

```text
workflow의 각 처리 단계를 함수 단위 node로 분리한다.
```

코드 위치:

```text
src/agent/failure_agent_graph.py

validate_question_node()
classify_intent_node()
call_failure_prediction_node()
build_dataset_schema_answer_node()
build_fallback_answer_node()
build_final_answer_node()
```

---

### 20.6 Conditional edge

개념:

```text
state 값에 따라 다음 node를 결정한다.
```

코드 위치:

```text
route_after_validation()
route_after_classification()
route_after_prediction()
```

---

### 20.7 Service layer 재사용

개념:

```text
LangGraph node가 모델 로딩과 예측 세부 로직을 직접 들고 있지 않고,
Day 12 service layer를 호출한다.
```

코드 위치:

```text
_run_failure_prediction_service()
```

---

## 21. 한계와 개선 방향

Day 13 구조에도 아직 한계가 있다.

```text
1. intent 종류가 아직 3개뿐이다.
2. dataset_schema_query는 정적 답변으로 처리한다.
3. raw_sample 추출은 아직 사용자가 명시적으로 제공해야 한다.
4. 자연어 문장에서 수치 입력값을 자동 추출하지 않는다.
5. multi-turn memory는 구조만 있고 실제 활용은 제한적이다.
6. OpenAI API 호출 비용과 latency를 고려해야 한다.
7. LangSmith, OpenTelemetry 같은 tracing은 아직 연결하지 않았다.
```

이후 개선 방향은 다음과 같다.

```text
1. intent 확장
   - failure_type_analysis
   - model_metric_query
   - shap_explanation_query
   - threshold_policy_query

2. 자연어에서 raw_sample 자동 추출
   - "Torque 62, Tool wear 220이면 위험해?" 같은 문장에서 수치 추출

3. dataset_schema_query를 RAG로 개선
   - docs/
   - reports/
   - metadata.json

4. LangGraph trace 강화
   - trace_id
   - LangSmith
   - OpenTelemetry

5. FastAPI endpoint와 LangGraph endpoint 연결
   - POST /agent/langgraph-query

6. multi-turn 대화 확장
   - chat_history 활용
   - 이전 입력값 재사용
```

---

## 22. Day 13 면접 답변

Day 13에서는 기존 `manufacturing-mcp-agent`의 rule-based intent routing 구조를 LangGraph 기반 Agent workflow로 확장했습니다.

기존 프로젝트에서는 사용자의 질문을 키워드 규칙으로 intent에 매핑하고, intent에 따라 정해진 Tool을 호출하는 구조였습니다. 이 방식은 단순하고 예측 가능하다는 장점이 있지만, 질문 표현이 조금만 달라져도 intent 분류가 제한적이라는 한계가 있었습니다.

이번 reference 프로젝트에서는 OpenAI gpt-4o-mini를 사용해 사용자의 자연어 질문을 `failure_prediction`, `dataset_schema_query`, `unknown` 같은 intent JSON으로 분류하도록 만들었습니다.

다만 LLM 출력은 그대로 신뢰하지 않고, `intent`, `confidence`, `reason` 형태의 JSON으로 제한한 뒤 검증 함수를 거치게 했습니다. 지원하지 않는 intent가 나오거나 API 오류, JSON 파싱 실패가 발생하면 기존 방식처럼 rule-based fallback으로 처리했습니다.

또한 LangGraph의 `StateGraph`를 사용해 workflow를 node 단위로 나누었습니다. `validate_question_node`, `classify_intent_node`, `call_failure_prediction_node`, `build_fallback_answer_node`처럼 각 node의 책임을 분리했고, `AgentState`를 통해 question, intent, raw_sample, prediction, evidence, answer, warnings, errors를 관리했습니다.

중요한 점은 LLM이 고장 예측을 직접 수행하지 않는다는 것입니다. LLM은 사용자의 질문을 어떤 workflow로 보낼지 분류하는 역할만 하고, 실제 고장 예측은 Day 12에서 만든 `failure_agent_service.py`를 재사용해 처리했습니다.

이를 통해 기존 rule-based routing의 한계를 개선하면서도, LLM 실패 시 fallback 가능한 안정적인 Agent workflow 구조를 만들었습니다.

---

## 23. 커밋 추천 메시지

Day 13 작업이 끝나면 아래 명령으로 커밋한다.

```bash
git add src/agent/intent_classifier.py src/agent/state.py src/agent/failure_agent_graph.py tests/test_intent_classifier.py tests/test_agent_state.py tests/test_failure_agent_graph.py scripts/run_failure_agent_graph_demo.py reports/day13_langgraph_agent_workflow_summary.md requirements.txt .gitignore
git commit -m "Add LangGraph failure agent workflow"
```

`.env`는 절대 커밋하지 않는다.

`.gitignore`에 `.env`가 들어갔는지 반드시 확인한다.
