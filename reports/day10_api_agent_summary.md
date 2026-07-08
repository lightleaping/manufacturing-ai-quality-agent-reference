# Day 10 API Agent Summary

## 1. 오늘의 목표

Day 10의 목표는 Day 5 inference, Day 8 SHAP explanation, Day 9 Agent evidence / answer builder 흐름을 FastAPI endpoint로 연결하는 것이다.

추천 endpoint는 다음과 같다.

POST /agent/failure-prediction

---

## 2. 만든 파일

```text
src/api/__init__.py
src/api/schemas.py
src/api/failure_agent_api.py
src/api/main.py

tests/test_api_failure_agent.py
reports/day10_api_agent_summary.md
```

---

## 3. Day 10에서 추가로 필요한 패키지

FastAPI endpoint 테스트에서 `TestClient`를 사용하려면 현재 Starlette 버전 기준으로 `httpx2` 패키지가 필요하다.

설치 명령:

```bash
python -m pip install httpx2
```

`requirements.txt`에도 아래 내용을 추가한다.

```txt
httpx2
```

에러 예시:

```text
RuntimeError: The starlette.testclient module requires the httpx2 package to be installed.
```

해석:

```text
테스트 코드에서 fastapi.testclient.TestClient를 import했는데,
내부적으로 Starlette TestClient가 사용되고,
현재 환경에 httpx2가 설치되어 있지 않아서 테스트 수집 단계에서 실패했다.
```

---

## 4. API 입력 구조

API 요청 예시는 다음과 같다.

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

API에서는 사용자가 이해하기 쉬운 snake_case field를 받는다.

내부 inference 함수에는 기존 AI4I feature 이름으로 변환해서 넘긴다.

```text
air_temperature      -> Air temperature [K]
process_temperature  -> Process temperature [K]
rotational_speed     -> Rotational speed [rpm]
torque               -> Torque [Nm]
tool_wear            -> Tool wear [min]
type                 -> Type
```

---

## 5. API 응답 구조

응답에는 다음 항목이 포함된다.

```text
prediction
probability
threshold
risk_level
recommended_action
evidence
answer
warnings
limitations
```

각 항목의 의미는 다음과 같다.

```text
prediction
= threshold 기준 최종 0/1 판단

probability
= 모델이 예측한 고장 확률

threshold
= 운영 판단 기준

risk_level
= probability를 사람이 이해하기 쉬운 LOW / MEDIUM / HIGH 등급으로 바꾼 값

recommended_action
= 위험도에 따른 권장 조치

evidence
= prediction_summary, rule_based, shap_local, global_importance를 포함하는 근거 목록

answer
= 사용자가 읽을 수 있는 자연어 Agent 답변

warnings
= 해석 시 주의사항

limitations
= 현재 시스템의 한계
```

---

## 6. 중요한 설계 원칙

FastAPI endpoint 안에 모델 로직을 직접 길게 쓰지 않는다.

endpoint는 다음 역할만 담당한다.

```text
1. request 받기
2. Pydantic schema로 입력값 검증하기
3. request를 raw_sample로 변환하기
4. Day 5 inference 함수 호출하기
5. Day 8 SHAP explanation 함수 호출 준비하기
6. Day 9 evidence builder 호출하기
7. Day 9 answer builder 호출하기
8. response schema로 반환하기
```

즉, API 계층은 얇게 유지한다.

좋은 구조:

```text
API endpoint
-> 기존 inference 함수 호출
-> 기존 SHAP 함수 호출
-> 기존 evidence builder 호출
-> 기존 answer builder 호출
-> JSON response 반환
```

나쁜 구조:

```text
API endpoint 안에서 직접 scaling
API endpoint 안에서 직접 torch model 실행
API endpoint 안에서 직접 SHAP 계산 로직 작성
API endpoint 안에서 직접 긴 answer 문장 조립
```

---

## 7. evidence 해석 주의점

Day 10 API에서도 evidence_type과 source를 분리한다.

```text
prediction_summary
= 모델의 최종 probability, threshold, prediction, risk_level 요약

rule_based
= 입력값을 사람이 정한 제조 기준으로 해석한 점검 신호

shap_local
= 특정 sample 하나에 대해 feature가 모델 output에 기여한 방향

global_importance
= 전체 test set 기준 모델 성능에 중요한 feature
```

따라서 Agent answer에서는 특정 feature를 실제 고장의 물리적 원인이라고 단정하지 않는다.

정확한 표현:

```text
입력값 기준으로 Torque가 제조 rule에서 점검 신호로 표시되었습니다.
SHAP 기준으로 Torque는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

부정확한 표현:

```text
모델은 Torque 때문에 고장이라고 판단했습니다.
```

---

## 8. 테스트

테스트 파일은 다음과 같다.

```text
tests/test_api_failure_agent.py
```

테스트 목적은 실제 PyTorch 모델 성능 검증이 아니다.

Day 10 테스트의 목적은 다음과 같다.

```text
FastAPI endpoint가 요청을 받을 수 있는지 확인한다.
응답 JSON 구조가 예상대로 나오는지 확인한다.
prediction, probability, threshold, risk_level이 포함되는지 확인한다.
evidence list가 포함되는지 확인한다.
answer, warnings, limitations가 포함되는지 확인한다.
```

실제 모델 로딩과 추론은 Day 5 테스트에서 이미 검증했다.

evidence builder와 answer builder는 Day 9 테스트에서 이미 검증했다.

따라서 Day 10 테스트에서는 monkeypatch를 사용해 API 계층만 검증한다.

---

## 9. 테스트 에러와 해결

이번에 발생한 에러:

```text
RuntimeError: The starlette.testclient module requires the httpx2 package to be installed.
```

원인:

```text
tests/test_api_failure_agent.py에서 TestClient를 import했다.
FastAPI의 TestClient는 내부적으로 Starlette TestClient를 사용한다.
현재 설치된 Starlette TestClient는 httpx2 패키지를 필요로 한다.
하지만 현재 .venv에 httpx2가 설치되어 있지 않다.
그래서 테스트 실행 전에 import 단계에서 실패했다.
```

해결:

```bash
python -m pip install httpx2
```

`requirements.txt`에 추가:

```txt
httpx2
```

다시 테스트:

```bash
pytest tests/test_api_failure_agent.py -v
```

---

## 10. 서버 실행 명령

프로젝트 루트에서 실행한다.

```bash
uvicorn src.api.main:app --reload
```

Swagger UI 접속 주소:

```text
http://127.0.0.1:8000/docs
```

---

## 11. Day 10에서 배운 개념

### FastAPI

FastAPI는 Python 함수로 API endpoint를 만들 수 있게 해주는 웹 프레임워크다.

이번 프로젝트에서는 사용자가 설비 sample 값을 POST 요청으로 보내면, 모델 예측 결과와 evidence, Agent answer를 JSON으로 반환하도록 사용했다.

---

### Pydantic BaseModel

Pydantic BaseModel은 API 입력값과 출력값의 구조를 정의하고 검증하는 schema 역할을 한다.

예를 들어 `air_temperature: float`라고 정의하면, 사용자는 해당 값에 숫자를 보내야 한다.

문자열이나 누락된 값이 들어오면 FastAPI가 자동으로 validation error를 반환한다.

---

### Field

`Field`는 Pydantic model field에 추가 정보를 넣는 함수다.

예:

```python
air_temperature: float = Field(
    ...,
    description="Air temperature [K]",
    examples=[303.0],
)
```

의미:

```text
air_temperature는 필수 입력값이다.
값은 float이어야 한다.
Swagger UI에는 Air temperature [K]라고 설명된다.
예시값으로 303.0을 보여준다.
```

`Field(...)`의 `...`은 필수값이라는 뜻이다.

---

### APIRouter

APIRouter는 endpoint들을 기능별로 묶는 도구다.

이번 프로젝트에서는 `/agent` prefix를 가진 router를 만들었다.

최종 endpoint는 다음과 같다.

```text
POST /agent/failure-prediction
```

---

### TestClient

TestClient는 실제 서버를 켜지 않고도 FastAPI endpoint를 테스트할 수 있게 해주는 도구다.

예:

```python
client = TestClient(app)

response = client.post(
    "/agent/failure-prediction",
    json={
        "air_temperature": 303.0,
        "process_temperature": 312.5,
        "rotational_speed": 1380.0,
        "torque": 62.0,
        "tool_wear": 220.0,
        "type": "L",
        "include_shap": True,
        "include_global_importance": True,
    },
)
```

이렇게 하면 실제 uvicorn 서버를 실행하지 않아도 API 응답 구조를 검증할 수 있다.

---

### monkeypatch

monkeypatch는 테스트에서 특정 함수를 임시로 fake 함수로 바꿔 끼우는 도구다.

이번 테스트에서는 실제 모델 artifact를 로드하지 않기 위해 사용했다.

이유:

```text
API 테스트의 목적은 모델 성능 검증이 아니다.
모델 로딩과 추론은 Day 5에서 이미 테스트했다.
Day 10에서는 endpoint의 요청/응답 구조만 검증하면 된다.
```

---

## 12. 기존 manufacturing-mcp-agent와 비교

기존 manufacturing-mcp-agent는 사용자의 질문을 받아 intent를 분류하고 tool을 호출한 뒤 evidence를 반환하는 구조였다.

Day 10의 manufacturing-ai-quality-agent-reference는 raw sensor sample을 입력받아 다음 흐름을 수행한다.

```text
raw sensor sample
-> model inference
-> probability
-> threshold comparison
-> prediction
-> risk_level
-> rule-based evidence
-> SHAP local evidence
-> global importance evidence
-> Agent answer
-> FastAPI response
```

즉, 단순히 tool 결과를 반환하는 API가 아니라, 모델 예측 결과와 설명 가능한 evidence를 함께 반환하는 제조 AI Agent API의 첫 형태다.

---

## 13. 면접 답변 문장

이번 프로젝트에서는 모델 추론 결과를 단순히 숫자로만 반환하지 않고, FastAPI endpoint를 통해 prediction, probability, threshold, risk level, rule-based evidence, SHAP local evidence, global importance, Agent answer를 함께 반환하도록 설계했습니다.

특히 endpoint 안에 모델 추론이나 설명 로직을 직접 길게 작성하지 않고, 기존 inference 함수, interpretability 함수, evidence builder, answer builder를 재사용하는 얇은 API 계층으로 구성했습니다.

또한 rule-based evidence, SHAP local evidence, global importance의 의미를 evidence_type과 source로 구분하여, 특정 feature를 실제 고장의 원인으로 단정하지 않고 모델 기준 위험 예측에 어떤 방향으로 작용했는지 설명하도록 했습니다.
