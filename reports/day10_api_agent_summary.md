# Day 10 API Agent Summary

## 1. 오늘의 목표

Day 10의 목표는 Day 5 inference, Day 8 SHAP explanation, Day 9 Agent evidence / answer builder 흐름을 FastAPI endpoint로 연결하는 것이다.

오늘 만드는 API의 핵심 endpoint는 다음과 같다.

```text
POST /agent/failure-prediction
```

이 endpoint는 사용자가 설비 sample 값을 JSON으로 보내면, 모델 예측 결과와 evidence, Agent answer를 JSON으로 반환한다.

---

## 2. Day 10에서 만든 파일

```text
src/api/__init__.py
src/api/schemas.py
src/api/failure_agent_api.py
src/api/main.py

tests/test_api_failure_agent.py
reports/day10_api_agent_summary.md
```

각 파일의 역할은 다음과 같다.

```text
src/api/__init__.py
= src/api를 Python 패키지로 인식시키는 파일

src/api/schemas.py
= API request / response 구조를 정의하는 Pydantic schema 파일

src/api/failure_agent_api.py
= POST /agent/failure-prediction endpoint를 정의하는 router 파일

src/api/main.py
= FastAPI app을 생성하고 router를 등록하는 파일

tests/test_api_failure_agent.py
= FastAPI endpoint의 request / response 구조를 검증하는 테스트 파일

reports/day10_api_agent_summary.md
= Day 10 학습 내용, 실행 결과, 면접 답변을 정리하는 보고서
```

---

## 3. Day 10에서 추가로 필요한 패키지

FastAPI endpoint 테스트에서 TestClient를 사용하려면 현재 Starlette 버전 기준으로 httpx2 패키지가 필요하다.

설치 명령은 다음과 같다.

```bash
python -m pip install httpx2
```

requirements.txt에도 아래 내용을 추가한다.

```text
httpx2
```

처음 발생한 에러는 다음과 같았다.

```text
RuntimeError: The starlette.testclient module requires the httpx2 package to be installed.
```

이 에러의 의미는 다음과 같다.

```text
tests/test_api_failure_agent.py에서 TestClient를 import했다.
FastAPI의 TestClient는 내부적으로 Starlette TestClient를 사용한다.
현재 설치된 Starlette TestClient는 httpx2 패키지를 필요로 한다.
하지만 현재 .venv에 httpx2가 설치되어 있지 않다.
그래서 테스트 실행 전에 import 단계에서 실패했다.
```

해결 방법은 다음과 같다.

```bash
python -m pip install httpx2
```

이후 다시 테스트를 실행했다.

```bash
pytest tests/test_api_failure_agent.py -v
```

---

## 4. API 입력 구조

Swagger UI 또는 HTTP client에서 보내는 요청 예시는 다음과 같다.

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

즉, API 입력 이름과 모델 내부 feature 이름을 분리했다.

이렇게 한 이유는 다음과 같다.

```text
API 사용자에게는 air_temperature처럼 쓰기 쉬운 이름을 제공한다.
내부 모델 로직은 Day 1~9에서 사용한 AI4I 원본 feature 이름을 그대로 유지한다.
기존 inference 함수를 수정하지 않고 재사용할 수 있다.
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

Day 10에서 가장 중요한 설계 원칙은 FastAPI endpoint를 얇게 유지하는 것이다.

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

좋은 구조는 다음과 같다.

```text
API endpoint
-> 기존 inference 함수 호출
-> 기존 SHAP 함수 호출 준비
-> 기존 evidence builder 호출
-> 기존 answer builder 호출
-> JSON response 반환
```

나쁜 구조는 다음과 같다.

```text
API endpoint 안에서 직접 scaling
API endpoint 안에서 직접 torch model 실행
API endpoint 안에서 직접 SHAP 계산 로직 작성
API endpoint 안에서 직접 긴 answer 문장 조립
```

Day 10에서는 기존 Day 5 / Day 8 / Day 9 흐름을 재사용하는 API 계층을 만드는 것이 목표였다.

---

## 7. src/api/schemas.py에서 배운 개념

schemas.py에서는 API request / response 구조를 Pydantic BaseModel로 정의했다.

예시 코드는 다음과 같다.

```python
air_temperature: float = Field(
    ...,
    description="Air temperature [K]",
    examples=[303.0],
)
```

이 코드의 의미는 다음과 같다.

```text
air_temperature는 API 요청 body에서 받을 입력값 이름이다.
값은 float 타입이어야 한다.
Field(...)의 ...은 필수 입력값이라는 뜻이다.
description은 Swagger UI에 표시될 설명이다.
examples는 Swagger UI에 표시될 예시값이다.
```

즉, 아래 JSON에서 air_temperature 값은 반드시 있어야 한다.

```json
{
  "air_temperature": 303.0
}
```

Field(...)의 ...이 의미하는 것은 다음과 같다.

```text
이 값은 필수 입력값이다.
사용자가 이 값을 빼고 요청하면 FastAPI가 422 validation error를 반환한다.
```

비교하면 다음과 같다.

```python
# 필수 입력값
air_temperature: float = Field(...)

# 기본값이 있는 선택 입력값
air_temperature: float = Field(default=303.0)

# 선택 입력값, 안 보내면 None
air_temperature: float | None = Field(default=None)
```

Day 10에서는 설비 sample 값이 없으면 예측을 할 수 없기 때문에 아래 값들은 필수 입력값으로 두었다.

```text
air_temperature
process_temperature
rotational_speed
torque
tool_wear
type
```

---

## 8. type 필드와 machine_type alias

API 요청에서는 type이라는 이름을 사용한다.

```json
{
  "type": "L"
}
```

하지만 Python에서 type은 내장 함수 이름이기도 하다.

그래서 내부 변수명은 machine_type으로 두고, API JSON에서는 alias="type"을 사용했다.

의미는 다음과 같다.

```text
외부 API 사용자 입장:
type이라는 이름으로 보낸다.

Python 코드 내부:
machine_type이라는 이름으로 사용한다.
```

이렇게 하면 API 입력은 자연스럽게 유지하면서, Python 코드에서는 내장 함수 type과 이름 충돌을 피할 수 있다.

---

## 9. src/api/failure_agent_api.py에서 배운 개념

failure_agent_api.py는 실제 endpoint를 정의하는 파일이다.

핵심 endpoint는 다음과 같다.

```text
POST /agent/failure-prediction
```

이 endpoint의 실행 흐름은 다음과 같다.

```text
1. FailurePredictionRequest로 request body 검증
2. request.to_raw_sample()로 내부 raw_sample 형식 변환
3. load_failure_model_artifacts()로 model.pt, scaler.joblib, metadata.json 로드
4. predict_failure_from_artifacts()로 실제 단일 sample 추론 실행
5. build_agent_evidence()로 prediction summary, rule-based evidence, global importance 통합
6. build_agent_answer()로 사용자용 Agent answer 생성
7. FailurePredictionResponse로 JSON response 반환
```

---

## 10. build_agent_evidence 인자 이름 오류

Swagger 실제 실행 중 다음 에러가 발생했다.

```text
TypeError: build_agent_evidence() got an unexpected keyword argument 'shap_local_evidence'
```

원인은 API 코드에서 build_agent_evidence()를 호출할 때 실제 함수 정의에 없는 인자 이름을 사용했기 때문이다.

실제 build_agent_evidence() 함수 signature는 다음과 같았다.

```text
(
  prediction_result: dict[str, Any],
  shap_local_explanation: Any | None = None,
  global_importance_items: Iterable[dict[str, Any]] | None = None,
  shap_top_n: int | None = 5
) -> list[dict[str, Any]]
```

따라서 잘못된 호출은 다음과 같다.

```python
agent_evidence = build_agent_evidence(
    prediction_result=prediction_result,
    shap_local_evidence=shap_evidence,
    global_importance_evidence=global_importance_evidence,
)
```

수정 후 올바른 호출은 다음과 같다.

```python
agent_evidence = build_agent_evidence(
    prediction_result=prediction_dict,
    shap_local_explanation=shap_local_explanation,
    global_importance_items=global_importance_evidence,
    shap_top_n=5,
)
```

중요한 점은 다음과 같다.

```text
keyword argument를 사용할 때는 실제 함수 정의부의 parameter 이름과 정확히 일치해야 한다.

맞는 이름:
prediction_result
shap_local_explanation
global_importance_items
shap_top_n

틀린 이름:
shap_local_evidence
global_importance_evidence
```

---

## 11. pytest 테스트 결과

테스트 명령은 다음과 같다.

```bash
pytest tests/test_api_failure_agent.py -v
```

실행 결과는 다음과 같다.

```text
tests/test_api_failure_agent.py::test_failure_prediction_agent_api_returns_expected_structure PASSED

1 passed in 7.27s
```

이 테스트가 검증한 것은 다음과 같다.

```text
FastAPI app 생성 성공
POST /agent/failure-prediction endpoint 등록 성공
request JSON 입력 성공
response JSON 반환 성공
prediction / probability / threshold / risk_level 포함 확인
evidence list 포함 확인
answer / warnings / limitations 포함 확인
```

단, 이 테스트는 실제 모델 추론 테스트가 아니다.

이 테스트에서는 monkeypatch를 사용해 아래 함수들을 fake 함수로 바꿨다.

```text
load_failure_model_artifacts
predict_failure_from_artifacts
build_agent_evidence
build_agent_answer
```

즉, Day 10 테스트의 목적은 실제 모델 성능 검증이 아니라 API 계층의 request / response 구조 검증이었다.

실제 모델 로딩과 추론은 Day 5에서 이미 검증했다.

evidence builder와 answer builder는 Day 9에서 이미 검증했다.

---

## 12. monkeypatch를 사용한 이유

monkeypatch는 테스트에서 특정 함수를 임시로 fake 함수로 바꿔 끼우는 도구다.

이번 테스트에서 monkeypatch를 사용한 이유는 다음과 같다.

```text
API 테스트의 목적은 모델 성능 검증이 아니다.
모델 artifact가 없어도 API 구조 테스트는 가능해야 한다.
실제 모델 로딩과 추론은 Day 5 테스트에서 이미 검증했다.
Day 10에서는 endpoint가 request를 받고 response를 반환하는지만 확인하면 된다.
```

따라서 tests/test_api_failure_agent.py에서는 실제 모델을 로드하지 않고 fake prediction result를 사용했다.

---

## 13. 서버 실행 명령

프로젝트 루트에서 아래 명령을 실행한다.

```bash
uvicorn src.api.main:app --reload
```

Swagger UI 접속 주소는 다음과 같다.

```text
http://127.0.0.1:8000/docs
```

Swagger에서 POST /agent/failure-prediction endpoint를 열고 sample JSON을 입력해 테스트한다.

---

## 14. Swagger 실제 실행 결과

Day 10 API endpoint를 Swagger UI에서 직접 실행했다.

실행 endpoint는 다음과 같다.

```text
POST /agent/failure-prediction
```

입력 sample은 다음과 같다.

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

실행 결과 핵심은 다음과 같다.

```text
prediction  : 1
probability : 0.9929707646369934
threshold   : 0.7
risk_level  : HIGH
```

해석은 다음과 같다.

```text
모델은 현재 sample의 고장 probability를 약 99.30%로 예측했다.
운영 threshold 0.7 기준으로 probability가 threshold보다 높기 때문에 prediction=1로 판단되었다.
risk_level은 HIGH다.
권장 조치는 설비 점검 및 생산 조건 확인이다.
```

---

## 15. Swagger 응답에 포함된 evidence

Swagger 응답에는 다음 evidence type이 포함되었다.

```text
prediction_summary
rule_based
global_importance
```

prediction_summary evidence는 모델 예측 결과를 요약한다.

```text
probability
threshold
prediction
risk_level
recommended_action
```

rule_based evidence에서는 다음 feature가 점검 신호로 표시되었다.

```text
Tool wear [min] = 220.0
Torque [Nm] = 62.0
```

global_importance evidence에서는 전체 test set 기준으로 다음 feature가 중요하게 표시되었다.

```text
Torque [Nm] importance = 0.3309
Air temperature [K] importance = 0.2725
Rotational speed [rpm] importance = 0.2292
```

---

## 16. evidence 해석 주의점

Day 10 API에서도 evidence_type과 source를 분리했다.

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

정확한 표현은 다음과 같다.

```text
입력값 기준으로 Torque가 제조 rule에서 점검 신호로 표시되었습니다.
SHAP 기준으로 Torque는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

부정확한 표현은 다음과 같다.

```text
모델은 Torque 때문에 고장이라고 판단했습니다.
```

---

## 17. SHAP 연결 상태

API 요청에서는 include_shap=true를 보냈다.

하지만 현재 응답에는 SHAP local explanation 결과가 포함되지 않았다.

응답 answer에는 다음 문장이 포함되었다.

```text
현재 응답에는 SHAP local explanation 결과가 포함되지 않았습니다.
```

이것은 실패가 아니라 Day 10에서 의도적으로 SHAP 연결부를 placeholder로 남겨두었기 때문이다.

현재 상태는 다음과 같다.

```text
FastAPI API 연결 완료
실제 모델 추론 연결 완료
rule-based evidence 연결 완료
global importance evidence 연결 완료
Agent answer 반환 완료
SHAP local explanation은 아직 API에 실제 연결 전
```

정확한 표현은 다음과 같다.

```text
Day 10에서는 FastAPI endpoint를 통해 실제 모델 추론, rule-based evidence, global importance, Agent answer 반환까지 연결했다.

다만 SHAP local explanation은 Day 8 함수와 API 연결부를 아직 완전히 붙이지 않았기 때문에 현재 응답에는 포함되지 않는다.
```

---

## 18. 현재 Day 10 완료 범위

Day 10 완료 범위는 다음과 같다.

```text
FastAPI endpoint 생성 완료
Pydantic request / response schema 생성 완료
TestClient 기반 API 구조 테스트 완료
Swagger UI 실제 실행 완료
실제 model artifact load 완료
실제 prediction 실행 완료
rule-based evidence 반환 완료
global importance evidence 반환 완료
Agent answer 반환 완료
```

아직 남은 선택 개선 사항은 다음과 같다.

```text
SHAP local explanation 실제 API 연결
include_shap=true인데 SHAP 결과가 없을 때 limitations에 명확한 안내 추가
global importance를 placeholder가 아니라 artifact 파일에서 로드하도록 개선
```

---

## 19. Day 10 완료 판정

현재 상태는 다음과 같이 볼 수 있다.

```text
Day 10-1 API 파일 생성: 완료
Day 10-2 pytest 구조 테스트: 완료
Day 10-3 Swagger 실제 실행: 완료
Day 10-4 실제 모델 artifact 추론 연결: 완료
Day 10-5 evidence / answer JSON 반환: 완료
Day 10-6 SHAP 실제 연결: 미완료, 다음 단계
```

따라서 Day 10은 완료로 봐도 된다.

단, SHAP local explanation을 실제 API 응답에 포함시키는 작업은 Day 11 또는 Day 10 추가 개선으로 진행하면 된다.

---

## 20. 기존 manufacturing-mcp-agent와 비교

기존 manufacturing-mcp-agent는 사용자의 질문을 받아 intent를 분류하고 tool을 호출한 뒤 evidence를 반환하는 구조였다.

Day 10의 manufacturing-ai-quality-agent-reference는 raw sensor sample을 입력받아 다음 흐름을 수행한다.

```text
raw sensor sample
-> model artifact load
-> model inference
-> probability
-> threshold comparison
-> prediction
-> risk_level
-> rule-based evidence
-> global importance evidence
-> Agent answer
-> FastAPI response
```

즉, 단순히 tool 결과를 반환하는 API가 아니라 모델 예측 결과와 설명 가능한 evidence를 함께 반환하는 제조 AI Agent API의 첫 형태다.

---

## 21. 오늘 배운 개념

### FastAPI

FastAPI는 Python 함수로 API endpoint를 만들 수 있게 해주는 웹 프레임워크다.

이번 프로젝트에서는 사용자가 설비 sample 값을 POST 요청으로 보내면, 모델 예측 결과와 evidence, Agent answer를 JSON으로 반환하도록 사용했다.

---

### Pydantic BaseModel

Pydantic BaseModel은 API 입력값과 출력값의 구조를 정의하고 검증하는 schema 역할을 한다.

예를 들어 air_temperature: float라고 정의하면, 사용자는 해당 값에 숫자를 보내야 한다.

문자열이나 누락된 값이 들어오면 FastAPI가 자동으로 validation error를 반환한다.

---

### Field

Field는 Pydantic model field에 추가 정보를 넣는 함수다.

예시는 다음과 같다.

```python
air_temperature: float = Field(
    ...,
    description="Air temperature [K]",
    examples=[303.0],
)
```

의미는 다음과 같다.

```text
air_temperature는 필수 입력값이다.
값은 float이어야 한다.
Swagger UI에는 Air temperature [K]라고 설명된다.
예시값으로 303.0을 보여준다.
```

Field(...)의 ...은 필수값이라는 뜻이다.

---

### APIRouter

APIRouter는 endpoint들을 기능별로 묶는 도구다.

이번 프로젝트에서는 /agent prefix를 가진 router를 만들었다.

최종 endpoint는 다음과 같다.

```text
POST /agent/failure-prediction
```

---

### TestClient

TestClient는 실제 서버를 켜지 않고도 FastAPI endpoint를 테스트할 수 있게 해주는 도구다.

예시는 다음과 같다.

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

이유는 다음과 같다.

```text
API 테스트의 목적은 모델 성능 검증이 아니다.
모델 로딩과 추론은 Day 5에서 이미 테스트했다.
Day 10에서는 endpoint의 요청 / 응답 구조만 검증하면 된다.
```

---

## 22. 면접 답변 문장

이번 프로젝트에서는 모델 추론 결과를 단순히 숫자로만 반환하지 않고, FastAPI endpoint를 통해 prediction, probability, threshold, risk level, rule-based evidence, global importance, Agent answer를 함께 반환하도록 설계했습니다.

특히 endpoint 안에 모델 추론이나 설명 로직을 직접 길게 작성하지 않고, 기존 inference 함수, evidence builder, answer builder를 재사용하는 얇은 API 계층으로 구성했습니다.

또한 rule-based evidence와 global importance의 의미를 evidence_type과 source로 구분하여, 특정 feature를 실제 고장의 원인으로 단정하지 않고 모델 기준 위험 예측과 참고 근거를 분리해서 설명하도록 했습니다.

SHAP local explanation은 Day 8에서 계산 구조를 만들었지만, Day 10 API에서는 아직 실제 연결 전이므로 다음 단계에서 API 응답에 포함되도록 확장할 예정입니다.
