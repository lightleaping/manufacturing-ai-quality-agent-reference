# Day 12 - API Error Handling, Artifact Caching, Response Stabilization Summary

## 1. Day 12 목표

Day 12의 목표는 FastAPI 기반 Failure Prediction Agent API의 안정성을 개선하는 것이다.

Day 11까지는 SHAP local explanation을 실제 API에 연결했고, `include_shap=true`일 때 SHAP local evidence가 응답에 포함되도록 만들었다. 또한 SHAP background tensor, reference values, global importance 결과를 artifact로 저장하고 API에서 로드하는 구조까지 구현했다.

하지만 운영 환경에 더 가깝게 만들기 위해서는 다음 문제가 남아 있었다.

1. 모델 artifact와 SHAP artifact를 매 요청마다 로드할 가능성이 있다.
2. SHAP artifact가 없거나 손상되면 API 전체가 실패할 수 있다.
3. global importance artifact가 없어도 prediction은 가능한데, 이를 전체 실패로 처리하면 응답 안정성이 떨어진다.
4. endpoint 안에 artifact loading, prediction, SHAP 계산, evidence 조립 로직이 길게 들어가면 유지보수가 어려워진다.
5. prediction 실패와 SHAP 실패를 같은 수준의 오류로 처리하면 API 설계가 부정확해진다.

따라서 Day 12에서는 다음을 개선했다.

```text
model artifact 실패
→ prediction 자체가 불가능하므로 API error로 처리

SHAP artifact 실패
→ prediction은 가능하므로 warnings에 기록하고 shap_local evidence만 생략

global importance 실패
→ prediction은 가능하므로 warnings에 기록하고 global_importance evidence만 생략
```

---

## 2. Day 12에서 만든 파일

```text
src/api/artifact_cache.py
src/api/failure_agent_service.py
```

---

## 3. Day 12에서 수정한 파일

```text
src/api/failure_agent_api.py
src/api/main.py
tests/test_api_failure_agent.py
```

---

## 4. `artifact_cache.py` 역할

`src/api/artifact_cache.py`는 모델 artifact와 SHAP artifact를 cache해서 로드하는 역할을 한다.

Day 11까지는 API 요청 시점마다 artifact를 다시 로드할 수 있는 구조였다. 하지만 운영 환경에서는 모델 파일이나 SHAP background tensor를 매 요청마다 디스크에서 읽는 것은 비효율적이다.

그래서 Day 12에서는 `functools.lru_cache`를 사용해 artifact를 한 번 로드하면 이후 같은 artifact directory에 대해서는 메모리에 저장된 객체를 재사용하도록 만들었다.

핵심 함수는 다음과 같다.

```text
get_cached_failure_model_artifacts
get_cached_shap_artifacts
clear_artifact_cache_for_tests
```

`get_cached_failure_model_artifacts()`는 Day 5에서 만든 `model.pt`, `scaler.joblib`, `metadata.json`을 로드한다.

`get_cached_shap_artifacts()`는 Day 11에서 만든 `shap_background.pt`, `shap_reference_values.json`, `global_importance.json`을 로드한다.

`clear_artifact_cache_for_tests()`는 테스트에서 cache를 초기화하기 위한 함수다. `lru_cache`는 한 번 저장한 값을 계속 재사용하기 때문에, monkeypatch로 fake load 함수를 적용하는 테스트에서는 cache를 비워야 테스트가 의도대로 동작한다.

---

## 5. `failure_agent_service.py` 역할

`src/api/failure_agent_service.py`는 Failure Prediction Agent API의 실제 실행 흐름을 담당한다.

Day 10~11에서는 FastAPI endpoint 안에서 prediction, SHAP, global importance, evidence 생성 로직이 길어질 수 있었다.

Day 12에서는 endpoint를 얇게 유지하기 위해 실제 로직을 service layer로 분리했다.

전체 흐름은 다음과 같다.

```text
request
→ request_to_raw_sample
→ cached model artifacts 로드
→ prediction 수행
→ 필요한 경우 SHAP artifacts 로드
→ SHAP local explanation 생성 시도
→ global importance evidence 생성 시도
→ Agent evidence 통합
→ Agent answer 생성
→ response 반환
```

중요한 점은 prediction 실패와 설명 기능 실패를 구분했다는 것이다.

모델 artifact 로드 실패나 prediction 실패는 API 전체 실패로 처리한다. 모델이 없거나 추론이 불가능하면 핵심 응답인 prediction을 만들 수 없기 때문이다.

반면 SHAP local explanation 실패나 global importance 실패는 API 전체 실패로 처리하지 않는다. 이들은 prediction을 보조하는 설명 정보이기 때문이다. 따라서 prediction과 rule-based evidence는 유지하고, 실패 이유는 `warnings`에 기록한다.

---

## 6. `failure_agent_api.py` 수정 내용

`src/api/failure_agent_api.py`는 endpoint만 담당하도록 단순화했다.

수정 후 endpoint의 역할은 다음과 같다.

```text
1. request를 받는다.
2. run_failure_prediction_agent(request)를 호출한다.
3. response를 반환한다.
```

즉, endpoint 안에서 직접 artifact loading, SHAP 계산, evidence 조립을 하지 않는다.

이렇게 분리한 이유는 다음과 같다.

1. endpoint 코드가 짧아진다.
2. API 계층과 service 계층의 책임이 분리된다.
3. 테스트하기 쉬워진다.
4. 나중에 LangGraph Agent나 다른 API endpoint에 service 로직을 재사용하기 쉬워진다.

---

## 7. `main.py` 수정 내용

`src/api/main.py`에서는 FastAPI app 설정과 router 등록을 담당한다.

Day 12에서는 입력 validation error를 더 친절하게 반환할 수 있도록 `RequestValidationError` handler를 추가했다.

FastAPI와 Pydantic은 기본적으로 잘못된 입력에 대해 422 응답을 반환한다. 하지만 기본 error 형식은 사용자에게 다소 어렵게 보일 수 있다.

그래서 Day 12에서는 어떤 field가 잘못되었는지, 어떤 message가 발생했는지, 어떤 type의 validation error인지 정리해서 반환하도록 했다.

예를 들어 `type` 값이 잘못되었거나 숫자 필드에 문자열이 들어가면 다음과 같은 구조로 반환할 수 있다.

```json
{
  "message": "입력값 형식이 올바르지 않습니다.",
  "errors": [
    {
      "field": "torque",
      "message": "Input should be a valid number",
      "type": "float_parsing"
    }
  ],
  "hint": "Swagger의 request schema를 확인하고, 숫자 필드와 type 값을 다시 확인하세요."
}
```

---

## 8. Day 12에서 추가한 테스트

`tests/test_api_failure_agent.py`에는 API 안정성 테스트를 추가했다.

핵심 테스트는 다음과 같다.

```text
test_failure_prediction_agent_returns_warning_when_shap_artifact_load_fails
test_failure_prediction_agent_skips_shap_when_include_shap_false
```

첫 번째 테스트는 SHAP artifact 로드가 실패해도 API 전체가 500으로 죽지 않는지 확인한다.

기대 결과는 다음과 같다.

```text
status_code == 200
prediction 존재
probability 존재
evidence 존재
warnings 존재
shap_local evidence 없음
prediction_summary evidence 있음
rule_based evidence 있음
```

두 번째 테스트는 `include_shap=false`일 때 SHAP 계산을 생략하는지 확인한다.

기대 결과는 다음과 같다.

```text
status_code == 200
prediction_summary evidence 있음
rule_based evidence 있음
shap_local evidence 없음
global_importance evidence 없음
```

---

## 9. Day 12에서 발생한 문제와 해결

### 문제 1. monkeypatch 경로 오류

초기 테스트에서는 다음 경로를 monkeypatch하려고 했다.

```text
src.api.failure_agent_api.load_failure_model_artifacts
```

하지만 Day 12에서 endpoint를 얇게 만들고 실제 artifact loading을 `artifact_cache.py`로 분리했기 때문에, `failure_agent_api.py`에는 더 이상 `load_failure_model_artifacts`가 존재하지 않는다.

따라서 monkeypatch 경로를 다음처럼 수정했다.

```text
src.api.artifact_cache.load_failure_model_artifacts
src.api.artifact_cache.load_shap_artifacts
src.api.failure_agent_service.predict_failure_from_artifacts
src.api.failure_agent_service.build_agent_evidence
src.api.failure_agent_service.build_agent_answer
```

핵심은 monkeypatch는 함수가 원래 정의된 위치가 아니라, 실제 실행 흐름에서 참조되는 위치에 적용해야 한다는 점이다.

---

### 문제 2. `build_agent_answer()` 인자 수 오류

초기 service 코드에서는 다음처럼 호출했다.

```python
answer = build_agent_answer(evidence_items)
```

하지만 실제 `build_agent_answer()` 함수는 다음처럼 `prediction_result`와 `evidence_items`를 함께 받는다.

```python
build_agent_answer(
    prediction_result=prediction_dict,
    evidence_items=evidence_items,
)
```

따라서 호출부를 수정했다.

이 문제를 통해 함수 signature와 호출부를 맞추는 것이 중요하다는 점을 확인했다.

---

## 10. Day 12 핵심 개념

### 10.1 Artifact caching

Artifact caching은 모델 파일이나 SHAP background tensor처럼 반복적으로 사용되는 객체를 매 요청마다 다시 로드하지 않고 메모리에 재사용하는 구조다.

이번 프로젝트에서는 `lru_cache(maxsize=1)`를 사용했다.

```python
@lru_cache(maxsize=1)
def get_cached_failure_model_artifacts(...):
    ...
```

이 구조는 신입 포트폴리오 기준으로 이해하기 쉽고, 실제 운영 구조의 기본 개념도 설명할 수 있다.

---

### 10.2 Hard failure와 soft failure

Day 12의 핵심 설계는 실패 종류를 나눈 것이다.

```text
hard failure
= API 핵심 기능을 수행할 수 없는 실패
= model artifact 없음, prediction 실패

soft failure
= 핵심 prediction은 가능하지만 부가 설명만 실패
= SHAP artifact 실패, SHAP 계산 실패, global importance artifact 실패
```

제조 AI API에서 prediction은 핵심 기능이다. 따라서 prediction이 불가능하면 명확한 error를 반환해야 한다.

반면 SHAP와 global importance는 설명 기능이다. 실패하더라도 prediction 결과와 rule-based evidence는 반환할 수 있다.

---

### 10.3 Endpoint와 service 분리

FastAPI endpoint 안에 모든 로직을 넣으면 코드가 길어지고 테스트가 어려워진다.

Day 12에서는 다음처럼 분리했다.

```text
failure_agent_api.py
= HTTP endpoint

failure_agent_service.py
= prediction, SHAP, evidence, answer 조립

artifact_cache.py
= artifact loading/cache
```

이 구조는 API를 유지보수하기 좋게 만들고, 나중에 LangGraph Agent나 다른 API에서 service 로직을 재사용하기 쉽게 한다.

---

## 11. Day 12 완료 범위

Day 12에서 완료한 내용은 다음과 같다.

```text
model artifact caching 구조 생성
SHAP artifact caching 구조 생성
테스트용 cache clear 함수 생성
Failure Agent service layer 생성
endpoint 얇게 리팩토링
prediction 실패와 SHAP 실패 처리 기준 분리
SHAP artifact 실패 시 warning fallback 처리
global importance 실패 시 warning fallback 처리
include_shap=false일 때 SHAP 계산 생략
입력 validation error response 개선
API fallback 테스트 추가
monkeypatch 경로 수정
build_agent_answer 호출부 수정
```

---

## 12. Day 12 면접 답변 문장

Day 12에서는 FastAPI endpoint의 안정성을 개선했습니다.

Day 11까지는 SHAP local explanation을 실제 API에 연결했지만, 모델 artifact와 SHAP artifact를 요청마다 로드할 수 있는 구조였고, SHAP artifact가 없거나 계산에 실패했을 때 API 전체가 불안정해질 수 있었습니다.

그래서 Day 12에서는 model artifact와 SHAP artifact 로딩을 `artifact_cache.py`로 분리하고, `lru_cache`를 사용해 같은 artifact를 반복 로드하지 않도록 개선했습니다.

또한 prediction에 필수적인 model artifact 실패는 API error로 처리하고, SHAP local explanation이나 global importance처럼 부가 설명에 해당하는 기능이 실패한 경우에는 prediction과 rule-based evidence는 정상 반환하되 `warnings`에 실패 이유를 기록하도록 만들었습니다.

마지막으로 endpoint 안에 예외 처리와 evidence 조립 로직이 길게 들어가지 않도록 `failure_agent_service.py`로 분리했습니다. 이를 통해 API endpoint는 얇게 유지하고, 테스트 가능한 service 구조로 개선했습니다.
