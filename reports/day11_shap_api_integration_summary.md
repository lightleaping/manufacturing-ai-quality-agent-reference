# Day 11 - SHAP API Integration Summary

## 1. Day 11 목표

Day 11의 목표는 Day 10에서 placeholder로 남겨두었던 SHAP local explanation을 FastAPI endpoint에 실제로 연결하는 것이었다.

Day 10에서는 `POST /agent/failure-prediction` endpoint가 다음 항목까지 반환했다.

* prediction
* probability
* threshold
* risk_level
* recommended_action
* prediction_summary evidence
* rule_based evidence
* global_importance evidence
* Agent answer

하지만 Day 10에서는 `include_shap=true`로 요청해도 실제 SHAP local explanation이 API 응답에 포함되지 않았다.

Day 11에서는 이 부분을 개선하여, `include_shap=true`일 때 실제 SHAP local explanation을 계산하고 `evidence_type="shap_local"`로 API 응답에 포함되도록 구현했다.

---

## 2. Day 11에서 만든 파일

Day 11에서 새로 만든 파일은 다음과 같다.

```text
src/interpretability/shap_artifacts.py
src/interpretability/shap_runtime.py
scripts/build_shap_artifacts.py
```

그리고 기존 파일을 수정했다.

```text
src/api/failure_agent_api.py
src/agent/evidence_builder.py
tests/test_api_failure_agent.py
```

---

## 3. 운영 환경에 가까운 구조로 변경한 이유

처음에는 API 요청이 들어올 때마다 AI4I CSV를 로드하고, train data를 전처리한 뒤 SHAP background tensor를 생성하는 방식도 가능했다.

하지만 이 방식은 운영 환경에 적합하지 않다.

이유는 다음과 같다.

1. API 요청마다 CSV를 다시 로드하면 응답 시간이 길어진다.
2. 요청마다 background sample을 다시 만들면 설명 기준이 흔들릴 수 있다.
3. API endpoint가 데이터 로드, 전처리, background 생성, SHAP 계산까지 모두 담당하면 구조가 복잡해진다.
4. 운영 환경에서는 모델 학습/배포 준비 단계와 API 추론 단계를 분리하는 것이 더 자연스럽다.

따라서 Day 11에서는 운영 환경에 더 가까운 구조로 변경했다.

변경한 구조는 다음과 같다.

```text
모델 준비 단계:
SHAP background artifact 생성

API 요청 단계:
저장된 SHAP artifact 로드
→ sample 전처리
→ prediction
→ include_shap=true이면 SHAP local explanation 계산
→ evidence/answer 반환
```

---

## 4. SHAP artifact 구조

Day 11에서는 `models/failure_mlp/` 아래에 SHAP 설명용 artifact를 추가했다.

```text
models/failure_mlp/
  model.pt
  scaler.joblib
  metadata.json
  shap_background.pt
  shap_reference_values.json
  global_importance.json
```

각 파일의 의미는 다음과 같다.

### model.pt

학습된 PyTorch FailureMLP 모델의 weight와 bias를 저장한 파일이다.

### scaler.joblib

train set 기준으로 fit된 StandardScaler를 저장한 파일이다.

API 추론과 SHAP 설명 모두 동일한 scaler를 사용해야 한다.

### metadata.json

모델 입력 차원, hidden dimension, dropout rate, threshold, feature columns 등의 메타데이터를 저장한 파일이다.

### shap_background.pt

SHAP DeepExplainer에 들어갈 background tensor다.

이 값은 API 요청마다 새로 만들지 않고, 미리 생성해서 저장한다.

### shap_reference_values.json

feature별 train set 평균값을 저장한 파일이다.

SHAP evidence에서 현재 sample 값과 비교할 기준값으로 사용한다.

### global_importance.json

Day 6 permutation importance 결과를 저장한 파일이다.

이는 전체 test set 기준 feature 중요도이며, 개별 sample의 직접 원인은 아니다.

---

## 5. SHAP artifact 생성 스크립트

Day 11에서는 다음 스크립트를 만들었다.

```text
scripts/build_shap_artifacts.py
```

실행 명령은 다음과 같다.

```bash
python -m scripts.build_shap_artifacts
```

실행 결과 다음 파일들이 생성되었다.

```text
models/failure_mlp/shap_background.pt
models/failure_mlp/shap_reference_values.json
models/failure_mlp/global_importance.json
```

실제 실행 로그는 다음과 같았다.

```text
[INFO] SHAP artifact build started
[INFO] artifact_dir    : models\failure_mlp
[INFO] feature_columns : ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type']
[INFO] background_source_tensor shape: (300, 6)
[INFO] shap_background_tensor shape  : (100, 6)
[INFO] saved: models/failure_mlp/shap_background.pt
[INFO] saved: models/failure_mlp/shap_reference_values.json
[INFO] saved: models/failure_mlp/global_importance.json
[INFO] SHAP artifact build completed
```

---

## 6. shap_artifacts.py 역할

`src/interpretability/shap_artifacts.py`는 SHAP artifact 저장/로드를 담당한다.

주요 구성은 다음과 같다.

```text
ShapArtifacts
save_shap_artifacts
load_shap_artifacts
```

`ShapArtifacts`는 다음 값을 묶어 관리한다.

```text
background_tensor
reference_values
global_importance_map
```

이 파일을 분리한 이유는 API endpoint가 직접 `torch.load`, `json.load`를 호출하지 않도록 하기 위해서다.

즉, artifact 로딩 책임은 `shap_artifacts.py`가 담당하고, API endpoint는 로드된 객체를 사용하는 역할만 한다.

---

## 7. shap_runtime.py 역할

`src/interpretability/shap_runtime.py`는 API 실행 시점에서 sample 하나에 대한 SHAP local explanation을 생성하는 helper를 담당한다.

주요 함수는 다음과 같다.

```text
normalize_sample_for_model_input
sample_to_model_tensor
build_raw_sample_values
build_global_importance_items_from_map
build_shap_local_explanation_for_sample
```

특히 핵심 함수는 다음이다.

```text
build_shap_local_explanation_for_sample
```

이 함수는 `include_shap=True`일 때만 실제 SHAP local explanation을 생성한다.

`include_shap=False`이면 `None`을 반환하고, SHAP 계산을 생략한다.

---

## 8. API 수정 내용

`src/api/failure_agent_api.py`에서는 Day 11에 다음 내용을 반영했다.

### 8.1 ARTIFACT_DIR 추가

```python
ARTIFACT_DIR = Path("models/failure_mlp")
```

모델 artifact와 SHAP artifact를 같은 디렉터리에서 관리한다.

### 8.2 request.machine_type 수정

처음에는 `_build_raw_sample_from_request()` 안에서 아래처럼 작성되어 있었다.

```python
"Type": request.type
```

하지만 Pydantic schema 내부 필드명은 `machine_type`이다.

API JSON에서는 `"type": "L"`로 입력하지만, Python 객체에서는 `request.machine_type`으로 접근해야 한다.

그래서 다음처럼 수정했다.

```python
"Type": request.machine_type
```

정리하면 변환 흐름은 다음과 같다.

```text
JSON "type"
→ request.machine_type
→ raw_sample["Type"]
→ Day 5 inference pipeline
```

### 8.3 SHAP artifact 로드 연결

Day 11에서는 API에서 저장된 SHAP artifact를 로드하도록 수정했다.

```python
if request.include_shap or request.include_global_importance:
    shap_artifacts = load_shap_artifacts(ARTIFACT_DIR)
```

### 8.4 global_importance_items 변수명 통일

중간에 `global_importance_evidence`와 `global_importance_items`가 섞이면서 `UnboundLocalError`가 발생했다.

이를 해결하기 위해 변수명을 `global_importance_items`로 통일했다.

```python
global_importance_items: list[dict[str, Any]] = []
```

그리고 `build_agent_evidence()` 호출부도 다음처럼 맞췄다.

```python
build_agent_evidence(
    prediction_result=prediction_dict,
    shap_local_explanation=shap_local_explanation,
    global_importance_items=global_importance_items,
    shap_top_n=5,
)
```

---

## 9. 테스트 수정 내용

Day 11부터 API가 SHAP artifact와 SHAP runtime helper를 호출하게 되었으므로, API layer 테스트도 수정했다.

`tests/test_api_failure_agent.py`에서는 실제 모델과 실제 SHAP 계산을 실행하지 않도록 다음 함수들을 monkeypatch했다.

```text
load_failure_model_artifacts
predict_failure_from_artifacts
load_shap_artifacts
build_global_importance_items_from_map
build_shap_local_explanation_for_sample
build_agent_evidence
build_agent_answer
```

이 테스트의 목적은 실제 추론 정확도가 아니라 API request/response 구조 검증이다.

실제 모델 추론은 Day 5 테스트에서 확인하고, 실제 SHAP 계산은 Day 8과 Day 11 Swagger 실행에서 확인한다.

---

## 10. 발생한 문제와 해결

### 문제 1. request.type 오류

에러 메시지:

```text
AttributeError: 'FailurePredictionRequest' object has no attribute 'type'
```

원인:

API JSON 입력 필드는 `"type"`이지만, Pydantic 모델 내부 필드명은 `machine_type`이었다.

해결:

```python
"Type": request.machine_type
```

---

### 문제 2. ARTIFACT_DIR 미정의

에러 메시지:

```text
NameError: name 'ARTIFACT_DIR' is not defined
```

원인:

Day 11 코드에서 `ARTIFACT_DIR`를 사용했지만 `failure_agent_api.py` 안에 상수를 정의하지 않았다.

해결:

```python
ARTIFACT_DIR = Path("models/failure_mlp")
```

---

### 문제 3. global_importance_evidence 변수명 오류

에러 메시지:

```text
UnboundLocalError: cannot access local variable 'global_importance_evidence'
```

원인:

어떤 곳에서는 `global_importance_items`, 다른 곳에서는 `global_importance_evidence`라는 이름을 사용했다.

해결:

변수명을 `global_importance_items`로 통일했다.

---

### 문제 4. SHAP summary 중복

초기 Swagger 응답에서는 SHAP summary가 다음처럼 중복되었다.

```text
SHAP 기준으로 Torque [Nm]=62.0는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
SHAP contribution=5.1592입니다.
SHAP 기준으로 Torque [Nm]는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
SHAP value=5.1592
```

원인:

`summary` 문장에 `contribution.reason`을 다시 붙였기 때문이다.

해결:

summary에서는 핵심 문장만 만들고, 원본 reason은 metadata 안에 보존했다.

---

### 문제 5. contribution 변수명 오류

에러 메시지:

```text
AttributeError: 'float' object has no attribute 'feature'
```

원인:

`contribution` 변수는 float인데, 실수로 `contribution.feature`, `contribution.value`, `contribution.contribution`처럼 객체로 접근했다.

해결:

```python
summary = (
    f"SHAP 기준으로 {feature}={value}는 "
    f"{direction_text}으로 작용했습니다. "
    f"SHAP contribution={contribution:.4f}입니다."
)
```

---

### 문제 6. direction_text 중복

초기 수정 후 summary가 다음처럼 출력되었다.

```text
모델의 고장 위험 logit을 모델의 고장 위험 logit을 높이는 방향 방향으로 작용했습니다.
```

원인:

`_direction_to_korean()` 함수가 이미 `"모델의 고장 위험 logit을 높이는 방향"`이라는 긴 문장 조각을 반환하는데, summary에서 다시 `"모델의 고장 위험 logit을 ... 방향"`을 붙였기 때문이다.

해결:

summary를 다음처럼 단순화했다.

```python
summary = (
    f"SHAP 기준으로 {feature}={value}는 "
    f"{direction_text}으로 작용했습니다. "
    f"SHAP contribution={contribution:.4f}입니다."
)
```

---

## 11. 테스트 결과

최종 테스트는 모두 통과했다.

실행 명령:

```bash
pytest tests/test_evidence_builder.py -v
pytest tests/test_answer_builder.py -v
pytest tests/test_api_failure_agent.py -v
```

결과:

```text
tests/test_evidence_builder.py 통과
tests/test_answer_builder.py 통과
tests/test_api_failure_agent.py 통과
```

---

## 12. Swagger 실제 실행 결과

서버 실행 명령:

```bash
uvicorn src.api.main:app --reload
```

Swagger URL:

```text
http://127.0.0.1:8000/docs
```

요청 JSON:

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

응답 핵심:

```text
prediction  : 1
probability : 0.9929707646369934
threshold   : 0.7
risk_level  : HIGH
recommended_action : 고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다.
```

응답에 포함된 evidence type:

```text
prediction_summary
rule_based
shap_local
global_importance
```

---

## 13. SHAP local evidence 결과

Swagger 실제 응답에서 SHAP local evidence는 다음과 같이 반환되었다.

### Torque [Nm]

```text
value        : 62.0
direction    : positive
contribution : 5.1592
importance   : 0.3309
summary      : SHAP 기준으로 Torque [Nm]=62.0는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

### Tool wear [min]

```text
value        : 220.0
direction    : positive
contribution : 2.8238
importance   : 0.1213
summary      : SHAP 기준으로 Tool wear [min]=220.0는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

### Process temperature [K]

```text
value        : 312.5
direction    : negative
contribution : -1.2535
importance   : 0.1651
summary      : SHAP 기준으로 Process temperature [K]=312.5는 모델의 고장 위험 logit을 낮추는 방향으로 작용했습니다.
```

### Air temperature [K]

```text
value        : 303.0
direction    : positive
contribution : 1.1895
importance   : 0.2725
summary      : SHAP 기준으로 Air temperature [K]=303.0는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

### Rotational speed [rpm]

```text
value        : 1380.0
direction    : negative
contribution : -0.8260
importance   : 0.2292
summary      : SHAP 기준으로 Rotational speed [rpm]=1380.0는 모델의 고장 위험 logit을 낮추는 방향으로 작용했습니다.
```

---

## 14. 최종 해석

이번 sample에 대해 모델은 고장 probability를 약 99.30%로 예측했다.

운영 threshold 0.7 기준으로 probability가 threshold보다 높기 때문에 prediction은 1이다.

risk_level은 HIGH이며, 권장 조치는 설비 점검 및 생산 조건 확인이다.

입력값 기준 rule-based evidence에서는 Tool wear와 Torque가 제조 rule 기준 점검 신호로 표시되었다.

SHAP local evidence에서는 Torque, Tool wear, Air temperature가 모델의 고장 위험 logit을 높이는 방향으로 작용했다.

반대로 Process temperature와 Rotational speed는 모델의 고장 위험 logit을 낮추는 방향으로 작용했다.

global importance에서는 전체 test set 기준으로 Torque, Air temperature, Rotational speed가 중요한 feature로 표시되었다.

---

## 15. 중요한 해석 주의점

SHAP value는 probability가 아니다.

현재 FailureMLP는 마지막에 Sigmoid가 없기 때문에 모델의 raw output은 logit이다.

따라서 이번 SHAP contribution은 probability 기준이 아니라 logit 기준 contribution이다.

정확한 표현:

```text
SHAP 기준으로 Torque [Nm]는 현재 sample에서 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

부정확한 표현:

```text
모델은 Torque [Nm] 때문에 고장이라고 판단했습니다.
```

또한 rule-based evidence, SHAP local evidence, global importance는 서로 다른 의미를 갖는다.

```text
rule_based evidence
= 사람이 정한 제조 rule 기준으로 입력값이 점검 신호인지 설명

SHAP local evidence
= 현재 sample에서 feature가 모델 output에 기여한 방향 설명

global importance
= 전체 test set 기준 모델 민감도 설명
```

---

## 16. Day 11 완료 범위

Day 11에서 완료한 범위는 다음과 같다.

```text
SHAP artifact 저장/로드 구조 생성 완료
SHAP background tensor 사전 생성 완료
reference values 저장 완료
global importance artifact 저장 완료
FastAPI에서 SHAP artifact 로드 완료
include_shap 옵션 실제 동작 완료
include_global_importance 옵션 실제 동작 완료
shap_local evidence API 응답 포함 완료
Agent answer의 SHAP 섹션 출력 완료
pytest 통과 완료
Swagger 실제 실행 성공
```

---

## 17. Day 11 미완료 / 이후 개선할 점

Day 11에서 핵심 기능은 완료되었지만, 이후 개선할 수 있는 점은 다음과 같다.

1. SHAP artifact를 매 요청마다 로드하지 않고, 서버 시작 시 한 번만 로드하도록 캐싱할 수 있다.
2. global_importance.json을 현재는 Day 6 결과를 직접 저장했지만, 이후 permutation importance 실행 결과에서 자동 생성하도록 개선할 수 있다.
3. SHAP 계산은 무거울 수 있으므로 timeout, fallback, warning 처리를 추가할 수 있다.
4. production 환경에서는 model version과 shap artifact version을 함께 관리해야 한다.
5. 입력값 validation error를 더 사용자 친화적으로 정리할 수 있다.
6. Type feature는 현재 L/M/H를 0/1/2로 mapping했지만, 이후 one-hot encoding으로 개선할 수 있다.

---

## 18. 면접 답변 문장

Day 11 작업은 다음처럼 설명할 수 있다.

```text
Day 11에서는 Day 10에서 placeholder로 남겨두었던 SHAP local explanation을 FastAPI API에 실제로 연결했습니다.

처음에는 API 요청마다 train data를 다시 로드해서 SHAP background를 생성하는 방식도 가능했지만, 운영 환경에 가깝게 만들기 위해 모델 배포 준비 단계에서 SHAP background tensor, reference values, global importance 결과를 artifact로 미리 저장하도록 분리했습니다.

API에서는 저장된 SHAP artifact를 로드하고, include_shap=True일 때만 SHAP local explanation을 계산하도록 구성했습니다.

또한 prediction_summary, rule_based evidence, shap_local evidence, global_importance evidence를 하나의 evidence list로 통합하되, evidence_type과 source를 통해 의미를 분리했습니다.

SHAP value는 probability가 아니라 현재 FailureMLP의 logit 기준 contribution이므로, 실제 고장의 물리적 원인으로 단정하지 않고 모델 출력에 대한 feature별 기여 방향으로 해석하도록 answer와 limitations에 명시했습니다.
```

---

## 19. Day 11 결론

Day 11에서는 FastAPI inference endpoint에 SHAP local explanation을 실제로 연결했다.

이제 API는 단순히 고장 여부만 반환하지 않고, 다음 정보를 함께 제공한다.

```text
모델 예측 요약
입력값 기준 rule evidence
개별 sample 기준 SHAP local evidence
전체 test set 기준 global importance
해석 시 주의점
```

이를 통해 기존 `manufacturing-mcp-agent`보다 더 설명 가능한 제조 AI Agent 구조에 가까워졌다.

Day 11은 완료되었다.
