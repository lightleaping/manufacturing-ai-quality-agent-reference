# Day 5. 모델 저장/로드와 단일 샘플 추론 구조 정리

## 1. Day 5 목표

Day 5의 목표는 Day 1~4에서 학습하고 평가한 설비 고장 예측 모델을 실제 서비스나 Agent에서 사용할 수 있는 형태로 바꾸는 것이다.

Day 4까지는 다음 흐름을 만들었다.

```text
AI4I 데이터 로드
↓
전처리
↓
feature scaling
↓
MLP 모델 학습
↓
평가
↓
threshold 비교
↓
운영 후보 threshold 선택
```

하지만 이 상태는 아직 실험 코드에 가깝다.

실제 FastAPI, AI Agent, 제조 AI 솔루션에서 사용하려면 학습된 모델을 매번 다시 학습하는 것이 아니라, 저장된 모델을 불러와 새로운 입력에 대해 추론할 수 있어야 한다.

따라서 Day 5에서는 다음 구조를 만들었다.

```text
학습된 model 저장
↓
scaler 저장
↓
threshold 및 feature metadata 저장
↓
저장된 artifact 로드
↓
raw input 입력
↓
학습 때와 동일한 전처리 및 scaling
↓
model inference
↓
probability 계산
↓
threshold 비교
↓
prediction / risk_level / recommended_action / evidence 반환
```

---

## 2. Day 5에서 만든 파일

Day 5에서 만든 주요 파일은 다음과 같다.

```text
src/inference/model_artifacts.py
src/inference/predict_failure.py

tests/test_model_artifacts.py
tests/test_predict_failure.py

scripts/run_predict_failure.py
```

각 파일의 역할은 다음과 같다.

```text
model_artifacts.py
- 학습된 model, scaler, metadata를 저장하고 다시 불러오는 파일

predict_failure.py
- 단일 raw sample을 입력받아 고장 확률, 예측 결과, 위험도, evidence를 반환하는 파일

test_model_artifacts.py
- model/scaler/metadata 저장 및 로드가 정상적으로 동작하는지 검증하는 테스트

test_predict_failure.py
- 단일 sample inference 흐름이 정상적으로 연결되는지 검증하는 테스트

run_predict_failure.py
- 실제 저장된 artifact를 불러와 정상 샘플과 위험 샘플을 비교 추론하는 실행 스크립트
```

---

## 3. Model artifact란 무엇인가

model artifact는 학습이 끝난 뒤 추론에 필요한 파일들을 의미한다.

이번 프로젝트에서는 다음 3개를 저장했다.

```text
models/failure_mlp/model.pt
models/failure_mlp/scaler.joblib
models/failure_mlp/metadata.json
```

각 파일의 의미는 다음과 같다.

```text
model.pt
- PyTorch 모델의 학습된 weight와 bias가 저장된 파일
- model.state_dict() 방식으로 저장

scaler.joblib
- train set에 fit된 StandardScaler 객체
- 추론 시에도 학습 때와 같은 평균/표준편차 기준으로 scaling하기 위해 필요

metadata.json
- threshold
- input_dim
- hidden_dim
- dropout_rate
- feature_columns
같은 추론 설정 정보 저장
```

모델만 저장하면 충분하지 않다.

이유는 모델은 단순한 숫자 tensor만 입력으로 받기 때문이다.

실제 raw input을 모델에 넣으려면 다음 정보가 필요하다.

```text
1. 어떤 feature를 사용할 것인가
2. feature 순서는 어떻게 맞출 것인가
3. Type 값을 어떻게 encoding할 것인가
4. numeric feature를 어떤 scaler로 변환할 것인가
5. probability를 어떤 threshold로 prediction으로 바꿀 것인가
```

따라서 실제 추론 구조에서는 model, scaler, metadata를 함께 관리해야 한다.

---

## 4. `state_dict`로 모델을 저장한 이유

PyTorch 모델은 크게 두 가지 방식으로 저장할 수 있다.

```text
1. 모델 객체 전체 저장
2. state_dict 저장
```

이번 프로젝트에서는 `state_dict` 방식을 사용했다.

```python
torch.save(model.state_dict(), paths.model_path)
```

`state_dict`는 모델의 구조 전체가 아니라, 학습된 weight와 bias 값만 담고 있는 dictionary다.

로드할 때는 같은 모델 구조를 다시 만든 뒤, 저장된 weight와 bias를 채운다.

```python
model = FailureMLP(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    dropout_rate=dropout_rate,
)

state_dict = torch.load(paths.model_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

이 방식은 모델 구조와 학습된 파라미터를 명확히 분리할 수 있다는 장점이 있다.

또한 `map_location="cpu"`를 사용하면 GPU에서 저장한 모델도 CPU 환경에서 안전하게 불러올 수 있다.

---

## 5. 추론 흐름

Day 5에서 만든 단일 샘플 추론 흐름은 다음과 같다.

```text
raw_sample
↓
validate_raw_sample
↓
build_single_sample_dataframe
↓
normalize_type_value
↓
scale_single_sample_dataframe
↓
dataframe_to_single_tensor
↓
model inference
↓
torch.sigmoid(logits)
↓
probability
↓
threshold 비교
↓
prediction
↓
risk_level
↓
recommended_action
↓
evidence
```

즉, raw input을 바로 모델에 넣지 않는다.

반드시 학습 때와 같은 feature column 순서, 같은 Type encoding, 같은 scaler를 적용한 뒤 모델에 입력한다.

---

## 6. Type 변환

AI4I 데이터의 `Type` 컬럼은 원래 문자열이다.

```text
L
M
H
```

Day 1 전처리에서는 이를 다음과 같이 mapping했다.

```text
L → 0
M → 1
H → 2
```

추론 시에도 같은 mapping을 사용해야 한다.

```python
def normalize_type_value(value: Any) -> int:
    if isinstance(value, str):
        type_mapping = {
            "L": 0,
            "M": 1,
            "H": 2,
        }
```

학습 때는 `L=0`, `M=1`, `H=2`로 넣었는데, 추론 때 다른 방식으로 encoding하면 모델 입력 의미가 달라진다.

따라서 학습과 추론의 전처리 방식은 반드시 일치해야 한다.

---

## 7. Feature column 순서가 중요한 이유

PyTorch 모델은 pandas DataFrame의 column 이름을 이해하지 않는다.

모델은 단순히 숫자 배열의 순서를 본다.

예를 들어 학습 때 feature 순서가 다음과 같았다고 하자.

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
Type
```

그런데 추론 때 순서가 바뀌면, 모델 입장에서는 전혀 다른 의미의 입력을 받게 된다.

예를 들어 `Torque [Nm]` 자리에 `Tool wear [min]` 값이 들어가면 모델 예측은 신뢰할 수 없다.

따라서 metadata에 저장된 `feature_columns` 순서대로 DataFrame을 만들어야 한다.

---

## 8. Scaling 방식

Day 4에서 `StandardScaler`를 적용했다.

중요한 원칙은 다음과 같다.

```text
scaler.fit()은 train set에만 한다.
test set과 inference input에는 transform()만 한다.
```

Day 5 추론에서도 저장된 scaler를 불러와 `transform()`만 사용한다.

```python
scaled_numeric_values = artifacts.scaler.transform(
    scaled_df[numeric_feature_columns]
)
```

현재 프로젝트에서는 `Type` 컬럼은 scaling하지 않는다.

이유는 `Type`은 숫자처럼 보이지만 실제로는 L/M/H를 0/1/2로 mapping한 범주형 feature이기 때문이다.

반면 다음 feature들은 numeric sensor feature로 보고 scaling한다.

```text
Air temperature [K]
Process temperature [K]
Rotational speed [rpm]
Torque [Nm]
Tool wear [min]
```

---

## 9. Logit, sigmoid, probability

모델의 마지막 출력은 probability가 아니라 logit이다.

Day 2에서 만든 `FailureMLP`는 마지막에 Sigmoid를 넣지 않았다.

이유는 학습 때 `BCEWithLogitsLoss`를 사용하기 때문이다.

따라서 추론 시에는 직접 sigmoid를 적용해야 한다.

```python
with torch.no_grad():
    logits = artifacts.model(input_tensor)
    probability = torch.sigmoid(logits).item()
```

정리하면 다음과 같다.

```text
model output
= logit

torch.sigmoid(logit)
= probability
```

probability는 0.0부터 1.0 사이 값이고, 고장일 가능성을 의미한다.

---

## 10. Threshold와 prediction

probability가 계산되면 threshold와 비교해 최종 prediction을 만든다.

```python
prediction = int(probability >= artifacts.threshold)
```

의미는 다음과 같다.

```text
probability >= threshold
→ prediction = 1
→ 고장 위험으로 판단

probability < threshold
→ prediction = 0
→ 정상으로 판단
```

이번 실행에서 저장된 threshold는 0.7000이었다.

정상 샘플 결과는 다음과 같았다.

```text
probability: 0.0384
threshold  : 0.7000
prediction : 0
risk_level : LOW
```

위험 샘플 결과는 다음과 같았다.

```text
probability: 0.9076
threshold  : 0.7000
prediction : 1
risk_level : HIGH
```

즉, 정상 샘플은 probability가 threshold보다 낮아 정상으로 판단되었고, 위험 샘플은 probability가 threshold보다 높아 고장 위험으로 판단되었다.

---

## 11. Risk level과 prediction의 차이

`prediction`과 `risk_level`은 비슷해 보이지만 역할이 다르다.

```text
prediction
- threshold 기준 최종 0/1 판단
- 0 = 정상
- 1 = 고장 위험

risk_level
- probability를 사람이 이해하기 쉬운 등급으로 표현
- LOW / MEDIUM / HIGH
```

현재 risk level 기준은 다음과 같다.

```text
probability >= 0.70 → HIGH
probability >= 0.40 → MEDIUM
그 외 → LOW
```

예를 들어 probability가 0.65이고 threshold가 0.70이면 prediction은 0이지만, risk_level은 MEDIUM이 될 수 있다.

이 경우 최종 고장 판단은 아니지만 모니터링이 필요한 상태라고 해석할 수 있다.

따라서 제조 AI에서는 prediction만 보여주는 것보다 risk_level을 함께 보여주는 것이 운영자에게 더 유용하다.

---

## 12. Recommended action

모델 예측 결과를 실제 운영자가 이해하려면 단순히 숫자만 보여주는 것보다 조치 문장이 필요하다.

그래서 Day 5에서는 `recommended_action`을 추가했다.

예시는 다음과 같다.

```text
LOW
→ 현재 입력 기준으로는 정상 범위로 판단됩니다.

MEDIUM
→ 즉시 고장으로 판단되지는 않지만, 상태 변화를 모니터링하세요.

HIGH
→ 고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다.
```

이번 위험 샘플에서는 다음 문장이 출력되었다.

```text
고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다.
```

이렇게 하면 모델 결과가 단순한 예측값에서 끝나지 않고, 실제 운영 의사결정에 연결된다.

---

## 13. Rule-based evidence

Day 5에서는 rule-based evidence를 추가했다.

현재 evidence는 모델 내부 설명이 아니다.

즉, 모델이 실제로 “Tool wear 때문에 고장이라고 판단했다”고 말하는 것은 아니다.

현재 evidence는 raw input 중 제조적으로 위험해 보이는 feature를 사람이 이해하기 쉽게 정리한 참고 근거다.

현재 rule 기준은 다음과 같다.

```text
Tool wear [min] >= 200
→ 공구 마모 시간이 높아 고장 위험 판단에 참고

Torque [Nm] >= 60
→ 토크 값이 높아 설비 부하 가능성 확인

Rotational speed [rpm] <= 1300
→ 회전 속도가 낮아 비정상 운전 가능성 확인

Process temperature - Air temperature >= 12
→ 공정 온도와 대기 온도의 차이가 커서 열적 이상 가능성 확인
```

정상 샘플에서는 특별한 위험 feature가 발견되지 않았다.

```text
feature: overall
value  : None
message: 현재 rule 기준에서 뚜렷한 위험 feature는 발견되지 않았습니다.
```

위험 샘플에서는 다음 evidence가 출력되었다.

```text
Tool wear [min] = 230.0
Torque [Nm] = 65.0
Rotational speed [rpm] = 1250.0
Process temperature [K] - Air temperature [K] = 16.0
```

이 evidence는 운영자가 입력 상태를 빠르게 이해하도록 돕는 역할을 한다.

---

## 14. Rule-based evidence의 한계

현재 rule-based evidence에는 한계가 있다.

가장 중요한 한계는 모델 내부 설명이 아니라는 점이다.

즉, 모델 probability가 높아진 이유를 직접 설명하는 것이 아니라, 입력값 중 사람이 보기에도 위험해 보이는 feature를 별도로 표시하는 것이다.

따라서 다음과 같은 표현은 부정확하다.

```text
모델은 Tool wear 때문에 고장이라고 판단했다.
```

현재 단계에서 더 정확한 표현은 다음과 같다.

```text
모델은 고장 probability를 높게 예측했고,
입력값을 rule 기준으로 확인했을 때 Tool wear, Torque, Rotational speed, 온도 차이가 위험 신호로 표시되었다.
```

이 차이를 구분하는 것이 중요하다.

---

## 15. 이후 확장 계획

이 프로젝트는 학습용 레퍼런스 프로젝트이므로 rule-based evidence에서 끝내지 않고, 이후 더 많은 설명 방법을 다룰 계획이다.

확장 순서는 다음과 같이 잡을 수 있다.

```text
1. rule-based evidence
2. feature importance
3. permutation importance
4. anomaly score
5. SHAP
6. evidence 통합 구조
```

각 방법의 의미는 다음과 같다.

```text
feature importance
- 모델 또는 데이터 기준으로 어떤 feature가 전체 예측에서 중요한지 확인

permutation importance
- 특정 feature 값을 섞었을 때 성능이 얼마나 떨어지는지 보고 중요도 판단

anomaly score
- 정상 패턴과 얼마나 다른지 점수화

SHAP
- 개별 예측에서 각 feature가 예측값을 높였는지 낮췄는지 설명
```

최종적으로는 다음과 같은 evidence 구조로 확장할 수 있다.

```text
evidence
├── rule_based_evidence
├── feature_importance_evidence
├── anomaly_score_evidence
└── shap_evidence
```

이렇게 하면 단순 고장 예측 모델을 설명 가능한 제조 AI Agent로 확장할 수 있다.

---

## 16. 테스트 결과

Day 5에서는 다음 테스트를 추가했다.

```text
tests/test_model_artifacts.py
tests/test_predict_failure.py
```

검증한 내용은 다음과 같다.

```text
model.pt 저장 여부
scaler.joblib 저장 여부
metadata.json 저장 여부
저장된 model/scaler/metadata 로드 여부
로드된 모델의 출력 일치 여부
Type 문자열 변환 여부
누락 feature 에러 처리 여부
risk_level 계산 여부
단일 sample inference 결과 반환 여부
recommended_action 반환 여부
rule-based evidence 반환 여부
```

전체 테스트는 다음과 같이 통과했다.

```text
28 passed
```

이후 inference 테스트가 추가되면서 다음 흐름도 검증했다.

```text
raw input
↓
artifact load
↓
scaling
↓
model inference
↓
probability
↓
prediction
↓
risk_level
↓
recommended_action
↓
evidence
```

---

## 17. Day 5 결론

Day 5에서는 실험용 학습 모델을 실제 추론 가능한 구조로 바꾸었다.

핵심은 단순히 model만 저장한 것이 아니라, scaler와 threshold, feature column 순서까지 함께 저장했다는 점이다.

또한 raw input을 학습 때와 같은 방식으로 변환하고, 저장된 model/scaler/metadata를 사용해 probability, prediction, risk_level을 반환하도록 만들었다.

마지막으로 recommended_action과 rule-based evidence를 추가해 모델 결과를 운영자가 이해할 수 있는 형태로 확장했다.

이를 통해 프로젝트는 단순한 모델 학습 코드에서 실제 AI Agent나 FastAPI 서비스로 연결할 수 있는 추론 구조를 갖추게 되었다.
