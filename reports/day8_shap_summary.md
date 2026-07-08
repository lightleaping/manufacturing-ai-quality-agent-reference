# Day 8 SHAP Explanation Summary

## 1. Day 8 목표

Day 8의 목표는 PyTorch 기반 `FailureMLP` 모델에 SHAP를 실제로 적용하여, 개별 AI4I sample 하나에 대한 feature contribution을 계산하고, 이를 Day 7에서 만든 local explanation schema와 Agent evidence 형식에 연결하는 것이다.

Day 6에서는 permutation importance를 통해 전체 test set 기준 feature importance를 계산했다.

Day 7에서는 SHAP를 바로 적용하기 전에 개별 sample 설명 결과를 담기 위한 구조를 먼저 만들었다.

Day 8에서는 실제 SHAP value를 계산하여 이 구조에 연결했다.

Day 8에서 다룬 핵심 주제는 다음과 같다.

```text
SHAP 기본 개념
SHAP value, expected value, background data
logit, sigmoid, probability, SHAP value의 차이
PyTorch MLP 모델에 SHAP DeepExplainer 적용
Day 7 LocalFeatureContribution 구조와 연결
Agent evidence 형식으로 변환
```

---

## 2. Day 8에서 만든 파일

Day 8에서 추가하거나 수정한 파일은 다음과 같다.

```text
src/interpretability/shap_explainer.py
tests/test_shap_explainer.py
scripts/run_shap_explanation.py
reports/day8_shap_summary.md
```

각 파일의 역할은 다음과 같다.

```text
src/interpretability/shap_explainer.py
= SHAP value 계산, 방향 해석, LocalExplanationResult 변환 담당

tests/test_shap_explainer.py
= SHAP 계산 코드와 local explanation 연결 구조 테스트

scripts/run_shap_explanation.py
= 실제 저장 모델과 실제 sample 기준 SHAP explanation 실행

reports/day8_shap_summary.md
= Day 8 학습 내용, 실행 결과, 해석, 면접 답변 정리
```

---

## 3. SHAP 기본 개념

SHAP는 모델 예측 결과를 feature별 contribution으로 나누어 설명하는 방법이다.

쉽게 말하면 다음 질문에 답한다.

```text
평균적인 기준 상태와 비교했을 때,
이 sample의 각 feature는 모델 출력을 어느 방향으로 얼마나 움직였는가?
```

SHAP에서 중요한 개념은 다음과 같다.

```text
expected value
= background data 기준 평균 모델 출력값

SHAP value
= 각 feature가 모델 출력을 움직인 정도

background data
= expected value를 계산하기 위한 기준 sample 집합

positive SHAP value
= 모델 출력을 높이는 방향으로 작용한 feature contribution

negative SHAP value
= 모델 출력을 낮추는 방향으로 작용한 feature contribution
```

---

## 4. logit, sigmoid, probability, SHAP value 차이

현재 `FailureMLP`는 마지막 layer에 Sigmoid가 없다.

따라서 모델의 raw output은 probability가 아니라 logit이다.

```text
model(sample) = logit
```

고장 확률은 추론 단계에서 sigmoid를 적용해서 만든다.

```text
probability = sigmoid(logit)
```

최종 예측은 probability와 threshold를 비교해서 만든다.

```text
prediction = 1 if probability >= threshold else 0
```

이번 Day 8에서 계산한 SHAP value는 모델의 raw output을 설명한다.

현재 모델의 raw output은 logit이므로, 이번 SHAP value는 probability 기준 contribution이 아니라 logit 기준 contribution이다.

즉, SHAP value는 probability 자체가 아니다.

정확한 표현은 다음과 같다.

```text
SHAP 기준으로 이 feature는 고장 위험 logit을 높이는 방향으로 작용했다.
```

부정확한 표현은 다음과 같다.

```text
모델은 이 feature 때문에 고장이라고 판단했다.
```

---

## 5. SHAP 버전과 가상환경 이슈

처음에는 `shap==0.52.0` 설치를 시도했지만, 현재 환경과 맞지 않아 설치가 실패했다.

이후 프로젝트 가상환경을 `.venv`로 맞추고, `shap==0.51.0`을 설치하여 진행했다.

```text
사용한 SHAP 버전: shap==0.51.0
사용한 가상환경: .venv
```

중요한 점은 패키지 설치와 pytest 실행이 같은 Python 가상환경에서 이루어져야 한다는 것이다.

```text
shap을 .venv에 설치했는데
pytest가 .venv-1에서 실행되면
shap을 찾지 못해 테스트가 skip될 수 있다.
```

환경 확인 명령은 다음과 같다.

```bash
python -c "import sys; print(sys.executable)"
python -c "import shap; print(shap.__version__)"
python -m pip show shap
```

---

## 6. 테스트 결과

`tests/test_shap_explainer.py`를 작성한 뒤 아래 명령을 실행했다.

```bash
pytest tests/test_shap_explainer.py -v
```

처음에는 SHAP가 현재 pytest 환경에 설치되어 있지 않아 테스트가 skip되었다.

이후 `.venv`로 전환하고 SHAP를 설치한 뒤 테스트를 다시 실행했고, 테스트가 통과했다.

이 테스트는 모델 성능을 검증하는 테스트가 아니다.

테스트의 목적은 다음과 같다.

```text
SHAP background tensor가 정상 생성되는지 확인
SHAP 반환값 shape normalize가 정상인지 확인
FailureMLP + SHAP DeepExplainer 연결이 되는지 확인
SHAP value가 LocalFeatureContribution으로 변환되는지 확인
LocalExplanationResult가 생성되는지 확인
```

---

## 7. 기존 함수 재사용 구조

Day 8 스크립트에서는 Day 5에서 만든 기존 함수를 재사용하는 방향으로 수정했다.

처음에는 스크립트 안에서 직접 `scaler.transform()`을 호출하려고 했지만, 다음 에러가 발생했다.

```text
ValueError: The feature names should match those that were passed during fit.
Feature names unseen at fit time:
- Type
```

이 에러의 의미는 다음과 같다.

```text
저장된 scaler는 Type 컬럼 없이 fit되었는데,
스크립트에서 Type까지 포함해서 scaler.transform()을 직접 호출했다.
```

따라서 `scripts/run_shap_explanation.py`에서는 scaler를 직접 만지지 않고, Day 5에서 만든 함수를 재사용하도록 수정했다.

재사용한 주요 함수는 다음과 같다.

```text
load_failure_model_artifacts
validate_raw_sample
normalize_type_value
build_single_sample_dataframe
scale_single_sample_dataframe
dataframe_to_single_tensor
calculate_risk_level
predict_failure_from_artifacts
```

이렇게 수정한 이유는 다음과 같다.

```text
Day 5 단일 sample 추론 흐름과
Day 8 SHAP explanation 흐름이
같은 전처리, 같은 scaling, 같은 threshold, 같은 risk_level 기준을 공유해야 하기 때문이다.
```

---

## 8. 실제 실행 명령

Day 8 SHAP explanation 실행 명령은 다음과 같다.

```bash
python -m scripts.run_shap_explanation
```

모듈 방식으로 실행한 이유는 프로젝트 루트 기준으로 `src` 패키지를 안정적으로 import하기 위해서다.

```text
python scripts/run_shap_explanation.py
```

위 방식으로 실행하면 환경에 따라 `ModuleNotFoundError: No module named 'src'`가 발생할 수 있다.

따라서 프로젝트 루트에서 아래처럼 실행하는 것이 더 안전하다.

```bash
python -m scripts.run_shap_explanation
```

---

## 9. 실제 실행 결과

실제 저장 모델 기준 SHAP explanation 실행 결과는 다음과 같다.

```text
[INFO] Day 8 SHAP explanation started
[INFO] artifact_dir: models\failure_mlp
[INFO] threshold   : 0.7
[INFO] features    : ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Type']
[INFO] background_source_tensor shape: (300, 6)
[INFO] background_tensor shape       : (100, 6)
```

설명에 사용한 raw sample은 다음과 같다.

```text
Air temperature [K]: 303.0
Process temperature [K]: 312.5
Rotational speed [rpm]: 1380.0
Torque [Nm]: 62.0
Tool wear [min]: 220.0
Type: L
```

Day 5 추론 흐름 기준 결과는 다음과 같다.

```text
probability        : 0.9930
threshold          : 0.7000
prediction         : 1
risk_level         : HIGH
recommended_action : 고장 위험이 높습니다. 설비 점검 및 생산 조건 확인을 권장합니다.
```

SHAP 입력 tensor 기준 모델 출력도 동일한 결과를 보였다.

```text
logit       : 4.9506
probability : 0.9930
prediction  : 1
risk_level  : HIGH
```

즉, Day 5 추론 흐름과 Day 8 SHAP 설명 흐름이 같은 sample에 대해 같은 예측 결과를 공유하고 있다.

---

## 10. Local explanation summary

실행 결과 생성된 local explanation summary는 다음과 같다.

```text
모델은 이 sample을 고장 위험으로 예측했습니다.
예측 probability는 0.9930입니다.
SHAP 기준으로 영향이 큰 feature는 Torque [Nm], Tool wear [min], Process temperature [K]입니다.
```

여기서 중요한 점은 “영향이 큰 feature”는 SHAP value의 절댓값 기준이라는 것이다.

즉, positive 방향뿐 아니라 negative 방향으로 크게 작용한 feature도 영향이 큰 feature로 볼 수 있다.

---

## 11. feature별 SHAP 해석

### 11.1 Torque [Nm]

```text
feature          : Torque [Nm]
value            : 62.0
reference_value  : 40.0033625
SHAP value       : +5.1592
direction        : positive
global_importance: 0.3309
```

해석:

```text
Torque [Nm]는 이 sample에서 모델의 고장 위험 logit을 가장 크게 높이는 방향으로 작용했다.
```

현재 Torque 값은 train 평균값보다 높고, SHAP value도 positive로 나왔다.

따라서 이 sample에서는 Torque가 모델의 고장 위험 예측을 높이는 데 크게 기여한 feature로 볼 수 있다.

---

### 11.2 Tool wear [min]

```text
feature          : Tool wear [min]
value            : 220.0
reference_value  : 107.685
SHAP value       : +2.8238
direction        : positive
global_importance: 0.1213
```

해석:

```text
Tool wear [min]는 이 sample에서 모델의 고장 위험 logit을 높이는 방향으로 작용했다.
```

현재 Tool wear 값은 train 평균값보다 높다.

SHAP value도 positive이므로, 이 sample에서는 Tool wear가 고장 위험 예측을 높이는 방향으로 기여했다.

---

### 11.3 Process temperature [K]

```text
feature          : Process temperature [K]
value            : 312.5
reference_value  : 310.0060625
SHAP value       : -1.2535
direction        : negative
global_importance: 0.1651
```

해석:

```text
Process temperature [K]는 이 sample에서 모델의 고장 위험 logit을 낮추는 방향으로 작용했다.
```

주의할 점은, Process temperature 값이 평균보다 높다고 해서 반드시 고장 위험을 높이는 방향으로 작용해야 하는 것은 아니라는 점이다.

SHAP는 사람이 정한 rule이 아니라, 현재 모델이 background data와 비교했을 때 이 sample에서 어떻게 반응했는지를 보여준다.

---

### 11.4 Air temperature [K]

```text
feature          : Air temperature [K]
value            : 303.0
reference_value  : 300.00545
SHAP value       : +1.1895
direction        : positive
global_importance: 0.2725
```

해석:

```text
Air temperature [K]는 이 sample에서 모델의 고장 위험 logit을 높이는 방향으로 작용했다.
```

Air temperature는 Day 6 permutation importance에서도 중요도가 높게 나왔고, 이번 sample에서도 positive contribution을 보였다.

---

### 11.5 Rotational speed [rpm]

```text
feature          : Rotational speed [rpm]
value            : 1380.0
reference_value  : 1539.356875
SHAP value       : -0.8260
direction        : negative
global_importance: 0.2292
```

해석:

```text
Rotational speed [rpm]는 이 sample에서 모델의 고장 위험 logit을 낮추는 방향으로 작용했다.
```

Rotational speed가 평균보다 낮지만, 이 sample에서 모델은 해당 feature를 고장 위험 logit을 낮추는 방향으로 반영했다.

이 역시 실제 원인 단정이 아니라 현재 모델의 local contribution으로 해석해야 한다.

---

## 12. Day 6 global explanation과 Day 8 local explanation의 차이

Day 6의 permutation importance는 global explanation이다.

```text
전체 test set 기준으로 어떤 feature가 모델 성능에 중요한지 설명한다.
```

Day 8의 SHAP explanation은 local explanation이다.

```text
특정 sample 하나에서 각 feature가 모델 출력에 어떤 방향으로 기여했는지 설명한다.
```

이번 결과에서 Day 6과 Day 8을 연결하면 다음처럼 볼 수 있다.

```text
Torque [Nm]
= Day 6 global importance 1위
= Day 8 sample에서도 가장 큰 positive SHAP contribution

Tool wear [min]
= Day 6 global importance는 상대적으로 낮았지만
= Day 8 sample에서는 큰 positive SHAP contribution

Process temperature [K]
= Day 6 global importance는 중간 정도
= Day 8 sample에서는 negative SHAP contribution
```

따라서 global importance와 local SHAP explanation은 서로 다른 질문에 답한다.

```text
Day 6:
전체적으로 모델이 어떤 feature에 민감한가?

Day 8:
이 sample 하나에서는 어떤 feature가 모델 출력을 어느 방향으로 움직였는가?
```

둘은 경쟁 관계가 아니라 보완 관계다.

---

## 13. Agent evidence 연결 결과

Day 7에서 만든 `format_local_explanation_as_evidence` 함수를 사용해 `LocalExplanationResult`를 Agent evidence 형식으로 변환했다.

변환된 evidence에는 다음 정보가 포함된다.

```text
source
evidence_type
explanation_method
prediction
probability
threshold
risk_level
feature
value
reference_value
contribution
direction
global_importance
message
```

예시 evidence는 다음과 같다.

```text
source: local_explanation
evidence_type: feature_contribution
explanation_method: shap_deep_explainer_logit
feature: Torque [Nm]
value: 62.0
reference_value: 40.0033625
contribution: 5.1592
direction: positive
global_importance: 0.3309
message: SHAP 기준으로 Torque [Nm]는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

이 구조를 통해 Agent는 단순히 “고장 위험입니다”라고 답하는 것이 아니라, 어떤 feature들이 모델 출력에 어떤 방향으로 기여했는지 함께 설명할 수 있다.

---

## 14. 주의할 표현

SHAP value를 해석할 때는 표현을 조심해야 한다.

정확한 표현:

```text
SHAP 기준으로 Torque [Nm]는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

부정확한 표현:

```text
모델은 Torque [Nm] 때문에 고장이라고 판단했습니다.
```

이유:

```text
SHAP value는 feature contribution이지 실제 고장의 물리적 원인 단정이 아니다.
```

또한 이번 구현의 SHAP value는 probability 기준이 아니라 logit 기준이다.

따라서 다음 표현도 피해야 한다.

```text
Torque가 고장 확률을 5.1592만큼 높였습니다.
```

올바른 표현은 다음과 같다.

```text
Torque는 모델의 고장 위험 logit을 5.1592만큼 높이는 방향으로 기여했습니다.
```

---

## 15. Day 8에서 배운 핵심 개념

Day 8에서 배운 핵심 개념은 다음과 같다.

```text
SHAP value는 probability가 아니라 feature contribution이다.

현재 FailureMLP는 Sigmoid가 없으므로 SHAP value는 logit 기준 contribution이다.

positive SHAP value는 모델 출력을 높이는 방향이다.

negative SHAP value는 모델 출력을 낮추는 방향이다.

background data는 SHAP expected value 계산의 기준이다.

SHAP explanation은 실제 원인 단정이 아니라 모델 출력 해석이다.

Day 5 추론 전처리와 Day 8 SHAP 전처리는 같은 함수를 재사용해야 한다.

scaler.transform을 직접 호출하면 scaler가 fit된 컬럼과 맞지 않아 오류가 날 수 있다.

global explanation과 local explanation은 서로 다른 목적을 가진다.
```

---

## 16. 현재 한계

현재 Day 8 구현에는 다음 한계가 있다.

```text
1. SHAP value는 logit 기준이라 probability 변화량으로 직접 해석할 수 없다.

2. background data 선택에 따라 expected value와 SHAP value가 달라질 수 있다.

3. SHAP explanation은 모델 출력 해석이지 실제 고장의 물리적 원인 분석은 아니다.

4. 현재 Type은 L/M/H를 0/1/2로 단순 mapping했기 때문에, 향후 one-hot encoding으로 개선할 수 있다.

5. 현재 SHAP 설명은 sample 하나 기준이며, 여러 sample batch 설명이나 시각화는 아직 구현하지 않았다.

6. Agent 답변에 실제로 통합하는 API 또는 workflow 연결은 다음 단계에서 진행해야 한다.
```

---

## 17. 개선 방향

이후 개선 방향은 다음과 같다.

```text
1. Type encoding을 one-hot encoding으로 개선한다.

2. SHAP explanation을 Agent 응답 생성 단계에 통합한다.

3. rule-based evidence, global importance, local SHAP evidence를 구분해서 보여준다.

4. SHAP summary plot 또는 waterfall plot을 추가해 시각화한다.

5. background sample 선택 기준을 실험한다.

6. logit 기준 SHAP와 probability 기준 설명 방식의 차이를 추가로 실험한다.

7. FastAPI endpoint에서 prediction 결과와 explanation 결과를 함께 반환하도록 확장한다.
```

---

## 18. 면접 답변

Day 8에서는 PyTorch 기반 FailureMLP 모델에 SHAP를 실제로 적용해 개별 예측 설명 기능을 구현했습니다.

Day 6에서 구현한 permutation importance는 전체 test set 기준으로 어떤 feature가 모델 성능에 중요한지 보여주는 global explanation이었습니다. 하지만 개별 sample 하나가 왜 고장 위험으로 예측되었는지는 직접 설명하지 못한다는 한계가 있었습니다.

그래서 Day 7에서 LocalFeatureContribution과 LocalExplanationResult 구조를 먼저 설계했고, Day 8에서는 SHAP DeepExplainer를 사용해 개별 sample의 feature별 contribution을 계산한 뒤 이 구조에 연결했습니다.

특히 현재 FailureMLP는 마지막에 Sigmoid가 없는 binary classification 모델이기 때문에 raw model output은 probability가 아니라 logit입니다. 따라서 이번 SHAP value는 probability 자체가 아니라 logit 기준 contribution으로 해석했습니다.

실제 실행 결과, sample의 probability는 0.9930으로 threshold 0.7000을 넘었고, prediction은 1, risk_level은 HIGH로 나왔습니다. SHAP 기준으로는 Torque와 Tool wear가 고장 위험 logit을 높이는 방향으로 크게 작용했습니다.

또한 Day 8 스크립트에서는 새로운 전처리나 scaling 로직을 직접 만들지 않고, Day 5에서 만든 `load_failure_model_artifacts`, `scale_single_sample_dataframe`, `predict_failure_from_artifacts`, `calculate_risk_level` 함수를 재사용했습니다. 이를 통해 단일 추론 결과와 SHAP explanation 결과가 같은 전처리 기준과 같은 위험도 기준을 공유하도록 만들었습니다.

결과적으로 Day 8을 통해 rule-based evidence, permutation importance 기반 global explanation, SHAP 기반 local explanation을 구분해서 사용할 수 있게 되었고, 이후 제조 AI Agent의 evidence schema에 개별 예측 설명을 통합할 수 있는 기반을 만들었습니다.
