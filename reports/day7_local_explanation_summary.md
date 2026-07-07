# Day 7 Local Explanation Summary

## 1. Day 7 목표

Day 7의 목표는 개별 sample 하나에 대해 모델 예측을 설명할 수 있는 구조를 설계하는 것이다.

Day 6에서는 permutation importance를 구현하여 전체 test set 기준으로 어떤 feature가 모델 성능에 중요한지 확인했다.

Day 6 실행 결과에서는 다음 순서로 feature importance가 높게 나왔다.

```text
1. Torque [Nm]
2. Air temperature [K]
3. Rotational speed [rpm]
4. Process temperature [K]
5. Tool wear [min]
6. Type
```

그러나 permutation importance는 전체 test set 기준 설명이다.

즉, 전체 데이터에서 어떤 feature가 모델 성능에 중요한지는 알려주지만, 특정 sample 하나가 왜 고장 위험으로 예측되었는지는 직접 설명하지 못한다.

따라서 Day 7에서는 global explanation과 local explanation의 차이를 정리하고, 이후 SHAP를 적용하기 전에 개별 sample explanation 결과를 담을 수 있는 구조를 먼저 설계했다.

---

## 2. Day 7에서 만든 파일

```text
src/interpretability/local_explanation.py
tests/test_local_explanation.py
reports/day7_local_explanation_summary.md
```

---

## 3. Day 7 핵심 개념

## 3.1 Global Explanation

Global explanation은 전체 데이터셋 기준으로 모델이 어떤 feature에 민감하게 반응하는지 설명하는 방식이다.

Day 6에서 구현한 permutation importance가 여기에 해당한다.

예를 들어 Day 6 결과에서 `Torque [Nm]`의 importance가 가장 높게 나왔다는 것은, 전체 test set에서 Torque 값을 섞었을 때 모델의 f1-score가 가장 크게 떨어졌다는 뜻이다.

즉, 현재 모델은 전체적으로 Torque 정보를 중요하게 사용하고 있다고 해석할 수 있다.

하지만 이것은 개별 sample 하나의 예측 이유가 아니다.

예를 들어 어떤 sample에서 모델이 고장 확률을 높게 예측했다고 해서, 반드시 Torque 때문에 그렇게 판단했다고 말할 수는 없다.

---

## 3.2 Local Explanation

Local explanation은 특정 sample 하나에 대해 feature들이 예측에 어떤 방향으로 영향을 주었는지 설명하는 방식이다.

예를 들어 어떤 sample의 예측 결과가 다음과 같다고 가정한다.

```text
probability: 0.82
threshold  : 0.70
prediction : 1
risk_level : HIGH
```

이 경우 local explanation은 다음과 같은 질문에 답하기 위한 구조다.

```text
이 sample은 왜 고장 위험으로 예측되었는가?
어떤 feature가 고장 위험을 높이는 방향으로 작용했는가?
어떤 feature가 고장 위험을 낮추는 방향으로 작용했는가?
```

예시 설명은 다음과 같다.

```text
Torque [Nm] 값은 고장 위험 예측을 높이는 방향으로 작용했다.
Tool wear [min] 값도 고장 위험 예측을 높이는 방향으로 작용했다.
Rotational speed [rpm] 값은 고장 위험 예측을 낮추는 방향으로 작용했다.
```

---

## 3.3 Permutation Importance가 개별 예측 이유가 아닌 이유

Permutation importance는 다음 방식으로 계산된다.

```text
importance = baseline_score - permuted_score
```

즉, 특정 feature 값을 무작위로 섞었을 때 전체 모델 성능이 얼마나 떨어지는지를 측정한다.

이 방식은 전체 test set 기준으로 feature의 중요도를 확인하는 데 유용하다.

하지만 특정 sample 하나에서 그 feature가 예측을 높였는지, 낮췄는지는 직접 알려주지 않는다.

따라서 다음 표현은 정확하다.

```text
현재 모델은 전체 test set 기준으로 Torque [Nm]에 가장 민감하게 반응했다.
```

하지만 다음 표현은 부정확하다.

```text
이 sample은 Torque [Nm] 때문에 고장으로 예측되었다.
```

개별 sample의 예측 이유를 말하려면 local explanation 또는 SHAP 같은 방법이 필요하다.

---

## 4. Rule-based Evidence, Permutation Importance, Local Explanation 차이

| 구분                     | 기준                | 설명 대상        | 예시                           |
| ---------------------- | ----------------- | ------------ | ---------------------------- |
| rule-based evidence    | 사람이 정한 제조 기준      | 개별 입력값       | Tool wear가 200 이상이면 위험 신호    |
| permutation importance | 전체 test set 성능 변화 | 전체 모델        | Torque를 섞으면 f1-score가 크게 떨어짐 |
| local explanation      | 개별 sample의 예측 변화  | 특정 sample 하나 | 이 sample에서 Torque가 위험 예측을 높임 |

중요한 점은 세 가지 evidence가 서로 다른 역할을 한다는 것이다.

rule-based evidence는 모델 내부 판단 근거가 아니라, 입력값을 제조 기준으로 해석한 참고 근거다.

permutation importance는 전체 데이터 기준으로 모델이 어떤 feature를 중요하게 사용하는지 보여준다.

local explanation은 특정 sample 하나에서 feature가 예측을 어느 방향으로 움직였는지 설명한다.

---

## 5. Day 7 구현 내용

Day 7에서는 실제 SHAP 라이브러리를 바로 붙이지 않고, SHAP 결과를 담을 수 있는 구조를 먼저 만들었다.

핵심 파일은 다음과 같다.

```text
src/interpretability/local_explanation.py
```

이 파일에서는 다음 구조와 함수를 만들었다.

```text
LocalFeatureContribution
LocalExplanationResult
determine_contribution_direction
create_local_feature_contribution
build_default_contribution_reason
get_top_local_contributions
build_local_explanation_result
build_local_explanation_summary
format_local_explanation_as_evidence
```

---

## 6. LocalFeatureContribution

`LocalFeatureContribution`은 개별 sample에서 feature 하나가 예측에 어떤 방향으로 작용했는지를 담는 dataclass다.

주요 필드는 다음과 같다.

```text
feature
value
contribution
direction
reference_value
global_importance
reason
```

각 필드의 의미는 다음과 같다.

| 필드                | 의미                             |
| ----------------- | ------------------------------ |
| feature           | 설명 대상 feature 이름               |
| value             | 현재 sample에서 해당 feature의 실제 값   |
| contribution      | 개별 예측에 대한 feature 기여도          |
| direction         | 위험 예측을 높였는지, 낮췄는지, 중립인지        |
| reference_value   | 비교 기준값                         |
| global_importance | Day 6 permutation importance 값 |
| reason            | 사람이 읽을 수 있는 설명 문장              |

예시:

```python
LocalFeatureContribution(
    feature="Torque [Nm]",
    value=65.0,
    contribution=0.31,
    direction="increases_risk",
    reference_value=40.0,
    global_importance=0.3309,
    reason="Torque [Nm] 값은 현재 샘플의 고장 위험 예측을 높이는 방향으로 작용했습니다.",
)
```

---

## 7. contribution과 direction

Day 7에서는 contribution 값을 기준으로 direction을 자동으로 정하도록 설계했다.

```text
contribution > 0
= 고장 위험을 높이는 방향

contribution < 0
= 고장 위험을 낮추는 방향

contribution ≈ 0
= 영향이 거의 없음
```

이를 코드에서는 다음 함수가 담당한다.

```python
determine_contribution_direction(
    contribution: float,
    neutral_tolerance: float = 1e-6,
)
```

`neutral_tolerance`가 필요한 이유는 float 계산 때문이다.

실제 계산에서는 0처럼 보이는 값도 아주 작은 소수값으로 표현될 수 있다.

따라서 정확히 `contribution == 0`만 확인하지 않고, 일정 범위 안이면 `neutral`로 처리한다.

---

## 8. LocalExplanationResult

`LocalExplanationResult`는 개별 sample 하나에 대한 전체 설명 결과를 담는 dataclass다.

주요 필드는 다음과 같다.

```text
prediction
probability
threshold
risk_level
explanation_method
contributions
summary
limitations
```

각 필드의 의미는 다음과 같다.

| 필드                 | 의미                             |
| ------------------ | ------------------------------ |
| prediction         | threshold 기준 최종 예측값            |
| probability        | 모델이 예측한 고장 확률                  |
| threshold          | prediction 판단 기준               |
| risk_level         | LOW / MEDIUM / HIGH / UNKNOWN  |
| explanation_method | local_proxy / shap / manual    |
| contributions      | feature별 local contribution 목록 |
| summary            | 전체 설명 요약 문장                    |
| limitations        | 설명 방식의 한계                      |

예시:

```text
모델은 고장 확률을 0.8200으로 예측했고,
threshold 0.7000 이상이므로 고장 위험으로 판단했습니다.
local explanation 기준 주요 위험 증가 feature는 Torque [Nm], Tool wear [min]입니다.
risk_level은 HIGH입니다.
```

---

## 9. Agent Evidence Schema 변환

Day 7의 중요한 목표는 local explanation을 Agent 답변에 넣을 수 있는 evidence 구조로 변환하는 것이다.

이를 위해 다음 함수를 만들었다.

```python
format_local_explanation_as_evidence(
    result: LocalExplanationResult,
    top_k: int = 3,
) -> list[dict[str, Any]]
```

반환 구조 예시는 다음과 같다.

```python
[
    {
        "source": "local_explanation",
        "evidence_type": "prediction_summary",
        "explanation_method": "local_proxy",
        "prediction": 1,
        "probability": 0.82,
        "threshold": 0.7,
        "risk_level": "HIGH",
        "message": "모델은 고장 확률을 0.8200으로 예측했고..."
    },
    {
        "source": "local_explanation",
        "evidence_type": "feature_contribution",
        "explanation_method": "local_proxy",
        "feature": "Torque [Nm]",
        "value": 65.0,
        "reference_value": 40.0,
        "contribution": 0.31,
        "direction": "increases_risk",
        "global_importance": 0.3309,
        "message": "Torque [Nm] 값은 현재 샘플의 고장 위험 예측을 높이는 방향으로 작용했습니다."
    }
]
```

이 구조를 사용하면 이후 Agent 답변에서 evidence를 다음처럼 조합할 수 있다.

```text
모델은 고장 확률을 0.82로 예측했고, threshold 0.70을 넘었기 때문에 고장 위험으로 판단했습니다.

개별 sample 설명 기준으로는 Torque [Nm]와 Tool wear [min]가 고장 위험 예측을 높이는 방향으로 작용했습니다.

또한 Day 6의 전체 test set 기준 permutation importance에서도 Torque [Nm]는 가장 중요한 feature로 나타났습니다.
```

---

## 10. 테스트 내용

테스트 파일은 다음과 같다.

```text
tests/test_local_explanation.py
```

테스트한 내용은 다음과 같다.

```text
1. contribution 값에 따라 direction이 올바르게 정해지는지 확인
2. create_local_feature_contribution 함수가 direction과 reason을 자동 생성하는지 확인
3. local contribution이 절댓값 기준으로 정렬되는지 확인
4. 위험을 높이는 feature만 필터링할 수 있는지 확인
5. LocalExplanationResult가 summary와 limitations를 생성하는지 확인
6. LocalExplanationResult를 Agent evidence schema로 변환할 수 있는지 확인
```

실행 명령어:

```powershell
pytest tests/test_local_explanation.py -v
```

예상 결과:

```text
tests/test_local_explanation.py::test_determine_contribution_direction PASSED
tests/test_local_explanation.py::test_create_local_feature_contribution_sets_direction PASSED
tests/test_local_explanation.py::test_get_top_local_contributions_sorts_by_absolute_value PASSED
tests/test_local_explanation.py::test_get_top_local_contributions_can_filter_risk_increasing PASSED
tests/test_local_explanation.py::test_build_local_explanation_result_creates_summary PASSED
tests/test_local_explanation.py::test_format_local_explanation_as_evidence PASSED
```

---

## 11. Day 7에서 아직 하지 않은 것

Day 7에서는 SHAP 라이브러리를 실제로 적용하지 않았다.

이유는 다음과 같다.

```text
1. SHAP를 바로 붙이면 구조보다 라이브러리 사용법에 집중하게 된다.
2. 먼저 Agent evidence에 넣을 공통 schema가 필요하다.
3. local explanation 결과를 어떤 형식으로 저장하고 전달할지 정해야 한다.
4. Day 8에서 실제 SHAP value를 이 구조에 연결하면 된다.
```

따라서 Day 7의 결과는 SHAP 이전 단계의 설계 작업이다.

정확한 표현은 다음과 같다.

```text
Day 7에서는 SHAP를 실제 적용하기 전에, 개별 sample explanation 결과를 담을 수 있는 local explanation schema를 설계했다.
```

부정확한 표현은 다음과 같다.

```text
Day 7에서 SHAP 기반 설명을 완성했다.
```

---

## 12. 기존 manufacturing-mcp-agent와 비교

기존 `manufacturing-mcp-agent`는 evidence를 반환하긴 했지만, evidence가 주로 rule 기반이거나 단순 조회 결과에 가까웠다.

즉, 다음 질문에 대한 답은 제한적이었다.

```text
모델이 왜 이 sample을 위험하다고 예측했는가?
어떤 feature가 예측을 높였는가?
feature 중요도와 개별 sample 설명을 구분했는가?
```

Day 7의 구조는 이 한계를 보완하기 위한 준비 단계다.

기존 프로젝트와 비교하면 다음과 같이 설명할 수 있다.

```text
기존 프로젝트는 규칙 기반 evidence를 중심으로 답변을 만들었다.
새 레퍼런스 프로젝트에서는 rule-based evidence, global importance, local explanation을 구분하여 evidence schema를 확장하고 있다.
이를 통해 단순히 위험 여부만 반환하는 것이 아니라, 모델 예측 결과와 설명 근거를 함께 제공하는 제조 AI Agent 구조로 발전시킬 수 있다.
```

---

## 13. Day 7 핵심 주의사항

Day 7에서 가장 중요한 주의사항은 다음과 같다.

```text
1. global importance와 local contribution을 혼동하지 않는다.
2. rule-based evidence를 모델 내부 판단 근거라고 말하지 않는다.
3. local_proxy를 SHAP 결과라고 말하지 않는다.
4. contribution이 양수라는 것은 위험 예측을 높이는 방향이라는 뜻이다.
5. contribution이 음수라는 것은 위험 예측을 낮추는 방향이라는 뜻이다.
6. probability와 contribution은 같은 값이 아니다.
7. threshold는 prediction을 결정하는 기준이고, contribution은 feature별 영향 방향이다.
```

---

## 14. Day 7 면접 답변

Day 7에서는 Day 6에서 구현한 permutation importance의 한계를 보완하기 위해 local explanation 구조를 설계했습니다.

Permutation importance는 전체 test set 기준으로 어떤 feature가 모델 성능에 중요한지를 보여주는 global explanation입니다. 하지만 개별 sample 하나가 왜 고장 위험으로 예측되었는지를 직접 설명하지는 못합니다.

그래서 Day 7에서는 개별 sample 기준 feature contribution을 담을 수 있는 `LocalFeatureContribution`과 `LocalExplanationResult` 구조를 만들었습니다. 이 구조는 feature별 contribution, 방향, 실제 입력값, reference value, global importance를 함께 담을 수 있도록 설계했습니다.

현재는 SHAP를 바로 붙이기 전 단계이기 때문에 `local_proxy` 방식으로 구조를 먼저 검증했습니다. 이후 Day 8에서는 SHAP 값을 실제로 계산해서 이 local explanation 구조에 연결하고, 최종적으로 Agent 답변의 evidence schema에 통합할 계획입니다.

---

## 15. Day 8 계획

Day 8에서는 Day 7에서 만든 local explanation schema에 실제 SHAP 값을 연결한다.

예상 작업은 다음과 같다.

```text
1. SHAP 기본 개념 정리
2. PyTorch MLP 모델에 SHAP 적용 가능 방식 확인
3. background sample / reference data 선택
4. 개별 sample에 대한 SHAP value 계산
5. SHAP value를 LocalFeatureContribution 구조에 매핑
6. Agent evidence schema로 변환
7. rule-based evidence + SHAP evidence를 함께 사용하는 답변 구조 설계
```

Day 8의 핵심 목표는 다음과 같다.

```text
개별 sample 하나에 대해 모델이 어떤 feature 때문에 고장 위험을 높게 또는 낮게 예측했는지 설명할 수 있도록 만든다.
```
