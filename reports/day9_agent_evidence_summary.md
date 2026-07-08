# Day 9 Agent Evidence Integration Summary

## 1. Day 9 목표

Day 9의 목표는 Day 5의 단일 sample inference 결과, Day 5의 rule-based evidence, Day 8의 SHAP local explanation evidence를 Agent가 사용할 수 있는 하나의 evidence schema로 통합하는 것이다.

최종적으로 Agent answer builder가 evidence를 읽고 prediction summary, rule-based evidence, SHAP local evidence를 구분해서 답변할 수 있도록 구성했다.

---

## 2. 오늘 만든 파일

* src/agent/evidence_builder.py
* src/agent/answer_builder.py
* tests/test_evidence_builder.py
* tests/test_answer_builder.py
* scripts/run_agent_evidence_demo.py
* reports/day9_agent_evidence_summary.md

---

## 3. 핵심 개념

### 3.1 rule-based evidence

rule-based evidence는 사람이 정한 제조 기준으로 입력값을 해석한 참고 근거다.

예를 들어 Tool wear가 높거나 Torque가 높은 경우, 제조 기준상 점검이 필요하다고 표시할 수 있다.

하지만 이것은 모델 내부 판단 근거가 아니다.

정확한 표현:

```
입력값을 rule 기준으로 확인했을 때 Tool wear가 점검 신호로 표시되었습니다.
```

부정확한 표현:

```
모델은 Tool wear 때문에 고장이라고 판단했습니다.
```

---

### 3.2 permutation importance

permutation importance는 전체 test set 기준으로 모델 성능에 중요한 feature를 설명한다.

특정 feature 값을 무작위로 섞었을 때 f1-score가 얼마나 떨어지는지를 기준으로 중요도를 계산한다.

이것은 global explanation이다.

따라서 개별 sample 하나의 직접적인 예측 이유로 말하면 안 된다.

정확한 표현:

```
전체 test set 기준으로 Torque [Nm]는 모델 성능에 중요한 feature로 나타났습니다.
```

부정확한 표현:

```
이 sample은 Torque 때문에 고장으로 예측되었습니다.
```

---

### 3.3 SHAP local evidence

SHAP local evidence는 특정 sample 하나에서 각 feature가 모델 output에 어느 방향으로 기여했는지를 설명한다.

현재 FailureMLP는 마지막에 Sigmoid가 없는 binary classification 모델이므로 raw model output은 probability가 아니라 logit이다.

따라서 Day 8에서 계산한 SHAP value는 probability 기준이 아니라 logit 기준 contribution이다.

정확한 표현:

```
SHAP 기준으로 Torque [Nm]는 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
```

부정확한 표현:

```
모델은 Torque [Nm] 때문에 고장이라고 판단했습니다.
```

---

## 4. AgentEvidence schema

Day 9에서는 AgentEvidence dataclass를 만들었다.

주요 필드는 다음과 같다.

* evidence_id
* evidence_type
* source
* title
* summary
* feature
* value
* direction
* contribution
* importance
* severity
* metadata

evidence_type은 evidence의 해석 종류를 구분한다.

* prediction_summary
* rule_based
* shap_local
* global_importance

source는 evidence가 어디에서 나온 것인지 구분한다.

* model_prediction
* rule_engine
* shap
* permutation_importance

이렇게 구분한 이유는 Agent가 답변할 때 서로 다른 성격의 evidence를 섞어 말하지 않도록 하기 위해서다.

---

## 5. evidence_builder.py 역할

src/agent/evidence_builder.py는 Day 5, Day 6, Day 8에서 나온 결과를 AgentEvidence 구조로 변환한다.

주요 함수는 다음과 같다.

* build_prediction_summary_evidence
* convert_rule_based_evidence
* convert_shap_local_evidence
* convert_global_importance_evidence
* build_agent_evidence
* filter_evidence_by_type
* group_evidence_by_type

핵심 함수는 build_agent_evidence다.

이 함수는 prediction_result, shap_local_explanation, global_importance_items를 받아 하나의 evidence list로 합친다.

합치는 순서는 다음과 같다.

1. prediction summary evidence
2. rule-based evidence
3. SHAP local evidence
4. optional global importance evidence

중요한 점은 evidence list는 하나로 합치지만, evidence_type과 source를 통해 의미는 분리한다는 것이다.

---

## 6. answer_builder.py 역할

src/agent/answer_builder.py는 evidence list를 읽고 최종 Agent answer를 만든다.

답변 문단은 다음처럼 나뉜다.

* 설비 고장 위험 예측 결과
* 입력값 기준 점검 신호
* 모델 설명 기준 SHAP 근거
* 전체 데이터 기준 참고 중요도
* 해석 시 주의점

이 구조 덕분에 rule-based evidence와 SHAP evidence가 섞이지 않는다.

rule-based evidence는 입력값 기준 점검 신호로 설명한다.

SHAP evidence는 모델 output에 대한 feature contribution으로 설명한다.

global importance는 전체 test set 기준 참고 중요도로 설명한다.

---

## 7. 테스트 내용

Day 9에서는 다음 내용을 테스트했다.

* prediction summary evidence 생성
* rule-based evidence 변환
* SHAP local evidence 변환
* source와 evidence_type 분리 유지
* evidence type별 grouping
* answer builder가 rule evidence와 SHAP evidence를 다른 문단으로 출력하는지 확인
* 잘못된 단정 표현이 포함되지 않는지 확인

특히 answer test에서는 다음과 같은 표현이 들어가지 않도록 확인했다.

* 때문에 고장
* 원인입니다

이 테스트는 Agent 답변이 feature를 실제 고장 원인으로 단정하지 않도록 하기 위한 안전장치다.

---

## 8. Day 9 실행 명령

테스트 실행:

```
pytest tests/test_evidence_builder.py -v
pytest tests/test_answer_builder.py -v
```

데모 스크립트 실행:

```
python -m scripts.run_agent_evidence_demo
```

직접 실행 방식은 피하는 것이 좋다.

```
python scripts/run_agent_evidence_demo.py
```

이 방식은 프로젝트 루트 import 문제가 날 수 있다.

---

## 9. Day 9 결과

Day 9를 통해 Agent가 사용할 수 있는 evidence 통합 구조를 만들었다.

이제 Day 5의 inference 결과와 Day 8의 SHAP explanation 결과를 하나의 Agent 응답으로 연결할 수 있다.

특히 rule-based evidence, SHAP local evidence, global importance를 각각 source와 evidence_type으로 구분했기 때문에 Agent 답변에서 서로 다른 근거를 섞어 말하지 않을 수 있다.

결과적으로 Agent는 다음처럼 안전하게 답변할 수 있다.

```
모델은 현재 sample의 고장 probability를 높게 예측했습니다.
입력값 기준으로는 Tool wear와 Torque가 rule 기반 점검 신호로 표시되었습니다.
SHAP 기준으로는 Torque와 Tool wear가 모델의 고장 위험 logit을 높이는 방향으로 작용했습니다.
다만 이 feature들이 실제 고장의 물리적 원인이라고 단정할 수는 없습니다.
```

---

## 10. 오늘 코드에서 가장 중요한 포인트

Day 9의 핵심은 다음 한 문장으로 정리할 수 있다.

```
AgentEvidence 하나로 통합하되, evidence_type과 source로 해석을 분리한다.
```

즉, 구조는 하나로 합치지만 의미는 섞지 않는다.

각 evidence의 의미는 다음과 같다.

* rule_based
  제조 rule 기준 입력값 점검 신호

* shap_local
  개별 sample 기준 모델 output contribution

* global_importance
  전체 test set 기준 모델 민감도

* prediction_summary
  모델의 최종 probability, threshold, prediction, risk_level 요약

---

## 11. 면접 답변

Day 9에서는 모델 예측 결과와 설명 근거를 Agent가 사용할 수 있는 evidence schema로 통합했습니다.

기존 Day 5에서는 단일 sample inference 결과와 rule-based evidence를 만들었습니다. 이 rule-based evidence는 입력값이 사람이 정한 제조 기준에서 위험해 보이는지를 표시하는 참고 근거였고, 모델 내부 판단 근거는 아니었습니다.

Day 6에서는 permutation importance를 통해 전체 test set 기준으로 모델이 어떤 feature에 민감한지 확인했습니다. 다만 permutation importance는 global explanation이기 때문에 개별 sample 하나의 예측 이유를 직접 설명하지는 못합니다.

Day 8에서는 SHAP DeepExplainer를 사용해 개별 sample 기준 local explanation을 계산했습니다. 현재 FailureMLP는 마지막에 Sigmoid가 없는 모델이므로 SHAP value는 probability가 아니라 logit 기준 contribution으로 해석했습니다.

Day 9에서는 이 세 종류의 근거를 하나의 AgentEvidence schema로 통합했습니다. evidence_type과 source를 분리해 prediction summary, rule-based evidence, SHAP local evidence, global importance를 구분했고, answer_builder에서는 이들을 각각 다른 문단으로 출력하도록 만들었습니다.

이를 통해 Agent가 답변할 때 “이 feature 때문에 고장”이라고 단정하지 않고, “입력값 기준 점검 신호”와 “모델 기준 위험 예측을 높이는 방향”을 구분해서 설명할 수 있게 했습니다.

---

## 12. Day 9 완료 기준

Day 9는 다음 기준을 만족하면 완료로 본다.

* src/agent/evidence_builder.py 작성
* src/agent/answer_builder.py 작성
* tests/test_evidence_builder.py 작성
* tests/test_answer_builder.py 작성
* scripts/run_agent_evidence_demo.py 작성
* pytest 통과
* demo script 실행 성공
* reports/day9_agent_evidence_summary.md 작성
* rule-based evidence와 SHAP evidence의 차이를 면접 답변으로 설명 가능
* feature를 실제 고장 원인으로 단정하지 않는 표현 정리 완료
