# Day 6 Interpretability Summary

## 1. Day 6 목표

Day 6의 목표는 모델 설명 가능성을 높이기 위해 feature importance와 permutation importance를 이해하고 구현하는 것이다.

기존 Day 5에서는 단일 샘플 inference 결과에 대해 rule-based evidence를 생성했다. 하지만 rule-based evidence는 입력값 중 제조 기준으로 위험해 보이는 feature를 표시하는 방식일 뿐, 모델 내부 판단 근거라고 말할 수는 없다.

따라서 Day 6에서는 모델 성능 기준으로 feature 중요도를 확인하기 위해 permutation importance를 구현했다.

---

## 2. Permutation Importance 개념

Permutation importance는 특정 feature 컬럼의 값을 무작위로 섞은 뒤 모델 성능이 얼마나 떨어지는지 측정하는 방법이다.

성능이 많이 떨어지는 feature는 모델이 해당 feature 정보에 더 민감하게 의존했을 가능성이 높다고 해석할 수 있다.

계산식은 다음과 같다.

```text
importance = baseline_score - permuted_score
```

예를 들어 원본 test set의 f1-score가 0.5043이고, Torque 컬럼을 섞은 뒤 f1-score가 0.1734라면 Torque의 importance는 다음과 같다.

```text
0.5043 - 0.1734 = 0.3309
```

즉, Torque 컬럼을 섞었을 때 f1-score가 평균 0.3309만큼 감소했으므로, 현재 모델은 Torque 정보에 가장 민감하게 반응하는 것으로 해석할 수 있다.

---

## 3. 실행 흐름

Day 6 permutation importance 실행 흐름은 다음과 같다.

```text
AI4I CSV 로드
→ Type encoding
→ train/test split
→ 저장된 scaler 로드
→ X_test scaling
→ 저장된 FailureMLP 로드
→ baseline f1 계산
→ feature별 permutation
→ feature별 importance 계산
→ model-based evidence 후보 생성
```

중요한 점은 test set에 scaler를 다시 fit하지 않고, Day 4~5에서 train set 기준으로 저장한 scaler를 그대로 불러와 transform만 적용했다는 것이다.

또한 feature column 순서는 metadata에 저장된 feature_columns 기준으로 고정했다. 모델은 column name을 직접 이해하는 것이 아니라 입력 숫자 배열의 순서에 의존하므로, feature 순서가 바뀌면 모델 입력의 의미가 달라질 수 있기 때문이다.

---

## 4. 실행 결과

실제 AI4I test set과 저장된 FailureMLP 모델로 permutation importance를 계산한 결과는 다음과 같다.

```text
baseline_score: 0.5043
threshold     : 0.7000
metric_name   : f1
```

feature별 중요도는 다음과 같다.

| rank | feature                 | importance | permuted_score | interpretation                                                       |
| ---: | ----------------------- | ---------: | -------------: | -------------------------------------------------------------------- |
|    1 | Torque [Nm]             |     0.3309 |         0.1734 | Torque를 섞었을 때 f1이 가장 크게 감소했다. 현재 모델이 가장 민감하게 사용하는 feature로 해석할 수 있다. |
|    2 | Air temperature [K]     |     0.2725 |         0.2319 | 대기 온도 정보도 모델 성능에 큰 영향을 주었다.                                          |
|    3 | Rotational speed [rpm]  |     0.2292 |         0.2752 | 회전 속도 정보 역시 고장 예측에 중요한 feature로 해석할 수 있다.                            |
|    4 | Process temperature [K] |     0.1651 |         0.3392 | 공정 온도도 일정 수준 모델 성능에 영향을 주었다.                                         |
|    5 | Tool wear [min]         |     0.1213 |         0.3830 | 공구 마모 시간도 모델이 참고한 feature지만, 이번 결과에서는 Torque나 온도, 회전 속도보다 낮게 나왔다.    |
|    6 | Type                    |     0.0186 |         0.4857 | Type을 섞어도 성능 감소가 작았다. 현재 모델은 Type 정보를 크게 활용하지 않았을 가능성이 있다.           |

---

## 5. 결과 해석

이번 결과에서 가장 중요한 feature는 Torque [Nm]였다. Torque 컬럼을 무작위로 섞었을 때 f1-score가 0.5043에서 0.1734로 크게 감소했다. 따라서 현재 모델은 test set 전체 기준으로 Torque 정보에 가장 민감하게 반응한다고 볼 수 있다.

Air temperature [K]와 Rotational speed [rpm]도 높은 중요도를 보였다. 이는 모델이 온도 조건과 회전 속도 정보를 고장 예측에 상당히 활용하고 있음을 의미한다.

반면 Type feature의 importance는 0.0186으로 가장 낮았다. 현재 프로젝트에서는 Type을 L/M/H에서 0/1/2로 단순 mapping했기 때문에, 모델이 범주형 정보를 충분히 의미 있게 활용하지 못했을 가능성이 있다. 이후 최종 프로젝트에서는 Type을 one-hot encoding으로 바꿔 범주형 변수 처리를 개선할 수 있다.

---

## 6. Rule-based Evidence와 Model-based Evidence 차이

Day 5의 rule-based evidence는 개별 입력 sample의 값 기준 설명이다.

예를 들어 Tool wear가 230이거나 Torque가 65이면, 제조 기준으로 위험해 보이는 feature로 표시한다.

반면 Day 6의 permutation importance는 test set 전체 기준 설명이다.

예를 들어 Torque를 섞었을 때 f1-score가 0.3309만큼 감소했다면, 모델이 전체 평가 데이터에서 Torque 정보에 크게 의존하고 있다고 해석할 수 있다.

따라서 두 evidence는 역할이 다르다.

```text
rule-based evidence
= 이 입력값에서 위험해 보이는 feature 설명

model-based evidence
= 전체 test set에서 모델 성능에 중요한 feature 설명
```

정확한 표현은 다음과 같다.

```text
모델은 고장 probability를 예측했고, 입력값을 rule 기준으로 확인했을 때 일부 feature가 위험 신호로 표시되었다. 또한 permutation importance 기준으로는 전체 test set에서 Torque, Air temperature, Rotational speed가 모델 성능에 큰 영향을 주는 feature로 확인되었다.
```

부정확한 표현은 다음과 같다.

```text
모델은 Torque 때문에 고장이라고 판단했다.
```

---

## 7. 한계

Permutation importance는 모델 구조와 무관하게 사용할 수 있다는 장점이 있다. PyTorch MLP뿐 아니라 RandomForest, XGBoost, 다른 딥러닝 모델에도 적용할 수 있다.

하지만 한계도 있다.

첫째, permutation importance는 test set 전체 기준 설명이므로 개별 sample 하나가 왜 고장으로 예측되었는지는 직접 설명하지 못한다.

둘째, feature 간 상관관계가 강한 경우 중요도가 왜곡될 수 있다. 예를 들어 Air temperature와 Process temperature가 서로 관련되어 있다면, 하나를 섞었을 때의 성능 하락만으로 실제 영향력을 완전히 설명하기 어렵다.

셋째, permutation은 무작위 과정이므로 반복 횟수와 random_state에 따라 결과가 조금 달라질 수 있다. 그래서 n_repeats를 여러 번 설정하고 평균과 표준편차를 함께 확인해야 한다.

---

## 8. 다음 확장 방향

이후에는 다음 순서로 설명 구조를 확장할 수 있다.

```text
1. rule-based evidence
2. permutation importance
3. feature importance summary
4. 개별 sample explanation
5. SHAP
6. Agent answer에 evidence 통합
```

최종적으로는 단순히 “고장 위험이 높다”고 말하는 모델이 아니라, 모델 예측값, rule-based evidence, model-based evidence를 함께 제공하는 제조 AI Agent로 확장할 수 있다.

---

## 9. 면접 답변

Day 6에서는 모델 설명 가능성을 높이기 위해 permutation importance를 구현했습니다.

기존 Day 5의 evidence는 rule-based 방식이어서 입력값 중 제조 기준으로 위험해 보이는 feature를 표시하는 구조였습니다. 다만 이것은 모델 내부 판단 근거라고 볼 수 없기 때문에, Day 6에서는 모델 성능 기준으로 feature 중요도를 확인하는 permutation importance를 추가했습니다.

Permutation importance는 특정 feature 값을 무작위로 섞은 뒤 f1-score가 얼마나 떨어지는지 측정하는 방법입니다. 이번 실행 결과에서는 Torque, Air temperature, Rotational speed 순으로 중요도가 높게 나왔습니다. 특히 Torque를 섞었을 때 f1-score가 0.3309만큼 감소해, 현재 모델이 Torque 정보에 가장 민감하게 반응하는 것으로 해석할 수 있었습니다.

다만 permutation importance는 test set 전체 기준 설명이므로, 개별 샘플 하나가 왜 고장으로 예측되었는지를 직접 설명하지는 못합니다. 따라서 이후에는 SHAP 같은 방법을 활용해 개별 예측 단위의 설명까지 확장할 계획입니다.
