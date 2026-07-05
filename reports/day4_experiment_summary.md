# Day 4. Manufacturing AI Failure Prediction 실험 결과 정리

## 1. Day 4 목표

Day 4의 목표는 단순히 모델을 학습시키는 것이 아니라, 실제 실행 결과를 해석하고 제조 고장 예측 문제에서 어떤 평가 지표를 봐야 하는지 이해하는 것이다.

주요 목표는 다음과 같다.

```text
1. 실제 AI4I CSV 데이터로 모델 학습 실행
2. loss 변화 해석
3. accuracy / precision / recall / f1-score 해석
4. confusion matrix 추가
5. class imbalance 문제 확인
6. threshold 0.3 / 0.5 / 0.7 비교
7. pos_weight 적용
8. feature scaling 적용
9. threshold 세분화 비교
10. 제조 고장 예측 관점에서 운영 threshold 후보 선택
```

---

## 2. 데이터 구조

사용한 데이터는 AI4I 2020 Predictive Maintenance Dataset이다.

실행 결과 전체 데이터 크기는 다음과 같았다.

```text
Raw dataframe shape: (10000, 14)
X_train shape: (8000, 6)
X_test shape: (2000, 6)
y_train shape: (8000,)
y_test shape: (2000,)
```

모델 입력 feature는 6개다.

```text
1. Air temperature [K]
2. Process temperature [K]
3. Rotational speed [rpm]
4. Torque [Nm]
5. Tool wear [min]
6. Type
```

target은 다음 컬럼이다.

```text
Machine failure
```

label 의미는 다음과 같다.

```text
0 = 정상
1 = 고장
```

---

## 3. Class Imbalance 확인

train set과 test set의 class 비율은 다음과 같았다.

```text
y_train class ratio:
0    0.966125
1    0.033875

y_test class ratio:
0    0.966
1    0.034
```

즉, 정상 class가 약 96.6%, 고장 class가 약 3.4%인 매우 불균형한 데이터다.

test set 2000개 기준으로 보면 다음과 같다.

```text
정상: 1932개
고장: 68개
```

따라서 모델이 모든 샘플을 정상으로 예측해도 accuracy는 약 96.6%가 나올 수 있다.

이 때문에 제조 고장 예측에서는 accuracy만 보면 위험하다. 실제 고장을 얼마나 잡았는지 확인하기 위해 precision, recall, f1-score, confusion matrix를 함께 봐야 한다.

---

## 4. Baseline 결과

처음 baseline 모델은 feature scaling 없이, class imbalance 보정 없이 학습했다.

threshold 0.5 기준 결과는 다음과 같았다.

```text
accuracy : 0.9660
precision: 0.0000
recall   : 0.0000
f1       : 0.0000
```

confusion matrix는 다음과 같았다.

```text
                 Predicted
                Normal   Failure
Actual Normal      1932         0
Actual Failure       68         0
```

해석은 다음과 같다.

```text
TN = 1932
FP = 0
FN = 68
TP = 0
```

즉, 모델은 모든 test sample을 정상으로 예측했다.

accuracy는 96.6%로 높게 나왔지만, 실제 고장 68개를 하나도 잡지 못했다. 따라서 이 accuracy는 좋은 성능이 아니라 class imbalance 때문에 발생한 착시다.

---

## 5. Confusion Matrix 용어 정리

이진 분류에서 confusion matrix는 예측 결과를 네 가지로 나누어 보여준다.

```text
TN = True Negative
실제 정상이고 모델도 정상이라고 예측한 경우

FP = False Positive
실제 정상인데 모델이 고장이라고 예측한 경우

FN = False Negative
실제 고장인데 모델이 정상이라고 예측한 경우

TP = True Positive
실제 고장이고 모델도 고장이라고 예측한 경우
```

제조 고장 예측에서는 특히 FN이 중요하다.

FN은 실제 고장이 발생했는데 모델이 정상으로 판단한 경우다. 현장에서는 고장 위험 설비를 놓치거나 예방 정비 타이밍을 놓치는 문제로 이어질 수 있다.

---

## 6. pos_weight 적용

baseline 모델이 고장 class를 거의 잡지 못했기 때문에 BCEWithLogitsLoss에 pos_weight를 적용했다.

pos_weight는 positive class, 즉 Machine failure = 1인 고장 class의 손실을 더 크게 반영하기 위한 값이다.

계산식은 다음과 같다.

```text
pos_weight = negative_count / positive_count
```

현재 프로젝트에서는 다음과 같이 해석할 수 있다.

```text
negative class = 0 = 정상
positive class = 1 = 고장
```

pos_weight를 적용한 목적은 accuracy를 높이는 것이 아니라, 실제 고장을 정상으로 놓치는 FN을 줄이고 recall을 개선하는 것이다.

pos_weight 적용 후 threshold 0.5 기준 결과는 다음과 같았다.

```text
accuracy : 0.8725
precision: 0.1445
recall   : 0.5588
f1       : 0.2296

TN = 1707
FP = 225
FN = 30
TP = 38
```

해석은 다음과 같다.

```text
실제 고장 68개 중 38개를 잡았다.
실제 고장 68개 중 30개는 아직 놓쳤다.
정상 1932개 중 225개를 고장으로 잘못 예측했다.
```

baseline과 비교하면 다음과 같다.

```text
TP: 0  → 38
FN: 68 → 30
recall: 0.0000 → 0.5588
```

즉, pos_weight 적용 후 모델이 고장 class를 학습하기 시작했다.

다만 FP가 증가했고 precision은 낮았다. 이는 모델이 고장을 더 적극적으로 예측하면서 생긴 precision-recall trade-off로 볼 수 있다.

---

## 7. Feature Scaling 적용

pos_weight 적용 후 다음 개선으로 feature scaling을 적용했다.

AI4I feature들은 단위 차이가 컸다.

```text
Air temperature [K]        약 300
Process temperature [K]    약 300
Rotational speed [rpm]     약 1000~3000
Torque [Nm]                약 10~80
Tool wear [min]            약 0~250
Type                       0~2
```

MLP는 입력 feature의 scale 차이에 영향을 받을 수 있다. 그래서 numeric sensor feature에 StandardScaler를 적용했다.

StandardScaler는 각 feature를 다음 기준으로 변환한다.

```text
평균 ≈ 0
표준편차 ≈ 1
```

중요한 점은 scaler를 train set에만 fit해야 한다는 것이다.

정상 흐름은 다음과 같다.

```text
X_train에 scaler.fit()
X_train에 scaler.transform()
X_test에는 scaler.transform()만 적용
```

test set의 평균이나 표준편차를 scaling 기준에 사용하면 data leakage가 발생할 수 있다.

현재 Type 컬럼은 L/M/H를 0/1/2로 mapping한 범주형 feature이므로 scaling 대상에서 제외했다. 이후 최종 프로젝트에서는 one-hot encoding으로 개선할 수 있다.

feature scaling과 pos_weight를 함께 적용한 후 threshold 0.5 기준 결과는 다음과 같았다.

```text
accuracy : 0.8730
precision: 0.2000
recall   : 0.9118
f1       : 0.3280

TN = 1684
FP = 248
FN = 6
TP = 62
```

해석은 다음과 같다.

```text
실제 고장 68개 중 62개를 잡았다.
실제 고장 68개 중 6개만 놓쳤다.
정상 1932개 중 248개를 고장으로 잘못 예측했다.
```

pos_weight만 적용했을 때와 비교하면 다음과 같다.

```text
TP: 38 → 62
FN: 30 → 6
recall: 0.5588 → 0.9118
precision: 0.1445 → 0.2000
f1: 0.2296 → 0.3280
```

feature scaling 적용 후 모델의 고장 탐지 성능이 크게 개선되었다.

---

## 8. Threshold 세분화 비교

기존에는 threshold 0.3, 0.5, 0.7만 비교했다.

feature scaling 이후 threshold 0.7 근처에서 f1-score가 좋아졌기 때문에, threshold를 0.50부터 0.90까지 0.05 간격으로 세분화해 비교했다.

결과는 다음과 같았다.

```text
threshold | accuracy | precision | recall | f1 | TN | FP | FN | TP
-------------------------------------------------------------------
     0.50 |   0.8730 |    0.2000 | 0.9118 | 0.3280 | 1684 | 248 |  6 | 62
     0.55 |   0.8915 |    0.2210 | 0.8676 | 0.3522 | 1724 | 208 |  9 | 59
     0.60 |   0.9080 |    0.2521 | 0.8676 | 0.3907 | 1757 | 175 |  9 | 59
     0.65 |   0.9190 |    0.2740 | 0.8382 | 0.4130 | 1781 | 151 | 11 | 57
     0.70 |   0.9305 |    0.3060 | 0.8235 | 0.4462 | 1805 | 127 | 12 | 56
     0.75 |   0.9385 |    0.3270 | 0.7647 | 0.4581 | 1825 | 107 | 16 | 52
     0.80 |   0.9510 |    0.3790 | 0.6912 | 0.4896 | 1855 |  77 | 21 | 47
     0.85 |   0.9565 |    0.4040 | 0.5882 | 0.4790 | 1873 |  59 | 28 | 40
     0.90 |   0.9675 |    0.5217 | 0.5294 | 0.5255 | 1899 |  33 | 32 | 36
```

---

## 9. Threshold 후보 해석

### 9.1 f1-score 기준 최적 후보

f1-score만 기준으로 보면 threshold 0.90이 가장 좋았다.

```text
threshold : 0.90
accuracy  : 0.9675
precision : 0.5217
recall    : 0.5294
f1        : 0.5255

TN = 1899
FP = 33
FN = 32
TP = 36
```

장점은 precision과 f1-score가 가장 높고 FP가 매우 적다는 것이다.

하지만 실제 고장 68개 중 32개를 놓친다. 제조 고장 예측에서는 FN이 위험할 수 있으므로, f1-score만 기준으로 threshold를 선택하는 것은 조심해야 한다.

### 9.2 recall 0.85 이상 조건에서 최적 후보

recall이 최소 0.85 이상인 후보 중 f1-score가 가장 높은 threshold는 0.60이었다.

```text
threshold : 0.60
accuracy  : 0.9080
precision : 0.2521
recall    : 0.8676
f1        : 0.3907

TN = 1757
FP = 175
FN = 9
TP = 59
```

threshold 0.60은 실제 고장 68개 중 59개를 잡았고, 놓친 고장은 9개였다.

threshold 0.50과 비교하면 다음과 같다.

```text
threshold 0.50:
FP = 248
FN = 6
TP = 62
f1 = 0.3280

threshold 0.60:
FP = 175
FN = 9
TP = 59
f1 = 0.3907
```

threshold 0.60은 threshold 0.50보다 고장 3개를 더 놓치지만, 정상 오탐을 73개 줄이고 f1-score도 개선했다.

따라서 현재 모델에서는 다음과 같이 해석할 수 있다.

```text
f1-score 기준 후보: threshold 0.90
제조 안전/recall 기준 운영 후보: threshold 0.60
```

---

## 10. Day 4 결론

Day 4 실험을 통해 다음을 확인했다.

```text
1. baseline 모델은 accuracy는 높았지만 실제 고장을 하나도 잡지 못했다.
2. 이 문제는 class imbalance 때문에 발생했다.
3. confusion matrix를 통해 accuracy의 착시를 확인할 수 있었다.
4. pos_weight를 적용하자 recall이 0에서 0.5588로 개선되었다.
5. feature scaling을 추가하자 recall이 0.9118까지 개선되었다.
6. threshold를 세분화해 비교하자 운영 기준 후보를 더 명확히 선택할 수 있었다.
7. f1-score 기준 최적 threshold는 0.90이었다.
8. 제조 고장 미탐을 줄이는 운영 후보는 threshold 0.60으로 볼 수 있다.
```

현재 모델의 추천 후보는 다음과 같다.

```text
운영 후보 threshold = 0.60
```

이유는 다음과 같다.

```text
1. recall이 0.8676으로 높다.
2. 실제 고장 68개 중 59개를 잡는다.
3. FN이 9개로 비교적 낮다.
4. threshold 0.50보다 FP를 크게 줄인다.
5. f1-score도 threshold 0.50보다 높다.
```

---

## 11. 다음 개선 방향

Day 5에서는 모델을 실제 추론에 사용할 수 있도록 저장과 로드를 구현한다.

다음 단계는 다음과 같다.

```text
1. 학습된 모델 저장
2. scaler 저장
3. threshold 저장
4. 저장된 모델 로드
5. 단일 sample inference 함수 작성
6. probability, prediction, risk_level 반환
7. 테스트 작성
8. 이후 FastAPI 또는 Agent 구조와 연결
```

최종 목표는 다음과 같은 추론 흐름을 만드는 것이다.

```text
raw input
↓
same preprocessing
↓
same scaling
↓
model inference
↓
probability
↓
threshold comparison
↓
prediction
↓
risk_level
↓
explanation/evidence
```
