"""
baseline_notes.py

이 파일은 manufacturing-ai-quality-agent-reference 프로젝트에서 사용하는
주요 baseline 설정값과 그 이유를 정리한 학습용 주석 파일입니다.

이 파일은 실제 모델 학습에 반드시 필요한 실행 코드라기보다,
프로젝트를 이해하고 설명하기 위한 기준 사전 역할을 합니다.

정리 목적:
1. 왜 이 값을 기본값으로 두었는지 설명할 수 있다.
2. 이 값이 최종 최적값이 아니라 baseline 출발점임을 명확히 한다.
3. 이후 모델 최적화 단계에서 어떤 값을 조정할 수 있는지 이해한다.
4. 면접에서 hyperparameter와 평가 기준을 근거 있게 설명할 수 있다.

중요:
baseline은 "최종 정답"이 아닙니다.

baseline이란:
- 처음 모델을 만들 때 사용하는 기본 출발점
- 너무 복잡하지 않은 구조
- 학습 루프가 정상 동작하는지 확인하기 위한 기준
- 이후 평가 결과를 보고 개선할 비교 대상

즉, baseline 모델을 먼저 만들고 나서
성능, loss 변화, precision, recall, f1-score 등을 보고
hidden_dim, dropout_rate, learning_rate, epochs, batch_size, threshold 등을 조정합니다.
"""


# ============================================================
# 1. input_dim = 6
# ============================================================

"""
input_dim은 모델이 입력으로 받는 feature 개수입니다.

이 프로젝트에서는 AI4I 데이터에서 다음 6개 feature를 사용합니다.

1. Air temperature [K]
2. Process temperature [K]
3. Rotational speed [rpm]
4. Torque [Nm]
5. Tool wear [min]
6. Type

따라서 모델의 input_dim은 6입니다.

왜 6인가?
Day 1에서 전처리할 때 feature columns를 다음처럼 정했습니다.

AI4I_FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

AI4I_CATEGORICAL_COLUMNS = [
    "Type"
]

숫자 feature 5개 + Type encoding 1개 = 총 6개입니다.

모델의 첫 번째 Linear layer는 입력 feature 개수와 반드시 맞아야 합니다.

예:
nn.Linear(input_dim, hidden_dim)
nn.Linear(6, 32)

만약 input_dim을 5로 잘못 설정하면,
실제 입력 Tensor shape [batch_size, 6]과 모델이 기대하는 shape이 맞지 않아
shape mismatch 오류가 발생합니다.

면접식 답변:
AI4I 데이터에서 모델 입력으로 사용할 feature를 6개로 정의했기 때문에
FailureMLP의 input_dim을 6으로 설정했습니다.
이 값은 임의로 정한 것이 아니라 전처리 단계에서 선택한 feature column 수와 연결됩니다.
"""

BASELINE_INPUT_DIM = 6


# ============================================================
# 2. hidden_dim = 32
# ============================================================

"""
hidden_dim은 MLP 은닉층의 뉴런 개수입니다.

현재 FailureMLP 구조는 대략 다음과 같습니다.

입력 6개
→ Linear(6, 32)
→ ReLU
→ Dropout
→ Linear(32, 16)
→ ReLU
→ Linear(16, 1)
→ logit 출력

hidden_dim=32로 설정하면 첫 번째 은닉층은 32개 뉴런을 가집니다.
두 번째 은닉층은 hidden_dim // 2 이므로 16개 뉴런을 가집니다.

왜 32인가?
AI4I baseline에서는 입력 feature가 6개로 많지 않습니다.
따라서 처음부터 128, 256처럼 큰 모델을 사용하는 것은 과할 수 있습니다.

hidden_dim이 너무 작으면:
- 모델 표현력이 부족할 수 있습니다.
- feature 간 복합 관계를 충분히 학습하지 못할 수 있습니다.
- 예: hidden_dim=4, 8

hidden_dim이 너무 크면:
- 데이터 규모에 비해 모델이 복잡해질 수 있습니다.
- 과적합 위험이 커질 수 있습니다.
- 학습용 baseline 구조가 불필요하게 복잡해질 수 있습니다.
- 예: hidden_dim=128, 256

따라서 hidden_dim=32는 작은 tabular dataset에서
baseline MLP를 시작하기 위한 적당한 출발값입니다.

중요:
hidden_dim=32가 최적값이라는 뜻은 아닙니다.
이후 최적화 단계에서 16, 32, 64, 128 등을 비교할 수 있습니다.

면접식 답변:
입력 feature가 6개로 많지 않기 때문에 처음부터 큰 신경망을 사용하지 않고,
hidden_dim=32의 작은 MLP를 baseline으로 설정했습니다.
이후 평가 결과에 따라 hidden_dim을 조정할 수 있도록 했습니다.
"""

BASELINE_HIDDEN_DIM = 32


# ============================================================
# 3. dropout_rate = 0.2
# ============================================================

"""
dropout_rate는 학습 중 일부 뉴런 출력을 무작위로 0으로 만드는 비율입니다.

dropout_rate=0.2는:
학습 중 약 20%의 뉴런 출력을 임의로 끈다는 뜻입니다.

Dropout을 사용하는 이유:
- 모델이 특정 뉴런에 지나치게 의존하는 것을 줄입니다.
- 과적합을 완화하는 regularization 역할을 합니다.
- train 모드에서는 활성화되고 eval 모드에서는 비활성화됩니다.

왜 0.2인가?
0.0이면 Dropout을 사용하지 않는 것입니다.
0.5 이상은 작은 MLP에서는 너무 강한 regularization이 될 수 있습니다.

현재 모델은 작은 tabular MLP이므로,
학습을 방해하지 않으면서 과적합을 조금 줄이는 baseline 값으로 0.2를 사용합니다.

dropout_rate가 너무 낮으면:
- 과적합 방지 효과가 거의 없습니다.

dropout_rate가 너무 높으면:
- 너무 많은 정보가 사라져 학습이 불안정할 수 있습니다.
- 작은 모델에서는 성능이 떨어질 수 있습니다.

중요:
dropout_rate=0.2도 최종값이 아닙니다.
이후 train loss와 validation/test 성능을 비교하면서
0.0, 0.1, 0.2, 0.3, 0.5 등을 실험할 수 있습니다.

면접식 답변:
Dropout은 과적합을 줄이기 위해 사용했고,
baseline에서는 작은 MLP 구조에 맞춰 0.2로 시작했습니다.
이 값은 최적값이 아니라 학습 안정성과 과적합 여부를 보며 조정할 hyperparameter입니다.
"""

BASELINE_DROPOUT_RATE = 0.2


# ============================================================
# 4. learning_rate = 0.001
# ============================================================

"""
learning_rate는 optimizer가 weight와 bias를 한 번에 얼마나 크게 수정할지 정하는 값입니다.

쉽게 말하면:
learning_rate = 학습 보폭

loss를 줄이기 위해 parameter를 어느 방향으로 움직일지는 gradient가 알려주고,
얼마나 크게 움직일지는 learning_rate가 결정합니다.

왜 0.001인가?
Adam optimizer에서는 0.001이 자주 쓰이는 baseline learning rate입니다.
처음 학습 루프를 만들 때 안정적인 출발값으로 많이 사용됩니다.

learning_rate가 너무 크면:
- weight가 너무 크게 움직입니다.
- loss가 튀거나 발산할 수 있습니다.
- 학습이 불안정해질 수 있습니다.
- 예: 0.1, 0.01

learning_rate가 너무 작으면:
- weight가 너무 조금씩 움직입니다.
- 학습 속도가 지나치게 느릴 수 있습니다.
- loss가 거의 줄지 않는 것처럼 보일 수 있습니다.
- 예: 0.00001

중요:
learning_rate=0.001은 Adam baseline 출발값입니다.
loss가 불안정하면 더 낮추고,
loss가 너무 천천히 줄면 조정할 수 있습니다.

면접식 답변:
Adam optimizer에서 일반적으로 많이 사용하는 0.001을 baseline learning rate로 설정했습니다.
이 값은 최종값이 아니라 loss 변화와 평가 성능을 보며 조정할 대상입니다.
"""

BASELINE_LEARNING_RATE = 0.001


# ============================================================
# 5. epochs = 10
# ============================================================

"""
epochs는 학습 데이터 전체를 몇 번 반복해서 학습할지 정하는 값입니다.

epochs=10은:
전체 학습 데이터를 10번 반복해서 모델에 보여준다는 뜻입니다.

왜 10인가?
현재 단계의 목적은 최고 성능을 얻는 것이 아닙니다.

현재 목적:
1. 데이터 전처리 결과가 Tensor로 변환되는지 확인
2. model forward가 정상 동작하는지 확인
3. loss 계산이 되는지 확인
4. backward가 되는지 확인
5. optimizer.step()으로 weight가 업데이트되는지 확인
6. epoch별 loss가 기록되는지 확인

즉, 학습 파이프라인 검증 단계입니다.
그래서 처음부터 100, 300 epoch처럼 오래 학습하지 않고
10 epoch 정도로 시작합니다.

epochs가 너무 적으면:
- 학습이 거의 진행되지 않을 수 있습니다.
- loss 변화 확인이 어렵습니다.
- 예: 1 epoch

epochs가 너무 많으면:
- 시간이 오래 걸립니다.
- 작은 데이터에서는 과적합될 수 있습니다.
- baseline 확인 단계에서는 불필요하게 무겁습니다.
- 예: 100, 300 epoch

중요:
epochs=10은 학습 루프 확인용 baseline입니다.
실제 성능 비교 단계에서는 30, 50, 100 등을 실험할 수 있습니다.

면접식 답변:
초기 단계에서는 모델 성능보다 학습 루프가 정상적으로 동작하는지 확인하는 것이 목적이므로
epochs를 10으로 설정했습니다.
이후 validation 성능과 loss 변화를 보면서 epoch 수를 늘리거나 early stopping을 적용할 수 있습니다.
"""

BASELINE_EPOCHS = 10


# ============================================================
# 6. batch_size = 32
# ============================================================

"""
batch_size는 한 번에 모델에 넣을 학습 샘플 개수입니다.

batch_size=32는:
DataLoader가 학습 데이터를 32개 샘플씩 나누어
(X_batch, y_batch) 형태로 반환한다는 뜻입니다.

예:
전체 학습 데이터가 8000개이고 batch_size=32라면
한 epoch 안에서 약 250개의 batch가 만들어집니다.

각 batch마다 다음 과정이 수행됩니다.

1. optimizer.zero_grad()
2. logits = model(X_batch)
3. loss = criterion(logits, y_batch)
4. loss.backward()
5. optimizer.step()

왜 32인가?
32는 딥러닝 학습에서 자주 사용하는 baseline batch size입니다.
너무 작지도 크지도 않아 초기 학습 루프 검증에 적당합니다.

batch_size가 너무 작으면:
- 한 번 업데이트할 때 참고하는 샘플 수가 적습니다.
- gradient가 불안정할 수 있습니다.
- loss가 많이 흔들릴 수 있습니다.
- 제조 고장 데이터처럼 고장 class가 적은 경우,
  어떤 batch에는 고장 샘플이 아예 없을 수도 있습니다.
- 예: 1, 4, 8

batch_size가 너무 크면:
- 메모리 사용량이 커집니다.
- 한 epoch 안에서 update 횟수가 줄어듭니다.
- 작은 데이터에서는 학습이 둔해질 수 있습니다.
- 예: 256, 512, 1024

중요:
batch_size=32는 최종값이 아닙니다.
이후 16, 32, 64, 128 등을 비교할 수 있습니다.

면접식 답변:
batch_size=32는 초기 mini-batch 학습을 위한 baseline 값으로 설정했습니다.
너무 작으면 gradient가 불안정할 수 있고,
너무 크면 메모리 부담과 update 횟수 감소 문제가 있어
32를 기본 출발점으로 두었습니다.
"""

BASELINE_BATCH_SIZE = 32


# ============================================================
# 7. threshold = 0.5
# ============================================================

"""
threshold는 모델이 출력한 probability를 최종 prediction label로 바꾸는 기준값입니다.

현재 모델은 학습 단계에서 logit을 출력합니다.

학습 단계:
model(x) → logits → BCEWithLogitsLoss(logits, y)

평가/추론 단계:
model(x) → logits → torch.sigmoid(logits) → probabilities

여기서 probabilities는 0과 1 사이 값입니다.

이 프로젝트에서는:
0 = 정상
1 = 고장

따라서 probability는 고장일 확률처럼 해석할 수 있습니다.

threshold=0.5는:
고장 확률이 50% 이상이면 고장으로 판단하고,
50% 미만이면 정상으로 판단한다는 뜻입니다.

예:
probability = 0.82 → 0.5 이상 → prediction = 1, 고장
probability = 0.21 → 0.5 미만 → prediction = 0, 정상

왜 0.5인가?
0.5는 이진 분류에서 가장 기본적인 baseline threshold입니다.
특별한 도메인 비용 기준이 아직 없을 때,
class 1 가능성이 class 0보다 크다고 보는 중간 기준입니다.

중요:
제조 고장 예측에서는 threshold=0.5가 최종 정답이 아닐 수 있습니다.

실제 고장을 정상으로 놓치는 미탐이 위험하다면:
threshold를 낮출 수 있습니다.

예:
threshold = 0.3
→ 조금만 위험해 보여도 고장으로 예측
→ recall 증가 가능
→ precision 감소 가능

오탐을 줄이고 싶다면:
threshold를 높일 수 있습니다.

예:
threshold = 0.7
→ 더 확실한 경우만 고장으로 예측
→ precision 증가 가능
→ recall 감소 가능

면접식 답변:
threshold=0.5는 이진 분류의 기본 baseline 기준으로 사용했습니다.
다만 제조 고장 예측에서는 미탐과 오탐의 비용이 다르기 때문에,
이후 precision, recall, f1-score를 보며 threshold를 조정할 수 있도록 설계했습니다.
"""

BASELINE_THRESHOLD = 0.5


# ============================================================
# 8. optimizer = Adam
# ============================================================

"""
optimizer는 loss.backward()로 계산된 gradient를 이용해
모델의 weight와 bias를 실제로 업데이트하는 역할을 합니다.

학습 흐름:
1. logits = model(X_batch)
2. loss = criterion(logits, y_batch)
3. loss.backward()
4. optimizer.step()

중요:
loss.backward()는 weight를 직접 바꾸지 않습니다.
각 parameter에 대한 gradient만 계산합니다.

실제로 weight와 bias를 바꾸는 코드는:
optimizer.step()

왜 Adam인가?
Adam은 딥러닝 baseline에서 자주 사용하는 optimizer입니다.

Adam의 장점:
- SGD보다 learning rate에 비교적 덜 민감한 편입니다.
- 각 parameter의 업데이트 크기를 적응적으로 조절합니다.
- baseline 모델을 빠르게 학습시킬 때 안정적인 편입니다.

다른 후보:
- SGD
- SGD + Momentum
- RMSprop
- AdamW

중요:
Adam이 항상 최고라는 뜻은 아닙니다.
baseline으로 Adam을 사용한 뒤,
필요하면 AdamW나 SGD와 비교할 수 있습니다.

면접식 답변:
optimizer는 모델의 weight와 bias를 업데이트하는 역할을 하며,
이 프로젝트에서는 baseline optimizer로 Adam을 사용했습니다.
Adam은 기본 성능이 안정적인 편이고 baseline 딥러닝 학습에서 자주 쓰이기 때문에
초기 모델 학습에 적합하다고 판단했습니다.
"""

BASELINE_OPTIMIZER_NAME = "Adam"


# ============================================================
# 9. loss function = BCEWithLogitsLoss
# ============================================================

"""
loss function은 모델 예측이 정답과 얼마나 다른지 계산하는 함수입니다.

이 프로젝트는 이진 분류 문제입니다.

0 = 정상
1 = 고장

이진 분류에서 대표적인 loss:
1. BCELoss
2. BCEWithLogitsLoss

BCELoss:
- 모델 출력이 이미 sigmoid를 거친 probability일 때 사용합니다.
- 흐름:
  logits → sigmoid → probability → BCELoss

BCEWithLogitsLoss:
- 모델 출력이 sigmoid 전 logit일 때 사용합니다.
- 내부에서 sigmoid와 binary cross entropy를 함께 처리합니다.
- 흐름:
  logits → BCEWithLogitsLoss

왜 BCEWithLogitsLoss인가?
PyTorch에서는 이진 분류 학습에서 BCEWithLogitsLoss를 많이 사용합니다.
Sigmoid와 BCE를 따로 계산하는 것보다 수치적으로 안정적입니다.

따라서 현재 FailureMLP는 마지막에 nn.Sigmoid()를 넣지 않고,
logit을 그대로 출력합니다.

학습:
logits = model(X_batch)
loss = criterion(logits, y_batch)

추론:
logits = model(X_test)
probabilities = torch.sigmoid(logits)

중요:
BCEWithLogitsLoss를 사용할 때 모델 마지막에 Sigmoid를 또 넣으면 안 됩니다.
그러면 sigmoid가 중복되어 학습이 부정확해질 수 있습니다.

면접식 답변:
이진 분류 문제이므로 손실함수로 BCEWithLogitsLoss를 사용했습니다.
모델은 logit을 출력하고, BCEWithLogitsLoss가 내부에서 sigmoid와 binary cross entropy를 함께 계산합니다.
이 방식은 수치적으로 안정적이기 때문에 모델 마지막에는 Sigmoid를 넣지 않았습니다.
"""

BASELINE_LOSS_FUNCTION_NAME = "BCEWithLogitsLoss"


# ============================================================
# 10. activation = ReLU
# ============================================================

"""
activation function은 모델에 비선형성을 추가하는 함수입니다.

현재 모델에서는 ReLU를 사용합니다.

ReLU:
입력이 0보다 작으면 0
입력이 0보다 크면 그대로 출력

예:
ReLU(-3) = 0
ReLU(2) = 2

왜 ReLU인가?
Linear layer만 여러 개 쌓으면 복잡한 비선형 관계를 학습하기 어렵습니다.
ReLU를 중간에 넣으면 feature 간 복합적인 패턴을 학습할 수 있습니다.

현재 구조:
Linear
→ ReLU
→ Dropout
→ Linear
→ ReLU
→ Linear

ReLU 장점:
- 계산이 단순합니다.
- 딥러닝 baseline에서 가장 자주 쓰입니다.
- sigmoid/tanh보다 깊은 모델에서 gradient 소실 문제가 상대적으로 덜합니다.

다른 후보:
- LeakyReLU
- GELU
- ELU
- Tanh

면접식 답변:
Linear layer만으로는 복잡한 비선형 패턴을 학습하기 어렵기 때문에
중간 activation으로 ReLU를 사용했습니다.
ReLU는 계산이 단순하고 baseline 신경망에서 일반적으로 많이 쓰입니다.
"""

BASELINE_ACTIVATION_NAME = "ReLU"


# ============================================================
# 11. model output = logit
# ============================================================

"""
현재 FailureMLP는 최종 출력으로 probability가 아니라 logit을 반환합니다.

logit이란:
Sigmoid를 통과하기 전의 원시 점수입니다.

예:
logit = -2.0
logit = 0.0
logit = 3.0

logit은 확률이 아니므로:
- 음수일 수 있습니다.
- 1보다 클 수 있습니다.
- 0과 1 사이로 제한되지 않습니다.

왜 logit을 출력하나?
학습 단계에서 BCEWithLogitsLoss를 사용하기 때문입니다.
BCEWithLogitsLoss는 logit을 직접 입력으로 받고,
내부에서 sigmoid 계산을 수행합니다.

학습 단계:
logits = model(X_batch)
loss = BCEWithLogitsLoss(logits, y_batch)

평가/추론 단계:
logits = model(X_test)
probabilities = torch.sigmoid(logits)
predictions = probabilities >= threshold

면접식 답변:
모델은 학습 단계에서 확률이 아니라 logit을 출력하도록 구성했습니다.
BCEWithLogitsLoss가 내부에서 sigmoid와 BCE 계산을 함께 처리하기 때문에,
학습 안정성을 위해 모델 마지막에는 Sigmoid를 넣지 않았습니다.
"""

BASELINE_MODEL_OUTPUT = "logit"


# ============================================================
# 12. target label = Machine failure
# ============================================================

"""
이 프로젝트의 target label은 AI4I 데이터의 Machine failure 컬럼입니다.

Machine failure:
0 = 정상
1 = 고장

모델은 feature X를 입력받아
Machine failure가 1일 가능성, 즉 고장 가능성을 학습합니다.

중요:
0과 1의 의미는 모델이 정하는 것이 아니라 데이터셋의 label 정의에서 옵니다.

이 프로젝트에서는:
Machine failure = 0 → 정상
Machine failure = 1 → 고장

따라서 sigmoid probability는 class 1,
즉 고장일 확률처럼 해석합니다.

주의:
다른 데이터셋에서는 0과 1의 의미가 반대일 수도 있습니다.
실무에서는 반드시 데이터 설명서와 target label 정의를 먼저 확인해야 합니다.

면접식 답변:
0을 정상, 1을 고장으로 해석한 근거는 AI4I 데이터의 Machine failure target 정의입니다.
모델이 임의로 정한 것이 아니라 데이터셋의 label 의미를 따른 것입니다.
"""

TARGET_COLUMN_NAME = "Machine failure"
NEGATIVE_CLASS_LABEL = 0
POSITIVE_CLASS_LABEL = 1


# ============================================================
# 13. Type encoding = L/M/H → 0/1/2
# ============================================================

"""
AI4I 데이터의 Type 컬럼은 문자열 범주형 데이터입니다.

원본 값:
L
M
H

baseline mapping:
L → 0
M → 1
H → 2

왜 encoding이 필요한가?
PyTorch의 nn.Linear는 문자열을 입력으로 받을 수 없습니다.
모델 입력은 숫자 Tensor여야 하므로 Type을 숫자로 변환해야 합니다.

주의:
단순 숫자 mapping은 모델이 L < M < H처럼
순서 관계가 있다고 오해할 수 있습니다.

예:
L = 0
M = 1
H = 2

이렇게 하면 모델 입장에서는 H가 M보다 크고, M이 L보다 크다는 식의
순서 정보가 있는 것처럼 해석될 수 있습니다.

최종 프로젝트 개선 방향:
one-hot encoding을 사용할 수 있습니다.

예:
Type_L
Type_M
Type_H

L → [1, 0, 0]
M → [0, 1, 0]
H → [0, 0, 1]

면접식 답변:
초기 baseline에서는 Type 컬럼을 L/M/H에서 0/1/2로 단순 encoding했습니다.
다만 숫자 mapping은 순서 관계를 암시할 수 있으므로,
최종 프로젝트에서는 one-hot encoding으로 개선할 수 있다고 판단했습니다.
"""

TYPE_MAPPING = {
    "L": 0,
    "M": 1,
    "H": 2,
}


# ============================================================
# 14. train/test split 기준
# ============================================================

"""
train/test split은 모델이 학습한 데이터에만 잘 맞는지,
처음 보는 데이터에도 일반화되는지 확인하기 위해 사용합니다.

train set:
모델 weight를 학습하는 데 사용합니다.

test set:
학습에 사용하지 않고,
학습 후 모델 성능을 평가하는 데 사용합니다.

baseline에서는 보통:
test_size = 0.2

즉:
전체 데이터의 80%는 train
전체 데이터의 20%는 test

random_state:
데이터를 나눌 때 난수를 고정하기 위한 값입니다.
같은 random_state를 사용하면 매번 같은 방식으로 split됩니다.
재현성을 위해 사용합니다.

stratify=y:
target class 비율을 train/test에 비슷하게 유지합니다.

왜 stratify가 중요한가?
제조 고장 데이터는 보통 정상 데이터가 많고 고장 데이터가 적은 불균형 데이터일 수 있습니다.
stratify 없이 나누면 test set에 고장 sample이 너무 적거나 없을 수 있습니다.

면접식 답변:
train/test split은 모델의 일반화 성능을 확인하기 위해 사용했습니다.
특히 제조 고장 데이터는 class imbalance가 있을 수 있으므로
stratify=y를 사용해 train/test set의 정상/고장 비율이 비슷하게 유지되도록 했습니다.
"""

BASELINE_TEST_SIZE = 0.2
BASELINE_RANDOM_STATE = 42
USE_STRATIFY = True


# ============================================================
# 15. evaluation metrics
# ============================================================

"""
제조 고장 예측에서는 accuracy만 보면 위험합니다.

이유:
정상 데이터가 매우 많고 고장 데이터가 적으면,
모델이 전부 정상이라고 예측해도 accuracy가 높게 나올 수 있습니다.

예:
정상 97개
고장 3개

모델이 전부 정상이라고 예측:
accuracy = 97%
하지만 실제 고장은 하나도 못 잡음

따라서 다음 지표를 함께 봐야 합니다.

accuracy:
전체 샘플 중 맞힌 비율입니다.

precision:
모델이 고장이라고 예측한 것 중 실제 고장인 비율입니다.
오탐을 얼마나 줄였는지와 관련됩니다.

recall:
실제 고장 중 모델이 고장이라고 잡아낸 비율입니다.
미탐을 얼마나 줄였는지와 관련됩니다.

f1-score:
precision과 recall의 균형을 나타내는 지표입니다.

제조 고장 예측에서 특히 중요한 것:
recall

이유:
실제 고장을 정상으로 놓치는 미탐은 큰 위험이 될 수 있습니다.
설비 고장을 놓치면 생산 중단, 품질 문제, 비용 증가로 이어질 수 있습니다.

하지만 recall만 높이면 정상도 고장으로 많이 잡을 수 있습니다.
따라서 precision과 recall의 균형을 같이 봐야 합니다.

면접식 답변:
제조 고장 예측은 class imbalance가 있을 수 있기 때문에 accuracy만으로 성능을 판단하지 않았습니다.
실제 고장을 놓치는 미탐이 중요하므로 recall을 함께 확인했고,
오탐과의 균형을 보기 위해 precision과 f1-score도 계산했습니다.
"""

BASELINE_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
]


# ============================================================
# 16. model.train() / model.eval()
# ============================================================

"""
PyTorch 모델은 학습 모드와 평가 모드를 구분합니다.

model.train():
학습 모드입니다.
Dropout이 활성화됩니다.
BatchNorm이 있다면 학습용 통계를 사용합니다.

model.eval():
평가/추론 모드입니다.
Dropout이 비활성화됩니다.
BatchNorm이 있다면 고정된 통계를 사용합니다.

현재 FailureMLP에는 Dropout이 있으므로
학습할 때는 반드시 model.train()을 호출하고,
평가/추론할 때는 model.eval()을 호출해야 합니다.

왜 중요한가?
평가할 때 Dropout이 켜져 있으면
같은 입력을 넣어도 매번 다른 출력이 나올 수 있습니다.
따라서 평가/추론 단계에서는 Dropout을 꺼야 합니다.

면접식 답변:
학습 시에는 Dropout이 작동해야 하므로 model.train()을 사용했고,
평가와 추론 시에는 안정적인 예측을 위해 model.eval()로 전환했습니다.
"""

USE_TRAIN_MODE_DURING_TRAINING = True
USE_EVAL_MODE_DURING_EVALUATION = True


# ============================================================
# 17. torch.no_grad()
# ============================================================

"""
torch.no_grad()는 gradient 추적을 끄는 context manager입니다.

학습 단계:
gradient가 필요합니다.
loss.backward()를 통해 gradient를 계산하고,
optimizer.step()으로 weight를 업데이트해야 합니다.

평가/추론 단계:
gradient가 필요 없습니다.
모델 weight를 업데이트하지 않고 예측만 하면 됩니다.

따라서 평가/추론에서는 다음처럼 사용합니다.

with torch.no_grad():
    logits = model(X_tensor)
    probabilities = torch.sigmoid(logits)

장점:
1. 불필요한 gradient 계산을 하지 않습니다.
2. 메모리 사용량을 줄입니다.
3. 추론 속도가 더 효율적입니다.

면접식 답변:
평가와 추론 단계에서는 모델을 업데이트하지 않으므로 gradient 계산이 필요 없습니다.
그래서 torch.no_grad()를 사용해 불필요한 연산과 메모리 사용을 줄였습니다.
"""

USE_NO_GRAD_DURING_EVALUATION = True


# ============================================================
# 18. DataLoader / TensorDataset
# ============================================================

"""
DataLoader는 데이터를 mini-batch 단위로 꺼내기 위한 PyTorch 도구입니다.

TensorDataset:
X_tensor와 y_tensor를 같은 index 기준으로 묶습니다.

예:
X_tensor[0]와 y_tensor[0]이 하나의 학습 샘플이 됩니다.

DataLoader:
TensorDataset을 batch_size 단위로 잘라서 반환합니다.

왜 DataLoader를 사용하는가?
전체 데이터를 한 번에 모델에 넣는 방식은 작은 데이터에서는 가능하지만,
데이터가 커지면 메모리 부담이 커질 수 있습니다.

mini-batch 학습을 사용하면:
- 데이터를 batch 단위로 나누어 학습할 수 있습니다.
- batch마다 weight update가 일어납니다.
- 실제 딥러닝 학습 방식에 더 가깝습니다.
- shuffle을 통해 데이터 순서 의존성을 줄일 수 있습니다.

shuffle=True:
학습 데이터는 보통 epoch마다 순서를 섞습니다.
모델이 데이터 순서에 과하게 의존하는 것을 줄이기 위함입니다.

평가 데이터는 보통 shuffle=False로 둡니다.
평가에서는 순서를 섞을 필요가 없고,
예측 결과와 원본 샘플 순서를 맞춰야 할 수 있기 때문입니다.

면접식 답변:
처음에는 전체 데이터를 한 번에 넣는 단순 학습 루프도 가능하지만,
실제 학습 구조를 고려해 TensorDataset과 DataLoader를 사용한 mini-batch 학습으로 구성했습니다.
이를 통해 데이터가 커져도 batch 단위로 안정적으로 학습할 수 있도록 했습니다.
"""

USE_DATALOADER = True
BASELINE_SHUFFLE_TRAIN = True
BASELINE_SHUFFLE_EVAL = False


# ============================================================
# 19. train_one_epoch 분리 이유
# ============================================================

"""
train_one_epoch 함수는 한 epoch 동안의 학습 과정을 담당합니다.

한 epoch:
학습 데이터 전체를 한 번 사용하는 단위입니다.

mini-batch 학습에서 한 epoch은:
DataLoader의 모든 batch를 한 번씩 사용하는 것을 의미합니다.

train_one_epoch 내부 흐름:
1. model.train()
2. DataLoader에서 batch 꺼내기
3. optimizer.zero_grad()
4. logits = model(X_batch)
5. loss = criterion(logits, y_batch)
6. loss.backward()
7. optimizer.step()
8. batch loss 기록
9. epoch 평균 loss 반환

왜 train_one_epoch로 분리하는가?
전체 학습 함수에 모든 코드를 넣으면 역할이 섞입니다.

분리하면:
- 한 epoch 학습 흐름을 독립적으로 이해할 수 있습니다.
- 테스트하기 쉽습니다.
- train_failure_model은 전체 학습 관리에 집중할 수 있습니다.
- 나중에 validation loop, early stopping 등을 추가하기 쉽습니다.

면접식 답변:
학습 전체 흐름과 한 epoch의 세부 학습 흐름을 분리하기 위해 train_one_epoch 함수를 따로 만들었습니다.
이를 통해 forward, loss 계산, backward, optimizer update 단계를 명확히 관리할 수 있고,
나중에 mini-batch, validation, early stopping으로 확장하기 쉽습니다.
"""


# ============================================================
# 20. optimizer.zero_grad()
# ============================================================

"""
optimizer.zero_grad()는 이전 batch에서 계산된 gradient를 초기화하는 코드입니다.

PyTorch는 기본적으로 gradient를 덮어쓰지 않고 누적합니다.

따라서 매 batch마다 loss.backward()를 호출하기 전에
optimizer.zero_grad()를 호출해야 합니다.

학습 흐름:
1. optimizer.zero_grad()
2. logits = model(X_batch)
3. loss = criterion(logits, y_batch)
4. loss.backward()
5. optimizer.step()

zero_grad()를 하지 않으면:
이전 batch의 gradient와 현재 batch의 gradient가 섞입니다.
그러면 의도하지 않은 방향으로 parameter가 업데이트될 수 있습니다.

면접식 답변:
PyTorch는 gradient를 기본적으로 누적하기 때문에,
각 batch 학습 전에 optimizer.zero_grad()로 이전 gradient를 초기화했습니다.
그 후 현재 batch 기준으로 loss.backward()를 수행해 gradient를 계산했습니다.
"""


# ============================================================
# 21. loss.backward()
# ============================================================

"""
loss.backward()는 loss를 기준으로 각 parameter의 gradient를 계산하는 코드입니다.

중요:
loss.backward()는 weight를 직접 업데이트하지 않습니다.
각 parameter의 .grad에 gradient를 계산해서 저장합니다.

gradient란:
loss를 줄이기 위해 parameter를 어느 방향으로 바꿔야 하는지 알려주는 값입니다.

실제 업데이트는 optimizer.step()이 수행합니다.

학습 흐름:
loss.backward()
→ gradient 계산

optimizer.step()
→ gradient를 이용해 weight/bias 업데이트

면접식 답변:
loss.backward()는 loss를 줄이기 위한 각 parameter의 gradient를 계산하는 단계입니다.
이 단계에서는 weight가 직접 바뀌지 않고,
optimizer.step()에서 계산된 gradient를 바탕으로 실제 업데이트가 이루어집니다.
"""


# ============================================================
# 22. optimizer.step()
# ============================================================

"""
optimizer.step()은 loss.backward()로 계산된 gradient를 사용해
모델의 weight와 bias를 실제로 업데이트하는 코드입니다.

학습 과정에서 weight가 바뀌는 시점은 optimizer.step()입니다.

학습 흐름:
1. loss.backward()
   → gradient 계산

2. optimizer.step()
   → weight/bias 업데이트

optimizer는 learning_rate를 기준으로
parameter를 얼마나 크게 움직일지 결정합니다.

면접식 답변:
optimizer.step()은 계산된 gradient를 바탕으로 모델의 weight와 bias를 실제로 업데이트하는 단계입니다.
즉 loss.backward()가 방향 정보를 계산한다면,
optimizer.step()은 그 정보를 사용해 parameter 값을 수정합니다.
"""


# ============================================================
# 23. probability to prediction
# ============================================================

"""
평가/추론 단계에서는 모델 logit을 probability로 변환한 뒤
threshold 기준으로 prediction을 만듭니다.

코드:
probabilities = torch.sigmoid(logits)
predictions = (probabilities >= threshold).int()

여기서 probabilities는 Tensor이고,
threshold는 Python float입니다.

PyTorch는 Tensor와 float을 비교할 때
float 값을 Tensor의 모든 원소에 적용합니다.

예:
probabilities = tensor([[0.10], [0.49], [0.50], [0.90]])
threshold = 0.5

probabilities >= threshold
→ tensor([[False], [False], [True], [True]])

.int()
→ tensor([[0], [0], [1], [1]])

즉:
False → 0
True → 1

이것은 element-wise operation입니다.
Tensor 내부의 모든 값을 for문 없이 한 번에 비교합니다.

면접식 답변:
PyTorch에서는 Tensor와 scalar를 비교하면 scalar가 Tensor의 각 원소에 적용됩니다.
따라서 probabilities >= threshold는 각 샘플의 고장 확률을 기준값과 원소별로 비교하고,
.int()를 통해 True/False를 1/0 label로 변환합니다.
"""


# ============================================================
# 24. baseline 이후 최적화 대상
# ============================================================

"""
baseline 모델이 완성된 뒤에는 다음 항목을 최적화할 수 있습니다.

1. feature scaling
- StandardScaler 적용
- feature 단위 차이 완화
- 예: rpm, temperature, torque, tool wear scale 차이 조정

2. class imbalance 대응
- class weight
- pos_weight in BCEWithLogitsLoss
- oversampling
- threshold 조정

3. hidden_dim 조정
- 16, 32, 64, 128 비교

4. dropout_rate 조정
- 0.0, 0.1, 0.2, 0.3, 0.5 비교

5. learning_rate 조정
- 0.01, 0.001, 0.0005, 0.0001 비교

6. batch_size 조정
- 16, 32, 64, 128 비교

7. epochs 조정
- 10, 30, 50, 100 비교
- early stopping 추가 가능

8. threshold 조정
- 0.3, 0.4, 0.5, 0.6, 0.7 비교
- recall/precision trade-off 확인

9. optimizer 비교
- Adam
- AdamW
- SGD

10. metric 기준 변경
- accuracy 중심이 아니라 recall, f1, PR-AUC 중심으로 평가

면접식 답변:
먼저 작은 MLP baseline을 구성해 학습 루프와 평가 흐름을 검증했습니다.
이후 feature scaling, class imbalance 대응, threshold 조정, hidden_dim과 learning_rate 변경 등을 통해
baseline 대비 성능을 개선하는 방식으로 최적화를 진행할 수 있도록 설계했습니다.
"""


# ============================================================
# 25. baseline summary
# ============================================================

"""
현재 baseline 요약:

input_dim = 6
- AI4I feature 5개 + Type 1개

hidden_dim = 32
- 작은 tabular MLP baseline

dropout_rate = 0.2
- 약한 regularization 출발값

learning_rate = 0.001
- Adam optimizer baseline learning rate

epochs = 10
- 학습 루프 검증용 초기 반복 수

batch_size = 32
- mini-batch 학습 baseline

threshold = 0.5
- 이진 분류 기본 판단 기준

optimizer = Adam
- 딥러닝 baseline optimizer

loss function = BCEWithLogitsLoss
- logit 기반 이진 분류 손실함수

activation = ReLU
- 기본 비선형 활성화 함수

output = logit
- 학습 시 BCEWithLogitsLoss에 직접 입력
- 추론 시 sigmoid로 probability 변환

metrics = accuracy, precision, recall, f1
- accuracy만 보지 않고 불균형 데이터 지표 포함

중요한 결론:
이 값들은 최종 성능을 보장하는 값이 아니라
학습과 평가 파이프라인을 검증하기 위한 baseline 기준입니다.

이후 성능 평가 결과를 보고 조정해야 합니다.
"""