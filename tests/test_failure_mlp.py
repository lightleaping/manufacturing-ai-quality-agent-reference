import torch

from src.models.failure_mlp import FailureMLP

def test_failure_mlp_forward_output_shape():
    """
    FailureMLP가 입력 batch를 받아
    [batch_size, 1] 형태의 logit을 반환하는지 확인합니다.
    """

    # AI4I feature 개수는 6입니다.
    input_dim = 6

    # 모델 인스턴스를 생성합니다.
    model = FailureMLP(input_dim=input_dim)

    # batch_size가 4인 가짜 입력 데이터를 만듭니다.
    # 즉, 4개의 설비 샘플이 있고 각 샘플은 6개 feature를 가집니다.
    x = torch.randn(4, input_dim)

    # 모델에 입력을 넣어 forward 계산을 수행합니다.
    # 현재 모델 출력은 확률이 아니라 logit입니다.
    output = model(x)

    # 출력 shape은 [batch_size, 1]이어야 합니다.
    assert output.shape == (4, 1)

def test_failure_mlp_logits_can_be_converted_to_probability():
    """
    모델 출력 logit에 torch.sigmoid를 적용하면
    0과 1 사이의 probability로 변환되는지 확인합니다.
    """

    # 모델을 생성합니다.
    model = FailureMLP(input_dim=6)

    # 8개의 가짜 설비 샘플을 만듭니다.
    #
    # torch.randn(8, 6)은 평균 0, 표준편차 1을 따르는
    # 임의의 숫자로 이루어진 Tensor를 생성합니다.
    #
    # 여기서 (8, 6)의 의미는 다음과 같습니다.
    #
    # 8 = batch_size
    #   -> 한 번에 모델에 넣을 샘플 개수입니다.
    #   -> AI4I 데이터에서는 다음 6개 feature를 사용합니다.
    #       1. Air temperature [K]
    #       2. Process temperature [K]
    #       3. Rotational speed [rpm]
    #       4. Torque [Nm]
    #       5. Tool wear [min]
    #       6. Type
    # 
    # 따라서 x는 "8개의 설비 샘플 x 각 샘플당 6개의 feature" 형태입니다.
    #
    # shape으로 표현하면:
    # x.shape == (8, 6)
    # 
    # 단, 여기서 만든 x는 실제 AI4I 데이터가 아닙니다.
    # 모델 구조가 정상적으로 동작하는지 확인하기 위한 가짜 입력 데이터입니다.
    x = torch.randn(8, 6)

    # model(x)는 PyTorch 모델에 입력 Tensor x를 넣는 코드입니다.
    #
    # 겉으로는 model(x)처럼 보이지만,
    # 내부적으로는 FailureMLP 클래스의 forward(x) 함수가 실행됩니다.
    #
    # 즉, 아래 코드는:
    # logit = model(x)
    #
    # 내부적으로 다음과 비슷하게 동작합니다.
    # logit = model.forward(x)
    #
    # 하지만 PyTorch에서는 model.forward(x)를 직접 호출하기보다
    # model(x) 형태로 호출하는 것이 표준입니다.
    #
    # 모델 출력은 logit입니다.
    
    # inference는 학습된 모델을 실제 입력 데이터에 적용해
    # 결과를 계산하는 전체 추론 과정입니다.
    #
    # 이 프로젝트의 inference 흐름은 다음과 같습니다.
    #
    # 1. model(x)
    #     -> 모델이 입력 x를 받아 logit을 출력합니다.
    #
    # 2, torch.sigmoid(logits)
    #     -> logit을 0과 1 사이의 probability로 변환합니다.
    #
    # 3. probabilities >= threshold
    #     -> probability를 기준값(threshold)과 비교합니다.
    # 4. prediction
    #     -> 최종 예측 label를 만듭니다.
    #         0 = 정상
    #         1 = 고장
    #
    # 따라서 inference는 전체 과정이고,
    # prediction은 그 과정의 최종 결과라고 볼 수 있습니다.
    logits = model(x)

    # logit은 확률이 아니므로 0~1 범위라고 보장할 수 없습니다.
    # 사람이 해석 가능한 고장 확률이 필요할 때 sigmoid를 적용합니다.
    probabilites = torch.sigmoid(logits)

    # probability >= 0은 probability Tensor의 모든 값에 대해
    # "각 값이 0 이상인가?"를 비교합니다.
    #
    # 예를 들어 probability가 다음과 같다면:
    #
    # probability = tensor([
    #     [0.12],
    #     [0.83],
    #     [0.51]
    # ])
    #
    # probability >= 0의 결과는:
    #
    # tensor([
    #     [True],
    #     [True],
    #     [True]
    # ])
    #
    # 처럼 Boolean Tensor가 됩니다.
    #
    # torch.all(...)은 이 Boolean Tensor 안의 값이
    # 전부 True인지 확인합니다.
    #
    # 즉, 아래 코드는:
    # "모델 출력값이 모두 0 이상인가?"
    # 를 테스트합니다.

    # sigmoid를 통과한 probability는 0 이상이어야 합니다.

    # Sigmoid는 어떤 logit 값이 들어와도
    # 0과 1 사이의 값으로 변환하기 때문에,
    # 모델의 마지막 layer가 Sigmoid라면 이 테스트는 통과해야 합니다.

    assert torch.all(probabilites >= 0)

    # sigmoid를 통과한 probability는 1 이하이어야 합니다.
    assert torch.all(probabilities <= 1)

def test_failure_mlp_batch_shape_meaning():
    """
    batch_size와 input_dim의 의미를 확인하는 테스트입니다.

    x.shape == (batch_size, input_dim)
    output.shape == (batch_size, 1)

    즉, 모델은 여러 개의 설비 샘플을 한 번에 입력받고,
    각 샘플마다 logit 1개를 반환해야 합니다.
    """

    # batch_size는 한 번에 모델에 넣을 샘플 개수입니다.
    batch_size = 16

    # input_dim은 각 샘플이 가진 feature 개수입니다.
    # AI4I 데이터에서는 숫자 feature 5개 + Type 1개 = 총 6개입니다.
    input_dim = 6

    # FailureMLP 모델을 생성합니다.
    model = FailureMLP(input_dim=input_dim)

    # 16개의 가짜 설비 샘플을 만듭니다.
    # 각 샘플은 6개의 feature를 가집니다.
    x = torch.randn(batch_size, input_dim)

    # 모델에 batch 입력을 넣어 forward 계산을 수행합니다.
    output = model(x)

    # 입력 shape은 [batch_size, input_dim]이어야 합니다.
    assert x.shape == (batch_size, input_dim)

    # 출력 shape은 [batch_size, 1]이어야 합니다.
    # 각 샘플마다 고장 확률 1개를 반환하기 때문입니다.
    assert output.shape == (batch_size, 1)

def test_failure_mlp_bce_with_logits_loss():
    """
    FailureMLP의 logit 출력과 binary target을 이용해
    BCEWithLogitsLoss가 정상적으로 계산되는지 확인합니다.

    이 테스트는 아직 모델 성능을 검증하는 것이 아니라,
    학습에 필요한 loss 계산 흐름이 정상적으로 연결되는지 확인하는 테스트입니다.
    """

    # 모델을 생성합니다.
    model = FailureMLP(input_dim=6)

    # 이진 분류에서 사용할 손실함수입니다.
    # BCEWithLogitsLoss는 모델 출력 logit을 직접 입력으로 받습니다.
    # 내부에서 sigmoid 처리를 포함해 binary cross entropy를 계산합니다.
    criterion = torch.nn.BCEWithLogitsLoss()

    # batch_size가 5인 가짜 입력 데이터를 만듭니다.
    x = torch.randn(5, 6)

    # 각 샘플의 정답 label을 만듭니다.
    # shape은 모델 출력과 맞추기 위해 [batch_size, 1]로 둡니다.
    #
    # 0.0 = 정상
    # 1.0 = 고장
    y = torch.tensor(
        [[0.0], [1.0], [0.0], [1.0], [0.0],
         dtype=torch.float32,
    )

    # 모델 출력은 logit입니다.
    logits = model(x)

    # logit과 정답 y를 비교해 loss를 계산합니다.
    # criterion은 loss function입니다.
    # 예: nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss(), nn.MSELoss() 등
    #
    # loss function은 모델의 예측값과 실제 정답을 비교해서
    # "모델이 얼마나 틀렸는지"를 하나의 숫자로 계산합니다.
    #
    # logits:
    # - 모델이 출력한 예측값입니다.
    # - 아직 확률로 변환되지 않은 원본 점수입니다.
    # - 예를 들어 이진 분류라면 "고장일 가능성이 높은지"를 나타내는 점수일 수 있습니다.
    #
    # y:
    # - 실제 정답 label입니다.
    # - 예를 들어 정상=0, 고장=1 같은 값입니다.
    #
    # criterion(logits, y):
    # - 모델 예측값 logits와 실제 정답 y를 비교합니다.
    # - 비교 결과로 loss를 계산합니다.
    # - loss가 작을수록 모델 예측이 정답에 가깝다는 뜻입니다.
    # - loss가 클수록 모델 예측이 정답과 많이 다르다는 뜻입니다.
    loss = criterion(logits, y)

    # loss는 Tensor 하나로 계산되어야 합니다.
    # loss.ndim은 loss Tensor의 차원 수를 의미합니다.
    #
    # loss.ndim == 0 이라는 것은
    # loss가 벡터나 행렬이 아니라 "숫자 하나"라는 뜻입니다.
    #
    # 예:
    # torch.tensor(0.53)
    # → 차원 수 0
    # → 스칼라 Tensor
    #
    # 반면 아래처럼 샘플별 loss가 여러 개 남아 있으면 ndim이 1입니다.
    # torch.tensor([0.3, 0.7, 0.2])
    # → 차원 수 1
    #
    # 일반적인 학습에서는 optimizer가 역전파할 수 있도록
    # batch 전체의 loss가 평균 또는 합계로 줄어든 숫자 하나여야 합니다.
    #
    # 그래서 이 테스트는
    # "loss가 학습에 사용할 수 있는 단일 스칼라 값으로 계산되었는가?"
    # 를 확인합니다.
    assert loss.ndim == 0

    # loss는 음수가 아니어야 합니다.
    # loss.item()은 Tensor 안에 들어 있는 숫자 값을
    # Python 숫자로 꺼내는 코드입니다.
    #
    # 예:
    # loss = torch.tensor(0.53)
    # loss.item()
    # → 0.53
    #
    # 대부분의 loss function은
    # 예측값과 정답의 차이, 또는 틀린 정도를 계산하기 때문에
    # 값이 0 이상이어야 합니다.
    #
    # loss가 0에 가까움:
    # - 예측이 정답에 매우 가까움
    #
    # loss가 큼:
    # - 예측이 정답과 많이 다름
    #
    # loss가 음수:
    # - 일반적인 분류/회귀 loss에서는 이상한 상황
    # - loss function 선택, 입력값 형태, target 형식 등에 문제가 있을 수 있음
    #
    # 그래서 이 테스트는
    # "loss가 정상적인 범위의 값으로 계산되었는가?"
    # 를 확인합니다.
    assert loss.item() >= 0
