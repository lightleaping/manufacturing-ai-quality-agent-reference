"""
run_train_failure_model.py

실제 AI4I CSV 데이터를 사용해
FailureMLP 모델을 학습하고 평가하는 실행 스크립트입니다.

이 파일의 목적은 다음과 같습니다.

1. data/raw/ai4i_2020.csv 파일을 읽습니다.
2. Day 1에서 만든 전처리 함수를 실행합니다.
3. Day 3에서 만든 학습 함수를 실행합니다.
4. Day 3-2에서 만든 평가 함수를 실행합니다.
5. 학습 loss와 평가 지표를 터미널에 출력합니다.

이 스크립트는 테스트용 코드가 아니라,
프로젝트 전체 파이프라인이 실제 데이터에서 연결되는지 확인하는 실행 파일입니다.
"""

# scripts 폴더 안의 파일을 직접 실행하면,
# Python이 프로젝트 루트를 import 경로로 자동 인식하지 못할 수 있습니다.
#
# 예:
# python scripts/run_train_failure_model.py
#
# 이때 from src... import가 실패할 수 있으므로,
# 현재 파일 위치 기준으로 프로젝트 루트 경로를 찾아 sys.path에 추가합니다.
#
# Path(__file__).resolve()
#   현재 파일의 절대 경로입니다.
#
# parents[1]
#   scripts 폴더의 상위 폴더,
#   즉 manufacturing-ai-quality-agent-reference 프로젝트 루트입니다.

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_ai4i import load_ai4i_csv
from src.data.preprocess_ai4i import preprocess_ai4i_dataframe
from src.training.train_failure_model import train_failure_model
from src.training.evaluate_failure_model import evaluate_failure_model

def main() -> None:
    """
    AI4I 데이터 로드부터 모델 학습, 평가까지 실행하는 메인 함수입니다.

    전체 흐름:

    CSV 로드
    -> 전처리
    -> 모델 학습
    -> 모델 평가
    -> 결과 출력
    """

    # 1. AI4I CSV 파일을 로드합니다.
    #
    # 기본 경로는 load_ai4i.py에 정의된
    # data/raw/ai4i/ai4i_2020.csv 입니다.
    #
    # 이 파일이 없으면 FileNotFoundError가 발생합니다.
    df = load_ai4i_csv()

    # 데이터 크기를 확인합니다.
    #
    # 예:
    # df.shape == (10000, 14)
    #
    # shape의 의미:
    #   첫 번째 값 = row 개수, 즉 샘플 수
    #   두 번째 값 = column 개수
    print("[INFO] Loaded AI4I CSV")
    print(f"[INFO] Raw dataframe shape: {df.shape}")

    # 2. AI4I 데이터를 모델 학습용으로 전처리합니다.
    #
    # preprocess_ai4i_dataframe 내부 흐름:
    #
    # 1. 필수 컬럼 인증
    # 2. Type 컬럼 L / M / H -> 0 / 1 / 2 encoding
    # 3. X, y 분리
    # 4. train / test split
    #
    # 반환값은 PreprocessedAI4IData dataclass입니다.
    processed = preprocess_ai4i_dataframe(df)

    # 전처리 후 train / test 데이터 크기를 확인합니다.
    print("[INFO] Preprocessing completed")
    print(f"[INFO] X_train shape: {processed.X_train.shape}")
    print(f"[INFO] X_test shape: {processed.X_test.shape}")
    print(f"[INFO] y_train shape: {processed.y_train.shape}")
    print(f"[INFO] y_test shape: {processed.y_test.shape}")

    # target class 비율을 확인합니다.
    #
    # 제조 고장 데이터는 보통 정상 데이터가 많고,
    # 고장 데이터가 적은 class imbalance 문제가 있을 수 있습니다.
    #
    # value_counts(normalize=True)를 사용하면
    # 각 class의 비율을 확인할 수 있습니다.
    print("[INFO] y_train class ratio:")
    print(processed.y_train.value_counts(normalize=True))

    print("[INFO] y_test class ratio:")
    print(processed.y_test.value_counts(normalize=True))

    # 3. FailureMLP 모델을 학습합니다.
    #
    # baseline 설정:
    #
    # input_dim=6
    #   AI4I feature 5개 + Type 1개
    #
    # hidden_dim=32
    #   작은 MLP baseline
    #
    # dropout_rate=0.2
    #   과적합 완화를 위한 기본값
    #
    # learning_rate=0.001
    #   Adam optimizer baseline learning rate
    #
    # epochs=10
    #   학습 루프 확인용 초기 반복 수
    #
    # batch_size=32
    #   mini-batch 학습 baseline
    training_result = train_failure_model(
        X_train=processed.X_train,
        y_train=processed.y_train,
        input_dim=6,
        hidden_dim=32,
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=10,
        batch_size=32,
    )

    print("[INFO] Training completed")
    print("[INFO] Epoch losses:")

    # epoch별 평균 loss를 출력합니다.
    #
    # loss가 전반적으로 줄어드는지 확인합니다.
    # 단, mini-batch와 dropout 때문에 매 epoch마다 반드시 감소하지는 않을 수 있습니다.

    # training_result.losses는 train_failure_model()에서 반환된
    # epoch별 평균 loss 리스트입니다.
    #
    # 예:
    # training_result.losses = [
    #   0.7213,
    #   0.6981,
    #   0.6814,
    #   ...
    # ]
    #
    # loss는 모델 예측이 정답과 얼마나 다른지 나타내는 값입니다.
    #
    # loss가 작아진다는 것은:
    #   모델의 예측이 학습 데이터의 정답에 가까워지고 있다는 의미입니다.
    #
    # loss가 전반적으로 감소하면:
    #   모델이 학습되고 있을 가능성이 높습니다.
    #
    # 단, loss가 매 epoch마다 반드시 계속 감소해야 하는 것은 아닙니다.
    #
    # 이유 1. mini-batch 학습
    #   DataLoader가 학습 데이터를 batch_size 단위로 나누어 학습합니다.
    #   각 batch마다 데이터 구성이 다르기 때문에 loss가 조금 흔들릴 수 있습니다.
    #
    # 이유 2. Dropout
    #   현재 FailureMLP에는 Dropout layer가 있습니다.
    #   Dropout은 학습 중 일부 뉴현 출력을 무작위로 0으로 만들기 때문에
    #   같은 데이터라도 학습 과정에서 loss가 약간 달라질 수 있습니다.
    #
    # 따라서 한 epoch의 loss가 직전 epoch보다 조금 올라갔다고 해서
    # 바로 학습 실패라고 판단하면 안 됩니다.
    #
    # 중요한 것은 전체적인 감소 추세입니다.
    #
    # enumerate(training_result.losses, start=1)는
    # loss 리스트를 반복하면서 epoch 번호와 loss 값을 함께 꺼냅니다.
    #
    # start=1을 사용하는 이유:
    #   Python의 기본 index는 0부터 시작하지만,
    #   사람이 보는 epoch 번호는 1부터 시작하는 것이 자연스럽기 때문입니다.
    #
    # 예:
    # losses = [0.72, 0.68, 0.65]
    #
    # enumerate(losses, start=1)
    # -> (1, 0.72), (2, 0.68), (3, 0.65)
    for epoch_index, loss in enumerate(training_result.losses, start=1):
        
        # f-string을 사용해 epoch 번호와 loss 값을 보기 좋게 출력합니다.
        #
        # {epoch_index:02d}
        #   epoch_index를 정수(d)로 출력합니다.
        #   최소 두 자리 수로 출력하고, 빈 자리는 0으로 채웁니다.
        #
        #   예:
        #   1 -> 01
        #   2 -> 02
        #   10 -> 10
        #
        # {loss:.6f}
        #   loss를 float 형식으로 출력합니다.
        #   소수점 아래 6자리까지 표시합니다.
        #
        #   예:
        #   0.721345678 -> 0.721346
        #
        # 최종 출력 예:
        #   epoch=01, loss=0.721346
        print(f"    epoch={epoch_index:02d}, loss={loss:.6f}")
    
    # 4. 학습된 모델을 test set으로 평가합니다.
    #
    # threshold=0.5:
    #   sigmoid probability가 0.5 이상이면 고장으로 판단하는 baseline 기준입니다.
    #
    # 제조 고장 예측에서는 threshold=0.5가 최종 정답이 아니며,
    # 이후 recall / precision trade-off를 보면서 조정할 수 있습니다.
    evaluation_result = evaluate_failure_model(
        model=training_result.model,
        X_test=processed.X_test,
        y_test=processed.y_test,
        threshold=0.5,
    )

    print("[INFO] Evaluation completed")
    print(f"[METRIC] accuracy : {evaluation_result.accuracy:.4f}")
    print(f"[METRIC] precision: {evaluation_result.precision:.4f}")
    print(f"[METRIC] recall   : {evaluation_result.recall:.4f}")
    print(f"[METRIC] f1       : {evaluation_result.f1:.4f}")
    print(f"[METRIC] threshold: {evaluation_result.threshold:.2f}")

    # 여기서 주의할 점:
    #
    # accuracy가 높다고 해서 모델이 좋은 것은 아닙니다.
    # AI4I 데이터에서 고장 class가 적다면,
    # 모델이 대부분 정상으로 예측해도 accuracy가 높게 나올 수 있습니다.
    #
    # 따라서 제조 고장 예측에서는 recall과 precision을 함께 봐야 합니다.
    #
    # recall:
    #   실제 고장 중 모델이 얼마나 잡아냈는가
    #
    # precision:
    #   모델이 고장이라고 한 것 중 실제 고장이 얼마나 되는가
    #
    # f1:
    #   precision과 recall의 균형

if __name__ == "__main__":
    # 이 파일을 직접 호출했을 때만 main()을 호출합니다.
    #
    # 예:
    # python scripts/run_train_failure_model.py
    #
    # 다른 파일에서 import 할 때는 main()이 자동 실행되지 않습니다.
    main()