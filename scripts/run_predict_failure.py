from pathlib import Path
import sys


# ------------------------------------------------------------
# 프로젝트 루트 경로를 Python import 경로에 추가합니다.
# ------------------------------------------------------------
# 이 스크립트는 scripts/ 폴더 안에 있습니다.
#
# PowerShell에서 아래처럼 실행하면:
#
# python scripts/run_predict_failure.py
#
# Python은 기본적으로 scripts/ 폴더를 기준으로 import를 찾을 수 있습니다.
# 그런데 src/ 폴더는 프로젝트 루트에 있으므로,
# 안전하게 프로젝트 루트를 sys.path에 추가합니다.
#
# .parents[1]은 현재 위치에서 두 단계 더 위로 올라간 부모 요소 ( 인덱스 )
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.inference.predict_failure import predict_failure

def print_prediction_result(
    sample_name: str,
    raw_sample: dict,
    artifact_dir: str,
) -> None:
    """
    하나의 raw sample에 대해 고장 예측을 실행하고 결과를 출력합니다.

    sample_name:
    - 출력에서 구분하기 위한 샘플 이름입니다.
    - 예: normal_sample, risky_sample

    raw_sample:
    - 아직 scaling되지 않은 원본 입력값입니다.

    artifact_dir:
    - model.pt, scaler.joblib, metadata.json이 저장된 폴더입니다.
    """

    print("=" * 80)
    print(f"[INFO] Sample name: {sample_name}")
    print("[INFO] Raw sample:")

    for key, value in raw_sample.items():
        print(f"    {key}: {value}")

    result = predict_failure(
        raw_sample=raw_sample,
        artifact_dir=artifact_dir,
    )

    print("[INFO] Failure prediction completed")
    print(f"[INFO] probability: {result.probability:.4f}")
    print(f"[INFO] threshold  : {result.threshold:.4f}")
    print(f"[INFO] prediction : {result.prediction}")
    print(f"[INFO] risk_level : {result.risk_level}")
    print(f"[INFO] recommended_action: {result.recommended_action}")

    print("[INFO] evidence:")
    for item in result.evidence:
        print(f"    - feature: {item['feature']}")
        print(f"      value  : {item['value']}")
        print(f"      message: {item['message']}")

    if result.prediction == 1:
        print("[INFO] Interpretation: 고장 위험으로 예측되었습니다.")
    else:
        print("[INFO] Interpretation: 정상으로 예측되었습니다.")


def main() -> None:
    """
    저장된 model/scaler/metadata를 사용해 단일 설비 샘플을 추론합니다.

    이 스크립트는 학습을 하지 않습니다.

    전제 조건:
    - scripts/run_train_failure_model.py를 먼저 실행해서
      models/failure_mlp/ 아래에 artifact가 저장되어 있어야 합니다.

    이번 단계의 목적:
    - 정상에 가까운 sample과 위험 신호가 있는 sample을 비교합니다.
    - probability, prediction, risk_level, evidence가 어떻게 달라지는지 확인합니다.

    사용되는 artifact:
    1. model.pt
       - 학습된 PyTorch 모델 weight

    2. scaler.joblib
       - train set에 fit된 StandardScaler

    3. metadata.json
       - threshold, feature_columns, input_dim 등 추론 설정값
    """

    artifact_dir = "models/failure_mlp"

    # ------------------------------------------------------------
    # 1. 정상에 가까운 샘플
    # ------------------------------------------------------------
    # Tool wear가 낮고, Torque도 높지 않은 입력입니다.
    # rule-based evidence 기준에서는 뚜렷한 위험 feature가 나오지 않을 가능성이 큽니다.
    normal_sample = {
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 312.0,
        "Rotational speed [rpm]": 1550.0,
        "Torque [Nm]": 42.0,
        "Tool wear [min]": 15.0,
        "Type": "L",
    }

    # ------------------------------------------------------------
    # 2. 위험 신호가 있는 샘플
    # ------------------------------------------------------------
    # 아래 값들은 일부러 위험 신호가 나오도록 구성한 입력입니다.
    #
    # Tool wear [min] >= 200
    # - 공구 마모 시간이 높음
    #
    # Torque [Nm] >= 60
    # - 설비 부하 가능성
    #
    # Rotational speed [rpm] <= 1300
    # - 회전 속도 저하 가능성
    #
    # Process temperature - Air temperature >= 12
    # - 공정 온도와 대기 온도의 차이가 큼
    risky_sample = {
        "Air temperature [K]": 298.0,
        "Process temperature [K]": 314.0,
        "Rotational speed [rpm]": 1250.0,
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 230.0,
        "Type": "H",
    }

    print("[INFO] Failure prediction comparison started")

    print_prediction_result(
        sample_name="normal_sample",
        raw_sample=normal_sample,
        artifact_dir=artifact_dir,
    )

    print_prediction_result(
        sample_name="risky_sample",
        raw_sample=risky_sample,
        artifact_dir=artifact_dir,
    )

    # print("=" * 80)
    # print("[INFO] Failure prediction comparison completed")


    # # ------------------------------------------------------------
    # # 단일 설비 raw input 예시
    # # ------------------------------------------------------------
    # # 이 값들은 아직 scaling되지 않은 원본 입력값입니다.
    # #
    # # 중요한 점:
    # # - feature 이름은 학습 때 사용한 column 이름과 같아야 합니다.
    # # - Type은 문자열 L/M/H로 넣어도 됩니다.
    # # - predict_failure 내부에서 Type을 0/1/2로 변환합니다.
    # raw_sample = {
    #     "Air temperature [K]": 302.0,
    #     "Process temperature [K]": 312.0,
    #     "Rotational speed [rpm]": 1550.0,
    #     "Torque [Nm]": 42.0,
    #     "Tool wear [min]": 15.0,
    #     "Type": "L",
    # }

    # print("[INFO] Failure prediction started")
    # print("[INFO] Raw sample:")
    # for key, value in raw_sample.items():
    #     print(f"    {key}: {value}")

    # result = predict_failure(
    #     raw_sample=raw_sample,
    #     artifact_dir=artifact_dir,
    # )

    # print("[INFO] Failure prediction completed")
    # print(f"[INFO] probability: {result.probability:.4f}")
    # print(f"[INFO] threshold  : {result.threshold:.4f}")
    # print(f"[INFO] prediction : {result.prediction}")
    # print(f"[INFO] risk_level : {result.risk_level}")
    # print(f"[INFO] recommended_action: {result.recommended_action}")

    # print("[INFO] evidence:")
    # for item in result.evidence:
    #     print(f"    - feature: {item['feature']}")
    #     print(f"      value  : {item['value']}")
    #     print(f"      message: {item['message']}")

    #     # ------------------------------------------------------------
    #     # 결과 해석
    #     # ------------------------------------------------------------
    #     # prediction = 0
    #     # - threshold 기준으로 정상으로 판단
    #     #
    #     # prediction = 1
    #     # - threshold 기준으로 고장 위험으로 판단
    #     #
    #     # risk_level
    #     # - probability를 사람이 이해하기 쉽게 LOW/MEDIUM/HIGH로 변환한 값
    #     if result.prediction == 1:
    #         print("[INFO] Interpretation: 고장 위험으로 예측되었습니다.")
    #     else:
    #         print("[INFO] Interpretation: 정상으로 예측되었습니다.")


if __name__ == "__main__":
    main()