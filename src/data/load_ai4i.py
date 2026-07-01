"""
AI4I CSV 데이터를 읽어오는 함수입니다.

이 파일은 데이터를 '로드'하는 책임만 가집니다.
결측치 처리, 컬럼 변환, train / test split 같은 전처리는 preprocess_ai4i.py에서 수행합니다.
"""

from pathlib import Path

import pandas as pd

# 기본 데이터 경로입니다.
# 실제 CSV 파일은 data/raw/ai4i/ai4i_2020.csv 위치에 둔다고 가정합니다.
DEFAULT_AI4I_CSV_PATH = Path("data/raw/ai4i/ai4i_2020.csv")

def load_ai4i_csv(csv_path: str | Path = DEFAULT_AI4I_CSV_PATH) -> pd.DataFrame:
    """
    AI4I CSV 파일을 pandas DataFrame으로 읽어옵니다.

    Args:
        csv_path:
            읽어올 CSV 파일 경로입니다.
            기본값은 data/raw/ai4i/ai4i_2020.csv입니다.

    Returns:
        pd.DataFrame:
            CSV 내용을 담은 pandas DataFrame입니다.

    Raises:
        FileNotFoundError:
            지정한 경로에 CSV 파일이 없을 애 발생합니다.
    """

    # 문자열 경로가 들어와도 Path 객체로 변환합니다.
    # Path를 쓰면 Windows와 Linux 경로 처리를 더 안정적으로 할 수 있습니다.
    csv_path = Path(csv_path)

    # 파일이 존재하지 않으면 명확한 에러 메시지를 보여줍니다.
    # 그냥 pd.read_csv에서 에러가 나게 두는 것보다,
    # 사용자가 무엇을 해야 하는지 알기 쉽습니다.
    if not csv_path.exists():
        raise FileNotFoundError(
            f"AI4I CSV 파일을 찾을 수 없습니다: {csv_path}]\n"
            "CSV 파일을 data/raw/ai4i/ai4i_2020.csv 위치에 넣어주세요."
        )

    # pandas로 CSV를 읽습니다.
    # 반환값은 DataFrame입니다.
    df = pd.read_csv(csv_path)

    # 읽어온 데이터프레임을 반환합니다.
    return df