# src/services/dataset_schema_service.py

"""
AI4I Dataset Schema application service.

이 파일의 역할
----------------
현재 프로젝트가 사용하는 AI4I 데이터셋의 정보를
공통 구조로 조합하여 반환합니다.

현재 이 정보는 다음 두 인터페이스에서 사용할 예정입니다.

1. LangGraph

    dataset_schema_query
        ↓

    build_dataset_schema_answer_node()
        ↓

    이 Service 호출
        ↓

    Agent answer와 evidence 생성


2. MCP

    MCP Client
        ↓

    MCP Server
        ↓

    get_dataset_schema Tool
        ↓

    이 Service 호출
        ↓

    구조화된 Dataset Schema 반환


중요한 설계 원칙
----------------
feature 이름, target 이름, 제외 column을
이 파일에 문자열로 다시 복사하지 않습니다.

실제 기준값은 이미 다음 파일에 있습니다.

    src/data/schemas.py

따라서 이 Service는 기존 상수를 import한 뒤
API·Agent·MCP가 사용하기 좋은 형태로 조합합니다.


계층 구조
---------
src/data/schemas.py

    데이터 column의 기준값


        ↓


src/services/dataset_schema_service.py

    기존 기준값을 application 결과로 조합


        ↓                    ↓


LangGraph node          MCP Tool
"""

from __future__ import annotations

# Any
# ---
# evidence item은 문자열, 숫자, None, list, dict 등
# 여러 JSON 호환 값을 포함합니다.
#
# 따라서 evidence의 value type을
# 하나의 고정 타입으로 제한하기보다 Any를 사용합니다.
#
#
# TypedDict
# ---------
# 일반 dict의 key와 value type을
# 타입 검사기와 IDE가 이해할 수 있게 정의합니다.
#
# 실행 중 새로운 객체를 만드는 class가 아니라,
# dict의 예상 구조를 설명하는 타입 힌트입니다.
from typing import Any, TypedDict

# AI4I 데이터 구조의 실제 기준값입니다.
#
# 이 Service 안에 feature 이름을 다시 문자열로 작성하지 않고,
# 기존 데이터 계층의 상수를 재사용합니다.
from src.data.schemas import (
    AI4I_CATEGORICAL_COLUMNS,
    AI4I_DROP_COLUMNS,
    AI4I_FEATURE_COLUMNS,
    AI4I_TARGET_COLUMN,
    AI4I_TYPE_MAPPING,
)


# Dataset 이름도 여러 함수에서 반복하지 않도록
# module 상수로 한 번만 정의합니다.
#
# 이 값은 모델 입력 column과 달리
# 데이터 전처리에 직접 사용되는 값은 아닙니다.
#
# 따라서 src/data/schemas.py가 아니라
# Dataset Schema 응답을 만드는 현재 Service에 둡니다.
AI4I_DATASET_NAME = (
    "AI4I 2020 Predictive Maintenance Dataset"
)


class AI4IDatasetSchema(TypedDict):
    """
    AI4I Dataset Schema Service가 반환하는
    구조화된 dict의 타입입니다.

    TypedDict를 사용하는 이유
    -------------------------
    일반 dict만 사용하면 IDE나 타입 검사기는
    어떤 key가 반드시 존재하는지 알기 어렵습니다.

    예:

        result["features"]

        result["target"]


    TypedDict를 사용하면 다음 구조를
    코드 수준에서 명확하게 표현할 수 있습니다.

        dataset

        features

        numeric_features

        categorical_features

        target

        excluded_columns

        categorical_mappings

        current_encoding_note

        improvement_note


    왜 Pydantic BaseModel을 사용하지 않는가?
    --------------------------------------
    이 객체는 HTTP request나 MCP request를
    직접 검증하는 transport schema가 아닙니다.

    현재는 내부 Application Service의
    단순하고 예측 가능한 반환 구조입니다.

    따라서 Pydantic 의존성을 추가하지 않고
    표준 Python dict + TypedDict를 사용합니다.
    """

    # 사람이 확인할 수 있는 Dataset 이름입니다.
    dataset: str

    # 현재 PyTorch 모델에 실제로 입력되는
    # 전체 feature 순서입니다.
    #
    # numeric feature 5개
    #
    # +
    #
    # categorical feature Type
    #
    # =
    #
    # 총 6개
    features: list[str]

    # scaling 대상인 숫자형 feature입니다.
    numeric_features: list[str]

    # 문자열 값을 숫자로 변환해야 하는
    # 범주형 feature입니다.
    categorical_features: list[str]

    # 모델이 예측하는 정답 column입니다.
    target: str

    # 모델 학습 feature에서 제외하는 column입니다.
    excluded_columns: list[str]

    # 범주형 feature의 현재 mapping 정보입니다.
    #
    # 현재:
    #
    # {
    #     "Type": {
    #         "L": 0,
    #         "M": 1,
    #         "H": 2,
    #     }
    # }
    categorical_mappings: dict[
        str,
        dict[str, int],
    ]

    # 현재 Type encoding 방식에 대한 설명입니다.
    current_encoding_note: str

    # 현재 encoding 방식의 한계와 개선 방향입니다.
    improvement_note: str


class DatasetSchemaAgentResult(TypedDict):
    """
    LangGraph dataset schema node가 사용할 결과 타입입니다.

    LangGraph node에는 다음 두 값이 필요합니다.

        answer

        evidence

    MCP Tool은 전체 AgentState가 필요하지 않으므로
    이 타입을 직접 사용하지 않고
    AI4IDatasetSchema 구조만 반환할 예정입니다.
    """

    # 사용자에게 보여줄 자연어 답변입니다.
    answer: str

    # 답변 근거로 사용하는 구조화 evidence입니다.
    evidence: list[dict[str, Any]]


def get_ai4i_dataset_schema() -> AI4IDatasetSchema:
    """
    현재 프로젝트가 사용하는
    AI4I Dataset Schema를 구조화하여 반환합니다.

    Parameters
    ----------
    없음

    이 함수는 외부 입력을 받지 않습니다.

    현재 프로젝트에서 사용할 Dataset과
    feature 기준이 이미 코드 상수로 정해져 있기 때문입니다.


    Returns
    -------
    AI4IDatasetSchema

    반환 예:

        {
            "dataset": (
                "AI4I 2020 "
                "Predictive Maintenance Dataset"
            ),
            "features": [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Type",
            ],
            "target": "Machine failure",
            ...
        }


    왜 새 list와 새 dict를 만드는가?
    -------------------------------
    다음과 같이 기존 module 상수를
    그대로 반환할 수도 있습니다.

        "features": AI4I_FEATURE_COLUMNS


    그러나 반환받은 코드가 list를 수정하면
    module의 원본 상수까지 변경될 가능성이 있습니다.

    따라서:

        list(...)

        dict(...)

    로 복사본을 만들어 반환합니다.


    데이터 흐름
    ----------
    src/data/schemas.py

        ↓

    get_ai4i_dataset_schema()

        ↓

    LangGraph 또는 MCP
    """

    # 숫자형 feature 5개와
    # 범주형 feature 1개를 합쳐
    # 실제 모델 입력 feature 순서를 만듭니다.
    features = (
        list(AI4I_FEATURE_COLUMNS)
        +
        list(AI4I_CATEGORICAL_COLUMNS)
    )

    # Type mapping도 새로운 dict로 복사합니다.
    #
    # 이렇게 하면 호출자가 반환값을 수정해도
    # src/data/schemas.py의 원본 mapping은 유지됩니다.
    type_mapping = dict(AI4I_TYPE_MAPPING)

    return {
        "dataset": AI4I_DATASET_NAME,
        "features": features,
        "numeric_features": list(
            AI4I_FEATURE_COLUMNS
        ),
        "categorical_features": list(
            AI4I_CATEGORICAL_COLUMNS
        ),
        "target": AI4I_TARGET_COLUMN,
        "excluded_columns": list(
            AI4I_DROP_COLUMNS
        ),
        "categorical_mappings": {
            "Type": type_mapping,
        },
        "current_encoding_note": (
            "Type은 현재 L/M/H 값을 "
            "0/1/2 숫자로 mapping해 사용합니다."
        ),
        "improvement_note": (
            "현재 mapping은 L/M/H 사이에 "
            "순서 관계가 있는 것처럼 "
            "모델이 해석할 가능성이 있으므로, "
            "향후 one-hot encoding을 "
            "검토할 수 있습니다."
        ),
    }


def build_ai4i_dataset_schema_answer(
    schema: AI4IDatasetSchema | None = None,
) -> str:
    """
    AI4I Dataset Schema를
    사용자용 자연어 답변으로 변환합니다.

    Parameters
    ----------
    schema:
        이미 조회한 AI4IDatasetSchema입니다.

        기본값은 None입니다.

        None이면 함수 내부에서
        get_ai4i_dataset_schema()를 호출합니다.


    왜 schema parameter를 선택적으로 받는가?
    --------------------------------------
    다음 함수에서는 schema를 이미 한 번 조회합니다.

        build_ai4i_dataset_schema_agent_result()


    그때 같은 schema를 다시 만들지 않고
    현재 함수에 전달할 수 있습니다.

    반면 이 함수만 독립적으로 사용할 때는
    argument 없이 호출할 수도 있습니다.


    Returns
    -------
    str

    LangGraph Agent가 사용자에게 보여줄
    Dataset Schema 설명입니다.
    """

    # 호출자가 schema를 전달하지 않았다면
    # 현재 프로젝트의 기본 AI4I schema를 조회합니다.
    if schema is None:
        schema = get_ai4i_dataset_schema()

    # feature 수가 바뀌더라도
    # "6개"를 직접 수정하지 않도록
    # 실제 list 길이에서 계산합니다.
    feature_count = len(
        schema["features"]
    )

    # feature list도 문자열로 직접 반복하지 않고
    # 현재 schema 결과로 동적으로 만듭니다.
    feature_lines = "\n".join(
        f"- {feature}"
        for feature in schema["features"]
    )

    # 제외 column도 현재 schema 결과를 사용해
    # 자연어 문자열로 조합합니다.
    excluded_column_text = "와 ".join(
        schema["excluded_columns"]
    )

    return (
        f"현재 프로젝트는 "
        f"{schema['dataset']}을 사용합니다.\n\n"
        f"모델 입력 feature는 "
        f"다음 {feature_count}개입니다.\n"
        f"{feature_lines}\n\n"
        f"target은 {schema['target']}입니다.\n"
        f"{excluded_column_text}는 "
        f"식별자이므로 학습 feature에서 제외합니다.\n"
        f"{schema['current_encoding_note']} "
        f"{schema['improvement_note']}"
    )


def build_ai4i_dataset_schema_evidence(
    schema: AI4IDatasetSchema | None = None,
) -> list[dict[str, Any]]:
    """
    AI4I Dataset Schema를
    Agent evidence 형식으로 변환합니다.

    Parameters
    ----------
    schema:
        이미 조회한 AI4IDatasetSchema입니다.

        None이면 함수 내부에서
        현재 프로젝트 schema를 조회합니다.


    Returns
    -------
    list[dict[str, Any]]

    현재 Dataset Schema evidence 한 건을
    list 안에 넣어 반환합니다.


    왜 list를 반환하는가?
    --------------------
    현재 AgentState의 evidence 타입이
    여러 evidence item을 담는 list 구조이기 때문입니다.

    Dataset Schema는 현재 evidence 한 건이지만,
    Agent 전체 구조와 일관성을 유지하기 위해
    list 형식을 사용합니다.
    """

    if schema is None:
        schema = get_ai4i_dataset_schema()

    feature_count = len(
        schema["features"]
    )

    return [
        {
            "evidence_id": (
                "dataset_schema_001"
            ),
            "evidence_type": (
                "dataset_schema"
            ),
            "source": (
                "project_schema"
            ),
            "title": (
                "AI4I 데이터셋 schema"
            ),
            "summary": (
                "현재 모델은 "
                f"AI4I feature {feature_count}개를 "
                f"사용하고, "
                f"{schema['target']}를 "
                "target으로 사용합니다."
            ),
            "feature": None,
            "value": None,
            "direction": None,
            "contribution": None,
            "importance": None,
            "severity": "LOW",
            "metadata": {
                "features": list(
                    schema["features"]
                ),
                "target": (
                    schema["target"]
                ),
                "excluded_columns": list(
                    schema[
                        "excluded_columns"
                    ]
                ),
            },
        }
    ]


def build_ai4i_dataset_schema_agent_result(
) -> DatasetSchemaAgentResult:
    """
    LangGraph dataset schema node가 사용할
    answer와 evidence를 함께 생성합니다.

    Parameters
    ----------
    없음


    Returns
    -------
    DatasetSchemaAgentResult

    반환 구조:

        {
            "answer": "...",
            "evidence": [
                {
                    ...
                }
            ],
        }


    실행 순서
    ----------
    1.

    get_ai4i_dataset_schema()

        현재 AI4I 구조 조회


    2.

    build_ai4i_dataset_schema_answer()

        사용자용 자연어 답변 생성


    3.

    build_ai4i_dataset_schema_evidence()

        구조화 evidence 생성


    4.

    answer와 evidence를 하나의 dict로 반환


    왜 이 함수가 필요한가?
    ---------------------
    LangGraph node가 Service 내부 세부 함수를
    여러 개 직접 호출하지 않도록 하기 위해서입니다.

    LangGraph node는 앞으로 다음 정도의
    단순한 연결 역할만 담당합니다.

        result = (
            build_ai4i_dataset_schema_agent_result()
        )

        state["answer"] = result["answer"]

        state["evidence"] = result["evidence"]


    이렇게 하면 실제 Dataset Schema 생성 책임은
    Service에 남고,
    LangGraph node는 workflow state 관리에 집중합니다.
    """

    # schema를 한 번만 생성합니다.
    schema = get_ai4i_dataset_schema()

    return {
        "answer": (
            build_ai4i_dataset_schema_answer(
                schema
            )
        ),
        "evidence": (
            build_ai4i_dataset_schema_evidence(
                schema
            )
        ),
    }