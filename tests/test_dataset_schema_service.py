# tests/test_dataset_schema_service.py

"""
Day 20 - AI4I Dataset Schema Service tests.

이 테스트 파일의 목적
---------------------
Day 20에서는 Dataset Schema 정보를
LangGraph node 안에서 직접 작성하지 않고
공통 Application Service로 분리했습니다.

현재 구조:

src/data/schemas.py

    AI4I feature·target 기준값

        ↓

src/services/dataset_schema_service.py

    Dataset Schema 구조 생성

    자연어 answer 생성

    Agent evidence 생성

        ↓                 ↓

LangGraph             MCP Tool


이번 테스트에서는 아직 MCP SDK를 사용하지 않습니다.

먼저 MCP와 독립적인 공통 Service가
정확하게 동작하는지 검증합니다.


검증 항목
---------
1.

Dataset 이름과 필수 key가 정상인가?


2.

기존 AI4I 상수를 재사용하는가?


3.

숫자형 feature와 범주형 feature가
올바른 순서로 결합되는가?


4.

반환된 list·dict를 수정해도
원본 AI4I 상수가 변경되지 않는가?


5.

사용자용 answer에 핵심 정보가 포함되는가?


6.

LangGraph용 answer·evidence 결과가
기존 Agent 계약을 유지하는가?
"""

from src.data.schemas import (
    AI4I_CATEGORICAL_COLUMNS,
    AI4I_DROP_COLUMNS,
    AI4I_FEATURE_COLUMNS,
    AI4I_TARGET_COLUMN,
    AI4I_TYPE_MAPPING,
)
from src.services.dataset_schema_service import (
    AI4I_DATASET_NAME,
    build_ai4i_dataset_schema_agent_result,
    build_ai4i_dataset_schema_answer,
    build_ai4i_dataset_schema_evidence,
    get_ai4i_dataset_schema,
)


def test_get_ai4i_dataset_schema_returns_expected_structure():
    """
    Dataset Schema Service가
    필수 key와 예상 데이터 타입을 반환하는지 검증합니다.

    이 테스트가 필요한 이유
    ----------------------
    MCP Tool은 이후 이 구조화 결과를
    그대로 반환할 예정입니다.

    따라서 MCP를 연결하기 전에
    Service의 기본 output 계약을 먼저 확인합니다.
    """

    # 현재 프로젝트의 AI4I Dataset Schema를 조회합니다.
    schema = get_ai4i_dataset_schema()

    # Dataset 이름이 Service에서 정의한
    # 공식 프로젝트 이름과 같아야 합니다.
    assert schema["dataset"] == AI4I_DATASET_NAME

    # 현재 모델 입력 feature는:
    #
    # numeric feature 5개
    #
    # +
    #
    # Type 1개
    #
    # =
    #
    # 총 6개입니다.
    assert len(schema["features"]) == 6

    # 반환값의 각 주요 field가
    # 예상 Python 타입인지 확인합니다.
    assert isinstance(schema["features"], list)

    assert isinstance(
        schema["numeric_features"],
        list,
    )

    assert isinstance(
        schema["categorical_features"],
        list,
    )

    assert isinstance(
        schema["excluded_columns"],
        list,
    )

    assert isinstance(
        schema["categorical_mappings"],
        dict,
    )

    assert isinstance(
        schema["current_encoding_note"],
        str,
    )

    assert isinstance(
        schema["improvement_note"],
        str,
    )


def test_get_ai4i_dataset_schema_reuses_project_constants():
    """
    Dataset Schema Service가
    src/data/schemas.py의 기존 기준값을
    정확하게 재사용하는지 검증합니다.

    잘못된 구현 예
    -------------
    Service 안에 다음 문자열을
    다시 직접 작성하는 경우입니다.

        "Air temperature [K]"

        "Machine failure"

        "UDI"


    올바른 구현
    -----------
    기존 상수를 import한 뒤
    현재 응답 구조로 조합합니다.
    """

    schema = get_ai4i_dataset_schema()

    # 전체 모델 입력 feature는
    # 숫자형 feature와 범주형 feature를
    # 순서대로 합친 결과여야 합니다.
    expected_features = (
        list(AI4I_FEATURE_COLUMNS)
        +
        list(AI4I_CATEGORICAL_COLUMNS)
    )

    assert (
        schema["features"]
        ==
        expected_features
    )

    # 숫자형 feature는
    # 기존 데이터 계층 상수와 같아야 합니다.
    assert (
        schema["numeric_features"]
        ==
        AI4I_FEATURE_COLUMNS
    )

    # 범주형 feature도
    # 기존 상수와 같아야 합니다.
    assert (
        schema["categorical_features"]
        ==
        AI4I_CATEGORICAL_COLUMNS
    )

    # Target과 제외 column도
    # 기존 기준값을 유지해야 합니다.
    assert (
        schema["target"]
        ==
        AI4I_TARGET_COLUMN
    )

    assert (
        schema["excluded_columns"]
        ==
        AI4I_DROP_COLUMNS
    )

    # 현재 Type mapping도
    # 기존 데이터 기준과 같아야 합니다.
    assert (
        schema["categorical_mappings"]["Type"]
        ==
        AI4I_TYPE_MAPPING
    )


def test_get_ai4i_dataset_schema_returns_safe_copies():
    """
    Service 반환값을 수정해도
    src/data/schemas.py의 원본 상수가
    변경되지 않는지 검증합니다.

    왜 필요한가?
    ------------
    Python의 list와 dict는 mutable 객체입니다.

    만약 Service가 원본 list를 그대로 반환하면:

        schema["features"].append(...)

    같은 코드가 원본 상수까지
    변경할 가능성이 있습니다.

    현재 Service는 list()와 dict()를 사용해
    새로운 복사본을 반환합니다.
    """

    # 첫 번째 Schema 결과를 가져옵니다.
    first_schema = get_ai4i_dataset_schema()

    # 반환된 list와 dict를 의도적으로 수정합니다.
    first_schema["features"].append(
        "Temporary test feature"
    )

    first_schema[
        "numeric_features"
    ].clear()

    first_schema[
        "categorical_features"
    ].append(
        "Temporary category"
    )

    first_schema[
        "excluded_columns"
    ].append(
        "Temporary excluded column"
    )

    first_schema[
        "categorical_mappings"
    ]["Type"]["L"] = 999

    # Service를 다시 호출합니다.
    #
    # 원본 상수가 보호되고 있다면
    # 첫 반환값의 수정 내용이
    # 새 결과에 남아 있지 않아야 합니다.
    second_schema = get_ai4i_dataset_schema()

    expected_features = (
        list(AI4I_FEATURE_COLUMNS)
        +
        list(AI4I_CATEGORICAL_COLUMNS)
    )

    assert (
        second_schema["features"]
        ==
        expected_features
    )

    assert (
        second_schema["numeric_features"]
        ==
        AI4I_FEATURE_COLUMNS
    )

    assert (
        second_schema[
            "categorical_features"
        ]
        ==
        AI4I_CATEGORICAL_COLUMNS
    )

    assert (
        second_schema["excluded_columns"]
        ==
        AI4I_DROP_COLUMNS
    )

    assert (
        second_schema[
            "categorical_mappings"
        ]["Type"]
        ==
        AI4I_TYPE_MAPPING
    )


def test_build_ai4i_dataset_schema_answer_contains_core_information():
    """
    사용자용 Dataset Schema answer에
    핵심 정보가 포함되는지 검증합니다.

    Answer 문장 전체를 완전히 동일하게
    비교하지 않는 이유
    -----------------
    자연어 표현은 이후 읽기 좋게
    일부 수정될 수 있습니다.

    따라서 현재 기능 계약에 중요한
    핵심 정보가 포함되는지를 검증합니다.
    """

    answer = (
        build_ai4i_dataset_schema_answer()
    )

    # Dataset 이름을 설명해야 합니다.
    assert (
        "AI4I 2020 "
        "Predictive Maintenance Dataset"
        in answer
    )

    # 현재 feature 수를 설명해야 합니다.
    assert (
        "다음 6개"
        in answer
    )

    # 대표 숫자형 feature가 있어야 합니다.
    assert (
        "Air temperature [K]"
        in answer
    )

    # 범주형 feature도 있어야 합니다.
    assert (
        "Type"
        in answer
    )

    # Target을 설명해야 합니다.
    assert (
        "Machine failure"
        in answer
    )

    # 제외 column도 설명해야 합니다.
    assert "UDI" in answer

    assert "Product ID" in answer

    # 현재 Type mapping 방식과
    # 개선 방향도 설명해야 합니다.
    assert "0/1/2" in answer

    assert (
        "one-hot encoding"
        in answer
    )


def test_build_ai4i_dataset_schema_evidence_keeps_agent_contract():
    """
    Dataset Schema evidence가
    기존 Agent evidence 계약을 유지하는지 검증합니다.

    기존 LangGraph·FastAPI는
    evidence list 안에서 다음 정보를 사용합니다.

        evidence_id

        evidence_type

        source

        title

        summary

        severity

        metadata
    """

    evidence = (
        build_ai4i_dataset_schema_evidence()
    )

    # 현재 Dataset Schema evidence는
    # 한 건입니다.
    assert len(evidence) == 1

    item = evidence[0]

    # 기존 LangGraph 테스트에서 사용하는
    # 핵심 evidence type을 유지해야 합니다.
    assert (
        item["evidence_type"]
        ==
        "dataset_schema"
    )

    assert (
        item["evidence_id"]
        ==
        "dataset_schema_001"
    )

    assert (
        item["source"]
        ==
        "project_schema"
    )

    assert (
        item["severity"]
        ==
        "LOW"
    )

    # Metadata 안에는 MCP와 Agent가
    # 재사용할 수 있는 구조화 Dataset 정보가
    # 들어 있어야 합니다.
    metadata = item["metadata"]

    expected_features = (
        list(AI4I_FEATURE_COLUMNS)
        +
        list(AI4I_CATEGORICAL_COLUMNS)
    )

    assert (
        metadata["features"]
        ==
        expected_features
    )

    assert (
        metadata["target"]
        ==
        AI4I_TARGET_COLUMN
    )

    assert (
        metadata["excluded_columns"]
        ==
        AI4I_DROP_COLUMNS
    )


def test_build_ai4i_dataset_schema_agent_result_returns_answer_and_evidence():
    """
    LangGraph node용 통합 Service 함수가
    answer와 evidence를 함께 반환하는지 검증합니다.

    실제 LangGraph node 흐름
    -----------------------
    result = (
        build_ai4i_dataset_schema_agent_result()
    )

    state["answer"] = result["answer"]

    state["evidence"] = result["evidence"]
    """

    result = (
        build_ai4i_dataset_schema_agent_result()
    )

    # LangGraph node가 사용하는
    # 두 key가 모두 있어야 합니다.
    assert set(result) == {
        "answer",
        "evidence",
    }

    # Answer는 비어 있지 않은 문자열이어야 합니다.
    assert isinstance(
        result["answer"],
        str,
    )

    assert result["answer"]

    # Evidence는 한 건 이상 존재해야 합니다.
    assert isinstance(
        result["evidence"],
        list,
    )

    assert len(
        result["evidence"]
    ) == 1

    assert (
        result["evidence"][0][
            "evidence_type"
        ]
        ==
        "dataset_schema"
    )

    # Answer와 Evidence가
    # 동일한 Target 기준을 사용해야 합니다.
    assert (
        AI4I_TARGET_COLUMN
        in result["answer"]
    )

    assert (
        result["evidence"][0][
            "metadata"
        ]["target"]
        ==
        AI4I_TARGET_COLUMN
    )