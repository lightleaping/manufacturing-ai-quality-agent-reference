"""
Day 19 - Agent 실행 이력 SQLite Persistence 테스트

이 파일의 역할
----------------
src/persistence/execution_history.py에서 구현한
Agent 실행 이력 저장·조회 기능을 검증합니다.

검증 대상:

    SQLite table 생성

    created_at index 생성

    Agent 실행 이력 INSERT

    SQLite 내부 execution id 반환

    trace_id 기준 상세 조회

    JSON TEXT 직렬화

    JSON TEXT 역직렬화

    Python None <-> SQL NULL

    Python bool <-> SQLite INTEGER

    동일 trace_id 중복 저장 방지

    없는 trace_id 조회

    최근 실행 목록

    최신 실행 우선 정렬

    LIMIT 적용

    잘못된 limit 검증

    SQLite connection 종료


중요
----
이 테스트는 실제 애플리케이션 DB를 사용하지 않습니다.

실제 DB:

    data/runtime/
    agent_execution_history.db

테스트 DB:

    pytest tmp_path/
    test_agent_execution_history.db

pytest의 tmp_path fixture가
각 테스트용 임시 폴더를 생성합니다.

따라서 테스트 데이터를 저장해도
실제 Agent 실행 이력이 오염되지 않습니다.


실제 OpenAI를 호출하지 않는 이유
-------------------------------
Persistence 단위 테스트의 목적은
SQLite 저장·조회 기능을 검증하는 것입니다.

OpenAI intent classifier까지 호출하면:

    네트워크 필요

    OPENAI_API_KEY 필요

    API 비용 발생 가능

    응답시간 증가

    외부 서비스 상태에 따라 테스트 불안정

문제가 생깁니다.

따라서 이미 완성된 final AgentState와 비슷한
고정 dictionary를 사용합니다.
"""

from __future__ import annotations

import json
import sqlite3

from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

import pytest

import src.persistence.execution_history as execution_history

from src.agent.state import AgentState
from src.persistence.execution_history import (
    build_execution_record,
    get_execution_by_trace_id,
    initialize_database,
    insert_execution,
    list_recent_executions,
)


# ---------------------------------------------------------------------
# 테스트용 AgentState 생성 helper
# ---------------------------------------------------------------------

def build_sample_state(
    *,
    trace_id: str = "day19-test-trace-001",
    question: str = (
        "이 설비의 고장 위험을 예측해줘."
    ),
    probability: float = 0.9929,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    include_raw_sample: bool = True,
) -> AgentState:
    """
    Persistence 테스트에 사용할
    고정 AgentState를 생성합니다.

    왜 helper 함수를 사용하는가?
    ----------------------------
    여러 테스트에서 같은 구조의 AgentState를
    반복해서 직접 작성하면 코드가 길어집니다.

    또한 필드 하나를 변경할 때
    여러 테스트를 모두 수정해야 할 수 있습니다.

    helper 함수를 사용하면:

        공통 기본값

        +

        테스트마다 필요한 일부 값 변경

    구조로 재사용할 수 있습니다.

    Parameters
    ----------
    trace_id:
        테스트 실행을 구분할 trace ID입니다.

    question:
        저장할 사용자 질문입니다.

    probability:
        테스트용 고장 probability입니다.

    warnings:
        저장할 warning 목록입니다.

    errors:
        저장할 error 목록입니다.

    include_raw_sample:
        True:

            raw_sample 포함

        False:

            raw_sample key 생략

            SQL NULL 동작 확인에 사용

    Returns
    -------
    AgentState
        Persistence 테스트용 AgentState입니다.
    """

    state: AgentState = {
        # -------------------------------------------------------------
        # 사용자 요청
        # -------------------------------------------------------------

        "question": question,

        # -------------------------------------------------------------
        # Intent 분류
        # -------------------------------------------------------------

        "intent": "failure_prediction",

        "intent_source": "openai",

        "confidence": 0.95,

        "intent_reason": (
            "사용자가 설비 고장 위험 예측을 "
            "요청했습니다."
        ),

        # -------------------------------------------------------------
        # Prediction
        # -------------------------------------------------------------

        "prediction": 1,

        "probability": probability,

        "threshold": 0.7,

        "risk_level": "HIGH",

        "recommended_action": (
            "설비 점검 및 생산 조건 확인을 "
            "권장합니다."
        ),

        # -------------------------------------------------------------
        # Evidence·Answer
        # -------------------------------------------------------------

        "answer": (
            "현재 입력 기준 고장 위험이 높습니다."
        ),

        "evidence": [
            {
                "evidence_type": (
                    "prediction_summary"
                ),
                "source": "model_prediction",
                "summary": (
                    "고장 probability는 "
                    f"{probability:.4f}입니다."
                ),
            },
            {
                "evidence_type": "feature_value",
                "source": "raw_sample",
                "feature": "Torque [Nm]",
                "value": 62.0,
            },
        ],

        "warnings": list(
            warnings or []
        ),

        "errors": list(
            errors or []
        ),

        "limitations": [
            (
                "현재 결과는 학습용 AI4I 모델의 "
                "예측입니다."
            )
        ],

        # -------------------------------------------------------------
        # Trace
        # -------------------------------------------------------------

        "trace_id": trace_id,

        "trace_status": "success",

        "trace_started_at": (
            "2026-07-10T04:00:00+00:00"
        ),

        "trace_finished_at": (
            "2026-07-10T04:00:02+00:00"
        ),

        "trace_duration_ms": 2000.0,

        "fallback_occurred": False,

        "selected_route": "final",

        "trace_events": [
            {
                "sequence": 1,
                "event_type": "node",
                "event_name": (
                    "validate_question"
                ),
                "status": "success",
                "started_at": (
                    "2026-07-10T04:00:00+00:00"
                ),
                "finished_at": (
                    "2026-07-10T04:00:00.001"
                    "+00:00"
                ),
                "duration_ms": 1.0,
                "metadata": {
                    "question_valid": True,
                },
            },
            {
                "sequence": 2,
                "event_type": "route",
                "event_name": (
                    "route_after_validation"
                ),
                "status": "success",
                "started_at": (
                    "2026-07-10T04:00:00.002"
                    "+00:00"
                ),
                "finished_at": (
                    "2026-07-10T04:00:00.003"
                    "+00:00"
                ),
                "duration_ms": 1.0,
                "metadata": {
                    "selected_route": "classify",
                },
            },
        ],
    }

    # raw_sample이 필요한 테스트에서만
    # State에 추가합니다.
    #
    # include_raw_sample=False이면
    # raw_sample key 자체가 없는 상태를 만듭니다.
    if include_raw_sample:
        state["raw_sample"] = {
            "Air temperature [K]": 303.0,
            "Process temperature [K]": 312.5,
            "Rotational speed [rpm]": 1380.0,
            "Torque [Nm]": 62.0,
            "Tool wear [min]": 220.0,
            "Type": "L",
        }

    return state


# ---------------------------------------------------------------------
# DB table·index 생성 테스트
# ---------------------------------------------------------------------

def test_initialize_database_creates_table_and_index(
    tmp_path: Path,
) -> None:
    """
    initialize_database()가

        agent_executions table

        created_at index

    를 생성하는지 확인합니다.
    """

    # tmp_path는 pytest가 제공하는
    # 테스트 전용 임시 폴더입니다.
    db_path = (
        tmp_path
        / "test_initialize_database.db"
    )

    initialize_database(
        db_path=db_path
    )

    # DB 파일이 실제 생성됐는지 확인합니다.
    assert db_path.exists()

    connection = sqlite3.connect(
        db_path
    )

    try:
        # sqlite_master는 SQLite 내부 schema 정보를
        # 확인할 수 있는 system table입니다.
        table_names = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = ?
                """,
                (
                    "table",
                ),
            ).fetchall()
        }

        index_names = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = ?
                """,
                (
                    "index",
                ),
            ).fetchall()
        }

    finally:
        connection.close()

    assert "agent_executions" in table_names

    assert (
        "idx_agent_executions_created_at"
        in index_names
    )


# ---------------------------------------------------------------------
# INSERT 기본 동작 테스트
# ---------------------------------------------------------------------

def test_insert_execution_returns_id_and_saves_core_fields(
    tmp_path: Path,
) -> None:
    """
    AgentState를 INSERT하면

        내부 execution id 반환

        핵심 컬럼 저장

    이 정상 동작하는지 확인합니다.
    """

    db_path = (
        tmp_path
        / "test_insert_execution.db"
    )

    # 작은따옴표가 포함된 질문을 사용합니다.
    #
    # parameter binding이 정상이라면
    # SQL 문법이 깨지지 않고 그대로 저장됩니다.
    question = (
        "이 설비가 '위험'한지 예측해줘."
    )

    state = build_sample_state(
        question=question,
    )

    execution_id = insert_execution(
        state=state,
        db_path=db_path,
    )

    # 첫 INSERT이므로 SQLite 내부 id는 1입니다.
    assert execution_id == 1

    connection = sqlite3.connect(
        db_path
    )

    try:
        row = connection.execute(
            """
            SELECT
                id,
                trace_id,
                question,
                intent,
                prediction,
                probability,
                warning_count,
                error_count
            FROM agent_executions
            """
        ).fetchone()

    finally:
        connection.close()

    assert row is not None

    assert row[0] == 1

    assert row[1] == (
        "day19-test-trace-001"
    )

    assert row[2] == question

    assert row[3] == "failure_prediction"

    assert row[4] == 1

    assert row[5] == pytest.approx(
        0.9929
    )

    assert row[6] == 0

    assert row[7] == 0


# ---------------------------------------------------------------------
# JSON 직렬화·역직렬화 테스트
# ---------------------------------------------------------------------

def test_detail_lookup_restores_json_and_boolean(
    tmp_path: Path,
) -> None:
    """
    JSON TEXT 컬럼이 실제 DB에서는 문자열이고,
    상세 조회 후에는 원래 Python 구조로
    복원되는지 확인합니다.
    """

    db_path = (
        tmp_path
        / "test_json_restore.db"
    )

    state = build_sample_state(
        warnings=[
            "SHAP evidence 생성에 실패했습니다."
        ],
    )

    insert_execution(
        state=state,
        db_path=db_path,
    )

    # 먼저 SQLite 내부 원본 값을 확인합니다.
    connection = sqlite3.connect(
        db_path
    )

    try:
        raw_row = connection.execute(
            """
            SELECT
                raw_sample_json,
                evidence_json,
                trace_events_json,
                warnings_json,
                fallback_occurred
            FROM agent_executions
            """
        ).fetchone()

    finally:
        connection.close()

    assert raw_row is not None

    # SQLite TEXT에는 실제 문자열이 저장됩니다.
    assert isinstance(
        raw_row[0],
        str,
    )

    assert isinstance(
        raw_row[1],
        str,
    )

    assert isinstance(
        raw_row[2],
        str,
    )

    assert isinstance(
        raw_row[3],
        str,
    )

    # JSON 문자열 자체도 정상적인 JSON인지 확인합니다.
    assert (
        json.loads(
            raw_row[0]
        )["Torque [Nm]"]
        == 62.0
    )

    # False는 SQLite INTEGER 0으로 저장됩니다.
    assert raw_row[4] == 0

    execution = get_execution_by_trace_id(
        trace_id=(
            "day19-test-trace-001"
        ),
        db_path=db_path,
    )

    assert execution is not None

    # JSON TEXT가 다시 Python dict로 복원됩니다.
    assert isinstance(
        execution["raw_sample"],
        dict,
    )

    # JSON TEXT가 다시 Python list로 복원됩니다.
    assert isinstance(
        execution["evidence"],
        list,
    )

    assert isinstance(
        execution["trace_events"],
        list,
    )

    assert isinstance(
        execution["warnings"],
        list,
    )

    assert (
        execution["raw_sample"][
            "Torque [Nm]"
        ]
        == 62.0
    )

    assert (
        execution["evidence"][0][
            "evidence_type"
        ]
        == "prediction_summary"
    )

    assert (
        execution["trace_events"][0][
            "event_name"
        ]
        == "validate_question"
    )

    assert execution["warnings"] == [
        "SHAP evidence 생성에 실패했습니다."
    ]

    # SQLite INTEGER 0이
    # Python bool False로 복원됩니다.
    assert (
        execution["fallback_occurred"]
        is False
    )


# ---------------------------------------------------------------------
# Python None <-> SQL NULL 테스트
# ---------------------------------------------------------------------

def test_none_values_are_saved_as_sql_null_and_restored_as_none(
    tmp_path: Path,
) -> None:
    """
    Python None이 SQL NULL로 저장되고,
    조회 후 다시 Python None으로
    복원되는지 확인합니다.
    """

    db_path = (
        tmp_path
        / "test_none_and_null.db"
    )

    state = build_sample_state(
        include_raw_sample=False,
    )

    # 예측하지 않은 실행처럼 일부 값을
    # 명시적으로 None으로 변경합니다.
    state["prediction"] = None

    state["probability"] = None

    state["threshold"] = None

    insert_execution(
        state=state,
        db_path=db_path,
    )

    connection = sqlite3.connect(
        db_path
    )

    try:
        row = connection.execute(
            """
            SELECT
                prediction,
                probability,
                threshold,
                raw_sample_json
            FROM agent_executions
            """
        ).fetchone()

    finally:
        connection.close()

    assert row is not None

    # sqlite3는 SQL NULL을
    # Python None으로 반환합니다.
    assert row[0] is None

    assert row[1] is None

    assert row[2] is None

    assert row[3] is None

    execution = get_execution_by_trace_id(
        trace_id=(
            "day19-test-trace-001"
        ),
        db_path=db_path,
    )

    assert execution is not None

    assert execution["prediction"] is None

    assert execution["probability"] is None

    assert execution["threshold"] is None

    assert execution["raw_sample"] is None


# ---------------------------------------------------------------------
# 없는 trace_id 조회 테스트
# ---------------------------------------------------------------------

def test_get_execution_by_trace_id_returns_none_when_not_found(
    tmp_path: Path,
) -> None:
    """
    존재하지 않는 trace_id를 조회하면
    예외가 아니라 None을 반환하는지 확인합니다.
    """

    db_path = (
        tmp_path
        / "test_not_found.db"
    )

    result = get_execution_by_trace_id(
        trace_id=(
            "not-existing-trace-id"
        ),
        db_path=db_path,
    )

    assert result is None


# ---------------------------------------------------------------------
# trace_id UNIQUE 제약 테스트
# ---------------------------------------------------------------------

def test_duplicate_trace_id_raises_integrity_error(
    tmp_path: Path,
) -> None:
    """
    같은 trace_id를 두 번 저장하면
    SQLite UNIQUE 제약에 의해
    sqlite3.IntegrityError가 발생하는지 확인합니다.
    """

    db_path = (
        tmp_path
        / "test_duplicate_trace_id.db"
    )

    first_state = build_sample_state(
        trace_id="duplicate-trace-id",
    )

    second_state = build_sample_state(
        trace_id="duplicate-trace-id",
        question=(
            "같은 trace_id를 가진 두 번째 질문"
        ),
    )

    insert_execution(
        state=first_state,
        db_path=db_path,
    )

    with pytest.raises(
        sqlite3.IntegrityError
    ):
        insert_execution(
            state=second_state,
            db_path=db_path,
        )

    # 두 번째 INSERT는 실패했으므로
    # DB에는 첫 번째 실행 한 건만 있어야 합니다.
    connection = sqlite3.connect(
        db_path
    )

    try:
        execution_count = (
            connection.execute(
                """
                SELECT COUNT(*)
                FROM agent_executions
                """
            ).fetchone()[0]
        )

    finally:
        connection.close()

    assert execution_count == 1


# ---------------------------------------------------------------------
# 최근 실행 목록·정렬·LIMIT 테스트
# ---------------------------------------------------------------------

def test_list_recent_executions_returns_latest_first_and_applies_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    최근 실행 목록이

        created_at DESC

        id DESC

        LIMIT

    정책에 따라 조회되는지 확인합니다.
    """

    db_path = (
        tmp_path
        / "test_recent_executions.db"
    )

    # 실제 시간 흐름에 의존하지 않도록
    # created_at 값을 고정된 순서로 만듭니다.
    #
    # time.sleep()을 사용하지 않으므로
    # 테스트가 빠르고 결정적입니다.
    created_at_values = iter(
        [
            datetime(
                2026,
                7,
                10,
                4,
                0,
                1,
                tzinfo=timezone.utc,
            ),
            datetime(
                2026,
                7,
                10,
                4,
                0,
                2,
                tzinfo=timezone.utc,
            ),
            datetime(
                2026,
                7,
                10,
                4,
                0,
                3,
                tzinfo=timezone.utc,
            ),
        ]
    )

    class FakeDateTime:
        """
        build_execution_record()에서 사용하는
        datetime.now()를 테스트용으로 대체합니다.
        """

        @classmethod
        def now(
            cls,
            tz: Any = None,
        ) -> datetime:
            return next(
                created_at_values
            )

    # execution_history 모듈 안에서 import한
    # datetime 이름만 테스트 중 임시 교체합니다.
    monkeypatch.setattr(
        execution_history,
        "datetime",
        FakeDateTime,
    )

    for index in range(
        1,
        4,
    ):
        state = build_sample_state(
            trace_id=(
                f"recent-trace-{index:03d}"
            ),
            question=(
                f"최근 실행 테스트 질문 {index}"
            ),
            probability=(
                0.90
                + index * 0.01
            ),
        )

        insert_execution(
            state=state,
            db_path=db_path,
        )

    executions = list_recent_executions(
        limit=2,
        db_path=db_path,
    )

    # LIMIT 2
    assert len(executions) == 2

    # 가장 최근 created_at을 가진
    # 세 번째 실행이 먼저 조회됩니다.
    assert (
        executions[0]["trace_id"]
        == "recent-trace-003"
    )

    assert executions[0]["id"] == 3

    assert executions[0][
        "probability"
    ] == pytest.approx(
        0.93
    )

    # 두 번째 실행이 다음 순서입니다.
    assert (
        executions[1]["trace_id"]
        == "recent-trace-002"
    )

    assert executions[1]["id"] == 2

    assert executions[1][
        "probability"
    ] == pytest.approx(
        0.92
    )

    # 목록 조회에는 큰 JSON 상세 데이터가
    # 포함되지 않아야 합니다.
    assert (
        "evidence"
        not in executions[0]
    )

    assert (
        "trace_events"
        not in executions[0]
    )

    assert (
        "raw_sample"
        not in executions[0]
    )


# ---------------------------------------------------------------------
# 빈 DB 목록 조회 테스트
# ---------------------------------------------------------------------

def test_list_recent_executions_returns_empty_list_for_empty_database(
    tmp_path: Path,
) -> None:
    """
    실행 이력이 없는 DB를 조회하면
    None이 아니라 빈 list를 반환하는지 확인합니다.
    """

    db_path = (
        tmp_path
        / "test_empty_database.db"
    )

    executions = list_recent_executions(
        limit=10,
        db_path=db_path,
    )

    assert executions == []

    assert isinstance(
        executions,
        list,
    )


# ---------------------------------------------------------------------
# 잘못된 limit 검증
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "invalid_limit",
    [
        True,
        False,
        0,
        -1,
        101,
        1.5,
        "10",
    ],
)
def test_list_recent_executions_rejects_invalid_limit(
    tmp_path: Path,
    invalid_limit: Any,
) -> None:
    """
    limit 허용 범위:

        정수

        1 이상

        100 이하

    조건을 위반하면 ValueError가 발생해야 합니다.
    """

    db_path = (
        tmp_path
        / "test_invalid_limit.db"
    )

    with pytest.raises(
        ValueError
    ):
        list_recent_executions(
            limit=invalid_limit,
            db_path=db_path,
        )


# ---------------------------------------------------------------------
# 필수 trace_id 검증
# ---------------------------------------------------------------------

def test_build_execution_record_requires_non_empty_trace_id() -> None:
    """
    trace_id는 DB schema에서

        TEXT NOT NULL UNIQUE

    이므로 비어 있으면 저장 record를
    만들 수 없어야 합니다.
    """

    state = build_sample_state()

    state["trace_id"] = "   "

    with pytest.raises(
        ValueError,
        match="trace_id",
    ):
        build_execution_record(
            state
        )


# ---------------------------------------------------------------------
# 필수 question 검증
# ---------------------------------------------------------------------

def test_build_execution_record_requires_non_empty_question() -> None:
    """
    question은 DB schema에서

        TEXT NOT NULL

    이므로 비어 있으면 저장 record를
    만들 수 없어야 합니다.
    """

    state = build_sample_state()

    state["question"] = ""

    with pytest.raises(
        ValueError,
        match="question",
    ):
        build_execution_record(
            state
        )


# ---------------------------------------------------------------------
# SQLite connection 종료 테스트
# ---------------------------------------------------------------------

def test_database_connections_are_closed_after_operations(
    tmp_path: Path,
) -> None:
    """
    INSERT·상세 조회·목록 조회 후
    SQLite connection이 닫혔는지 확인합니다.

    Windows에서는 DB 파일이 열린 상태로 남아 있으면
    파일 이름 변경이 실패할 수 있습니다.

    모든 Persistence 함수 실행 후
    DB 파일 이름을 변경할 수 있다면
    connection이 정상적으로 닫혔음을 확인하는
    실용적인 검증이 됩니다.
    """

    db_path = (
        tmp_path
        / "test_connection_close.db"
    )

    state = build_sample_state(
        trace_id=(
            "connection-close-trace"
        ),
    )

    insert_execution(
        state=state,
        db_path=db_path,
    )

    detail = get_execution_by_trace_id(
        trace_id=(
            "connection-close-trace"
        ),
        db_path=db_path,
    )

    assert detail is not None

    recent = list_recent_executions(
        limit=10,
        db_path=db_path,
    )

    assert len(recent) == 1

    renamed_db_path = (
        tmp_path
        / "renamed_connection_close.db"
    )

    # connection이 열려 있는 채로 남지 않았다면
    # DB 파일 이름을 변경할 수 있습니다.
    db_path.rename(
        renamed_db_path
    )

    assert (
        renamed_db_path.exists()
    )

    assert not db_path.exists()