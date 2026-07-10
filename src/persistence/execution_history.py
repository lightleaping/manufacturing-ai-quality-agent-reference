"""
Day 19 - Agent 실행 이력 SQLite Persistence

이 파일의 역할
----------------
LangGraph Agent 실행 결과를 저장할 SQLite 데이터베이스의
기본 구조와 연결 기능을 정의합니다.

Day 19 전체 목표:

    Agent 실행

    -> final AgentState 생성

    -> 저장용 record 변환

    -> SQLite INSERT

    -> trace_id 기준 상세 조회

    -> 최근 실행 목록 조회

    -> FastAPI 조회 endpoint

하지만 이번 첫 구현 단계에서는
아래 기능만 작성합니다.

    1. 기본 SQLite DB 파일 경로 정의

    2. DB 파일을 저장할 부모 폴더 자동 생성

    3. SQLite connection 생성

    4. agent_executions table 생성

    5. created_at 조회용 index 생성

실행 이력 INSERT와 조회 함수는
다음 단계에서 추가합니다.


Persistence란?
--------------
Persistence는 프로그램이 종료된 뒤에도
데이터가 사라지지 않고 유지되도록 저장하는 것을 의미합니다.

현재 AgentState:

    Python 메모리

    -> FastAPI 요청 처리

    -> 응답 반환

    -> 서버 종료 또는 프로세스 종료

    -> 이전 상태를 다시 조회하기 어려움

SQLite Persistence 추가 후:

    AgentState

    -> SQLite INSERT

    -> .db 파일 저장

    -> 서버 재시작

    -> 이전 실행 이력 조회 가능


왜 LangGraph 내부가 아니라 별도 persistence 계층인가?
---------------------------------------------------
LangGraph:

    Agent workflow 실행 책임

Persistence:

    Agent 실행 결과 저장·조회 책임

FastAPI 또는 service:

    LangGraph 실행 결과와
    Persistence 계층을 연결하는 책임

책임을 분리하면 Agent graph를 테스트할 때
SQLite가 반드시 필요하지 않습니다.

또한 향후 SQLite를 PostgreSQL로 변경하더라도
LangGraph workflow를 직접 수정할 필요가 줄어듭니다.


현재 저장하지 않는 값
--------------------
chat_history:

    이전 사용자 질문과 Agent 답변 전체가 들어 있으므로
    개인정보와 데이터 중복 위험을 줄이기 위해
    Day 19 초기 버전에서는 영구 저장하지 않습니다.

intent_raw_response:

    OpenAI 원본 응답 전체는 저장하지 않습니다.

OpenAI API key:

    절대 저장하지 않습니다.

환경 변수:

    절대 저장하지 않습니다.
"""

from __future__ import annotations

import json
import sqlite3

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.agent.state import AgentState


# ---------------------------------------------------------------------
# 프로젝트 경로
# ---------------------------------------------------------------------

# __file__은 현재 Python 파일의 경로를 나타냅니다.
#
# 현재 파일:
#
# manufacturing-ai-quality-agent-reference/
# └─ src/
#    └─ persistence/
#       └─ execution_history.py
#
# Path(__file__).resolve():
#
# 현재 파일의 절대 경로를 얻습니다.
#
# 예:
#
# C:\Users\kflow\Downloads\
# manufacturing-ai-quality-agent-reference\
# src\persistence\execution_history.py
#
# parents[0]:
#
# src/persistence
#
# parents[1]:
#
# src
#
# parents[2]:
#
# manufacturing-ai-quality-agent-reference
#
# 따라서 parents[2]를 프로젝트 루트 경로로 사용합니다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# 실제 애플리케이션에서 사용할 기본 SQLite DB 경로입니다.
#
# 최종 경로:
#
# data/
# └─ runtime/
#    └─ agent_execution_history.db
#
# data/raw:
#
# 외부에서 받은 원본 데이터
#
# data/processed:
#
# 전처리한 데이터
#
# data/runtime:
#
# 애플리케이션 실행 중 생성되는 데이터
#
# SQLite DB는 애플리케이션 실행 이력이므로
# data/runtime에 저장합니다.
DEFAULT_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "runtime"
    / "agent_execution_history.db"
)


# ---------------------------------------------------------------------
# SQL schema
# ---------------------------------------------------------------------

# CREATE TABLE IF NOT EXISTS
# --------------------------
#
# CREATE TABLE:
#
# SQLite에 새 table을 생성합니다.
#
# IF NOT EXISTS:
#
# 같은 이름의 table이 이미 있으면
# 오류를 발생시키지 않고 기존 table을 유지합니다.
#
# 따라서 애플리케이션을 여러 번 실행해도
# table 생성 코드 때문에 바로 실패하지 않습니다.
#
#
# id INTEGER PRIMARY KEY AUTOINCREMENT
# ------------------------------------
#
# id:
#
# SQLite 내부에서 각 실행 이력 행을 구분하는 번호입니다.
#
# 예:
#
# 1
# 2
# 3
#
# INTEGER PRIMARY KEY:
#
# 이 컬럼을 table의 기본 식별자로 사용합니다.
#
# AUTOINCREMENT:
#
# INSERT할 때 id 값을 직접 전달하지 않아도
# SQLite가 다음 번호를 자동 생성합니다.
#
#
# trace_id TEXT NOT NULL UNIQUE
# -----------------------------
#
# trace_id:
#
# LangGraph Agent 실행 한 건을 구분하는 고유 ID입니다.
#
# 예:
#
# 5d0507ebae9a490db99ac61ab2477dea
#
# TEXT:
#
# 문자열로 저장합니다.
#
# NOT NULL:
#
# trace_id 없이 실행 이력을 저장할 수 없습니다.
#
# UNIQUE:
#
# 같은 trace_id를 두 번 저장할 수 없습니다.
#
# id:
#
# DB 내부 식별자
#
# trace_id:
#
# Agent 실행과 외부 조회를 위한 식별자
#
#
# SQLite 자료형
# -------------
#
# TEXT:
#
# Python str
#
# INTEGER:
#
# Python int
#
# REAL:
#
# Python float
#
# NULL:
#
# Python None
#
#
# JSON 컬럼
# ---------
#
# SQLite에 Python dict나 list를 그대로 저장하지 않습니다.
#
# 다음 단계에서:
#
# Python dict 또는 list
#
# -> json.dumps()
#
# -> JSON 문자열
#
# -> SQLite TEXT
#
# 순서로 저장합니다.
#
# 조회할 때는:
#
# SQLite TEXT
#
# -> json.loads()
#
# -> Python dict 또는 list
#
# 순서로 복원합니다.
CREATE_AGENT_EXECUTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS agent_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    trace_id TEXT NOT NULL UNIQUE,

    question TEXT NOT NULL,

    intent TEXT,

    intent_source TEXT,

    confidence REAL,

    intent_reason TEXT,

    selected_route TEXT,

    prediction INTEGER,

    probability REAL,

    threshold REAL,

    risk_level TEXT,

    recommended_action TEXT,

    answer TEXT,

    trace_status TEXT,

    trace_started_at TEXT,

    trace_finished_at TEXT,

    fallback_occurred INTEGER NOT NULL DEFAULT 0,

    trace_duration_ms REAL,

    warning_count INTEGER NOT NULL DEFAULT 0,

    error_count INTEGER NOT NULL DEFAULT 0,

    raw_sample_json TEXT,

    evidence_json TEXT NOT NULL DEFAULT '[]',

    trace_events_json TEXT NOT NULL DEFAULT '[]',

    warnings_json TEXT NOT NULL DEFAULT '[]',

    errors_json TEXT NOT NULL DEFAULT '[]',

    limitations_json TEXT NOT NULL DEFAULT '[]',

    created_at TEXT NOT NULL
);
"""


# 최근 실행 목록은 created_at을 기준으로 정렬할 예정입니다.
#
# 예:
#
# SELECT *
# FROM agent_executions
# ORDER BY created_at DESC
# LIMIT 10;
#
# created_at 조회와 정렬이 반복될 수 있으므로
# 별도의 index를 만듭니다.
#
#
# UNIQUE와 INDEX 차이
# ------------------
#
# UNIQUE:
#
# 데이터 중복을 방지하는 제약 조건입니다.
#
# 예:
#
# trace_id TEXT UNIQUE
#
# 같은 trace_id 저장 금지
#
#
# INDEX:
#
# 검색과 정렬 성능을 개선하기 위한 자료 구조입니다.
#
# 예:
#
# created_at index
#
# 최근 실행 목록 조회 성능 개선
#
#
# IF NOT EXISTS:
#
# 같은 이름의 index가 이미 있으면
# 다시 생성하지 않습니다.
CREATE_CREATED_AT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS
idx_agent_executions_created_at
ON agent_executions(created_at);
"""

# ---------------------------------------------------------------------
# Agent 실행 이력 INSERT SQL
# ---------------------------------------------------------------------

# final AgentState를 agent_executions table에 저장할 SQL입니다.
#
# id 컬럼이 없는 이유
# ------------------
#
# id는 다음과 같이 정의되어 있습니다.
#
#     id INTEGER PRIMARY KEY AUTOINCREMENT
#
# 따라서 INSERT할 때 직접 값을 전달하지 않습니다.
#
# SQLite가 자동으로:
#
#     1
#     2
#     3
#
# 순서의 내부 id를 생성합니다.
#
#
# :trace_id, :question 등의 의미
# --------------------------------
#
# 아래 표현은 SQL parameter binding입니다.
#
#     :trace_id
#
#     :question
#
#     :intent
#
# 실제 값은 SQL 문자열에 직접 넣지 않고,
# 별도의 dictionary로 전달합니다.
#
# 예:
#
#     {
#         "trace_id": "abc123",
#         "question": "고장 위험을 예측해줘.",
#     }
#
# SQLite가 각 이름에 맞는 값을 안전하게 연결합니다.
#
#
# 왜 f-string으로 값을 넣지 않는가?
# ---------------------------------
#
# 잘못된 예:
#
#     f"""
#     INSERT INTO agent_executions (
#         question
#     )
#     VALUES (
#         '{question}'
#     )
#     """
#
# 사용자 질문에 작은따옴표가 포함되면
# SQL 문법이 깨질 수 있습니다.
#
# 또한 외부 입력이 SQL 구조에 직접 들어가면
# SQL Injection 위험이 생길 수 있습니다.
#
# parameter binding을 사용하면:
#
# SQL 구조:
#
#     INSERT INTO ...
#
# 실제 데이터:
#
#     question
#
# 를 분리할 수 있습니다.
INSERT_AGENT_EXECUTION_SQL = """
INSERT INTO agent_executions (
    trace_id,
    question,
    intent,
    intent_source,
    confidence,
    intent_reason,
    selected_route,
    prediction,
    probability,
    threshold,
    risk_level,
    recommended_action,
    answer,
    trace_status,
    trace_started_at,
    trace_finished_at,
    fallback_occurred,
    trace_duration_ms,
    warning_count,
    error_count,
    raw_sample_json,
    evidence_json,
    trace_events_json,
    warnings_json,
    errors_json,
    limitations_json,
    created_at
)
VALUES (
    :trace_id,
    :question,
    :intent,
    :intent_source,
    :confidence,
    :intent_reason,
    :selected_route,
    :prediction,
    :probability,
    :threshold,
    :risk_level,
    :recommended_action,
    :answer,
    :trace_status,
    :trace_started_at,
    :trace_finished_at,
    :fallback_occurred,
    :trace_duration_ms,
    :warning_count,
    :error_count,
    :raw_sample_json,
    :evidence_json,
    :trace_events_json,
    :warnings_json,
    :errors_json,
    :limitations_json,
    :created_at
);
"""

# ---------------------------------------------------------------------
# trace_id 기준 실행 이력 상세 조회 SQL
# ---------------------------------------------------------------------

# trace_id를 사용하여
# Agent 실행 이력 한 건을 조회합니다.
#
# SELECT *
# --------
#
# agent_executions table의
# 모든 컬럼을 가져옵니다.
#
# Day 19 상세 조회에서는 다음 데이터를 모두 사용합니다.
#
#     기본 실행 정보
#
#     intent 결과
#
#     prediction 결과
#
#     Agent answer
#
#     trace 요약
#
#     JSON으로 저장한 상세 데이터
#
#
# WHERE trace_id = ?
# ------------------
#
# ?는 SQL parameter placeholder입니다.
#
# 실제 trace_id 값은 SQL 문자열 안에
# f-string으로 직접 넣지 않습니다.
#
# 실행 예:
#
#     cursor.execute(
#         SELECT_EXECUTION_BY_TRACE_ID_SQL,
#         ("abc123",),
#     )
#
# SQLite가 ? 위치에
# "abc123" 값을 안전하게 연결합니다.
#
#
# LIMIT 1
# -------
#
# trace_id에는 UNIQUE 제약이 있으므로
# 원래 최대 한 행만 존재할 수 있습니다.
#
# LIMIT 1은 이 조회가 한 건만 필요하다는 의도를
# SQL에서도 명확하게 표현합니다.
SELECT_EXECUTION_BY_TRACE_ID_SQL = """
SELECT *
FROM agent_executions
WHERE trace_id = ?
LIMIT 1;
"""
# ---------------------------------------------------------------------
# 최근 Agent 실행 목록 조회 SQL
# ---------------------------------------------------------------------

# 최근 Agent 실행 이력을
# created_at 기준 내림차순으로 조회합니다.
#
# 목록 조회에서는 상세 JSON 데이터를 제외합니다.
#
# 제외:
#
#     raw_sample_json
#
#     evidence_json
#
#     trace_events_json
#
#     warnings_json
#
#     errors_json
#
#     limitations_json
#
# 이유:
#
# 최근 실행 목록은 여러 실행을 빠르게 확인하기 위한
# 요약 조회이기 때문입니다.
#
# 각 실행의 전체 상세 데이터는:
#
#     get_execution_by_trace_id()
#
# 를 사용하여 별도로 조회합니다.
#
#
# ORDER BY created_at DESC
# ------------------------
#
# created_at:
#
# DB 실행 이력 record가 생성된 UTC 시각
#
# DESC:
#
# descending
#
# 큰 값부터 작은 값 순서
#
# 시간 기준으로는:
#
# 최신 실행
#
# -> 이전 실행
#
# -> 더 오래된 실행
#
# 순서가 됩니다.
#
#
# LIMIT ?
# -------
#
# 반환할 최대 실행 이력 개수를 제한합니다.
#
# 예:
#
#     LIMIT 10
#
# 최근 실행 최대 10건 반환
#
# limit 값도 SQL 문자열에 직접 넣지 않고
# parameter binding을 사용합니다.
SELECT_RECENT_EXECUTIONS_SQL = """
SELECT
    id,
    trace_id,
    question,
    intent,
    intent_source,
    confidence,
    selected_route,
    prediction,
    probability,
    threshold,
    risk_level,
    trace_status,
    fallback_occurred,
    trace_duration_ms,
    warning_count,
    error_count,
    created_at
FROM agent_executions
ORDER BY created_at DESC, id DESC
LIMIT ?;
"""

# ---------------------------------------------------------------------
# DB 경로 처리
# ---------------------------------------------------------------------

def resolve_db_path(
    db_path: str | Path | None = None,
) -> Path:
    """
    사용할 SQLite DB 경로를 Path 객체로 반환합니다.

    Parameters
    ----------
    db_path:
        사용할 SQLite DB 파일 경로입니다.

        None:

            실제 애플리케이션 기본 DB를 사용합니다.

            data/runtime/
            agent_execution_history.db

        str 또는 Path:

            전달한 경로를 사용합니다.

            단위 테스트에서는 tmp_path가 만든
            임시 DB 파일 경로를 전달할 예정입니다.

    Returns
    -------
    Path
        최종 SQLite DB 파일 경로입니다.


    왜 db_path를 외부에서 받을 수 있게 하는가?
    ---------------------------------------
    실제 애플리케이션:

        data/runtime/
        agent_execution_history.db

    pytest:

        tmp_path/
        test_agent_execution_history.db

    테스트가 실제 운영 DB 파일을 사용하면
    기존 실행 이력이 오염될 수 있습니다.

    따라서 DB 경로를 함수 매개변수로 주입할 수 있게 만듭니다.
    """

    # db_path가 None이면
    # 실제 애플리케이션 기본 DB 경로를 반환합니다.
    if db_path is None:
        return DEFAULT_DB_PATH

    # str이 전달되어도 Path 객체로 변환합니다.
    #
    # 예:
    #
    # "temp/test.db"
    #
    # ->
    #
    # Path("temp/test.db")
    return Path(db_path)


# ---------------------------------------------------------------------
# SQLite connection
# ---------------------------------------------------------------------

def get_connection(
    *,
    db_path: str | Path | None = None,
) -> sqlite3.Connection:
    """
    SQLite connection을 생성하여 반환합니다.

    실행 흐름:

        DB 경로 결정

        -> 부모 폴더 생성

        -> sqlite3.connect()

        -> row_factory 설정

        -> foreign key 기능 활성화

        -> connection 반환

    Parameters
    ----------
    db_path:
        사용할 SQLite DB 파일 경로입니다.

        전달하지 않으면:

            data/runtime/
            agent_execution_history.db

        를 사용합니다.

    Returns
    -------
    sqlite3.Connection
        SQLite DB와 연결된 connection 객체입니다.


    connection이란?
    ---------------
    connection은 Python 프로그램과
    SQLite 데이터베이스의 연결을 나타냅니다.

    connection을 통해 다음 작업을 수행합니다.

        cursor 생성

        commit

        rollback

        close


    cursor란?
    ---------
    cursor는 connection 위에서
    실제 SQL 문을 실행하고 결과를 읽는 객체입니다.

    관계:

        Python 프로그램

        -> connection

        -> cursor

        -> SQL 실행

        -> SQLite DB


    connection과 cursor 차이
    ------------------------
    connection:

        DB 연결 전체 관리

        commit

        rollback

        close

    cursor:

        CREATE

        INSERT

        SELECT

        UPDATE

        DELETE

        등의 SQL 실행


    row_factory란?
    ---------------
    SQLite의 기본 조회 결과는 tuple입니다.

    기본:

        row[0]

        row[1]

        row[2]

    sqlite3.Row를 사용하면
    컬럼 이름으로도 값을 읽을 수 있습니다.

    예:

        row["trace_id"]

        row["question"]

        row["intent"]

    이후 실행 이력 조회 결과를
    Python dict로 변환할 때 더 읽기 쉽습니다.
    """

    # 실제로 사용할 DB 경로를 결정합니다.
    resolved_db_path = resolve_db_path(
        db_path
    )

    # SQLite는 DB 파일 자체는 자동 생성할 수 있지만,
    # 존재하지 않는 부모 폴더까지 자동 생성하지는 않습니다.
    #
    # 예:
    #
    # data/runtime 폴더가 없는데
    #
    # sqlite3.connect(
    #     "data/runtime/history.db"
    # )
    #
    # 를 바로 실행하면
    # DB 파일을 열 수 없다는 오류가 발생할 수 있습니다.
    #
    # mkdir:
    #
    # 폴더 생성
    #
    # parents=True:
    #
    # 상위 폴더도 없으면 함께 생성
    #
    # exist_ok=True:
    #
    # 폴더가 이미 있어도 오류를 발생시키지 않음
    resolved_db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # SQLite DB에 연결합니다.
    #
    # DB 파일이 없으면:
    #
    # 새 .db 파일 생성
    #
    # DB 파일이 있으면:
    #
    # 기존 파일에 연결
    connection = sqlite3.connect(
        resolved_db_path
    )

    # SELECT 조회 결과를
    # tuple뿐 아니라 컬럼 이름으로도 읽을 수 있게 설정합니다.
    #
    # 예:
    #
    # row["trace_id"]
    connection.row_factory = sqlite3.Row

    # SQLite는 foreign key 검사를
    # connection마다 명시적으로 활성화하는 것이 안전합니다.
    #
    # 현재 Day 19 초기 table은 하나뿐이므로
    # foreign key를 직접 사용하지 않습니다.
    #
    # 하지만 향후:
    #
    # agent_executions
    #
    # evidence
    #
    # trace_events
    #
    # 등을 별도 table로 분리할 경우를 대비합니다.
    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ---------------------------------------------------------------------
# DB 초기화
# ---------------------------------------------------------------------

def initialize_database(
    *,
    db_path: str | Path | None = None,
) -> None:
    """
    Agent 실행 이력 DB의 table과 index를 생성합니다.

    실행 흐름:

        get_connection()

        -> cursor 생성

        -> CREATE TABLE

        -> CREATE INDEX

        -> commit

        -> cursor close

        -> connection close

    Parameters
    ----------
    db_path:
        사용할 SQLite DB 파일 경로입니다.

        전달하지 않으면 실제 애플리케이션 DB를 사용합니다.

        테스트에서는 tmp_path가 만든
        임시 DB 경로를 전달할 예정입니다.


    commit이란?
    -----------
    commit은 현재 transaction에서 수행한 변경을
    DB에 최종 반영하는 작업입니다.

    예:

        CREATE TABLE

        INSERT

        UPDATE

        DELETE

        -> commit

        -> 변경 내용 확정


    rollback이란?
    -------------
    rollback은 transaction 수행 중 오류가 발생했을 때
    아직 commit하지 않은 변경을 취소하는 작업입니다.

    예:

        CREATE TABLE 성공

        CREATE INDEX 중 오류

        -> rollback

        -> 미완료 transaction 취소


    왜 finally에서 close하는가?
    ---------------------------
    SQL 실행이 성공해도 connection을 닫아야 합니다.

    SQL 실행 중 오류가 발생해도 connection을 닫아야 합니다.

    finally는 성공·실패와 관계없이 실행되므로
    자원 정리에 적합합니다.
    """

    # SQLite DB connection을 생성합니다.
    connection = get_connection(
        db_path=db_path
    )

    # cursor는 SQL을 실행하는 객체입니다.
    cursor = connection.cursor()

    try:
        # agent_executions table을 생성합니다.
        #
        # 이미 존재하면:
        #
        # IF NOT EXISTS에 의해
        # 기존 table을 유지합니다.
        cursor.execute(
            CREATE_AGENT_EXECUTIONS_TABLE_SQL
        )

        # 최근 실행 조회용 created_at index를 생성합니다.
        #
        # 이미 존재하면:
        #
        # IF NOT EXISTS에 의해
        # 기존 index를 유지합니다.
        cursor.execute(
            CREATE_CREATED_AT_INDEX_SQL
        )

        # CREATE TABLE과 CREATE INDEX 작업을
        # DB에 최종 반영합니다.
        connection.commit()

    except sqlite3.Error:
        # SQLite 관련 오류가 발생하면
        # 아직 commit되지 않은 transaction을 취소합니다.
        connection.rollback()

        # 오류를 숨기지 않고 다시 발생시킵니다.
        #
        # 초기 DB 생성 실패는
        # 애플리케이션 초기화 문제이므로
        # 호출한 코드가 실패 사실을 알 수 있어야 합니다.
        raise

    finally:
        # cursor를 먼저 닫습니다.
        cursor.close()

        # DB connection을 닫습니다.
        connection.close()

# ---------------------------------------------------------------------
# JSON 직렬화
# ---------------------------------------------------------------------

def serialize_json(
    value: Any,
) -> str:
    """
    Python 값을 JSON 문자열로 직렬화합니다.

    Parameters
    ----------
    value:
        JSON 문자열로 변환할 Python 값입니다.

        예:

            dict

            list

            str

            int

            float

            bool

            None

    Returns
    -------
    str
        JSON 형식의 문자열입니다.


    직렬화란?
    ----------
    Python 메모리에 있는 객체를
    저장하거나 전송할 수 있는 형식으로 변환하는 과정입니다.

    예:

        Python list

        [
            "warning 1",
            "warning 2",
        ]

        -> json.dumps()

        JSON 문자열

        '[
            "warning 1",
            "warning 2"
        ]'


    왜 SQLite TEXT에 저장하는가?
    ---------------------------
    현재 evidence와 trace_events는
    여러 dict가 들어 있는 중첩 list입니다.

    예:

        [
            {
                "event_name": "validate_question",
                "status": "success",
            },
            {
                "event_name": "classify_intent",
                "status": "success",
            },
        ]

    Day 19에서는 evidence와 trace event를
    별도 table로 정규화하지 않습니다.

    Python list와 dict를 JSON 문자열로 바꾼 뒤
    SQLite TEXT 컬럼 하나에 저장합니다.


    ensure_ascii=False
    ------------------
    기본 json.dumps()는 한글을
    Unicode escape 형식으로 변환할 수 있습니다.

    예:

        "\\uace0\\uc7a5"

    ensure_ascii=False를 사용하면
    한글을 읽을 수 있는 형태로 유지합니다.

    예:

        "고장 위험이 높습니다."
    """

    return json.dumps(
        value,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------
# AgentState -> 저장용 record
# ---------------------------------------------------------------------

def build_execution_record(
    state: AgentState,
) -> dict[str, Any]:
    """
    final AgentState를 SQLite INSERT용 dictionary로 변환합니다.

    Parameters
    ----------
    state:
        LangGraph 실행이 끝난 final AgentState입니다.

    Returns
    -------
    dict[str, Any]
        INSERT parameter binding에 사용할 저장용 record입니다.


    왜 AgentState를 바로 INSERT하지 않는가?
    -------------------------------------
    AgentState와 DB table은 구조가 완전히 같지 않습니다.

    예:

        AgentState:

            warnings: list[str]

        DB:

            warning_count: INTEGER

            warnings_json: TEXT

    따라서 저장 전에 변환 단계가 필요합니다.

    변환 흐름:

        AgentState

        -> warning_count 계산

        -> error_count 계산

        -> dict와 list JSON 직렬화

        -> bool을 SQLite용 0 또는 1로 변환

        -> created_at 생성

        -> INSERT용 record
    """

    # trace_id는 Agent 실행 한 건을 구분하는
    # 외부 조회 식별자입니다.
    #
    # DB schema에서:
    #
    #     TEXT NOT NULL UNIQUE
    #
    # 로 정의했으므로 반드시 필요합니다.
    trace_id = state.get("trace_id")

    # isinstance(..., str):
    #
    # trace_id가 실제 문자열인지 확인합니다.
    #
    # trace_id.strip():
    #
    # 공백만 있는 문자열인지 확인합니다.
    #
    # 예:
    #
    # ""
    #
    # "   "
    #
    # 는 유효한 trace ID가 아닙니다.
    if (
        not isinstance(trace_id, str)
        or not trace_id.strip()
    ):
        raise ValueError(
            "Agent 실행 이력을 저장하려면 "
            "비어 있지 않은 trace_id가 필요합니다."
        )

    # question은 Agent 실행의 원본 질문입니다.
    #
    # DB schema에서:
    #
    #     question TEXT NOT NULL
    #
    # 로 정의했으므로 반드시 필요합니다.
    question = state.get("question")

    if (
        not isinstance(question, str)
        or not question.strip()
    ):
        raise ValueError(
            "Agent 실행 이력을 저장하려면 "
            "비어 있지 않은 question이 필요합니다."
        )

    # warning과 error는 State에서 list로 관리합니다.
    #
    # state에 key가 없으면 빈 list를 사용합니다.
    #
    # list(...)를 사용하여
    # 저장 변환 과정에서 별도의 최상위 list를 만듭니다.
    warnings = list(
        state.get("warnings", []) or []
    )

    errors = list(
        state.get("errors", []) or []
    )

    limitations = list(
        state.get("limitations", []) or []
    )

    evidence = list(
        state.get("evidence", []) or []
    )

    trace_events = list(
        state.get("trace_events", []) or []
    )

    # raw_sample은 제공되지 않을 수도 있습니다.
    #
    # 예:
    #
    # dataset_schema_query
    #
    # unknown
    #
    # 에서는 raw_sample이 없을 수 있습니다.
    raw_sample = state.get("raw_sample")

    # raw_sample이 None이면
    # Python None을 그대로 저장합니다.
    #
    # sqlite3는 Python None을
    # SQL NULL로 자동 변환합니다.
    #
    # raw_sample이 dict이면:
    #
    # json.dumps()
    #
    # 를 사용해 JSON TEXT로 변환합니다.
    if raw_sample is None:
        raw_sample_json = None
    else:
        raw_sample_json = serialize_json(
            raw_sample
        )

    # INSERT parameter 이름과
    # dictionary key 이름을 동일하게 맞춥니다.
    #
    # 예:
    #
    # SQL:
    #
    #     :trace_id
    #
    # dictionary:
    #
    #     "trace_id": trace_id
    #
    # SQLite가 같은 이름을 찾아 값을 연결합니다.
    return {
        "trace_id": trace_id,

        "question": question,

        "intent": state.get(
            "intent"
        ),

        "intent_source": state.get(
            "intent_source"
        ),

        "confidence": state.get(
            "confidence"
        ),

        "intent_reason": state.get(
            "intent_reason"
        ),

        "selected_route": state.get(
            "selected_route"
        ),

        "prediction": state.get(
            "prediction"
        ),

        "probability": state.get(
            "probability"
        ),

        "threshold": state.get(
            "threshold"
        ),

        "risk_level": state.get(
            "risk_level"
        ),

        "recommended_action": state.get(
            "recommended_action"
        ),

        "answer": state.get(
            "answer"
        ),

        "trace_status": state.get(
            "trace_status"
        ),

        "trace_started_at": state.get(
            "trace_started_at"
        ),

        "trace_finished_at": state.get(
            "trace_finished_at"
        ),

        # SQLite에서는 bool 값을 일반적으로
        # INTEGER 0 또는 1로 저장합니다.
        #
        # False:
        #
        #     int(False)
        #
        #     -> 0
        #
        # True:
        #
        #     int(True)
        #
        #     -> 1
        "fallback_occurred": int(
            bool(
                state.get(
                    "fallback_occurred",
                    False,
                )
            )
        ),

        "trace_duration_ms": state.get(
            "trace_duration_ms"
        ),

        # warnings 전체는 warnings_json에 저장하고,
        # 검색과 통계를 위해 개수도 별도로 저장합니다.
        "warning_count": len(
            warnings
        ),

        # errors 전체는 errors_json에 저장하고,
        # 검색과 통계를 위해 개수도 별도로 저장합니다.
        "error_count": len(
            errors
        ),

        "raw_sample_json": (
            raw_sample_json
        ),

        "evidence_json": serialize_json(
            evidence
        ),

        "trace_events_json": serialize_json(
            trace_events
        ),

        "warnings_json": serialize_json(
            warnings
        ),

        "errors_json": serialize_json(
            errors
        ),

        "limitations_json": serialize_json(
            limitations
        ),

        # DB record를 생성한 UTC 시각입니다.
        #
        # trace_finished_at:
        #
        # Agent workflow가 끝난 시각
        #
        # created_at:
        #
        # 실행 이력 record를 DB에 저장하기 위해
        # 생성한 시각
        #
        # 두 값은 역할이 다릅니다.
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    }


# ---------------------------------------------------------------------
# Agent 실행 이력 INSERT
# ---------------------------------------------------------------------

def insert_execution(
    *,
    state: AgentState,
    db_path: str | Path | None = None,
) -> int:
    """
    final AgentState를 SQLite에 저장합니다.

    Parameters
    ----------
    state:
        저장할 final AgentState입니다.

    db_path:
        사용할 SQLite DB 파일 경로입니다.

        None:

            실제 애플리케이션 기본 DB

            data/runtime/
            agent_execution_history.db

        임시 경로:

            pytest 또는 수동 테스트 DB

    Returns
    -------
    int
        SQLite가 자동 생성한 실행 이력의 내부 id입니다.


    전체 실행 흐름
    --------------
    initialize_database()

    -> table과 index 존재 보장

    -> build_execution_record()

    -> AgentState를 저장용 dictionary로 변환

    -> get_connection()

    -> cursor 생성

    -> parameter binding INSERT

    -> 새 id 확인

    -> commit

    -> cursor close

    -> connection close


    Agent 성공과 DB 저장 성공은 다릅니다.
    -----------------------------------
    이 함수에서 저장 오류가 발생하면
    예외를 호출한 계층으로 전달합니다.

    이후 FastAPI에 연결할 때는:

        Agent 실행 성공

        +

        DB 저장 실패

        -> Agent 응답은 유지

        -> logging 또는 warning 기록

    정책으로 처리할 예정입니다.

    즉, Persistence 함수는
    저장 실패를 숨기지 않습니다.

    FastAPI 또는 service 계층이
    저장 실패가 사용자 응답에 어떤 영향을 줄지 결정합니다.
    """

    # table과 index가 없는 경우 생성합니다.
    #
    # CREATE TABLE IF NOT EXISTS를 사용하므로
    # 이미 존재해도 기존 table을 유지합니다.
    initialize_database(
        db_path=db_path
    )

    # AgentState를 INSERT 가능한 구조로 변환합니다.
    record = build_execution_record(
        state
    )

    # SQLite connection을 엽니다.
    connection = get_connection(
        db_path=db_path
    )

    # SQL 실행용 cursor를 만듭니다.
    cursor = connection.cursor()

    try:
        # INSERT SQL과 실제 record 값을
        # 별도로 전달합니다.
        #
        # SQL:
        #
        #     :trace_id
        #
        #     :question
        #
        # record:
        #
        #     {
        #         "trace_id": "...",
        #         "question": "...",
        #     }
        #
        # SQLite가 이름을 기준으로
        # parameter를 안전하게 연결합니다.
        cursor.execute(
            INSERT_AGENT_EXECUTION_SQL,
            record,
        )

        # cursor.lastrowid는
        # 방금 INSERT한 행의 id입니다.
        #
        # 예:
        #
        # 첫 번째 저장:
        #
        #     1
        #
        # 두 번째 저장:
        #
        #     2
        execution_id = cursor.lastrowid

        # INSERT 변경을 DB에 최종 반영합니다.
        connection.commit()

    except sqlite3.Error:
        # INSERT 중 SQLite 오류가 발생하면
        # 아직 commit되지 않은 변경을 취소합니다.
        connection.rollback()

        # 오류를 숨기지 않고 호출한 코드에 전달합니다.
        #
        # 같은 trace_id를 두 번 저장하면:
        #
        # sqlite3.IntegrityError
        #
        # 가 발생할 수 있습니다.
        raise

    finally:
        # 성공·실패와 관계없이
        # cursor와 connection을 닫습니다.
        cursor.close()

        connection.close()

    # 정상 INSERT 후에는
    # SQLite가 생성한 내부 id가 있어야 합니다.
    if execution_id is None:
        raise RuntimeError(
            "Agent 실행 이력은 저장됐지만 "
            "SQLite execution id를 확인하지 못했습니다."
        )

    return int(
        execution_id
    )

# ---------------------------------------------------------------------
# JSON 역직렬화
# ---------------------------------------------------------------------

def deserialize_json(
    value: str | None,
    *,
    default: Any,
) -> Any:
    """
    SQLite TEXT에 저장된 JSON 문자열을
    Python 값으로 역직렬화합니다.

    Parameters
    ----------
    value:
        SQLite에서 읽은 JSON 문자열입니다.

        예:

            '{"Torque [Nm]": 62.0}'

            '["warning 1", "warning 2"]'

        SQL NULL이 저장되어 있으면
        Python에서는 None으로 전달됩니다.

    default:
        value가 None일 때 반환할 기본값입니다.

        예:

            raw_sample:

                default=None

            evidence:

                default=[]

            warnings:

                default=[]

    Returns
    -------
    Any
        JSON 문자열에서 복원한 Python 값입니다.


    역직렬화란?
    -----------
    저장하거나 전송하기 위해 문자열로 변환했던 데이터를
    다시 Python 객체로 복원하는 과정입니다.

    저장:

        Python list

        -> json.dumps()

        -> JSON 문자열

        -> SQLite TEXT

    조회:

        SQLite TEXT

        -> json.loads()

        -> Python list


    예
    --

    SQLite:

        '[{"feature": "Torque [Nm]"}]'

    json.loads():

        [
            {
                "feature": "Torque [Nm]"
            }
        ]


    None 처리
    ---------
    raw_sample은 없는 실행도 있습니다.

    예:

        dataset_schema_query

    이 경우 DB에는:

        SQL NULL

    Python 조회 결과에는:

        None

    이 함수는 value가 None이면
    json.loads()를 실행하지 않고
    전달받은 default를 반환합니다.
    """

    # SQL NULL은 sqlite3 조회 시
    # Python None으로 변환됩니다.
    #
    # None은 JSON 문자열이 아니므로
    # json.loads(None)을 실행하면 오류가 발생합니다.
    #
    # 따라서 먼저 None 여부를 확인합니다.
    if value is None:
        return default

    # JSON 문자열을 원래 Python 구조로 복원합니다.
    #
    # 예:
    #
    # '{"Torque [Nm]": 62.0}'
    #
    # ->
    #
    # {
    #     "Torque [Nm]": 62.0
    # }
    return json.loads(
        value
    )

# ---------------------------------------------------------------------
# SQLite Row -> Python 실행 이력 dictionary
# ---------------------------------------------------------------------

def execution_row_to_dict(
    row: sqlite3.Row,
) -> dict[str, Any]:
    """
    SQLite 조회 결과 한 행을
    일반 Python dictionary로 변환합니다.

    JSON TEXT 컬럼은 json.loads()를 사용하여
    원래 Python dict 또는 list 구조로 복원합니다.
    """

    return {
        # SQLite 내부 실행 이력 ID
        "id": row["id"],

        # Agent 실행 식별자
        "trace_id": row["trace_id"],

        # 사용자 질문
        "question": row["question"],

        # Intent 결과
        "intent": row["intent"],
        "intent_source": row["intent_source"],
        "confidence": row["confidence"],
        "intent_reason": row["intent_reason"],

        # 최종 AgentState에 남아 있던 최근 route
        "selected_route": row["selected_route"],

        # Prediction 결과
        "prediction": row["prediction"],
        "probability": row["probability"],
        "threshold": row["threshold"],
        "risk_level": row["risk_level"],
        "recommended_action": row[
            "recommended_action"
        ],

        # 최종 Agent 답변
        "answer": row["answer"],

        # Trace 요약
        "trace_status": row["trace_status"],
        "trace_started_at": row[
            "trace_started_at"
        ],
        "trace_finished_at": row[
            "trace_finished_at"
        ],

        # SQLite INTEGER 0/1을
        # Python bool로 복원합니다.
        "fallback_occurred": bool(
            row["fallback_occurred"]
        ),

        "trace_duration_ms": row[
            "trace_duration_ms"
        ],

        # Warning·Error 개수
        "warning_count": row[
            "warning_count"
        ],
        "error_count": row[
            "error_count"
        ],

        # JSON TEXT를 Python 객체로 복원합니다.
        #
        # raw_sample은 SQL NULL일 수 있으므로
        # 기본값을 None으로 사용합니다.
        "raw_sample": deserialize_json(
            row["raw_sample_json"],
            default=None,
        ),

        # 목록 데이터는 기본값으로
        # 빈 list를 사용합니다.
        "evidence": deserialize_json(
            row["evidence_json"],
            default=[],
        ),

        "trace_events": deserialize_json(
            row["trace_events_json"],
            default=[],
        ),

        "warnings": deserialize_json(
            row["warnings_json"],
            default=[],
        ),

        "errors": deserialize_json(
            row["errors_json"],
            default=[],
        ),

        "limitations": deserialize_json(
            row["limitations_json"],
            default=[],
        ),

        # DB record 생성 시각
        "created_at": row["created_at"],
    }

# ---------------------------------------------------------------------
# trace_id 기준 실행 이력 상세 조회
# ---------------------------------------------------------------------

def get_execution_by_trace_id(
    *,
    trace_id: str,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """
    trace_id를 사용하여
    Agent 실행 이력 한 건을 조회합니다.

    Parameters
    ----------
    trace_id:
        조회할 Agent 실행의 trace ID입니다.

        예:

            "5d0507ebae9a490db99ac61ab2477dea"

    db_path:
        사용할 SQLite DB 파일 경로입니다.

        None:

            실제 애플리케이션 DB

            data/runtime/
            agent_execution_history.db

        임시 경로:

            pytest 또는 수동 테스트 DB

    Returns
    -------
    dict[str, Any] | None

        실행 이력이 있으면:

            상세 실행 이력 dictionary

        실행 이력이 없으면:

            None


    전체 실행 흐름
    --------------
    trace_id 검증

    -> initialize_database()

    -> SQLite connection

    -> SELECT WHERE trace_id = ?

    -> fetchone()

    -> 조회 결과 없음

        -> None

    -> 조회 결과 있음

        -> execution_row_to_dict()

        -> JSON 역직렬화

        -> 상세 dictionary 반환

    -> cursor close

    -> connection close


    왜 없는 경우 예외가 아니라 None인가?
    ------------------------------------
    Repository 또는 Persistence 계층은
    데이터가 존재하는지 조회합니다.

    없는 데이터는 정상적인 조회 결과일 수 있습니다.

    따라서:

        Persistence:

            None 반환

        FastAPI:

            None 확인

            -> HTTP 404 Not Found

    로 책임을 나눌 예정입니다.
    """

    # trace_id가 문자열인지 확인합니다.
    #
    # 빈 문자열이나 공백만 있는 값은
    # 유효한 조회 키가 아닙니다.
    if (
        not isinstance(trace_id, str)
        or not trace_id.strip()
    ):
        raise ValueError(
            "실행 이력을 조회하려면 "
            "비어 있지 않은 trace_id가 필요합니다."
        )

    # table이 아직 없는 새 DB에서도
    # 조회 함수가 동작할 수 있도록 초기화합니다.
    #
    # 새 DB:
    #
    #     table 생성
    #
    #     SELECT
    #
    #     결과 없음
    #
    #     None 반환
    initialize_database(
        db_path=db_path
    )

    # SQLite connection을 엽니다.
    connection = get_connection(
        db_path=db_path
    )

    # SQL 실행용 cursor를 생성합니다.
    cursor = connection.cursor()

    try:
        # parameter binding을 사용합니다.
        #
        # SQL:
        #
        #     WHERE trace_id = ?
        #
        # 실제 값:
        #
        #     (trace_id,)
        #
        # 를 별도로 전달합니다.
        cursor.execute(
            SELECT_EXECUTION_BY_TRACE_ID_SQL,
            (
                trace_id,
            ),
        )

        # fetchone()은 조회 결과 중
        # 첫 번째 행 하나를 반환합니다.
        #
        # 조회 결과가 있으면:
        #
        #     sqlite3.Row
        #
        # 조회 결과가 없으면:
        #
        #     None
        row = cursor.fetchone()

        # 해당 trace_id의 실행 이력이 없으면
        # None을 반환합니다.
        if row is None:
            return None

        # SQLite Row를 일반 Python dict로 변환하고,
        # JSON TEXT도 원래 dict·list로 복원합니다.
        return execution_row_to_dict(
            row
        )

    finally:
        # SELECT는 DB 데이터를 변경하지 않으므로
        # commit은 필요하지 않습니다.
        #
        # 하지만 cursor와 connection은
        # 성공·실패와 관계없이 닫아야 합니다.
        cursor.close()

        connection.close()

# ---------------------------------------------------------------------
# SQLite Row -> 최근 실행 요약 dictionary
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# SQLite Row -> 최근 실행 요약 dictionary
# ---------------------------------------------------------------------

def execution_summary_row_to_dict(
    row: sqlite3.Row,
) -> dict[str, Any]:
    """
    SQLite 조회 결과 한 행을
    최근 실행 목록용 요약 dictionary로 변환합니다.

    상세 JSON 데이터는 제외하고
    목록 화면에 필요한 핵심 값만 반환합니다.
    """

    return {
        "id": row["id"],

        "trace_id": row["trace_id"],

        "question": row["question"],

        "intent": row["intent"],

        "intent_source": row[
            "intent_source"
        ],

        "confidence": row[
            "confidence"
        ],

        "selected_route": row[
            "selected_route"
        ],

        "prediction": row[
            "prediction"
        ],

        "probability": row[
            "probability"
        ],

        "threshold": row[
            "threshold"
        ],

        "risk_level": row[
            "risk_level"
        ],

        "trace_status": row[
            "trace_status"
        ],

        # SQLite INTEGER 0·1을
        # Python bool로 복원합니다.
        "fallback_occurred": bool(
            row["fallback_occurred"]
        ),

        "trace_duration_ms": row[
            "trace_duration_ms"
        ],

        "warning_count": row[
            "warning_count"
        ],

        "error_count": row[
            "error_count"
        ],

        "created_at": row[
            "created_at"
        ],
    }


# ---------------------------------------------------------------------
# 최근 Agent 실행 목록 조회
# ---------------------------------------------------------------------

def list_recent_executions(
    *,
    limit: int = 20,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    최근 Agent 실행 이력을
    최신 실행 순서로 조회합니다.

    Parameters
    ----------
    limit:
        반환할 최대 실행 이력 개수입니다.

        허용 범위:

            1 ~ 100

    db_path:
        사용할 SQLite DB 경로입니다.

        None이면 실제 애플리케이션 기본 DB를 사용합니다.

    Returns
    -------
    list[dict[str, Any]]
        최근 실행 이력 요약 목록입니다.

        실행 이력이 없으면
        빈 list를 반환합니다.
    """

    # Python에서 bool은 int의 하위 타입입니다.
    #
    # isinstance(True, int)
    #
    # 결과:
    #
    # True
    #
    # 그러나 limit=True는 조회 개수로
    # 의미가 적절하지 않으므로 따로 차단합니다.
    if isinstance(limit, bool):
        raise ValueError(
            "limit은 1 이상의 정수여야 합니다."
        )

    if not isinstance(limit, int):
        raise ValueError(
            "limit은 1 이상의 정수여야 합니다."
        )

    if limit < 1:
        raise ValueError(
            "limit은 1 이상의 정수여야 합니다."
        )

    if limit > 100:
        raise ValueError(
            "limit은 100 이하이어야 합니다."
        )

    # 새 DB에서도 조회할 수 있도록
    # table과 index를 먼저 보장합니다.
    initialize_database(
        db_path=db_path
    )

    connection = get_connection(
        db_path=db_path
    )

    cursor = connection.cursor()

    try:
        # LIMIT 값도 parameter binding으로 전달합니다.
        cursor.execute(
            SELECT_RECENT_EXECUTIONS_SQL,
            (
                limit,
            ),
        )

        rows = cursor.fetchall()

        return [
            execution_summary_row_to_dict(
                row
            )
            for row in rows
        ]

    finally:
        cursor.close()

        connection.close()