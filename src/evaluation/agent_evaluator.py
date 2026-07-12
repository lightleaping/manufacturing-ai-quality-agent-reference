"""
Day 21 Agent 평가 실행 로직입니다.

이 파일의 책임
--------------
agent_evaluation_cases.py에 정의한 평가 Case를 읽고,
기존 LangGraph Agent를 실제로 실행한 뒤,
기대값과 실제 AgentState를 비교합니다.


전체 데이터 흐름
----------------

AgentEvaluationCase

↓

deterministic classifier 적용

↓

필요한 경우
deterministic prediction service 적용

↓

기존 run_failure_agent_graph()

↓

기존 LangGraph workflow 실행

↓

최종 AgentState

↓

expected 값과 actual 값 비교

↓

EvaluationCheckResult

↓

AgentEvaluationResult

↓

AgentEvaluationSummary


중요한 설계 원칙
----------------
1. 기존 Agent 로직을 복사하지 않습니다.

2. 기존 run_failure_agent_graph()를 그대로 호출합니다.

3. 기본 평가에서는 실제 OpenAI API를 호출하지 않습니다.

4. 평가 결과가 실행할 때마다 달라지지 않도록
   Intent Classification 결과를 고정합니다.

5. 고장 예측 정합성 Case에서는
   prediction service 결과도 고정합니다.

6. 실제 API Key, .env 내용, Authorization Header,
   전체 환경 변수를 읽거나 저장하지 않습니다.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from math import isclose
from typing import Any
from unittest.mock import patch

from src.agent import failure_agent_graph
from src.agent.intent_classifier import (
    IntentClassificationResult,
)
from src.evaluation.agent_evaluation_cases import (
    AgentEvaluationCase,
)


@dataclass(
    frozen=True,
    slots=True,
)
class EvaluationCheckResult:
    """
    평가 조건 한 개의 결과입니다.

    예
    --
    expected intent:

        failure_prediction

    actual intent:

        failure_prediction

    result:

        PASS


    field 설명
    ----------
    check_name:
        어떤 조건을 검사했는지 나타냅니다.

    passed:
        조건 통과 여부입니다.

    expected:
        평가 Case에 정의한 기대값입니다.

    actual:
        실제 AgentState에서 확인한 값입니다.

    message:
        사람이 결과를 읽을 때 사용할 설명입니다.
    """

    check_name: str

    passed: bool

    expected: Any

    actual: Any

    message: str


@dataclass(
    frozen=True,
    slots=True,
)
class AgentEvaluationResult:
    """
    Agent 평가 Case 한 건의 최종 결과입니다.

    한 Case 안에는 여러 Check가 존재합니다.

    예
    --
    high_risk_prediction_consistency

        intent 일치

        prediction 일치

        probability 일치

        threshold 일치

        risk_level 일치

        fallback 일치

        evidence 수 확인

        answer 문구 확인


    Case 통과 기준
    --------------
    모든 Check가 통과해야
    Case 전체를 PASS로 판정합니다.
    """

    case_id: str

    category: str

    description: str

    passed: bool

    checks: tuple[
        EvaluationCheckResult,
        ...,
    ]

    # 실제 AgentState 전체를 저장하지 않고,
    # 평가와 보고서에 필요한 안전한 필드만 저장합니다.
    #
    # 실제 환경 변수나 Secret은 포함하지 않습니다.
    actual_output: dict[str, Any]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        평가 결과를 JSON 저장 가능한 dict로 변환합니다.

        dataclasses.asdict()
        --------------------
        dataclass 내부의 중첩 dataclass도
        재귀적으로 dict 형태로 변환합니다.
        """

        return asdict(self)


@dataclass(
    frozen=True,
    slots=True,
)
class AgentEvaluationSummary:
    """
    여러 Agent 평가 Case의 전체 결과입니다.

    저장 정보
    ---------
    total_count:

        전체 Case 수

    passed_count:

        통과 Case 수

    failed_count:

        실패 Case 수

    pass_rate:

        전체 통과율

    category_summary:

        평가 영역별 결과

    results:

        Case별 상세 결과
    """

    total_count: int

    passed_count: int

    failed_count: int

    pass_rate: float

    category_summary: dict[
        str,
        dict[str, int | float],
    ]

    results: tuple[
        AgentEvaluationResult,
        ...,
    ]

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        전체 평가 결과를
        JSON 저장 가능한 dict로 변환합니다.
        """

        return asdict(self)


def _build_fake_classify_intent(
    case: AgentEvaluationCase,
):
    """
    평가 Case 전용 deterministic classifier를 생성합니다.

    Parameter
    ---------
    case:
        현재 실행할 AgentEvaluationCase입니다.


    Return
    ------
    callable

        기존 classify_intent()와 호환되는
        fake classifier 함수입니다.


    왜 factory 함수인가?
    --------------------
    평가 Case마다 반환해야 하는 Intent가 다릅니다.

    예:

        Dataset Schema Case

        -> dataset_schema_query


        Missing raw_sample Case

        -> failure_prediction


        Unsupported Question Case

        -> unknown


    따라서 현재 Case를 기억하는
    별도 fake 함수를 만들어 반환합니다.
    """

    def fake_classify_intent(
        question: str,
        *,
        chat_history=None,
    ) -> IntentClassificationResult:
        """
        실제 OpenAI API를 호출하지 않고
        현재 평가 Case에 정의된 Intent를 반환합니다.

        question과 chat_history를 parameter로 받는 이유
        ----------------------------------------------
        기존 classify_intent()와
        같은 함수 interface를 유지하기 위해서입니다.

        현재 deterministic 평가에서는
        question 내용을 다시 분석하지 않습니다.

        Case에 정의된 classifier_intent를
        그대로 반환합니다.
        """

        return IntentClassificationResult(
            intent=case.classifier_intent,
            confidence=1.0,
            reason=(
                "Day 21 deterministic "
                "Agent evaluation intent."
            ),
            # 기존 IntentClassificationResult가
            # 허용하는 source 값을 사용합니다.
            #
            # 실제 OpenAI API를 호출했다는 의미가 아니라,
            # 기존 결과 schema와 호환하기 위한 값입니다.
            source="openai",
            raw_response=None,
            error=None,
        )

    return fake_classify_intent


def _build_fake_prediction_service(
    case: AgentEvaluationCase,
):
    """
    평가 Case 전용 deterministic prediction service를 생성합니다.

    Parameter
    ---------
    case:
        prediction_service_result가 정의된
        AgentEvaluationCase입니다.


    Return
    ------
    callable

        기존 _run_failure_prediction_service()와
        호환되는 fake 함수입니다.


    왜 실제 PyTorch 결과를 사용하지 않는가?
    --------------------------------------
    Day 21의 answer consistency 평가는:

        prediction

        probability

        threshold

        risk_level

        answer

        evidence

    사이의 정합성을 평가합니다.

    모델 파일이나 실행 환경 변화 때문에
    결과가 달라지면 평가 재현성이 낮아질 수 있습니다.

    따라서 해당 Case에서는
    고정된 prediction 결과를 사용합니다.
    """

    def fake_run_failure_prediction_service(
        raw_sample,
        include_shap=True,
        include_global_importance=True,
    ) -> dict[str, Any]:
        """
        기존 prediction helper와
        동일한 parameter 구조를 유지합니다.

        deepcopy()를 사용하는 이유
        --------------------------
        LangGraph node가 반환 dict나
        내부 evidence list를 수정하더라도

        AgentEvaluationCase에 저장된
        원본 기대 데이터가 변경되지 않도록 합니다.
        """

        if (
            case.prediction_service_result
            is None
        ):
            raise ValueError(
                "prediction_service_result가 "
                "정의되지 않은 평가 Case입니다."
            )

        return deepcopy(
            case.prediction_service_result
        )

    return fake_run_failure_prediction_service


def _values_match(
    expected: Any,
    actual: Any,
) -> bool:
    """
    expected와 actual의 일치 여부를 계산합니다.

    float 비교
    ----------
    float는 내부 이진 표현 때문에

        0.1 + 0.2

    결과가 정확히 0.3과 같지 않을 수 있습니다.

    따라서 expected가 float이면
    math.isclose()를 사용합니다.


    그 외 값
    --------
    문자열, bool, int, None 등은
    == 연산자로 비교합니다.
    """

    if isinstance(
        expected,
        float,
    ):
        if not isinstance(
            actual,
            (int, float),
        ):
            return False

        return isclose(
            float(actual),
            expected,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )

    return actual == expected


def _build_value_check(
    *,
    check_name: str,
    expected: Any,
    actual: Any,
) -> EvaluationCheckResult:
    """
    단일 값 비교 결과를 생성합니다.
    """

    passed = _values_match(
        expected,
        actual,
    )

    if passed:
        message = (
            f"{check_name} 값이 "
            "기대값과 일치합니다."
        )
    else:
        message = (
            f"{check_name} 값이 "
            "기대값과 일치하지 않습니다."
        )

    return EvaluationCheckResult(
        check_name=check_name,
        passed=passed,
        expected=expected,
        actual=actual,
        message=message,
    )


def _build_minimum_count_check(
    *,
    check_name: str,
    minimum_count: int,
    actual_count: int,
) -> EvaluationCheckResult:
    """
    실제 개수가 최소 기준 이상인지 평가합니다.

    사용 예
    -------
    evidence가 최소 1개 이상 필요한 경우:

        minimum_count = 1

        actual_count = 3

        -> PASS
    """

    passed = (
        actual_count
        >= minimum_count
    )

    if passed:
        message = (
            f"{check_name} 개수가 "
            "최소 기준 이상입니다."
        )
    else:
        message = (
            f"{check_name} 개수가 "
            "최소 기준보다 작습니다."
        )

    return EvaluationCheckResult(
        check_name=check_name,
        passed=passed,
        expected=(
            f">= {minimum_count}"
        ),
        actual=actual_count,
        message=message,
    )


def _build_required_substring_check(
    *,
    check_name: str,
    required_substring: str,
    actual_text: str,
) -> EvaluationCheckResult:
    """
    필수 문자열이 실제 텍스트에 포함되는지 평가합니다.
    """

    passed = (
        required_substring
        in actual_text
    )

    if passed:
        message = (
            "필수 문자열이 "
            "정상적으로 포함되어 있습니다."
        )
    else:
        message = (
            "필수 문자열이 "
            "포함되어 있지 않습니다."
        )

    return EvaluationCheckResult(
        check_name=check_name,
        passed=passed,
        expected=(
            f"contains: "
            f"{required_substring}"
        ),
        actual=actual_text,
        message=message,
    )


def _build_forbidden_substring_check(
    *,
    check_name: str,
    forbidden_substring: str,
    actual_text: str,
) -> EvaluationCheckResult:
    """
    금지 문자열이 실제 텍스트에 없는지 평가합니다.

    Secret 안전성 평가에서는
    실제 API Key를 읽지 않습니다.

    다음과 같은 일반적인 노출 형태만 확인합니다.

        sk-

        OPENAI_API_KEY=

        Authorization: Bearer
    """

    passed = (
        forbidden_substring
        not in actual_text
    )

    if passed:
        message = (
            "금지 문자열이 "
            "출력되지 않았습니다."
        )
    else:
        message = (
            "금지 문자열이 "
            "출력되었습니다."
        )

    return EvaluationCheckResult(
        check_name=check_name,
        passed=passed,
        expected=(
            f"not contains: "
            f"{forbidden_substring}"
        ),
        actual=actual_text,
        message=message,
    )


def _build_actual_output(
    final_state: dict[str, Any],
) -> dict[str, Any]:
    """
    최종 AgentState에서
    평가·보고서용 안전 필드만 선택합니다.

    저장 필드
    ---------
    intent

    confidence

    intent_source

    prediction

    probability

    threshold

    risk_level

    fallback_occurred

    answer

    evidence_count

    errors

    trace_status


    저장하지 않는 정보
    -------------------
    환경 변수

    OPENAI_API_KEY

    Authorization Header

    .env 내용

    OpenAI 원본 전체 응답
    """

    evidence = (
        final_state.get(
            "evidence"
        )
        or []
    )

    errors = (
        final_state.get(
            "errors"
        )
        or []
    )

    return {
        "intent": final_state.get(
            "intent"
        ),
        "confidence": final_state.get(
            "confidence"
        ),
        "intent_source": (
            final_state.get(
                "intent_source"
            )
        ),
        "prediction": final_state.get(
            "prediction"
        ),
        "probability": final_state.get(
            "probability"
        ),
        "threshold": final_state.get(
            "threshold"
        ),
        "risk_level": final_state.get(
            "risk_level"
        ),
        "fallback_occurred": (
            final_state.get(
                "fallback_occurred"
            )
        ),
        "answer": final_state.get(
            "answer"
        ),
        "evidence_count": (
            len(evidence)
            if isinstance(
                evidence,
                list,
            )
            else 0
        ),
        "errors": (
            list(errors)
            if isinstance(
                errors,
                list,
            )
            else []
        ),
        "trace_status": (
            final_state.get(
                "trace_status"
            )
        ),
    }


def evaluate_agent_case(
    case: AgentEvaluationCase,
) -> AgentEvaluationResult:
    """
    Agent 평가 Case 한 건을 실행합니다.

    실행 순서
    ---------
    1. deterministic classifier 생성

    2. failure_agent_graph.classify_intent 교체

    3. prediction 결과가 정의된 Case이면
       prediction helper도 교체

    4. 기존 run_failure_agent_graph() 실행

    5. 최종 AgentState 수집

    6. expected 값과 actual 값 비교

    7. Check별 PASS / FAIL 생성

    8. Case 전체 PASS / FAIL 반환


    Parameter
    ---------
    case:
        실행할 AgentEvaluationCase입니다.


    Return
    ------
    AgentEvaluationResult

        Case 한 건의 전체 평가 결과입니다.
    """

    fake_classify_intent = (
        _build_fake_classify_intent(
            case
        )
    )

    # ExitStack
    # ---------
    # 적용할 patch 수가 Case마다 다를 때
    # 여러 context manager를 동적으로 관리합니다.
    #
    # 모든 Case:
    #
    #     classify_intent patch
    #
    # Prediction Case:
    #
    #     classify_intent patch
    #     +
    #     prediction service patch
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                failure_agent_graph,
                "classify_intent",
                fake_classify_intent,
            )
        )

        if (
            case.prediction_service_result
            is not None
        ):
            fake_prediction_service = (
                _build_fake_prediction_service(
                    case
                )
            )

            stack.enter_context(
                patch.object(
                    failure_agent_graph,
                    (
                        "_run_failure_"
                        "prediction_service"
                    ),
                    fake_prediction_service,
                )
            )

        # tuple로 보관된 chat_history를
        # runner가 기대하는 list 형태로 변환합니다.
        #
        # deepcopy()를 사용하여
        # Agent 실행 중 내부 dict가 수정되어도
        # 평가 Case 원본이 변경되지 않도록 합니다.
        chat_history = (
            deepcopy(
                list(
                    case.chat_history
                )
            )
            if case.chat_history
            else None
        )

        final_state = (
            failure_agent_graph
            .run_failure_agent_graph(
                question=case.question,
                raw_sample=deepcopy(
                    case.raw_sample
                ),
                # Day 21 기본 평가는
                # Intent, Routing, Fallback,
                # Answer, Evidence 정합성에 집중합니다.
                #
                # SHAP과 Permutation Importance는
                # 기존 전용 테스트에서 검증했으므로
                # 기본 평가에서는 비활성화합니다.
                include_shap=False,
                include_global_importance=False,
                chat_history=chat_history,
            )
        )

    actual_output = (
        _build_actual_output(
            final_state
        )
    )

    checks: list[
        EvaluationCheckResult
    ] = []

    # 1. Intent
    checks.append(
        _build_value_check(
            check_name="intent",
            expected=(
                case.expected_intent
            ),
            actual=(
                actual_output[
                    "intent"
                ]
            ),
        )
    )

    # 2. Prediction
    checks.append(
        _build_value_check(
            check_name="prediction",
            expected=(
                case.expected_prediction
            ),
            actual=(
                actual_output[
                    "prediction"
                ]
            ),
        )
    )

    # 3. Probability
    checks.append(
        _build_value_check(
            check_name="probability",
            expected=(
                case.expected_probability
            ),
            actual=(
                actual_output[
                    "probability"
                ]
            ),
        )
    )

    # 4. Threshold
    checks.append(
        _build_value_check(
            check_name="threshold",
            expected=(
                case.expected_threshold
            ),
            actual=(
                actual_output[
                    "threshold"
                ]
            ),
        )
    )

    # 5. Risk Level
    checks.append(
        _build_value_check(
            check_name="risk_level",
            expected=(
                case.expected_risk_level
            ),
            actual=(
                actual_output[
                    "risk_level"
                ]
            ),
        )
    )

    # 6. Fallback 여부
    checks.append(
        _build_value_check(
            check_name=(
                "fallback_occurred"
            ),
            expected=(
                case
                .expected_fallback_occurred
            ),
            actual=(
                actual_output[
                    "fallback_occurred"
                ]
            ),
        )
    )

    # 7. Evidence 최소 개수
    checks.append(
        _build_minimum_count_check(
            check_name="evidence_count",
            minimum_count=(
                case
                .minimum_evidence_count
            ),
            actual_count=(
                actual_output[
                    "evidence_count"
                ]
            ),
        )
    )

    # 8. Error 개수
    checks.append(
        _build_value_check(
            check_name="error_count",
            expected=(
                case.expected_error_count
            ),
            actual=len(
                actual_output[
                    "errors"
                ]
            ),
        )
    )

    answer = (
        actual_output.get(
            "answer"
        )
        or ""
    )

    # 모든 Agent 결과는
    # 최소한 비어 있지 않은 answer를
    # 반환해야 합니다.
    checks.append(
        EvaluationCheckResult(
            check_name=(
                "answer_is_not_empty"
            ),
            passed=bool(
                answer.strip()
            ),
            expected=(
                "non-empty answer"
            ),
            actual=answer,
            message=(
                "최종 answer가 "
                "비어 있지 않은지 평가합니다."
            ),
        )
    )

    # 9. Answer 필수 문구
    for index, substring in enumerate(
        case.required_answer_substrings,
        start=1,
    ):
        checks.append(
            _build_required_substring_check(
                check_name=(
                    "required_answer_"
                    f"substring_{index}"
                ),
                required_substring=(
                    substring
                ),
                actual_text=answer,
            )
        )

    # errors는 list[str] 형태이므로
    # 하나의 문자열로 결합한 뒤
    # 필수 오류 문구를 검색합니다.
    combined_errors = "\n".join(
        str(error)
        for error in actual_output[
            "errors"
        ]
    )

    # 10. Error 필수 문구
    for index, substring in enumerate(
        case.required_error_substrings,
        start=1,
    ):
        checks.append(
            _build_required_substring_check(
                check_name=(
                    "required_error_"
                    f"substring_{index}"
                ),
                required_substring=(
                    substring
                ),
                actual_text=(
                    combined_errors
                ),
            )
        )

    # 11. Answer 금지 문구
    for index, substring in enumerate(
        case.forbidden_answer_substrings,
        start=1,
    ):
        checks.append(
            _build_forbidden_substring_check(
                check_name=(
                    "forbidden_answer_"
                    f"substring_{index}"
                ),
                forbidden_substring=(
                    substring
                ),
                actual_text=answer,
            )
        )

    # 모든 Check가 통과해야
    # Case 전체를 PASS로 판정합니다.
    passed = all(
        check.passed
        for check in checks
    )

    return AgentEvaluationResult(
        case_id=case.case_id,
        category=case.category,
        description=case.description,
        passed=passed,
        checks=tuple(
            checks
        ),
        actual_output=actual_output,
    )


def _build_category_summary(
    results: tuple[
        AgentEvaluationResult,
        ...,
    ],
) -> dict[
    str,
    dict[str, int | float],
]:
    """
    평가 영역별 통과 결과를 계산합니다.

    예
    --
    {
        "safety": {
            "total_count": 2,
            "passed_count": 2,
            "failed_count": 0,
            "pass_rate": 100.0
        }
    }
    """

    category_counts: dict[
        str,
        dict[str, int],
    ] = {}

    for result in results:
        if (
            result.category
            not in category_counts
        ):
            category_counts[
                result.category
            ] = {
                "total_count": 0,
                "passed_count": 0,
                "failed_count": 0,
            }

        current = category_counts[
            result.category
        ]

        current[
            "total_count"
        ] += 1

        if result.passed:
            current[
                "passed_count"
            ] += 1
        else:
            current[
                "failed_count"
            ] += 1

    category_summary: dict[
        str,
        dict[str, int | float],
    ] = {}

    for (
        category,
        counts,
    ) in category_counts.items():
        total_count = counts[
            "total_count"
        ]

        passed_count = counts[
            "passed_count"
        ]

        pass_rate = (
            passed_count
            / total_count
            * 100.0
            if total_count > 0
            else 0.0
        )

        category_summary[
            category
        ] = {
            "total_count": (
                total_count
            ),
            "passed_count": (
                passed_count
            ),
            "failed_count": (
                counts[
                    "failed_count"
                ]
            ),
            "pass_rate": round(
                pass_rate,
                2,
            ),
        }

    return category_summary


def evaluate_agent_cases(
    cases: tuple[
        AgentEvaluationCase,
        ...,
    ],
) -> AgentEvaluationSummary:
    """
    여러 Agent 평가 Case를 순서대로 실행합니다.

    Parameter
    ---------
    cases:
        실행할 AgentEvaluationCase tuple입니다.


    Return
    ------
    AgentEvaluationSummary

        전체 통과율과
        Case별 상세 결과를 포함합니다.
    """

    results = tuple(
        evaluate_agent_case(
            case
        )
        for case in cases
    )

    total_count = len(
        results
    )

    passed_count = sum(
        1
        for result in results
        if result.passed
    )

    failed_count = (
        total_count
        - passed_count
    )

    pass_rate = (
        passed_count
        / total_count
        * 100.0
        if total_count > 0
        else 0.0
    )

    return AgentEvaluationSummary(
        total_count=total_count,
        passed_count=passed_count,
        failed_count=failed_count,
        pass_rate=round(
            pass_rate,
            2,
        ),
        category_summary=(
            _build_category_summary(
                results
            )
        ),
        results=results,
    )