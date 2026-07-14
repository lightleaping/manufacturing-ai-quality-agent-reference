# src/dashboard/pages/failure_prediction.py

"""Beginner-first Streamlit page for manufacturing failure-risk analysis."""

from __future__ import annotations

import math
import re
from typing import Any

import streamlit as st

from src.dashboard.api_client import (
    DashboardApiClient,
    DashboardApiClientError,
)
from src.dashboard.session_state import (
    initialize_dashboard_session_state,
)
from src.dashboard.ui_helpers import (
    build_failure_prediction_payload,
    group_evidence_by_type,
)


FEATURE_LABELS = {
    "Air temperature [K]": "공기 온도",
    "Process temperature [K]": "공정 온도",
    "Rotational speed [rpm]": "회전 속도",
    "Torque [Nm]": "토크",
    "Tool wear [min]": "공구 마모 시간",
    "Type": "제품 유형",
}

FEATURE_UNITS = {
    "Air temperature [K]": "K",
    "Process temperature [K]": "K",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "Nm",
    "Tool wear [min]": "분",
    "Type": "",
}


def format_percentage(
    value: Any,
    *,
    digits: int = 2,
) -> str:
    """Format a backend ratio as a percentage for display only."""
    if value is None:
        return "-"

    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return str(
            value
        )

    if not math.isfinite(
        numeric_value
    ):
        return "-"

    return (
        f"{numeric_value * 100:.{digits}f}%"
    )


def safe_float(
    value: Any,
) -> float | None:
    """Return a finite float or None."""
    try:
        normalized = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        normalized
    ):
        return None

    return normalized


def get_prediction_display(
    prediction: Any,
) -> str:
    """Convert backend prediction 0/1 into a plain-language label."""
    try:
        normalized = int(
            prediction
        )

    except (
        TypeError,
        ValueError,
    ):
        return "판정 확인 필요"

    if normalized == 1:
        return "고장 위험 있음"

    if normalized == 0:
        return "고장 위험 낮음"

    return "판정 확인 필요"


def format_risk_level(
    risk_level: Any,
) -> str:
    """Add a Korean explanation while preserving the backend level."""
    normalized = str(
        risk_level
        or "UNKNOWN"
    ).strip().upper()

    labels = {
        "HIGH": "높음 (HIGH)",
        "MEDIUM": "보통 (MEDIUM)",
        "LOW": "낮음 (LOW)",
        "UNKNOWN": "확인 필요 (UNKNOWN)",
    }

    return labels.get(
        normalized,
        f"확인 필요 ({normalized})",
    )


def get_risk_copy(
    risk_level: Any,
) -> tuple[str, str, str]:
    """
    Return a conclusion, plain-language meaning, and immediate action.
    """
    normalized = str(
        risk_level
        or "UNKNOWN"
    ).strip().upper()

    copies = {
        "HIGH": (
            "현재 입력값에서는 고장 위험 신호가 높게 나타났습니다.",
            (
                "AI 모델이 온도, 회전 속도, 토크, 공구 마모 시간 등을 "
                "함께 확인한 결과, 위험 판정 기준선을 넘는 신호를 찾았습니다. "
                "이 결과는 실제 고장이 이미 발생했다는 확정 진단이 아닙니다."
            ),
            (
                "먼저 공구 상태와 설비 부하를 확인하고, "
                "현재 운전 조건이 현장 기준에서 벗어나지 않았는지 점검하세요."
            ),
        ),
        "MEDIUM": (
            "현재 입력값에서는 주의해서 살펴볼 신호가 나타났습니다.",
            (
                "AI 모델이 일부 입력값을 주의 신호로 보았습니다. "
                "즉시 고장이라는 뜻은 아니며, 평소 운전 조건과 비교해 "
                "달라진 값이 있는지 확인하는 단계입니다."
            ),
            (
                "운전 기록과 최근 점검 기록을 함께 확인하고, "
                "평소와 다른 변화가 있는지 살펴보세요."
            ),
        ),
        "LOW": (
            "현재 입력값에서는 고장 위험 신호가 낮게 나타났습니다.",
            (
                "AI 모델이 현재 입력값을 비교적 낮은 위험 상태로 보았습니다. "
                "다만 이 결과가 실제 설비의 정상 상태를 보장하지는 않습니다."
            ),
            (
                "현재 운전 상태를 유지하되, "
                "정기 점검과 현장 기준에 따른 확인은 계속하세요."
            ),
        ),
        "UNKNOWN": (
            "현재 위험 단계를 확인할 수 없습니다.",
            (
                "Backend 응답에서 위험 단계 정보를 확인하지 못했습니다. "
                "입력값과 서버 응답을 다시 확인해야 합니다."
            ),
            (
                "입력값을 확인한 뒤 다시 분석하고, "
                "문제가 계속되면 상세 오류 정보를 확인하세요."
            ),
        ),
    }

    return copies.get(
        normalized,
        copies["UNKNOWN"],
    )


def clean_explanation_item(
    value: Any,
) -> str:
    """
    Remove generated labels and broken-text remnants from old responses.

    The Dashboard owns section labels.
    OpenAI provides only the explanatory sentence.
    """
    text = str(
        value
        or ""
    ).strip()

    text = re.sub(
        r"^\[[^\]]*\]\s*",
        "",
        text,
    )

    text = text.replace(
        "\ufffd",
        "",
    )

    text = re.sub(
        r"\?{2,}",
        "",
        text,
    )

    return text.strip()


def feature_label(
    feature: Any,
) -> str:
    """Translate a backend feature name into a beginner-friendly label."""
    normalized = str(
        feature
        or ""
    ).strip()

    return FEATURE_LABELS.get(
        normalized,
        normalized
        or "입력값",
    )


def format_feature_value(
    feature: Any,
    value: Any,
) -> str:
    """Render a feature value with a familiar label and unit."""
    normalized_feature = str(
        feature
        or ""
    ).strip()

    label = feature_label(
        normalized_feature
    )

    unit = FEATURE_UNITS.get(
        normalized_feature,
        "",
    )

    numeric_value = safe_float(
        value
    )

    if numeric_value is None:
        value_text = str(
            value
            if value is not None
            else "-"
        )

    elif numeric_value.is_integer():
        value_text = str(
            int(
                numeric_value
            )
        )

    else:
        value_text = (
            f"{numeric_value:.2f}"
            .rstrip("0")
            .rstrip(".")
        )

    if unit:
        return (
            f"{label} {value_text} {unit}"
        )

    return (
        f"{label} {value_text}"
    )


def split_readable_sentences(
    text: Any,
) -> list[str]:
    """Split a long explanation into short readable sentences."""
    normalized = clean_explanation_item(
        text
    )

    if not normalized:
        return []

    parts = re.split(
        r"(?<=[.!?。])\s+",
        normalized,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def render_sentence_list(
    text: Any,
) -> None:
    """Render long explanatory text as short paragraphs."""
    sentences = split_readable_sentences(
        text
    )

    if not sentences:
        return

    for sentence in sentences:
        st.write(
            sentence
        )


def render_page_intro() -> None:
    """Explain what the page does in three short steps."""
    st.title(
        "설비 고장 위험 분석"
    )

    st.write(
        "설비 운전 값을 입력하면 AI가 고장 위험 신호를 예측하고, "
        "그 결과를 처음 보는 사람도 이해하기 쉬운 순서로 설명합니다."
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "#### 이 화면은 이렇게 사용합니다"
        )

        step_columns = st.columns(
            3
        )

        steps = [
            (
                "1. 설비 값 입력",
                "온도, 회전 속도, 토크, 공구 마모 시간과 제품 유형을 입력합니다.",
            ),
            (
                "2. 위험 신호 확인",
                "AI 모델 위험 점수와 위험 판정 기준선을 비교해 결과를 확인합니다.",
            ),
            (
                "3. 이유와 점검 순서 확인",
                "왜 이런 결과가 나왔는지와 무엇부터 확인할지 살펴봅니다.",
            ),
        ]

        for (
            column,
            (
                title,
                description,
            ),
        ) in zip(
            step_columns,
            steps,
        ):
            with column:
                st.markdown(
                    f"**{title}**"
                )

                st.write(
                    description
                )

    st.caption(
        "Dashboard는 예측을 다시 계산하지 않습니다. "
        "입력값을 FastAPI Backend에 전달하고, "
        "Backend가 반환한 예측 결과와 판단 근거를 화면에 표시합니다."
    )


def render_input_guide() -> None:
    """Provide compact examples without crowding the main form."""
    with st.expander(
        "입력값과 단위를 처음 보는 경우",
        expanded=False,
    ):
        st.markdown(
            "#### 입력값을 읽는 예시"
        )

        st.write(
            "**303 K**는 약 **29.9°C**, "
            "**312.5 K**는 약 **39.4°C**입니다."
        )

        st.write(
            "**1,380 rpm**은 설비가 1분에 약 1,380회 회전한다는 뜻입니다."
        )

        st.write(
            "**토크 62 Nm**는 회전축에 걸리는 힘을 나타내고, "
            "**공구 마모 시간 220분**은 공구 사용과 관련된 누적 입력값입니다."
        )

        st.write(
            "**제품 유형 L, M, H**는 AI4I 학습 데이터에서 사용한 제품 유형 코드입니다."
        )

        st.caption(
            "위 설명은 단위를 이해하기 위한 예시입니다. "
            "실제 정상 범위와 점검 기준은 설비 종류와 현장 운영 기준에 따라 달라질 수 있습니다."
        )


def render_input_form() -> tuple[
    bool,
    dict[str, Any],
]:
    """Render the prediction form and return submission state plus values."""
    with st.form(
        "failure_prediction_form"
    ):
        st.markdown(
            "### 설비 운전 값 입력"
        )

        st.caption(
            "현재 설비에서 측정한 값을 입력하세요. "
            "아래 기본값은 Dashboard 기능 확인을 위한 학습용 예시입니다."
        )

        columns = st.columns(
            2
        )

        with columns[0]:
            st.markdown(
                "#### 온도와 회전"
            )

            air_temperature = (
                st.number_input(
                    "공기 온도 (K)",
                    value=303.0,
                    step=0.1,
                    help=(
                        "설비 주변 공기의 절대 온도입니다. "
                        "303 K는 약 29.9°C입니다."
                    ),
                )
            )

            process_temperature = (
                st.number_input(
                    "공정 온도 (K)",
                    value=312.5,
                    step=0.1,
                    help=(
                        "공정에서 측정한 절대 온도입니다. "
                        "312.5 K는 약 39.4°C입니다."
                    ),
                )
            )

            rotational_speed = (
                st.number_input(
                    "회전 속도 (rpm)",
                    value=1380.0,
                    step=1.0,
                    help=(
                        "설비가 1분 동안 회전하는 횟수입니다. "
                        "1,380 rpm은 1분에 약 1,380회 회전한다는 뜻입니다."
                    ),
                )
            )

        with columns[1]:
            st.markdown(
                "#### 부하와 공구 사용"
            )

            torque = st.number_input(
                "토크 (Nm)",
                value=62.0,
                step=0.1,
                help=(
                    "회전축에 걸리는 힘의 크기입니다. "
                    "값의 의미는 설비의 정상 운전 기준과 함께 확인해야 합니다."
                ),
            )

            tool_wear = (
                st.number_input(
                    "공구 마모 시간 (분)",
                    value=220.0,
                    step=1.0,
                    help=(
                        "공구 사용과 관련된 누적 시간 입력값입니다. "
                        "실제 교체 기준은 현장 기준을 따라야 합니다."
                    ),
                )
            )

            machine_type = (
                st.selectbox(
                    "제품 유형",
                    options=[
                        "L",
                        "M",
                        "H",
                    ],
                    index=0,
                    help=(
                        "AI4I 학습 데이터에서 사용한 제품 유형 코드입니다."
                    ),
                )
            )

        with st.expander(
            "상세 분석 옵션",
            expanded=False,
        ):
            st.write(
                "처음 보는 사용자는 기본 설정을 그대로 사용해도 됩니다."
            )

            option_columns = (
                st.columns(
                    2
                )
            )

            with option_columns[0]:
                include_shap = (
                    st.checkbox(
                        "이번 입력값의 영향 분석 포함",
                        value=True,
                        help=(
                            "각 입력값이 이번 AI 판단을 높이거나 "
                            "낮춘 방향을 함께 확인합니다."
                        ),
                    )
                )

            with option_columns[1]:
                include_global_importance = (
                    st.checkbox(
                        "전체 데이터 중요도 포함",
                        value=True,
                        help=(
                            "전체 참고 데이터에서 AI가 자주 중요하게 본 "
                            "입력값을 함께 확인합니다."
                        ),
                    )
                )

        submitted = (
            st.form_submit_button(
                "고장 위험 분석하기",
                use_container_width=True,
            )
        )

    values = {
        "air_temperature": (
            air_temperature
        ),
        "process_temperature": (
            process_temperature
        ),
        "rotational_speed": (
            rotational_speed
        ),
        "torque": torque,
        "tool_wear": (
            tool_wear
        ),
        "machine_type": (
            machine_type
        ),
        "include_shap": (
            include_shap
        ),
        "include_global_importance": (
            include_global_importance
        ),
    }

    return (
        submitted,
        values,
    )


def render_result_overview(
    result: dict[str, Any],
) -> None:
    """Render the conclusion and immediate action before any detail."""
    (
        conclusion,
        meaning,
        next_action,
    ) = get_risk_copy(
        result.get(
            "risk_level"
        )
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "### 한눈에 보는 결론"
        )

        st.markdown(
            f"## {conclusion}"
        )

        st.write(
            meaning
        )

        st.divider()

        st.markdown(
            "#### 지금 할 일"
        )

        st.write(
            next_action
        )

        backend_action = (
            result.get(
                "recommended_action"
            )
        )

        if backend_action:
            st.caption(
                "Backend 권장 조치: "
                + str(
                    backend_action
                )
            )

        st.caption(
            "이 결과는 학습용 AI 모델의 예측입니다. "
            "실제 고장을 확정하는 진단 결과가 아니며, "
            "현장 점검과 설비 운영 기준을 대신하지 않습니다."
        )


def render_key_metrics(
    result: dict[str, Any],
) -> None:
    """Render four metrics in a beginner-friendly reading order."""
    probability = result.get(
        "probability"
    )

    threshold = result.get(
        "threshold"
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "### 핵심 결과"
        )

        st.caption(
            "처음 보는 사용자는 "
            "아래 네 가지를 순서대로 확인하면 됩니다."
        )

        first_row = st.columns(
            2
        )

        first_row[0].metric(
            label="한눈에 보는 판정",
            value=get_prediction_display(
                result.get(
                    "prediction"
                )
            ),
            help=(
                "Backend의 Prediction 0과 1을 "
                "쉬운 한국어 문장으로 표시합니다."
            ),
        )

        first_row[1].metric(
            label="AI 모델 위험 점수",
            value=format_percentage(
                probability
            ),
            help=(
                "현재 입력값을 보고 AI 모델이 "
                "고장 위험 신호를 얼마나 높게 보았는지 "
                "나타내는 모델 출력입니다."
            ),
        )

        second_row = st.columns(
            2
        )

        second_row[0].metric(
            label="위험 판정 기준선",
            value=format_percentage(
                threshold
            ),
            help=(
                "Backend가 위험 있음과 위험 낮음을 "
                "나눌 때 사용한 운영 기준입니다."
            ),
        )

        second_row[1].metric(
            label="위험 단계",
            value=format_risk_level(
                result.get(
                    "risk_level"
                )
            ),
            help=(
                "Backend가 반환한 위험 단계를 "
                "한국어 설명과 함께 표시합니다."
            ),
        )

        probability_value = (
            safe_float(
                probability
            )
        )

        if probability_value is not None:
            st.progress(
                min(
                    max(
                        probability_value,
                        0.0,
                    ),
                    1.0,
                ),
                text=(
                    "AI 모델 위험 점수 "
                    + format_percentage(
                        probability
                    )
                ),
            )

        st.caption(
            "AI 모델 위험 점수는 "
            "실제 고장 확률을 확정하는 값이 아닙니다. "
            "현재 입력값을 모델이 얼마나 위험한 방향으로 "
            "보았는지 나타내는 예측 참고값입니다."
        )


def render_score_example(
    result: dict[str, Any],
) -> None:
    """Explain the score and threshold with a dynamic example."""
    probability = safe_float(
        result.get(
            "probability"
        )
    )

    threshold = safe_float(
        result.get(
            "threshold"
        )
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "### 숫자를 예로 들어 이해하기"
        )

        if (
            probability is None
            or threshold is None
        ):
            st.write(
                "현재 응답에서는 모델 위험 점수 또는 기준선을 확인할 수 없습니다."
            )

            return

        probability_score = (
            probability
            * 100
        )

        threshold_score = (
            threshold
            * 100
        )

        st.write(
            f"100점 척도로 비유하면, "
            f"위험 판정 기준선은 **{threshold_score:.1f}점**이고 "
            f"현재 AI 모델 위험 점수는 **{probability_score:.1f}점**입니다."
        )

        if probability >= threshold:
            st.write(
                "현재 모델 점수가 기준선을 넘었기 때문에 "
                "Backend는 '고장 위험 있음'으로 판정했습니다."
            )

        else:
            st.write(
                "현재 모델 점수가 기준선보다 낮기 때문에 "
                "Backend는 '고장 위험 낮음'으로 판정했습니다."
            )

        st.caption(
            "이 비유는 AI 판정 방식을 쉽게 설명하기 위한 것입니다. "
            f"실제 설비가 고장 날 현실 확률이 정확히 "
            f"{probability_score:.1f}%라는 뜻은 아닙니다."
        )

        with st.expander(
            "용어를 하나씩 알아보기",
            expanded=False,
        ):
            st.markdown(
                """
**AI 모델 위험 점수**  
현재 입력값을 보고 AI 모델이 고장 위험 신호를 얼마나 높게 보았는지 나타내는 모델 출력입니다.

**위험 판정 기준선**  
모델 출력을 위험 있음과 위험 낮음으로 나누기 위해 Backend가 사용한 운영 기준입니다.

**위험 단계**  
결과를 빠르게 읽을 수 있도록 높음, 보통, 낮음으로 구분한 표시입니다.

**판단 근거**  
어떤 입력값과 참고 기준이 이번 예측을 이해하는 데 사용됐는지 보여주는 설명 자료입니다.
"""
            )


def evidence_plain_meaning(
    item: dict[str, Any],
) -> str:
    """Translate one evidence item into a short beginner-friendly meaning."""
    evidence_type = str(
        item.get(
            "evidence_type",
            "",
        )
    ).strip()

    feature = item.get(
        "feature"
    )

    value = item.get(
        "value"
    )

    observed = format_feature_value(
        feature,
        value,
    )

    if evidence_type == "rule_based":
        return (
            f"{observed} 값이 사람이 미리 정한 점검 기준에 해당했습니다. "
            "현장 기준과 함께 확인할 필요가 있다는 뜻이며, "
            "이 값 하나가 실제 고장의 원인이라는 뜻은 아닙니다."
        )

    if evidence_type == "shap_local":
        direction = str(
            item.get(
                "direction",
                "",
            )
        ).strip().lower()

        if direction in {
            "positive",
            "increases_risk",
        }:
            direction_text = (
                "이번 AI 판단에서 위험 점수를 높이는 방향으로 작용했습니다."
            )

        elif direction in {
            "negative",
            "decreases_risk",
        }:
            direction_text = (
                "이번 AI 판단에서 위험 점수를 낮추는 방향으로 작용했습니다."
            )

        else:
            direction_text = (
                "이번 AI 판단에 영향을 준 입력값으로 표시됐습니다."
            )

        return (
            f"{observed} 값은 {direction_text} "
            "이는 AI 판단의 방향을 설명하는 참고자료이며, "
            "실제 물리적 고장 원인을 증명하지는 않습니다."
        )

    if evidence_type == "global_importance":
        return (
            f"{feature_label(feature)}은 전체 참고 데이터에서 "
            "AI가 비교적 중요하게 사용한 입력값으로 표시됐습니다. "
            "이번 설비의 직접적인 고장 원인이라는 뜻은 아닙니다."
        )

    summary = clean_explanation_item(
        item.get(
            "summary",
            "",
        )
    )

    if summary:
        return summary

    return (
        "이 항목은 Backend가 반환한 판단 근거입니다."
    )


def select_top_evidence(
    evidence_items: Any,
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Choose the most useful evidence items for the first screen."""
    if not isinstance(
        evidence_items,
        list,
    ):
        return []

    candidates = [
        item
        for item in evidence_items
        if (
            isinstance(
                item,
                dict,
            )
            and item.get(
                "evidence_type"
            )
            in {
                "rule_based",
                "shap_local",
            }
        )
    ]

    def score(
        item: dict[str, Any],
    ) -> float:
        contribution = safe_float(
            item.get(
                "contribution"
            )
        )

        if contribution is None:
            return 0.0

        return abs(
            contribution
        )

    candidates.sort(
        key=score,
        reverse=True,
    )

    selected: list[
        dict[str, Any]
    ] = []

    seen: set[
        tuple[str, str]
    ] = set()

    for item in candidates:
        key = (
            str(
                item.get(
                    "evidence_type",
                    "",
                )
            ),
            str(
                item.get(
                    "feature",
                    "",
                )
            ),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        selected.append(
            item
        )

        if len(
            selected
        ) >= limit:
            break

    return selected


def render_top_evidence(
    evidence_items: Any,
) -> None:
    """Show only the most useful evidence before the detailed sections."""
    selected = select_top_evidence(
        evidence_items
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "### 왜 이런 판정이 나왔나요?"
        )

        st.write(
            "AI는 한 가지 값만 보고 결론을 내리지 않습니다. "
            "입력한 설비 조건과 여러 판단 근거를 함께 확인합니다."
        )

        if not selected:
            st.write(
                "현재 응답에서는 먼저 보여줄 주요 판단 근거를 찾지 못했습니다."
            )

            return

        for (
            index,
            item,
        ) in enumerate(
            selected,
            start=1,
        ):
            with st.container(
                border=True
            ):
                st.markdown(
                    f"#### 주요 근거 {index}"
                )

                feature = item.get(
                    "feature"
                )

                value = item.get(
                    "value"
                )

                if (
                    feature is not None
                    or value is not None
                ):
                    st.markdown(
                        "**확인된 입력값**"
                    )

                    st.write(
                        format_feature_value(
                            feature,
                            value,
                        )
                    )

                st.markdown(
                    "**쉽게 말하면**"
                )

                st.write(
                    evidence_plain_meaning(
                        item
                    )
                )

        st.caption(
            "더 많은 판단 근거와 기술 수치는 "
            "'판단 근거 자세히 보기' 화면 또는 아래 기술 상세 영역에서 확인할 수 있습니다."
        )


def build_check_steps(
    evidence_items: Any,
) -> list[
    tuple[str, str]
]:
    """Build practical checks from confirmed evidence without changing prediction."""
    steps: list[
        tuple[str, str]
    ] = []

    features: set[
        str
    ] = set()

    if isinstance(
        evidence_items,
        list,
    ):
        for item in evidence_items:
            if not isinstance(
                item,
                dict,
            ):
                continue

            feature = str(
                item.get(
                    "feature",
                    "",
                )
            ).strip()

            if feature:
                features.add(
                    feature
                )

    if "Tool wear [min]" in features:
        steps.append(
            (
                "공구 상태 확인",
                (
                    "공구 마모 시간 입력값이 점검 신호로 표시됐습니다. "
                    "공구의 사용 기록과 현재 상태를 현장 교체 기준과 비교하세요."
                ),
            )
        )

    if "Torque [Nm]" in features:
        steps.append(
            (
                "설비 부하와 토크 확인",
                (
                    "토크 값이 주요 판단 근거에 포함됐습니다. "
                    "현재 부하가 평소 운전 범위와 다른지 작업 기록과 함께 확인하세요."
                ),
            )
        )

    temperature_features = {
        "Air temperature [K]",
        "Process temperature [K]",
    }

    if (
        features
        & temperature_features
    ):
        steps.append(
            (
                "온도 조건 확인",
                (
                    "온도 입력값이 AI 판단에 영향을 준 항목으로 표시됐습니다. "
                    "센서 값과 현장 운전 기준을 함께 확인하세요."
                ),
            )
        )

    if (
        "Rotational speed [rpm]"
        in features
    ):
        steps.append(
            (
                "회전 조건 확인",
                (
                    "회전 속도가 AI 판단에 영향을 준 항목으로 표시됐습니다. "
                    "현재 회전 조건이 작업 기준과 일치하는지 확인하세요."
                ),
            )
        )

    if not steps:
        steps = [
            (
                "입력값 다시 확인",
                (
                    "입력한 값이 실제 설비 측정값과 일치하는지 확인하세요."
                ),
            ),
            (
                "현장 기준과 비교",
                (
                    "현재 운전 조건이 설비의 정상 운영 범위와 다른지 확인하세요."
                ),
            ),
        ]

    return steps[
        :3
    ]


def render_check_steps(
    result: dict[str, Any],
) -> None:
    """Render a clear practical checking order."""
    steps = build_check_steps(
        result.get(
            "evidence"
        )
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "### 무엇부터 확인하면 되나요?"
        )

        st.write(
            "아래 순서는 AI가 확정 진단을 내린 절차가 아니라, "
            "현재 결과를 보고 현장에서 확인해볼 수 있는 권장 순서입니다."
        )

        for (
            index,
            (
                title,
                description,
            ),
        ) in enumerate(
            steps,
            start=1,
        ):
            with st.container(
                border=True
            ):
                st.markdown(
                    f"#### {index}단계. {title}"
                )

                st.write(
                    description
                )

        backend_action = (
            result.get(
                "recommended_action"
            )
        )

        if backend_action:
            st.caption(
                "Backend 권장 조치: "
                + str(
                    backend_action
                )
            )


def render_evidence_summary(
    evidence_items: Any,
) -> None:
    """Explain the three evidence groups vertically without clipped cards."""
    if not isinstance(
        evidence_items,
        list,
    ):
        return

    grouped = group_evidence_by_type(
        evidence_items
    )

    items = [
        (
            "사람이 미리 정한 점검 기준",
            len(
                grouped.get(
                    "rule_based",
                    [],
                )
            ),
            (
                "현재 입력값이 사람이 미리 정한 제조 점검 기준에 "
                "해당했는지 보여줍니다."
            ),
            (
                "예: 공구 마모 시간이 점검 기준에 해당하면 "
                "공구 상태를 다시 확인할 신호로 사용합니다."
            ),
        ),
        (
            "이번 입력값이 AI 판단에 준 영향",
            len(
                grouped.get(
                    "shap_local",
                    [],
                )
            ),
            (
                "각 입력값이 이번 AI 위험 판단을 "
                "높이거나 낮춘 방향을 보여줍니다."
            ),
            (
                "예: 토크가 위험 점수를 높이는 방향으로 작용했다면 "
                "이번 입력에서 토크가 AI 판단에 큰 영향을 줬다는 뜻입니다."
            ),
        ),
        (
            "전체 데이터에서 중요했던 입력값",
            len(
                grouped.get(
                    "global_importance",
                    [],
                )
            ),
            (
                "전체 참고 데이터에서 AI가 자주 중요하게 사용한 "
                "입력값을 보여줍니다."
            ),
            (
                "예: 전체 데이터에서 토크가 중요했다고 해도 "
                "이번 설비 고장의 직접 원인이라는 뜻은 아닙니다."
            ),
        ),
    ]

    with st.container(
        border=True
    ):
        st.markdown(
            "### 판단 근거는 세 종류입니다"
        )

        st.write(
            "세 종류는 서로 의미가 다릅니다. "
            "같은 원인을 세 번 말하는 것이 아니라, "
            "서로 다른 관점에서 AI 결과를 설명합니다."
        )

        for (
            index,
            (
                title,
                count,
                meaning,
                example,
            ),
        ) in enumerate(
            items,
            start=1,
        ):
            with st.container(
                border=True
            ):
                st.markdown(
                    f"#### {index}. {title}"
                )

                st.markdown(
                    f"**확인된 근거 수: {count}개**"
                )

                st.write(
                    meaning
                )

                st.caption(
                    example
                )

        st.caption(
            "각 근거의 전체 목록은 왼쪽 메뉴의 "
            "'판단 근거 자세히 보기' 화면에서 확인할 수 있습니다."
        )


def render_explanation_failure(
    error: Any,
) -> None:
    """Show an explanation-only failure without making prediction look failed."""
    with st.container(
        border=True
    ):
        st.markdown(
            "#### 쉬운 설명을 생성하지 못했습니다"
        )

        st.write(
            "고장 위험 예측 결과와 판단 근거는 정상적으로 유지됩니다. "
            "추가 설명 기능만 현재 사용할 수 없습니다."
        )

        st.caption(
            "예측 실패가 아니므로 위에 표시된 결과는 그대로 확인하면 됩니다."
        )

        with st.expander(
            "개발자용 상세 정보",
            expanded=False,
        ):
            st.write(
                str(
                    error
                )
            )


def render_operational_explanation_card(
    explanation: dict[str, Any],
) -> None:
    """Render the optional explanation as full-width vertical cards."""
    error = explanation.get(
        "error"
    )

    if error:
        render_explanation_failure(
            error
        )

        return

    summary = clean_explanation_item(
        explanation.get(
            "summary",
            "",
        )
    )

    raw_key_signals = (
        explanation.get(
            "key_signals"
        )
    )

    raw_recommended_checks = (
        explanation.get(
            "recommended_checks"
        )
    )

    caution = clean_explanation_item(
        explanation.get(
            "caution",
            "",
        )
    )

    key_signals: list[
        str
    ] = []

    if isinstance(
        raw_key_signals,
        list,
    ):
        key_signals = [
            cleaned
            for cleaned in (
                clean_explanation_item(
                    item
                )
                for item in raw_key_signals
            )
            if cleaned
        ]

    recommended_checks: list[
        str
    ] = []

    if isinstance(
        raw_recommended_checks,
        list,
    ):
        recommended_checks = [
            cleaned
            for cleaned in (
                clean_explanation_item(
                    item
                )
                for item in raw_recommended_checks
            )
            if cleaned
        ]

    st.divider()

    st.markdown(
        "### AI가 한 번 더 쉽게 설명한 내용"
    )

    st.caption(
        "이 설명은 이미 확정된 Backend 예측 결과와 판단 근거를 "
        "쉬운 한국어로 정리한 것입니다. 기존 예측값은 바뀌지 않습니다."
    )

    with st.container(
        border=True
    ):
        st.markdown(
            "#### 현재 상태를 한 문장씩 이해하기"
        )

        if summary:
            render_sentence_list(
                summary
            )

        else:
            st.write(
                "표시할 요약 설명이 없습니다."
            )

    with st.container(
        border=True
    ):
        st.markdown(
            "#### AI와 점검 기준이 중요하게 본 내용"
        )

        st.write(
            "아래 내용은 확인할 가치가 있는 신호입니다. "
            "실제 고장 원인을 확정한 내용은 아닙니다."
        )

        if key_signals:
            for (
                index,
                item,
            ) in enumerate(
                key_signals,
                start=1,
            ):
                with st.container(
                    border=True
                ):
                    st.markdown(
                        f"##### 확인할 근거 {index}"
                    )

                    render_sentence_list(
                        item
                    )

        else:
            st.write(
                "표시할 주요 판단 근거가 없습니다."
            )

    with st.container(
        border=True
    ):
        st.markdown(
            "#### 현장에서 확인할 순서"
        )

        st.write(
            "각 단계는 점검을 시작하기 위한 참고 순서입니다."
        )

        if recommended_checks:
            for (
                index,
                item,
            ) in enumerate(
                recommended_checks,
                start=1,
            ):
                with st.container(
                    border=True
                ):
                    st.markdown(
                        f"##### {index}단계"
                    )

                    render_sentence_list(
                        item
                    )

        else:
            st.write(
                "표시할 확인 항목이 없습니다."
            )

    if caution:
        with st.container(
            border=True
        ):
            st.markdown(
                "#### 반드시 기억할 점"
            )

            render_sentence_list(
                caution
            )

    with st.expander(
        "AI 설명 생성 정보",
        expanded=False,
    ):
        st.write(
            "**설명 출처**"
        )

        st.write(
            str(
                explanation.get(
                    "source",
                    "openai",
                )
            )
        )

        model = explanation.get(
            "model"
        )

        if model:
            st.write(
                "**사용 모델**"
            )

            st.write(
                str(
                    model
                )
            )

        st.caption(
            "이 정보는 설명을 생성한 기술 정보를 확인하기 위한 것이며, "
            "설비 위험 판정 자체를 변경하지 않습니다."
        )


def render_operational_explanation_section(
    prediction_result: dict[str, Any],
) -> None:
    """Render the optional OpenAI explanation section."""
    st.divider()

    with st.container(
        border=True
    ):
        st.markdown(
            "### 설명이 더 필요한가요?"
        )

        st.write(
            "현재 예측 결과와 판단 근거를 AI가 "
            "처음 보는 사람도 이해하기 쉬운 문장으로 한 번 더 정리합니다."
        )

        st.caption(
            "선택 기능입니다. 버튼을 누를 때만 OpenAI API를 사용하며, "
            "추가 설명은 기존 예측 결과를 바꾸지 않습니다."
        )

        generate_explanation = (
            st.button(
                "AI에게 쉽게 설명받기",
                key=(
                    "generate_failure_operational_explanation"
                ),
                use_container_width=True,
            )
        )

        if generate_explanation:
            request_payload = {
                "prediction_result": (
                    prediction_result
                )
            }

            try:
                with st.spinner(
                    "예측 결과를 쉬운 문장으로 정리하고 있습니다..."
                ):
                    with (
                        DashboardApiClient()
                        as api_client
                    ):
                        explanation = (
                            api_client
                            .generate_failure_prediction_explanation(
                                request_payload
                            )
                        )

            except (
                DashboardApiClientError,
                ValueError,
            ) as exc:
                render_explanation_failure(
                    exc
                )

            else:
                st.session_state[
                    "failure_prediction_explanation"
                ] = explanation

        stored_explanation = (
            st.session_state.get(
                "failure_prediction_explanation"
            )
        )

        if isinstance(
            stored_explanation,
            dict,
        ):
            render_operational_explanation_card(
                stored_explanation
            )


def render_backend_messages(
    *,
    title: str,
    items: Any,
) -> None:
    """Render backend warning or limitation text without rewriting it."""
    if not isinstance(
        items,
        list,
    ):
        st.caption(
            f"{title} 없음"
        )

        return

    normalized = [
        str(
            item
        ).strip()
        for item in items
        if str(
            item
        ).strip()
    ]

    if not normalized:
        st.caption(
            f"{title} 없음"
        )

        return

    st.markdown(
        f"#### {title}"
    )

    for item in normalized:
        st.write(
            f"- {item}"
        )


def render_technical_details(
    result: dict[str, Any],
) -> None:
    """Keep technical backend content available but out of the primary flow."""
    st.divider()

    st.markdown(
        "### 기술 상세와 모델 한계"
    )

    st.caption(
        "아래 내용은 개발자, 데이터 분석가, 면접 검토자처럼 "
        "기술 정보를 확인해야 하는 사용자를 위한 영역입니다."
    )

    with st.expander(
        "Backend Agent 원본 분석 보기",
        expanded=False,
    ):
        st.caption(
            "아래는 Backend Agent가 생성한 원본 분석입니다. "
            "기술 용어와 상세 수치가 포함됩니다."
        )

        answer = result.get(
            "answer"
        )

        if answer:
            st.markdown(
                str(
                    answer
                )
            )

        else:
            st.write(
                "표시할 원본 분석 내용이 없습니다."
            )

    with st.expander(
        "주의사항과 이 모델의 한계",
        expanded=False,
    ):
        st.write(
            "이 영역에서는 결과를 어디까지 참고할 수 있는지와 "
            "해석할 때 주의할 점을 확인합니다."
        )

        render_backend_messages(
            title="주의사항",
            items=result.get(
                "warnings"
            ),
        )

        st.divider()

        render_backend_messages(
            title="이 모델의 한계",
            items=result.get(
                "limitations"
            ),
        )


def render_failure_prediction_result(
    result: dict[str, Any],
) -> None:
    """Render the result in a clear beginner-first reading order."""
    st.divider()

    st.subheader(
        "고장 위험 예측 결과"
    )

    st.caption(
        "위에서 아래 순서대로 읽으면 됩니다. "
        "결론을 먼저 보고, 필요한 경우 이유와 기술 상세를 확인하세요."
    )

    render_result_overview(
        result
    )

    render_key_metrics(
        result
    )

    render_check_steps(
        result
    )

    render_top_evidence(
        result.get(
            "evidence"
        )
    )

    render_score_example(
        result
    )

    render_evidence_summary(
        result.get(
            "evidence"
        )
    )

    render_operational_explanation_section(
        result
    )

    render_technical_details(
        result
    )


def main() -> None:
    """Render the Streamlit page."""
    initialize_dashboard_session_state(
        st.session_state
    )

    render_page_intro()

    render_input_guide()

    (
        submitted,
        values,
    ) = render_input_form()

    if submitted:
        payload = (
            build_failure_prediction_payload(
                air_temperature=(
                    values[
                        "air_temperature"
                    ]
                ),
                process_temperature=(
                    values[
                        "process_temperature"
                    ]
                ),
                rotational_speed=(
                    values[
                        "rotational_speed"
                    ]
                ),
                torque=(
                    values[
                        "torque"
                    ]
                ),
                tool_wear=(
                    values[
                        "tool_wear"
                    ]
                ),
                machine_type=(
                    values[
                        "machine_type"
                    ]
                ),
                include_shap=(
                    values[
                        "include_shap"
                    ]
                ),
                include_global_importance=(
                    values[
                        "include_global_importance"
                    ]
                ),
            )
        )

        try:
            with st.spinner(
                "설비 값을 분석하고 있습니다..."
            ):
                with (
                    DashboardApiClient()
                    as api_client
                ):
                    result = (
                        api_client
                        .predict_failure(
                            payload
                        )
                    )

        except (
            DashboardApiClientError,
            ValueError,
        ) as exc:
            st.error(
                "분석 요청을 처리하지 못했습니다. "
                + str(
                    exc
                )
            )

            with st.expander(
                "상세 오류 정보",
                expanded=False,
            ):
                st.write(
                    str(
                        exc
                    )
                )

        else:
            st.session_state[
                "failure_prediction_result"
            ] = result

            st.session_state[
                "failure_prediction_explanation"
            ] = None

            st.success(
                "고장 위험 분석이 완료되었습니다."
            )

    stored_result = (
        st.session_state.get(
            "failure_prediction_result"
        )
    )

    if isinstance(
        stored_result,
        dict,
    ):
        render_failure_prediction_result(
            stored_result
        )


main()
