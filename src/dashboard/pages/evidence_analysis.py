# src/dashboard/pages/evidence_analysis.py

"""
Failure Prediction Response의 Evidence를 시각화하는 Streamlit Page입니다.

화면 역할:
- Evidence Type별 개수 요약
- Evidence 비교용 Table
- Evidence별 독립 Card
- SHAP Contribution Chart
- Global Importance Chart
- 원본 Evidence·Metadata 확인

중요:
- Evidence 전용 API를 새로 호출하지 않습니다.
- SHAP 값을 다시 계산하지 않습니다.
- Global Importance를 다시 계산하지 않습니다.
- Backend가 반환한 Evidence 내용은 변경하지 않습니다.
- Dashboard에서는 표시 구조만 정리합니다.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import streamlit as st

from src.dashboard.session_state import (
    initialize_dashboard_session_state,
)
from src.dashboard.ui_helpers import (
    group_evidence_by_type,
)


KNOWN_EVIDENCE_TYPES = {
    "prediction_summary",
    "rule_based",
    "shap_local",
    "global_importance",
}


def display_text(
    value: Any,
    *,
    fallback: str = "-",
) -> str:
    """
    화면 표시용 문자열을 반환합니다.

    None 또는 빈 문자열은 fallback으로 표시합니다.

    Backend 원본 값 자체는 수정하지 않습니다.
    """

    if value is None:
        return fallback

    normalized = str(
        value,
    ).strip()

    if not normalized:
        return fallback

    return normalized


def display_decimal(
    value: Any,
    *,
    digits: int = 4,
) -> str:
    """
    숫자를 일정한 소수점 자릿수로 표시합니다.

    계산 결과를 변경하는 것이 아니라
    Presentation 형식만 통일합니다.
    """

    if value is None:
        return "-"

    try:
        numeric_value = float(
            value,
        )
    except (
        TypeError,
        ValueError,
    ):
        return display_text(
            value,
        )

    if not math.isfinite(
        numeric_value,
    ):
        return "-"

    return f"{numeric_value:.{digits}f}"


def display_severity(
    value: Any,
) -> str:
    """
    Severity를 Dashboard에서 읽기 쉽게 표시합니다.

    UNKNOWN은 Backend 오류가 아니라
    별도 Severity가 지정되지 않았다는 뜻이므로
    카드 상단에서는 '미지정'으로 표시합니다.

    원본 UNKNOWN 값은
    원본 Evidence 영역에서 그대로 확인할 수 있습니다.
    """

    normalized = display_text(
        value,
        fallback="UNKNOWN",
    ).upper()

    if normalized == "UNKNOWN":
        return "미지정"

    return normalized


def build_evidence_table(
    evidence_items: list[
        dict[str, Any]
    ],
) -> pd.DataFrame:
    """
    전체 Backend Evidence를
    원본 확인용 DataFrame으로 정리합니다.
    """

    rows = []

    for evidence in evidence_items:
        rows.append(
            {
                "Type": evidence.get(
                    "evidence_type",
                ),
                "Source": evidence.get(
                    "source",
                ),
                "Title": evidence.get(
                    "title",
                ),
                "Feature": evidence.get(
                    "feature",
                ),
                "Value": evidence.get(
                    "value",
                ),
                "Direction": evidence.get(
                    "direction",
                ),
                "Contribution": evidence.get(
                    "contribution",
                ),
                "Importance": evidence.get(
                    "importance",
                ),
                "Severity": evidence.get(
                    "severity",
                ),
                "Summary": evidence.get(
                    "summary",
                ),
            }
        )

    return pd.DataFrame(
        rows,
    )


def render_raw_evidence(
    evidence: dict[str, Any],
) -> None:
    """
    원본 Evidence 필드와 Metadata를
    접기 영역에서 확인할 수 있도록 표시합니다.
    """

    with st.expander(
        "원본 Evidence · Metadata",
        expanded=False,
    ):
        raw_fields = {
            key: evidence.get(
                key,
            )
            for key in [
                "evidence_id",
                "evidence_type",
                "source",
                "title",
                "summary",
                "feature",
                "value",
                "direction",
                "contribution",
                "importance",
                "severity",
            ]
        }

        st.json(
            raw_fields,
        )

        metadata = evidence.get(
            "metadata",
        )

        if isinstance(
            metadata,
            dict,
        ) and metadata:
            st.markdown(
                "##### Metadata"
            )

            st.json(
                metadata,
            )


def render_full_summary(
    evidence: dict[str, Any],
) -> None:
    """
    Summary를 DataFrame 안에서 잘리지 않도록
    Card 본문에 전체 문자열로 표시합니다.
    """

    st.markdown(
        "##### Summary"
    )

    summary = display_text(
        evidence.get(
            "summary",
        ),
        fallback=(
            "현재 Evidence에 Summary가 없습니다."
        ),
    )

    st.write(
        summary,
    )


def render_prediction_card(
    evidence: dict[str, Any],
) -> None:
    """
    Prediction Summary Evidence를
    하나의 전체 Card로 표시합니다.
    """

    with st.container(
        border=True,
    ):
        st.markdown(
            "###  모델 예측 요약"
        )

        original_title = display_text(
            evidence.get(
                "title",
            ),
            fallback="Prediction Summary",
        )

        st.caption(
            original_title,
        )

        render_full_summary(
            evidence,
        )

        severity = evidence.get(
            "severity",
        )

        if severity is not None:
            st.caption(
                "Severity · "
                f"{display_severity(severity)}"
            )

        render_raw_evidence(
            evidence,
        )


def render_rule_card(
    evidence: dict[str, Any],
) -> None:
    """
    Rule-based Evidence 하나를
    독립된 전체 Card로 표시합니다.
    """

    feature = display_text(
        evidence.get(
            "feature",
        ),
        fallback="Overall Rule",
    )

    with st.container(
        border=True,
    ):
        header_columns = st.columns(
            [
                4,
                1,
            ]
        )

        with header_columns[0]:
            st.markdown(
                f"###  {feature}"
            )

            st.caption(
                "Rule-based Evidence"
            )

        with header_columns[1]:
            st.markdown(
                "**Severity**"
            )

            st.write(
                display_severity(
                    evidence.get(
                        "severity",
                    )
                )
            )

        value_columns = st.columns(
            3,
        )

        with value_columns[0]:
            st.markdown(
                "**현재 값**"
            )

            st.write(
                display_text(
                    evidence.get(
                        "value",
                    )
                )
            )

        with value_columns[1]:
            st.markdown(
                "**방향**"
            )

            st.write(
                display_text(
                    evidence.get(
                        "direction",
                    )
                )
            )

        with value_columns[2]:
            st.markdown(
                "**Source**"
            )

            st.write(
                display_text(
                    evidence.get(
                        "source",
                    )
                )
            )

        st.divider()

        render_full_summary(
            evidence,
        )

        render_raw_evidence(
            evidence,
        )


def render_shap_card(
    evidence: dict[str, Any],
) -> None:
    """
    SHAP Local Evidence 하나를
    독립된 전체 Card로 표시합니다.
    """

    feature = display_text(
        evidence.get(
            "feature",
        ),
        fallback="Unknown Feature",
    )

    direction = display_text(
        evidence.get(
            "direction",
        ),
    ).lower()

    with st.container(
        border=True,
    ):
        header_columns = st.columns(
            [
                4,
                1.2,
            ]
        )

        with header_columns[0]:
            st.markdown(
                f"###  {feature}"
            )

            st.caption(
                "SHAP Local Evidence"
            )

        with header_columns[1]:
            st.metric(
                label="Contribution",
                value=display_decimal(
                    evidence.get(
                        "contribution",
                    )
                ),
            )

        value_columns = st.columns(
            3,
        )

        with value_columns[0]:
            st.markdown(
                "**입력값**"
            )

            st.write(
                display_text(
                    evidence.get(
                        "value",
                    )
                )
            )

        with value_columns[1]:
            st.markdown(
                "**Direction**"
            )

            st.write(
                display_text(
                    evidence.get(
                        "direction",
                    )
                )
            )

        with value_columns[2]:
            st.markdown(
                "**Source**"
            )

            st.write(
                display_text(
                    evidence.get(
                        "source",
                    )
                )
            )

        if direction in {
            "positive",
            "increases_risk",
        }:
            st.warning(
                "↑ 모델 출력의 고장 위험 방향을 "
                "높이는 Contribution입니다."
            )

        elif direction in {
            "negative",
            "decreases_risk",
        }:
            st.success(
                "↓ 모델 출력의 고장 위험 방향을 "
                "낮추는 Contribution입니다."
            )

        else:
            st.info(
                "현재 Evidence에서 "
                "Contribution 방향이 명확하지 않습니다."
            )

        render_full_summary(
            evidence,
        )

        render_raw_evidence(
            evidence,
        )


def render_global_card(
    evidence: dict[str, Any],
) -> None:
    """
    Global Importance Evidence 하나를
    독립된 전체 Card로 표시합니다.
    """

    feature = display_text(
        evidence.get(
            "feature",
        ),
        fallback="Unknown Feature",
    )

    with st.container(
        border=True,
    ):
        header_columns = st.columns(
            [
                4,
                1.2,
            ]
        )

        with header_columns[0]:
            st.markdown(
                f"###  {feature}"
            )

            st.caption(
                "Global Importance · "
                "전체 Test Set 기준"
            )

        with header_columns[1]:
            st.metric(
                label="Importance",
                value=display_decimal(
                    evidence.get(
                        "importance",
                    )
                ),
            )

        st.info(
            "이 값은 전체 데이터 기준 모델 민감도이며, "
            "현재 Sample의 직접적인 고장 원인을 뜻하지 않습니다."
        )

        render_full_summary(
            evidence,
        )

        render_raw_evidence(
            evidence,
        )


def render_evidence_details(
    evidence_items: list[
        dict[str, Any]
    ],
) -> None:
    """
    일반 Evidence 목록을
    전체 Card 형태로 표시합니다.

    향후 새로운 Evidence Type이 추가되어도
    Summary가 잘리지 않도록 구성합니다.
    """

    for index, evidence in enumerate(
        evidence_items,
        start=1,
    ):
        with st.container(
            border=True,
        ):
            st.markdown(
                "###  "
                + display_text(
                    evidence.get(
                        "title",
                    ),
                    fallback=(
                        f"Evidence {index}"
                    ),
                )
            )

            st.caption(
                display_text(
                    evidence.get(
                        "evidence_type",
                    ),
                )
            )

            render_full_summary(
                evidence,
            )

            render_raw_evidence(
                evidence,
            )


def render_prediction_summary(
    evidence_items: list[
        dict[str, Any]
    ],
) -> None:
    """
    Prediction Summary Evidence를 표시합니다.
    """

    st.subheader(
        "Prediction Summary",
    )

    st.caption(
        "모델 Prediction과 운영 Threshold에 대한 "
        "Backend 요약입니다."
    )

    if not evidence_items:
        st.info(
            "Prediction Summary Evidence가 없습니다."
        )
        return

    for evidence in evidence_items:
        render_prediction_card(
            evidence,
        )


def render_rule_based(
    evidence_items: list[
        dict[str, Any]
    ],
) -> None:
    """
    Rule-based Evidence를
    비교용 Table과 개별 Card로 표시합니다.
    """

    st.subheader(
        "Rule-based Evidence",
    )

    st.caption(
        "표는 Feature 간 빠른 비교용이고, "
        "각 Card에서는 Summary 전체 내용을 확인합니다."
    )

    if not evidence_items:
        st.info(
            "Rule-based Evidence가 없습니다."
        )
        return

    comparison_rows = [
        {
            "Feature": item.get(
                "feature",
            ),
            "Value": item.get(
                "value",
            ),
            "Direction": item.get(
                "direction",
            ),
            "Severity": (
                display_severity(
                    item.get(
                        "severity",
                    )
                )
            ),
        }
        for item in evidence_items
    ]

    st.dataframe(
        pd.DataFrame(
            comparison_rows,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "### 상세 Evidence"
    )

    for evidence in evidence_items:
        render_rule_card(
            evidence,
        )


def render_shap(
    evidence_items: list[
        dict[str, Any]
    ],
) -> None:
    """
    Backend SHAP Contribution을
    비교용 Table·Chart·개별 Card로 표시합니다.
    """

    st.subheader(
        "SHAP Local Evidence",
    )

    st.caption(
        "양수 Contribution은 모델 출력의 고장 위험 방향을 높이고, "
        "음수 Contribution은 낮추는 방향입니다. "
        "실제 물리적 고장 원인을 단정하지 않습니다."
    )

    if not evidence_items:
        st.info(
            "SHAP Local Evidence가 없습니다."
        )
        return

    comparison_rows = [
        {
            "Feature": item.get(
                "feature",
            ),
            "Value": item.get(
                "value",
            ),
            "Direction": item.get(
                "direction",
            ),
            "Contribution": item.get(
                "contribution",
            ),
        }
        for item in evidence_items
    ]

    comparison_table = pd.DataFrame(
        comparison_rows,
    )

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    chart = (
        comparison_table[
            [
                "Feature",
                "Contribution",
            ]
        ]
        .dropna()
        .set_index(
            "Feature",
        )
    )

    if not chart.empty:
        with st.container(
            border=True,
        ):
            st.markdown(
                "#### SHAP Contribution 비교"
            )

            st.bar_chart(
                chart,
            )

    st.markdown(
        "### 상세 Evidence"
    )

    for evidence in evidence_items:
        render_shap_card(
            evidence,
        )


def render_global_importance(
    evidence_items: list[
        dict[str, Any]
    ],
) -> None:
    """
    Backend Global Importance를
    비교용 Table·Chart·개별 Card로 표시합니다.
    """

    st.subheader(
        "Global Importance",
    )

    st.caption(
        "현재 FastAPI Response에 포함된 "
        f"{len(evidence_items)}개 Global Importance Evidence를 "
        "그대로 표시합니다. "
        "Dashboard에서 상위 항목을 다시 선택하지 않습니다."
    )

    if not evidence_items:
        st.info(
            "Global Importance Evidence가 없습니다."
        )
        return

    comparison_rows = [
        {
            "Feature": item.get(
                "feature",
            ),
            "Importance": item.get(
                "importance",
            ),
        }
        for item in evidence_items
    ]

    comparison_table = pd.DataFrame(
        comparison_rows,
    )

    st.dataframe(
        comparison_table,
        use_container_width=True,
        hide_index=True,
    )

    chart = (
        comparison_table[
            [
                "Feature",
                "Importance",
            ]
        ]
        .dropna()
        .set_index(
            "Feature",
        )
    )

    if not chart.empty:
        with st.container(
            border=True,
        ):
            st.markdown(
                "#### Global Importance 비교"
            )

            st.bar_chart(
                chart,
            )

    st.markdown(
        "### 상세 Evidence"
    )

    for evidence in evidence_items:
        render_global_card(
            evidence,
        )


def render_additional_evidence(
    grouped: dict[
        str,
        list[
            dict[str, Any]
        ],
    ],
) -> None:
    """
    새로운 Evidence Type도
    손실 없이 Card 형태로 표시합니다.
    """

    unknown_types = [
        evidence_type
        for evidence_type in grouped
        if evidence_type
        not in KNOWN_EVIDENCE_TYPES
    ]

    if not unknown_types:
        return

    st.subheader(
        "Additional Evidence",
    )

    for evidence_type in unknown_types:
        st.markdown(
            f"### {evidence_type}"
        )

        render_evidence_details(
            grouped[
                evidence_type
            ]
        )


def main() -> None:
    """
    Evidence 분석 Page를 렌더링합니다.
    """

    initialize_dashboard_session_state(
        st.session_state,
    )

    st.title(
        "Evidence 분석",
    )

    st.write(
        "마지막 설비 고장 위험 분석에서 "
        "Backend가 반환한 Evidence를 유형별로 확인합니다."
    )

    st.caption(
        "SHAP·Global Importance를 다시 계산하지 않습니다. "
        "현재 Prediction Response의 Evidence만 시각화합니다."
    )

    prediction_result = (
        st.session_state.get(
            "failure_prediction_result",
        )
    )

    if not isinstance(
        prediction_result,
        dict,
    ):
        st.info(
            "먼저 '설비 고장 위험 분석' 화면에서 "
            "Prediction을 실행해주세요."
        )
        return

    evidence_items = (
        prediction_result.get(
            "evidence",
        )
    )

    if not isinstance(
        evidence_items,
        list,
    ) or not evidence_items:
        st.info(
            "현재 Prediction Response에 "
            "표시할 Evidence가 없습니다."
        )
        return

    grouped = group_evidence_by_type(
        evidence_items,
    )

    with st.container(
        border=True,
    ):
        st.markdown(
            "#### Evidence 구성"
        )

        summary_columns = st.columns(
            5,
        )

        summary_columns[0].metric(
            "전체",
            len(
                evidence_items,
            ),
        )

        summary_columns[1].metric(
            "Prediction",
            len(
                grouped.get(
                    "prediction_summary",
                    [],
                )
            ),
        )

        summary_columns[2].metric(
            "Rule",
            len(
                grouped.get(
                    "rule_based",
                    [],
                )
            ),
        )

        summary_columns[3].metric(
            "SHAP",
            len(
                grouped.get(
                    "shap_local",
                    [],
                )
            ),
        )

        summary_columns[4].metric(
            "Global",
            len(
                grouped.get(
                    "global_importance",
                    [],
                )
            ),
        )

        st.caption(
            "각 탭의 표는 빠른 비교용이며, "
            "Summary 전체 내용은 아래 개별 Card에서 확인합니다."
        )

    (
        summary_tab,
        rule_tab,
        shap_tab,
        global_tab,
    ) = st.tabs(
        [
            " Prediction Summary",
            " Rule-based",
            " SHAP Local",
            " Global Importance",
        ]
    )

    with summary_tab:
        render_prediction_summary(
            grouped.get(
                "prediction_summary",
                [],
            )
        )

    with rule_tab:
        render_rule_based(
            grouped.get(
                "rule_based",
                [],
            )
        )

    with shap_tab:
        render_shap(
            grouped.get(
                "shap_local",
                [],
            )
        )

    with global_tab:
        render_global_importance(
            grouped.get(
                "global_importance",
                [],
            )
        )

    render_additional_evidence(
        grouped,
    )

    with st.expander(
        "전체 원본 Evidence Table 보기",
        expanded=False,
    ):
        st.caption(
            "Backend Response의 전체 필드를 "
            "원본 확인 목적으로 표시합니다."
        )

        st.dataframe(
            build_evidence_table(
                evidence_items,
            ),
            use_container_width=True,
            hide_index=True,
        )


main()
