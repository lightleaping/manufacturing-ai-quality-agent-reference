# Day 24 - Streamlit Manufacturing AI Dashboard and Beginner-First UX Summary

## 1. Day 24 목표

Day 24의 목표는 기존 Backend 비즈니스 로직을 다시 구현하지 않고, FastAPI 기반 제조 AI 기능을 Streamlit Dashboard에서 사용할 수 있도록 연결하는 것이었다.

핵심 목표는 다음과 같다.

1. Streamlit Dashboard는 화면 입력과 결과 표시를 담당한다.
2. Dashboard는 PyTorch Model, LangGraph workflow, SQLite를 직접 실행하지 않는다.
3. 예측·Agent·실행 이력 기능은 기존 FastAPI API를 통해 호출한다.
4. Backend가 반환한 Prediction, Probability, Threshold, Risk Level, Evidence, Answer를 Dashboard에서 다시 계산하지 않는다.
5. AI와 제조 배경지식이 없는 사용자도 결과를 이해할 수 있도록 화면 순서와 설명을 구성한다.
6. 기술 정보는 필요한 사용자만 확인할 수 있도록 접힌 상세 영역으로 분리한다.
7. 기존 Backend 기능과 회귀 테스트를 유지한다.

---

## 2. 최종 Architecture

```text
사용자
  ↓
Streamlit Dashboard
  ↓
DashboardApiClient
  ↓ HTTP
FastAPI
  ├─ LangGraph Agent
  ├─ OpenAI Intent Classification
  ├─ PyTorch Failure Prediction
  ├─ Evidence Builder
  ├─ Answer Builder
  ├─ Trace / Observability
  └─ SQLite Execution Persistence
  ↓
FastAPI Response
  ↓
Streamlit Dashboard
```

역할은 다음과 같이 분리했다.

### Streamlit Dashboard

- 사용자 입력 수집
- FastAPI 요청 생성
- Backend Response 표시
- 초보자용 설명 구성
- Session State 관리
- 화면용 대화 기록 관리

### DashboardApiClient

- FastAPI Endpoint 호출
- HTTP 오류를 Dashboard 전용 예외로 변환
- Response JSON 기본 구조 확인
- Dashboard와 Backend 사이의 통신 경계 제공

### FastAPI Backend

- 입력 검증
- PyTorch 예측 실행
- LangGraph Agent 실행
- OpenAI 호출
- Evidence 생성
- Answer 생성
- Trace 생성
- SQLite 실행 이력 저장·조회

---

## 3. Dashboard가 Model·LangGraph·SQLite를 직접 실행하지 않는 이유

Dashboard가 Backend 기능을 직접 실행하지 않도록 설계한 이유는 역할 분리와 결과 일관성 때문이다.

Dashboard와 FastAPI가 각각 Model·LangGraph·SQLite를 직접 실행하면 같은 예측 로직이 두 곳에 중복될 수 있고, Threshold·Risk Level·Evidence 정책이 서로 달라질 수 있다. Model이나 전처리 방식이 바뀔 때 두 영역을 함께 수정해야 하며, UI 테스트도 실제 Model·OpenAI·DB 실행에 의존하게 된다.

현재 구조에서는 FastAPI만 실제 실행을 담당한다.

```text
Streamlit
→ 입력과 결과 표시

DashboardApiClient
→ FastAPI 통신

FastAPI
→ 실행 진입점과 비즈니스 로직

LangGraph
→ Agent 흐름

PyTorch
→ 고장 위험 예측

SQLite
→ 실행 이력 저장
```

이를 통해 Prediction, Probability, Threshold, Risk Level, Evidence, Answer의 결정 지점을 Backend 한 곳으로 유지했다.

---

## 4. 구현한 Dashboard Page

### 4.1 설비 고장 위험 분석

기존 FastAPI 고장 위험 예측 기능을 Dashboard에서 사용할 수 있도록 연결했다.

Backend Endpoint:

```text
POST /agent/failure-prediction
```

입력 항목:

- 공기 온도
- 공정 온도
- 회전 속도
- 토크
- 공구 마모 시간
- 제품 유형
- 이번 입력값의 영향 분석 포함 여부
- 전체 데이터 중요도 포함 여부

결과 화면은 다음 순서로 구성했다.

1. 한눈에 보는 결론
2. 지금 할 일
3. 핵심 결과
4. 현장 확인 순서
5. 주요 판단 근거
6. 숫자를 예로 들어 이해하기
7. 판단 근거 세 종류 설명
8. 선택형 OpenAI 쉬운 설명
9. 기술 상세와 Model 한계

핵심 결과는 다음 순서로 표시한다.

```text
1. 한눈에 보는 판정
2. AI 모델 위험 점수
3. 위험 판정 기준선
4. 위험 단계
```

기본 예시 결과:

```text
한눈에 보는 판정
고장 위험 있음

AI 모델 위험 점수
약 99.30%

위험 판정 기준선
70.00%

위험 단계
높음 (HIGH)
```

AI 모델 위험 점수는 실제 고장 발생 확률을 확정하는 값이 아니라, 현재 입력값을 Model이 얼마나 위험한 방향으로 판단했는지 보여주는 예측 참고값임을 명시했다.

또한 다음과 같은 비유를 제공했다.

```text
100점 척도로 비유하면,
위험 판정 기준선은 70점이고
현재 AI 모델 위험 점수는 99.3점이다.

현재 모델 점수가 기준선을 넘었기 때문에
Backend는 고장 위험 있음으로 판정했다.

단, 실제 고장 발생 확률이 정확히 99.3%라는 뜻은 아니다.
```

### 4.2 판단 근거 자세히 보기

Failure Prediction Response에 포함된 Evidence를 유형별로 확인할 수 있도록 구성했다.

Evidence 유형:

- Prediction Summary
- Rule-based Evidence
- SHAP Local Evidence
- Global Importance

초보자용 화면에서는 다음처럼 구분한다.

#### 사람이 미리 정한 점검 기준

현재 입력값이 사람이 정한 제조 점검 기준에 해당했는지 보여준다.

#### 이번 입력값이 AI 판단에 준 영향

각 입력값이 이번 AI 위험 판단을 높이거나 낮춘 방향을 보여준다.

#### 전체 데이터에서 중요했던 입력값

전체 참고 데이터에서 AI가 자주 중요하게 사용한 입력값을 보여준다.

각 Evidence는 실제 고장의 물리적 원인을 확정하는 자료가 아니라 Model 판단을 이해하기 위한 참고자료임을 명시했다.

### 4.3 AI 질의 응답

기존 LangGraph Agent API를 자연어 Chat 화면에서 사용할 수 있도록 연결했다.

Backend Endpoint:

```text
POST /agent/langgraph-query
```

지원 질문 유형:

- 설비 입력값 기반 고장 위험 예측
- AI4I 데이터셋 feature, target, schema 설명

중요 정책:

```text
Chat History
→ 질문 문맥 이해용

이전 Raw Sample
→ 다음 요청에 자동 재사용하지 않음
```

고장 위험 예측 질문에서는 사용자가 매 요청마다 다음 옵션을 선택해야 한다.

```text
이번 질문에 설비 입력값 함께 보내기
```

질문 문장에 `Torque 62, Tool wear 220이면 고장 위험이 높아?`라고 적는 것만으로는 Structured Raw Sample이 자동 생성되지 않는다. 고장 예측을 실행하려면 현재 요청에 구조화된 설비 입력값을 함께 전달해야 한다.

화면에서는 다음 두 기록을 분리했다.

```text
chat_messages
→ 사용자 화면에 표시할 쉬운 대화 기록

agent_context_messages
→ Backend에 전달할 문맥 기록
```

Raw Sample 누락 시 기존 긴 fallback 원문 대신 다음 행동을 안내한다.

```text
이번 질문에는 설비 입력값이 함께 전달되지 않아
고장 위험을 계산하지 못했습니다.

고장 위험을 예측하려면
'이번 질문에 설비 입력값 함께 보내기'를 선택한 뒤
현재 설비 값을 확인하고 질문을 다시 보내세요.
```

Intent, Confidence, Trace, Metadata는 기본 Chat 화면에서 숨기고 `기술 상세 보기` 안으로 이동했다.

### 4.4 실행 기록과 처리 과정

SQLite에 저장된 Agent 실행 이력을 FastAPI 조회 API를 통해 표시한다.

Backend Endpoint:

```text
GET /agent/executions

GET /agent/executions/{trace_id}
```

Dashboard는 SQLite 파일을 직접 열거나 SQL Query를 직접 실행하지 않는다.

---

## 5. OpenAI 쉬운 운영 해설

기존 Prediction과 Evidence를 초보자가 이해하기 쉬운 한국어로 다시 설명하는 선택 기능을 추가했다.

Backend Endpoint:

```text
POST /agent/failure-prediction/explanation
```

추가 파일:

```text
src/agent/operational_explainer.py

src/api/failure_explanation_service.py
```

추가 Schema:

```text
FailurePredictionExplanationRequest

FailurePredictionExplanationResponse
```

설명 Response 주요 필드:

```text
summary

key_signals

recommended_checks

caution

source

model

error
```

중요 정책:

1. Prediction을 다시 실행하지 않는다.
2. Probability를 다시 계산하지 않는다.
3. Threshold를 변경하지 않는다.
4. Risk Level을 변경하지 않는다.
5. Evidence를 변경하지 않는다.
6. 실제 고장이 확정됐다고 말하지 않는다.
7. 실제 물리적 고장 원인을 확정하지 않는다.
8. 사용자가 버튼을 누를 때만 OpenAI API를 호출한다.
9. OpenAI 실패가 기존 Prediction 실패처럼 보이지 않도록 분리한다.

구조:

```text
확정된 Prediction Result
  ↓
확정된 Evidence
  ↓
OpenAI 쉬운 설명
  ↓
Dashboard 표시

Prediction·Evidence
→ 변경 없음
```

---

## 6. OpenAI Prompt 개선

초기 Prompt에 깨진 한글 문자열이 포함된 문제를 수정했다.

문제 예:

```text
[??? ?? ?? ??]

[?? ???? AI ??? ? ??]
```

개선 내용:

- Prompt 내부 깨진 문자열 제거
- OpenAI가 분류 라벨을 직접 생성하지 않도록 변경
- 분류 제목은 Dashboard가 고정된 한글로 표시
- OpenAI는 설명 문장만 생성
- 이모티콘과 장식 기호 생성 금지
- 기술 용어 최소화
- 짧은 한국어 문장 사용
- 결론을 기술 정보보다 먼저 설명
- 100점 척도 비유 허용
- Model 점수와 실제 현실 고장 확률 구분
- 점검 행동을 구체적으로 설명
- 실제 원인 확정 금지

---

## 7. 초보자 중심 UX 개선

사용자는 AI, 제조, Probability, Threshold, SHAP, Feature Importance를 모른다고 가정했다.

화면 구성 원칙:

```text
결론
  ↓
지금 할 일
  ↓
핵심 숫자
  ↓
현장 확인 순서
  ↓
왜 이런 결과인지
  ↓
쉬운 예시
  ↓
상세 근거
  ↓
기술 정보
```

적용한 개선:

- 이모티콘 제거
- 한글 시스템 Font 우선 적용
- 긴 제목과 문장 줄바꿈
- 4열 Metric 축소
- 결과 Metric 2×2 배치
- 긴 AI 설명 세로형 Card 구성
- 기술 상세 기본 접기
- 입력 단위 예시 제공
- 전문 용어를 쉬운 한국어로 변경
- Backend 원본 용어는 기술 상세에서 유지

Font Stack:

```text
-apple-system
BlinkMacSystemFont
Segoe UI
Noto Sans KR
Malgun Gothic
sans-serif
```

---

## 8. Session State

추가 파일:

```text
src/dashboard/session_state.py
```

주요 상태:

```text
failure_prediction_result

failure_prediction_explanation

chat_messages

last_agent_result

execution_history

selected_execution_trace_id

selected_execution_detail
```

Agent Chat에서는 Backend 문맥용 상태를 추가로 사용한다.

```text
agent_context_messages
```

중요:

```text
이전 Raw Sample
→ Session State에서 다음 예측 요청에 자동 첨부하지 않음
```

---

## 9. Dashboard API Client

수정 파일:

```text
src/dashboard/api_client.py
```

지원 기능:

```text
Failure Prediction

Failure Prediction Explanation

LangGraph Agent Query

Agent Execution List

Agent Execution Detail
```

주요 Endpoint:

```text
POST /agent/failure-prediction

POST /agent/failure-prediction/explanation

POST /agent/langgraph-query

GET /agent/executions

GET /agent/executions/{trace_id}
```

---

## 10. 주요 추가·수정 파일

### Backend

```text
src/agent/operational_explainer.py
src/api/failure_explanation_service.py
src/api/failure_agent_api.py
src/api/schemas.py
```

### Dashboard

```text
src/dashboard/__init__.py
src/dashboard/api_client.py
src/dashboard/config.py
src/dashboard/app.py
src/dashboard/styles.py
src/dashboard/session_state.py
src/dashboard/ui_helpers.py
src/dashboard/pages/__init__.py
src/dashboard/pages/failure_prediction.py
src/dashboard/pages/evidence_analysis.py
src/dashboard/pages/agent_chat.py
src/dashboard/pages/execution_history.py
```

### Dependency

```text
requirements.txt
```

### Test

```text
tests/test_operational_explainer.py
tests/test_api_failure_prediction_explanation.py
tests/test_dashboard_api_client.py
tests/test_dashboard_ui_helpers.py
tests/test_dashboard_session_state.py
tests/test_dashboard_failure_prediction_page.py
tests/test_dashboard_failure_prediction_explanation.py
tests/test_dashboard_evidence_analysis_page.py
tests/test_dashboard_agent_chat_page.py
tests/test_dashboard_agent_chat_beginner_layout.py
tests/test_dashboard_execution_history_page.py
tests/test_dashboard_app.py
tests/test_dashboard_styles.py
tests/test_dashboard_text_style.py
tests/test_day24_beginner_explanation_text.py
tests/test_dashboard_beginner_layout.py
```

---

## 11. 테스트 결과

최종 전체 회귀 테스트:

```text
307 passed
```

검증 범위:

- 기존 PyTorch 고장 위험 예측
- FastAPI
- LangGraph Agent
- OpenAI Intent Classification
- Rule-based Evidence
- SHAP Local Evidence
- Global Importance
- Trace / Observability
- SQLite Persistence
- MCP
- Dashboard API Client
- Streamlit Page
- Session State
- OpenAI 쉬운 설명
- Beginner-first Layout
- Agent Chat 문맥 분리
- 이전 Raw Sample 자동 재사용 금지
- 깨진 문자열 방지
- 이모티콘 제거
- Font·Typography

---

## 12. 로컬 실행 방법

### Terminal 1 - FastAPI

```powershell
uvicorn src.api.main:app --reload
```

기본 주소:

```text
http://127.0.0.1:8000
```

### Terminal 2 - Streamlit

```powershell
python -m streamlit run src\dashboard\app.py
```

기본 주소:

```text
http://localhost:8501
```

Windows 환경에서는 `streamlit run ...`보다 Python Module 실행을 사용한다.

---

## 13. 실제 동작 확인

기본 입력:

```text
Air temperature
303.0 K

Process temperature
312.5 K

Rotational speed
1380 rpm

Torque
62 Nm

Tool wear
220 min

Type
L
```

확인 결과:

```text
Prediction
1

Probability
약 0.993

Threshold
0.7

Risk Level
HIGH
```

Dashboard 표시:

```text
고장 위험 있음

AI 모델 위험 점수
약 99.30%

위험 판정 기준선
70.00%

위험 단계
높음 (HIGH)
```

OpenAI 쉬운 설명:

- 버튼 클릭 시에만 API 호출
- gpt-4o-mini 응답 확인
- 기존 Prediction 유지
- 기존 Evidence 유지
- 설명 실패 시 Prediction 결과 유지

Agent Chat:

```text
Raw Sample 미포함
→ 고장 위험 예측 실행 안 함
→ 현재 요청에 설비 입력값을 포함하라는 쉬운 안내

Raw Sample 포함
→ FastAPI
→ LangGraph Agent
→ Failure Prediction
→ Answer·Evidence·Trace
```

---

## 14. 설계상 유지한 안전 원칙

### 이전 Raw Sample 자동 재사용 금지

Chat History는 질문 문맥 이해용이다.

```text
이전 대화의 설비 조건
→ 다음 예측 요청에 자동 재사용하지 않음
```

새 고장 예측에는 현재 요청의 Raw Sample을 다시 포함해야 한다.

### OpenAI는 설명 계층

```text
Prediction
→ 고정

Evidence
→ 고정

OpenAI
→ 설명만 생성
```

### Dashboard는 Backend 결과를 다시 계산하지 않음

다음 값을 Dashboard에서 재계산하지 않는다.

```text
Prediction
Probability
Threshold
Risk Level
Evidence
Answer
```

---

## 15. 현재 한계

1. AI4I 기반 학습용 예시 Model이다.
2. 실제 설비 진단을 대체하지 않는다.
3. Rule-based Evidence는 사람이 정한 참고 기준이다.
4. SHAP Local Evidence는 Model Output 방향 설명이다.
5. SHAP 값은 실제 물리적 고장 원인을 확정하지 않는다.
6. Global Importance는 전체 Test Set 기준 참고 중요도다.
7. Global Importance는 개별 Sample의 직접 원인이 아니다.
8. 실제 운영 환경에서는 인증, 권한, 배포, Monitoring 보완이 필요하다.
9. OpenAI 쉬운 설명은 외부 API 상태와 환경 변수에 의존한다.
10. Streamlit Session State는 현재 Browser Session의 화면 상태 관리용이다.

---

## 16. Day 24 완료 기준

Day 24는 다음 기준을 모두 만족했다.

- Streamlit Dashboard 구성
- FastAPI Client 연결
- Failure Prediction Page
- Evidence Analysis Page
- Agent Chat Page
- Execution History Page
- Session State 관리
- OpenAI 쉬운 설명 API
- Beginner-first UX
- 긴 기술 정보 접기
- 이모티콘 제거
- 한글 깨짐 방지
- 이전 Raw Sample 자동 재사용 금지
- Backend 비즈니스 로직 중복 없음
- 전체 회귀 테스트 통과

최종 결과:

```text
307 passed
```

---

## 17. Day 25 연결

Day 25에서는 다음 작업으로 이어갈 수 있다.

1. 전체 Architecture 최종 정리
2. README 최종 업데이트
3. Dashboard 실행 Screenshot 정리
4. Portfolio 설명 문장 작성
5. 면접 예상 질문·답변 작성
6. 프로젝트 전체 최종 검토
7. Repository 최종 정리

Day 24에서 구현한 Dashboard는 기존 Backend 구조를 유지하면서, 사용자에게 AI 예측·판단 근거·Agent 실행 이력을 이해하기 쉽게 제공하는 표현 계층으로 완성했다.
