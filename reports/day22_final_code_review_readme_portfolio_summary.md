# Day 22 - Final Code Review, Regression Test, README, Architecture, and Portfolio Summary

## 1. Day 22 목표

Day 22의 목표는 Day 1부터 Day 21까지 구현한 전체 프로젝트를 다시 검토하고, 최종 포트폴리오 기준으로 코드·테스트·문서를 정리하는 것입니다.

주요 목표:

1. 중복 코드와 불필요한 Helper 제거
2. Agent State의 Warning·Error 누적 정책 점검
3. OpenAI Intent Payload 방어 로직 강화
4. 비정상 수치(`NaN`, `Infinity`) 처리 강화
5. Evidence 정규화와 Direction 호환성 개선
6. 내부 예외 정보의 사용자 응답 노출 방지
7. 전체 회귀 테스트 실행
8. README 전면 개편
9. Architecture·실행 방법·한계·면접 설명 정리

---

## 2. 최종 프로젝트 핵심 구조

```text
User Request
    ↓
FastAPI
    ↓
Pydantic Validation
    ↓
LangGraph Workflow
    ↓
OpenAI Intent Classification
    ├── success → validated intent
    └── failure → rule-based fallback
    ↓
Conditional Routing
    ├── failure_prediction
    │       ↓
    │   PyTorch FailureMLP
    │       ↓
    │   Probability / Prediction / Risk Level
    │       ↓
    │   Rule-based Evidence
    │   SHAP Local Evidence
    │   Permutation Global Importance
    │       ↓
    │   Deterministic Answer Builder
    │
    ├── dataset_schema_query
    │       ↓
    │   Dataset Schema Service
    │
    └── unknown / error
            ↓
        Safe Fallback Answer
    ↓
Trace Finalization
    ↓
SQLite Execution History
    ↓
FastAPI Response
```

MCP는 기존 Dataset Schema Service를 재사용합니다.

```text
MCP Client
    ↓
FastMCP stdio Server
    ↓
get_dataset_schema
    ↓
Dataset Schema Service
    ↓
Structured Result
```

---

## 3. Day 22 코드 리뷰 및 개선 사항

### 3.1 `failure_agent_service.py`

#### 문제

Request를 AI4I Raw Sample 형식으로 변환하는 로직이 중복되어 있었습니다.

#### 개선

기존 Request Model의 메서드를 재사용하도록 변경했습니다.

```python
raw_sample = request.to_raw_sample()
```

#### 의미

- 변환 기준 단일화
- 중복 코드 제거
- Request Schema 변경 시 수정 지점 감소

---

### 3.2 `failure_agent_graph.py`

#### 문제

기존 State에 이미 Warning 또는 Error가 있을 때 새 결과로 덮어쓸 가능성을 점검했습니다.

#### 개선

기존 List를 유지한 상태에서 새 항목을 누적하도록 변경했습니다.

```python
state.setdefault("warnings", []).extend(
    prediction_result.get("warnings") or []
)

state.setdefault("errors", []).extend(
    prediction_result.get("errors") or []
)
```

#### 의미

LangGraph 여러 Node에서 생성된 Warning·Error가 최종 응답까지 보존됩니다.

---

### 3.3 사용하지 않는 `has_raw_sample()` 제거

#### 문제

`state.py`에 정의된 `has_raw_sample()` Helper가 실제 Workflow에서 사용되지 않았습니다.

#### 개선

다음 위치에서 제거했습니다.

- `src/agent/state.py`
- `src/agent/failure_agent_graph.py`
- 관련 Test

#### 의미

- 사용하지 않는 API 제거
- State Helper 역할 단순화
- 유지보수 대상 감소

---

### 3.4 `intent_classifier.py` Reason 정규화

#### 문제

OpenAI Payload의 `reason`이 `None`일 때 문자열 `"None"`으로 변환될 수 있었습니다.

#### 개선

다음 경우 기본 Reason을 사용하도록 정규화했습니다.

- `None`
- 문자열이 아닌 값
- 공백만 있는 문자열

기본 문구:

```text
분류 이유가 제공되지 않았습니다.
```

#### 의미

사용자 응답과 Trace에 의미 없는 `"None"` 문자열이 남지 않습니다.

---

### 3.5 Intent Confidence의 `NaN`·`Infinity` 방어

#### 문제

`float()` 변환에 성공하더라도 `NaN`, `Infinity`는 정상 Confidence로 사용할 수 없습니다.

#### 개선

유한한 수치인지 확인하고, 유효하지 않으면 `0.0`으로 정규화합니다.

#### 의미

Confidence 범위 검증 이전에 비정상 부동소수점 값이 Workflow에 유입되는 것을 방지합니다.

---

### 3.6 OpenAI 내부 예외 정보 비노출

#### 문제

OpenAI 호출 실패 시 내부 Exception 문자열이 API Warning에 포함될 가능성이 있었습니다.

예:

```text
mock_openai_error
```

#### 개선

사용자 응답에는 일반화된 안전 문구만 반환하도록 변경했습니다.

```text
OpenAI intent 분류에 실패하여 rule-based fallback을 사용했습니다.
```

#### 추가 Test

사용자 응답에 내부 Exception 문자열이 포함되지 않는지 검증했습니다.

#### 의미

- 내부 구현 정보 노출 방지
- Secret·환경 정보 노출 위험 감소
- 사용자에게 필요한 복구 상태만 전달

---

### 3.7 `answer_builder.py` 비정상 수치 방어

#### 문제

Prediction Result에 `NaN` 또는 `Infinity`가 포함되면 최종 Answer에 비정상 문자열이 노출될 수 있었습니다.

#### 개선

`_safe_float()`에서 유한한 수치인지 검증하도록 변경했습니다.

#### 의미

최종 Answer의 Probability·Threshold 표현 안정성이 향상되었습니다.

---

### 3.8 `evidence_builder.py` 숫자 정규화 강화

#### 개선

다음 Helper를 강화했습니다.

```text
_safe_float

_safe_int
```

처리 대상:

- 숫자 문자열
- 잘못된 문자열
- `None`
- `NaN`
- `Infinity`

#### 의미

외부 Tool Result 또는 모델 설명 결과의 형식이 불완전해도 Evidence Builder가 안전하게 처리할 수 있습니다.

---

### 3.9 Risk Severity 정규화

#### 문제

Risk Level 또는 Severity가 예상하지 않은 형식으로 들어올 수 있습니다.

#### 개선

지원하는 Severity 형식으로 정규화했습니다.

```text
LOW

MEDIUM

HIGH

UNKNOWN
```

#### 의미

Evidence별 Severity 표현이 일관됩니다.

---

### 3.10 SHAP Direction 호환성 개선

#### 문제

설명 Source에 따라 Direction 표현이 다를 수 있습니다.

지원이 필요한 예:

```text
positive

negative

increases_risk

decreases_risk
```

#### 개선

기존 표현과 새 표현을 모두 처리하도록 확장했습니다.

#### 의미

SHAP 또는 외부 설명 Source의 Direction Naming 차이로 인한 Evidence 누락을 줄였습니다.

---

### 3.11 Global Importance 누락값 정책

#### 정책

SHAP Evidence에 `global_importance`가 없으면 임의의 `0.0`을 넣지 않고 `None`을 유지합니다.

#### 이유

```text
정보가 없음
```

과

```text
중요도가 정확히 0
```

은 서로 다른 의미이기 때문입니다.

---

## 4. README 전면 개편

기존 README는 Day 1 단계에 머물러 있었고, 현재 구현과 다른 목표도 포함하고 있었습니다.

기존 문제:

- Day 1 중심 설명
- 현재 구현되지 않은 AutoEncoder·CNN 내용
- 현재 Architecture 미반영
- FastAPI Endpoint 미반영
- LangGraph·Trace·SQLite·MCP 미반영
- 실제 Test·Evaluation 결과 미반영
- 설치·실행 방법 부족
- 현재 한계와 안전 정책 부족

Day 22에서 다음 내용을 기준으로 README를 다시 작성했습니다.

1. 프로젝트 배경과 기존 프로젝트의 한계
2. PyTorch FailureMLP
3. Threshold 선택 정책
4. OpenAI Intent Classification
5. Rule-based Fallback
6. Multi-turn Context 정책
7. Evidence 구조
8. 전체 Architecture
9. LangGraph Workflow
10. Trace와 Observability
11. SQLite Execution History
12. MCP Server와 실제 stdio 검증
13. FastAPI Endpoint
14. Artifact Cache와 오류 정책
15. Agent Safety
16. Day 18 E2E Benchmark
17. Day 21 Agent Evaluation
18. 전체 Test 결과
19. 실제 프로젝트 구조
20. 설치·환경 변수·Dataset 준비
21. 주요 실행 명령
22. Day 1~22 개발 단계
23. 현재 한계
24. 향후 Streamlit Dashboard 확장
25. 포트폴리오·면접 설명
26. AI 개발 도구 활용 방식

---

## 5. 현재 API

### 5.1 Direct Prediction

```http
POST /agent/failure-prediction
```

역할:

- 정형 설비 입력 검증
- PyTorch Prediction
- Rule-based Evidence
- SHAP Local Evidence
- Permutation Global Importance
- 결정론적 Answer 생성

---

### 5.2 LangGraph Agent

```http
POST /agent/langgraph-query
```

역할:

- 자연어 Question
- OpenAI Intent Classification
- Rule-based Intent Fallback
- Multi-turn Context
- Conditional Routing
- Prediction 또는 Dataset Schema
- Trace 생성
- SQLite 저장

---

### 5.3 Execution History

```http
GET /agent/executions
```

역할:

최근 Agent 실행 Summary 조회

---

```http
GET /agent/executions/{trace_id}
```

역할:

특정 실행의 Evidence·Trace·Warning·Error·Raw Sample 상세 조회

---

## 6. 현재 MCP

Server:

```text
manufacturing-ai-quality-agent
```

Transport:

```text
stdio
```

Tool:

```text
get_dataset_schema
```

구현 방식:

> 공식 MCP Python SDK의 FastMCP를 사용해 기존 Dataset Schema Service를 MCP Tool로 노출하고, 실제 subprocess 기반 stdio 연결·initialize·Tool 목록 조회·Tool 호출을 검증했습니다.

---

## 7. 최종 회귀 테스트

실행 명령:

```powershell
pytest -v
```

최종 결과:

```text
232 passed
```

검증 범위:

- AI4I Schema
- 데이터 전처리
- PyTorch Model
- Class Imbalance 처리
- Threshold 선택
- Model Artifact
- Prediction
- Permutation Importance
- Rule-based Local Explanation
- SHAP
- Evidence Builder
- Answer Builder
- OpenAI Intent Validation
- Rule-based Fallback
- Multi-turn
- LangGraph Node·Route
- Trace
- FastAPI
- Error Handling
- Artifact Cache
- SQLite Persistence
- MCP
- Agent Evaluation
- Day 22 방어 로직

---

## 8. Git Diff 검사

실행:

```powershell
git diff --check
```

결과:

```text
Whitespace error 없음
```

Windows 환경의 다음 메시지는 줄바꿈 변환 안내이며 코드 오류가 아닙니다.

```text
LF will be replaced by CRLF
```

README에 존재했던 Trailing Whitespace는 제거했습니다.

---

## 9. Day 18 Real E2E Benchmark

총 실행:

```text
3 scenarios × 3 runs = 9 runs
```

결과:

| 항목 | 결과 |
|---|---:|
| Total Runs | 9 |
| Successful Runs | 9 |
| Failed Runs | 0 |
| Success Rate | 100% |
| Intent Match Rate | 100% |
| OpenAI Source Rate | 100% |
| Route Match Rate | 100% |
| Trace Status Match Rate | 100% |
| Fallback Rate | 0% |

Scenario 평균 실행 시간:

| Scenario | Mean |
|---|---:|
| Schema | 3276.33 ms |
| Prediction | 2425.81 ms |
| API Prediction | 2584.33 ms |

한계:

- 제한된 Local 실행
- Scenario별 3회
- 성공 Run만 Latency 통계에 포함
- FastAPI TestClient 기반
- 작은 표본의 P95
- 운영 SLA로 해석 불가

---

## 10. Day 21 Deterministic Agent Evaluation

총 결과:

```text
6 passed / 6 total

Pass Rate: 100%
```

Category:

| Category | Result |
|---|---:|
| Routing | 1 / 1 |
| Safety | 2 / 2 |
| Intent | 1 / 1 |
| Answer Consistency | 1 / 1 |
| Multi-turn | 1 / 1 |

Case:

1. `dataset_schema_success`
2. `prediction_missing_raw_sample`
3. `unsupported_question_fallback`
4. `high_risk_prediction_consistency`
5. `multi_turn_does_not_reuse_raw_sample`
6. `secret_request_safe_fallback`

---

## 11. Threshold 설명 정리

Threshold는 `0.70`을 직접 하드코딩해서 선택한 것이 아닙니다.

선택 정책:

```text
Threshold 후보 평가
        ↓
Recall >= 0.85 조건
        ↓
조건을 만족하는 후보 중
F1-score가 가장 높은 결과 선택
        ↓
선택 결과를 metadata.json에 저장
```

현재 저장된 Artifact의 Threshold:

```text
0.70
```

주의:

Day 4 초기 보고서에서는 당시 학습 결과 기준 `0.60`이 Recall 제약 조건을 만족하는 결과로 기록되었습니다.

현재 Artifact는 이후 학습 실행에서 생성된 결과이며 `0.70`이 저장되어 있습니다.

Train/Test Split은 `random_state=42`로 고정했지만 현재 PyTorch Seed는 완전히 고정하지 않았으므로, 재학습 시 모델 확률 분포와 선택 Threshold가 달라질 수 있습니다.

면접 설명:

> 제조 고장 예측에서는 고장 미탐을 줄이는 것이 중요하다고 판단해 Recall 0.85 이상이라는 제약을 먼저 적용했습니다. 그 조건을 만족하는 Threshold 후보 중 F1-score가 가장 높은 값을 자동으로 선택하도록 구현했고, 현재 저장된 Model Artifact에는 0.70이 기록되어 있습니다.

---

## 12. 현재 한계

1. AI4I 2020 공개 Dataset 기반이며 실제 제조 현장 데이터와 차이가 있습니다.

2. 현재 지원 Intent는 다음 세 가지입니다.

```text
failure_prediction

dataset_schema_query

unknown
```

3. 실시간 Sensor Streaming은 구현하지 않았습니다.

4. Online Learning·자동 재학습 Pipeline은 구현하지 않았습니다.

5. 현재 MCP Tool은 Dataset Schema 조회 중심입니다.

6. 현재 MCP Transport는 stdio입니다.

7. PyTorch Seed를 완전히 고정하지 않았으므로 재학습 결과가 달라질 수 있습니다.

8. SHAP과 Permutation Importance는 모델 해석 결과이며 인과관계를 의미하지 않습니다.

9. Day 18 Benchmark는 제한된 Local TestClient 결과이며 운영 SLA가 아닙니다.

10. SQLite는 단일 Application 학습용 구조입니다.

11. 현재 Question 원문과 Prediction Raw Sample을 SQLite에 저장합니다.

12. 운영 환경에서는 개인정보 마스킹, 접근 권한, 암호화, 보존 기간 정책이 필요합니다.

13. 인증·인가, Rate Limit, 운영 배포 보안은 현재 범위에 포함하지 않았습니다.

---

## 13. 향후 계획

Day 23:

- Streamlit Dashboard 요구사항 정리
- Dashboard Architecture
- FastAPI Client Layer
- 화면별 데이터 계약

Day 24:

- 설비 고장 위험 분석 화면
- Evidence 분석 화면
- LangGraph Agent Chat 화면
- Trace·Execution History 화면

Day 25:

- Dashboard Test
- Screenshot
- README Dashboard 보완
- 최종 Portfolio 정리

---

## 14. 포트폴리오 설명

### 프로젝트 한 문장

> AI4I 제조 데이터를 기반으로 PyTorch 설비 고장 예측 모델을 구현하고, OpenAI Intent Classification, LangGraph Routing, Evidence 기반 답변, Trace, SQLite Persistence, MCP를 FastAPI로 통합한 제조 AI Agent 프로젝트입니다.

### 문제 상황

> 기존 프로젝트는 규칙 기반 Intent와 단순 Tool 호출 중심이어서 모델 Prediction, 설명 가능성, Multi-turn, Trace, 실행 이력, 실제 MCP 연결, Agent 품질 평가가 부족했습니다.

### 해결

> PyTorch 설비 고장 모델을 중심으로 OpenAI JSON Intent Classification, LangGraph Conditional Routing, Rule-based Fallback, SHAP·Permutation Evidence, 결정론적 Answer Builder, Trace, SQLite Persistence, FastMCP stdio Server, Deterministic Evaluation을 단계적으로 연결했습니다.

### 의미

> LLM이 Prediction 수치나 Evidence를 임의로 생성하지 않도록 역할을 Intent Classification으로 제한하고, 모델 결과와 검증된 Evidence를 답변의 기준으로 사용했습니다. 또한 OpenAI 실패, 입력 누락, 비정상 수치, Secret 요청, 내부 Exception 노출과 같은 실패 상황을 Test로 검증했습니다.

---

## 15. AI 개발 도구 활용 설명

> AI를 개발 보조 도구로 활용해 코드 초안을 빠르게 구성했고, 이후 제가 직접 실행·검증·수정하면서 코드 구조와 처리 흐름을 제 것으로 만들었습니다. 단순히 코드를 생성하는 데서 끝내지 않고 Endpoint 응답, LangGraph Routing, PyTorch Prediction, Evidence, Trace, Fallback, SQLite, MCP, Test 결과와 한계까지 직접 확인하고 문서화했습니다.

---

## 16. Day 22 완료 기준

- [x] 중복 Request 변환 제거
- [x] Warning·Error 누적 정책 강화
- [x] 사용하지 않는 Helper 제거
- [x] Intent Reason 방어 로직 강화
- [x] Confidence `NaN`·`Infinity` 방어
- [x] OpenAI 내부 Exception 비노출
- [x] Answer Numeric 방어
- [x] Evidence Numeric 방어
- [x] Risk Severity 정규화
- [x] SHAP Direction 호환성 개선
- [x] 관련 Unit Test 추가
- [x] 전체 회귀 테스트 통과
- [x] `232 passed`
- [x] README 전면 개편
- [x] Architecture 정리
- [x] 설치·실행 방법 정리
- [x] Benchmark·Evaluation 결과 정리
- [x] 한계와 향후 계획 정리
- [x] 포트폴리오·면접 설명 정리

---

## 17. 최종 결론

Day 22에서는 새로운 기능을 크게 추가하기보다, Day 1부터 Day 21까지 구현한 전체 구조를 다시 검토하고 포트폴리오 수준으로 정리했습니다.

최종적으로 다음 흐름을 하나의 프로젝트 안에서 연결했습니다.

```text
AI4I Dataset
    ↓
PyTorch FailureMLP
    ↓
Prediction
    ↓
SHAP / Permutation / Rule Evidence
    ↓
OpenAI JSON Intent
    ↓
LangGraph Routing
    ↓
Deterministic Answer
    ↓
FastAPI
    ↓
Trace
    ↓
SQLite Persistence
    ↓
FastMCP stdio
    ↓
Deterministic Agent Evaluation
```

전체 회귀 테스트는 최종적으로 다음 결과를 확인했습니다.

```text
232 passed
```

Day 22 완료 후 다음 단계는 Streamlit 기반 제조 AI Dashboard 구현입니다.
