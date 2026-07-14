"""
Streamlit Dashboard의 화면별 Page 모듈을 관리하는 패키지입니다.

각 Page 모듈은 다음 역할에 집중합니다.

- Streamlit Widget을 통해 사용자 입력을 받습니다.
- DashboardApiClient를 사용하여 기존 FastAPI를 호출합니다.
- FastAPI Response를 Metric, Table, Chart, Message 형태로 표시합니다.
- Loading, Success, Warning, Error 상태를 사용자에게 보여줍니다.

중요:
- PyTorch 모델을 직접 로드하거나 실행하지 않습니다.
- LangGraph workflow를 직접 실행하지 않습니다.
- SQLite를 직접 조회하지 않습니다.
- Prediction, Risk Level, Evidence, Answer를 다시 계산하지 않습니다.
- 검증된 기존 Backend 결과를 Presentation Layer에서 시각화합니다.
"""
