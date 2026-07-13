# src/dashboard/__init__.py

"""
dashboard 패키지입니다.

이 패키지는 Streamlit Dashboard를 지원하는 Presentation Layer를 담당합니다.

주요 역할:
- Dashboard에서 사용할 FastAPI Client를 제공합니다.
- FastAPI Base URL과 HTTP Timeout 설정을 관리합니다.
- API 응답을 Streamlit 화면에서 사용할 수 있도록 전달합니다.

중요:
- PyTorch 모델을 직접 로드하거나 실행하지 않습니다.
- LangGraph workflow를 직접 실행하지 않습니다.
- SQLite 실행 이력을 직접 조회하지 않습니다.
- 고장 위험도, Evidence, 최종 Answer를 다시 계산하지 않습니다.
- 기존 FastAPI endpoint를 호출하여 검증된 Backend 결과를 재사용합니다.

예상 요청 흐름:

    Streamlit UI
        ↓
    Dashboard FastAPI Client
        ↓
    Existing FastAPI Endpoint
        ↓
    Service / LangGraph / Model / Persistence
"""
