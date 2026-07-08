# src/api/main.py

from fastapi import FastAPI

from src.api.failure_agent_api import router as failure_agent_router


def create_app() -> FastAPI:
    """
    FastAPI app 객체를 생성합니다.

    create_app 함수를 따로 두는 이유:
    - 테스트에서 app을 쉽게 가져올 수 있습니다.
    - 나중에 router가 늘어나도 main 구조를 깔끔하게 유지할 수 있습니다.
    - 실제 서버 실행과 테스트 실행을 분리하기 좋습니다.
    """

    app = FastAPI(
        title="Manufacturing AI Quality Agent Reference API",
        description=(
            "AI4I 기반 설비 고장 예측, rule-based evidence, "
            "SHAP local explanation, Agent answer를 제공하는 학습용 API입니다."
        ),
        version="0.1.0",
    )

    app.include_router(failure_agent_router)

    return app


app = create_app()