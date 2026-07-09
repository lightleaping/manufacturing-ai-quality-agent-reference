# src/api/main.py

"""
FastAPI app entrypoint

Day 12 수정 내용
----------------
1. failure_agent_api router 등록
2. 입력 validation error를 사용자 친화적으로 반환

왜 validation handler가 필요한가?
--------------------------------
Pydantic 기본 validation error는 개발자에게는 유용하지만,
Swagger나 API 사용자에게는 다소 딱딱하게 보일 수 있다.

Day 12에서는 "어떤 필드가 왜 잘못됐는지"를 조금 더 명확하게 반환한다.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.failure_agent_api import router as failure_agent_router


app = FastAPI(
    title="Manufacturing AI Quality Agent Reference API",
    description="AI4I 기반 설비 고장 예측 + evidence 기반 Agent 응답 API",
    version="0.12.0",
)


app.include_router(failure_agent_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    FastAPI/Pydantic 입력 검증 실패를 사용자 친화적으로 반환한다.

    예:
        type에 "X"가 들어감
        torque가 문자열로 들어감
        required field가 빠짐

    기본 422 응답도 충분히 좋지만,
    포트폴리오에서는 이런 식으로 error response를 정리하면
    API 품질 개선 포인트로 설명하기 좋다.
    """
    formatted_errors: list[dict[str, str]] = []

    for error in exc.errors():
        # loc 예시:
        #   ("body", "torque")
        #   ("body", "type")
        loc = error.get("loc", [])

        # "body"는 API 내부 위치 정보라 사용자에게는 굳이 보여줄 필요가 적다.
        field_path = ".".join(
            str(part) for part in loc if part != "body"
        )

        formatted_errors.append(
            {
                "field": field_path,
                "message": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
        )

    return JSONResponse(
        status_code=422,
        content={
            "message": "입력값 형식이 올바르지 않습니다.",
            "errors": formatted_errors,
            "hint": "Swagger의 request schema를 확인하고, 숫자 필드와 type 값을 다시 확인하세요.",
        },
    )