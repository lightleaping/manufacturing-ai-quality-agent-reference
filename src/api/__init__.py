# src/api/__init__.py

"""
api 패키지입니다.

이 패키지는 FastAPI endpoint와 API request/response schema를 담당합니다.

중요:
- 모델 학습 로직은 src/training에 둡니다.
- 단일 sample 추론 로직은 src/inference에 둡니다.
- SHAP 설명 로직은 src/interpretability에 둡니다.
- evidence 생성과 answer 생성 로직은 src/agent에 둡니다.
- API 계층은 위 기능들을 호출해서 HTTP 응답으로 감싸는 역할만 합니다.
"""