# 1. Python 3.10 슬림 이미지 사용 (python:3.13이 없으므로 최신 안정 버전인 python:3.10-slim 사용)
FROM python:3.10-slim

# 2. 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev  # 예시로 PostgreSQL을 사용하는 경우 필요한 패키지, 필요 없으면 삭제 \
    && rm -rf /var/lib/apt/lists/*

# 3. requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 환경 변수 파일 추가 (필요한 경우)
# COPY .env .env  # 필요에 따라 주석을 해제하거나 사용

# 5. 애플리케이션 코드 복사
COPY . .

# 6. FastAPI 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
