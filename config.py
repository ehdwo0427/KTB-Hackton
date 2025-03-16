from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

class Settings:
    BACK_URL = os.getenv("BACK_URL")
    FRONT_URL = os.getenv("FRONT_URL")
settings = Settings()