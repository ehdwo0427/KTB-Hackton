# Heart API
간단한 하트 문자열을 반환하는 FastAPI 애플리케이션입니다.

## 폴더 구조
```
heart-api/
├── main.py
└── requirements.txt
```

## 파일 별 설명
- `main.py`: FastAPI 애플리케이션을 정의하는 메인 파일입니다.  `/heart` 엔드포인트에 접근하면 하트 모양 문자열을 JSON 형태로 반환합니다. Pydantic을 사용하여 응답 데이터의 형식을 검증합니다.
- `requirements.txt`: 프로젝트에 필요한 패키지와 버전을 명시하는 파일입니다.


## 배포 작업 순서 설명
1. `pip install -r requirements.txt` 명령어를 사용하여 필요한 패키지를 설치합니다.
2. `uvicorn main:app --reload` 명령어를 사용하여 FastAPI 애플리케이션을 실행합니다.
3. 웹 브라우저 또는 curl을 이용하여 `http://127.0.0.1:8000/docs` 에 접속하여 API 문서를 확인하고, `http://127.0.0.1:8000/heart` 에 접속하여 하트 문자열을 확인할 수 있습니다.