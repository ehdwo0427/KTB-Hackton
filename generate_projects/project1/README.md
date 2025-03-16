# Lunch Recommendation API
간단한 점심 메뉴 추천 API

## 폴더 구조
```
lunch_api/
├── main.py
└── requirements.txt
```

## 파일 별 설명
- `main.py`: FastAPI를 이용한 점심 메뉴 CRUD API 구현 파일입니다.  SQLite in-memory database를 사용하여 간편하게 데이터를 관리합니다.  Pydantic을 사용하여 데이터 모델을 정의하고 입력값을 검증합니다.
- `requirements.txt`: 프로젝트에 필요한 라이브러리와 버전 정보를 명시하는 파일입니다.


## 배포 작업 순서 설명
1.  터미널을 열고 프로젝트 디렉토리로 이동합니다.
2.  `pip install -r requirements.txt` 명령어를 실행하여 필요한 패키지를 설치합니다.
3.  `uvicorn main:app --reload` 명령어를 실행하여 API 서버를 시작합니다.
4.  Postman이나 curl과 같은 도구를 사용하여 API를 테스트할 수 있습니다.  `/lunch/` 엔드포인트에 POST 요청을 보내 새로운 점심 메뉴를 추가하고, GET 요청을 보내 점심 메뉴 목록을 가져올 수 있습니다.  `/lunch/{lunch_id}` 엔드포인트를 사용하여 특정 점심 메뉴를 조회, 수정 또는 삭제할 수 있습니다.