import logging
import uvicorn

from fastapi.middleware.cors import CORSMiddleware  # CORS 미들웨어 임포트
from config import settings  # 환경 변수 로드
from fastapi import FastAPI, HTTPException
from config import settings
from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi.responses import StreamingResponse
import asyncio
from prompt_engineering import CodeGenerator


app = FastAPI(
    title="Python Code Generator API (LangChain)",
    description="LangChain + Gemini 기반 Python 코드 생성 API",
    version="2.0.0"
)
    
origins = [
    settings.BACK_URL,      # "http://frontend:8000"
    settings.FRONT_URL,      # "http://frontend:8000"
]
# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 허용된 출처 목록
    allow_credentials=True,         # 인증 정보(쿠키, Authorization 헤더 등) 허용
    allow_methods=["*"],            # 모든 HTTP 메서드 허용 (GET, POST, PUT 등)
    allow_headers=["*"],            # 모든 헤더 허용
)

# 요청 데이터 모델 정의
class Room(BaseModel):
    id: int
    name: Optional[str] = None  # None 허용하도록 수정
    created_at: Optional[str] = None

class MessageHistory(BaseModel):  # 메시지 히스토리용 새로운 모델
    content: str
    sender: str
    created_at: str

class Message(BaseModel):
    content: str
    role: str

class RequestData(BaseModel):
    room: Room
    message_history: List[MessageHistory]  # 새로운 모델로 변경
    new_message: Message
    type: Optional[str] = "makecode"

# 기본 엔드포인트: 루트 경로
@app.get("/")
async def root():
    return {"message": "Welcome to the RAG System API"}

# 스트리밍 응답을 위한 제너레이터 함수
async def stream_response(request_data: RequestData):
    query = request_data.new_message.content
    request_type = request_data.type
    room_name = request_data.room.name or "Unnamed Room"  # name이 None일 경우 기본값 설정

    # # type에 따라 초기 메시지 설정
    # if request_type == "makecode":
    #     stages = ["처리중", "코드제작중", "파일화 완료"]
    # elif request_type == "moreinfo":
    #     stages = ["처리중", "정보수집중", "완료"]
    # else:
    #     yield f"data: Invalid type: {request_type}. Use 'makecode' or 'moreinfo'\n\n"
    #     return

    # # 5초 간격으로 스트리밍 메시지 전송
    # for stage in stages:
    #     yield f"data: {stage} - Room: {room_name}, Query: {query}\n\n"
    #     await asyncio.sleep(5)

    # 모델 Return값 전송
    key, value = await CodeGenerator.run_code_generation(request_data)
    logging.info(key, value)
    print(key, value)
    yield {key: value}
    await asyncio.sleep(1)


# RAG 시스템 요청 처리 엔드포인트 (스트리밍 형태)
@app.post("/generate-code/")
async def process_test_request(data: RequestData):
    return StreamingResponse(
        stream_response(data),
        media_type="text/event-stream"
    )


# @app.post("/generate-code")
# async def generate_code_api(data: RequestData):
#     """LangChain 기반 비동기 코드 생성 API"""
#     try:
#         key, value = await CodeGenerator.run_code_generation(data)
#         return {key: value}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))