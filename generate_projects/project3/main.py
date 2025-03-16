# main.py

import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "곰인형 만들기 프로젝트입니다."}