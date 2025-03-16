# main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Heart(BaseModel):
    rows: List[str]

@app.get("/heart", response_model=Heart)
async def create_heart():
    heart = [
        "     _,-._     ",
        "    / \_/ \    ",
        "    >-(_)-<    ",
        "    \_/ \_/    ",
        "      `-'      "
    ]
    return {"rows": heart}