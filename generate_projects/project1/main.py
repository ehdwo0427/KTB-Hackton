# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3

app = FastAPI()

class LunchItem(BaseModel):
    name: str
    ingredients: str
    instructions: str


# In-memory database for simplicity.  For production, use a persistent database.
conn = sqlite3.connect(':memory:')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS lunches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        ingredients TEXT NOT NULL,
        instructions TEXT NOT NULL
    )
''')
conn.commit()


@app.post("/lunch/", response_model=LunchItem)
async def create_lunch(lunch: LunchItem):
    cursor.execute("INSERT INTO lunches (name, ingredients, instructions) VALUES (?, ?, ?)",
                   (lunch.name, lunch.ingredients, lunch.instructions))
    conn.commit()
    return lunch


@app.get("/lunch/")
async def read_lunches():
    cursor.execute("SELECT * FROM lunches")
    lunches = cursor.fetchall()
    lunch_list = []
    for lunch in lunches:
        lunch_item = LunchItem(name=lunch[1], ingredients=lunch[2], instructions=lunch[3])
        lunch_list.append(lunch_item)
    return lunch_list


@app.get("/lunch/{lunch_id}")
async def read_lunch(lunch_id: int):
    cursor.execute("SELECT * FROM lunches WHERE id = ?", (lunch_id,))
    lunch = cursor.fetchone()
    if lunch:
        return LunchItem(name=lunch[1], ingredients=lunch[2], instructions=lunch[3])
    else:
        return {"message": "Lunch not found"}


@app.put("/lunch/{lunch_id}", response_model=LunchItem)
async def update_lunch(lunch_id: int, lunch: LunchItem):
    cursor.execute("UPDATE lunches SET name = ?, ingredients = ?, instructions = ? WHERE id = ?",
                   (lunch.name, lunch.ingredients, lunch.instructions, lunch_id))
    conn.commit()
    cursor.execute("SELECT * FROM lunches WHERE id = ?", (lunch_id,))
    updated_lunch = cursor.fetchone()
    return LunchItem(name=updated_lunch[1], ingredients=updated_lunch[2], instructions=updated_lunch[3])


@app.delete("/lunch/{lunch_id}")
async def delete_lunch(lunch_id: int):
    cursor.execute("DELETE FROM lunches WHERE id = ?", (lunch_id,))
    conn.commit()
    return {"message": "Lunch deleted"}