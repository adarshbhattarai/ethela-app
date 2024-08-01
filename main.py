from fastapi import FastAPI
import random

app = FastAPI()

@app.get("/")
def read_root():
    random_number = random.randint(1, 100)
    return {"Hello": random_number}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}