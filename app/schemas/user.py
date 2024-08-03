# app/schemas/user.py

from pydantic import BaseModel

class UserCreate(BaseModel):
    uid:str
    name: str
    question:str

class UserResponse(BaseModel):
    uid: str
    name: str
    answer: str
