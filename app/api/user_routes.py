# app/api/user_routes.py
import random
from typing import List,Optional
from fastapi import APIRouter,Depends, HTTPException
from app.schemas.user import UserCreate, UserResponse

from app.services.user_service import UserService
from app.dependencies import get_user_service

router = APIRouter()
@router.post("/", response_model=UserResponse)
def create_user(user: UserCreate,
                service: UserService = Depends(get_user_service)):
    return service.create_user(user)

@router.get("/")
def read_ftt():  # Accepting query parameter 'question'
    random_number = random.randint(1, 100)
    return {"Hello": random_number}

@router.get("/ques")
def read_root(question: Optional[str] = None):  # Accepting query parameter 'question'
    random_number = random.randint(1, 100)
    return {"Hello": random_number, "Question": question}
