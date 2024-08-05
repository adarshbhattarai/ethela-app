import random
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from app.models.prediction_request import PredictionRequest
from app.schemas.user import UserCreate, UserResponse
from app.services.user_service import UserService
from app.dependencies import get_user_service
from fastapi.responses import JSONResponse
from pydantic import ValidationError


class ThelaRoutes:
    def __init__(self):
        self.router = APIRouter()
        self.setup_routes()

    def setup_routes(self):
        @self.router.post("/", response_model=UserResponse)
        def create_user(user: UserCreate, service: UserService = Depends(get_user_service)):
            return service.create_user(user)

        @self.router.get("/restoreContent")
        def read_restore(question: Optional[str] = None):
            random_number = random.randint(1, 100)
            return {"Hello": random_number, "Question": question}

        @self.router.get("/")
        def read_ftt():
            random_number = random.randint(1, 100)
            return {"Hello": random_number}

        @self.router.get("/ques")
        def answer_dental_question(question: Optional[str] = None, service: UserService = Depends(get_user_service)):
            return service.answer_dental_question(question=question)


        @self.router.post("/predict")
        async def predict(request: PredictionRequest, service: UserService = Depends(get_user_service)):
            """ Prediction endpoint """
            try:
                instances = request.instances
                parameters = request.parameters
                predictions = get_user_service().predict_function(instances, parameters)
                return JSONResponse(content={"predictions": predictions})
            except ValidationError as e:
                raise HTTPException()



thela_routes = ThelaRoutes()
