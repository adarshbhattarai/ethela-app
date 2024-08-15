import random
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from app.models.prediction_request import PredictionRequest
from app.models.question_request import QuestionRequest
from app.schemas.user import UserCreate, UserResponse
from app.services.user_service import UserService
from app.dependencies import get_user_service
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from pydantic import BaseModel
class TextResponse(BaseModel):
    text: str

class ThelaRoutes:
    def __init__(self):
        self.router = APIRouter()
        self.setup_routes()
        self.chat_history=[]


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

        @self.router.post("/ques")
        def answer_dental_question(
                request: QuestionRequest,
                service: UserService = Depends(get_user_service)
        ):
            response_text= service.answer_dental_question(question=request.question, history=self.chat_history)
            self.chat_history.extend([
                HumanMessage(content=request.question),
                AIMessage(content=response_text),
            ])
            return TextResponse(text=response_text)



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
