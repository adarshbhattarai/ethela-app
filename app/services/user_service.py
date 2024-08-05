# app/services/user_service.py

from typing import List
from app.schemas.user import UserCreate, UserResponse
from ..start import db


class UserService:
    def __init__(self):
        self.users = []  # Mock in-memory database

    def create_user(self, user: UserCreate) -> UserResponse:
        new_user = UserResponse(uid="131TetAns",
                                name="TestAI",
                                answer="RandomAnswer")
        self.users.append(new_user)
        return new_user

    def get_users(self) -> List[UserResponse]:
        return self.users

    def predict_function(self,instances, parameters):
        """ Prediction logic goes here
            Right now i'm returning both without a model
        """
        print("Executed ##### ")

        return [instances, parameters]

    def answer_dental_question(self, question=""):
        print("DB",db.index)

        print("Wow")








print("Executed ### ")
# Create a singleton instance
user_service_instance = UserService()