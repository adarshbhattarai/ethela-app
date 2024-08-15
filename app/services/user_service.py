# app/services/user_service.py

from typing import List
from app.schemas.user import UserCreate, UserResponse
from ..start import db, chatbot


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

    def answer_dental_question(self, question="", history=None):
        print("DB",db.index)
        patient_details = {
            "name": "Bhattarai A.",
            "age": 30,
            "symptoms": "None",
            "medical_condition": "Nothing surgery once in past",
            "allergy_history": "Sometimes",
            "smoker_status": "yes",
            "current_dental_history": "None"
        }

        answer = chatbot.answer_question(patient_details=patient_details,query=question,conversation_history=history)

        return answer








print("Executed ### ")
# Create a singleton instance
user_service_instance = UserService()