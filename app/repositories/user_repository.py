# app/repositories/user_repository.py

from typing import List
from app.schemas.user import UserCreate, UserResponse

class UserRepository:
    def __init__(self):
        self.users = []  # Mock in-memory database

    def add_user(self, user: UserResponse) -> None:
        self.users.append(user)

    def get_all_users(self) -> List[UserResponse]:
        return self.users
