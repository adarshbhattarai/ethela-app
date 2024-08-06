from typing import List, Optional
from pydantic import BaseModel

class ConversationMessage(BaseModel):
    sender: str
    message: str

class QuestionRequest(BaseModel):
    question: Optional[str] = None
    history: Optional[List[ConversationMessage]] = None
