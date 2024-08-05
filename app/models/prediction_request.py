from pydantic import BaseModel
from typing import List, Optional, Any
class PredictionRequest(BaseModel):
    """ Pydantic Model to enforce types"""
    instances: List[Any]
    parameters: Optional[dict] = None