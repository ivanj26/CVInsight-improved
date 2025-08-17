from typing import Optional
from pydantic import BaseModel, Field

class EducationRequest(BaseModel):
    current_major: str = Field(..., min_length=3, max_length=50)