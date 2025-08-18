from pydantic import BaseModel, Field

class SkillRequest(BaseModel):
    current_role: str = Field(..., min_length=3, max_length=50)