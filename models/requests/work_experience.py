from typing import Optional
from pydantic import BaseModel, Field

class WorkExperienceRequest(BaseModel):
    current_role: str = Field(..., min_length=3, max_length=50)
    current_job_type: Optional[str] = Field(None)
    description: Optional[str] = Field(None)