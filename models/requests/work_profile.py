from typing import Optional
from pydantic import BaseModel, Field

class WorkProfileRequest(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    role: str = Field(...,)
    target_role: str = Field(...,)
    my_company: Optional[str] = Field(None)