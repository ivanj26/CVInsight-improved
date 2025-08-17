from typing import List
from pydantic import BaseModel, Field

class CollectionGenerative(BaseModel):
    recommendations: list[str]

class CollectionGenerativeResponse(BaseModel):
    data: CollectionGenerative