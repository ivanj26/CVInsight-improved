from typing import Dict, List
from pydantic import BaseModel, Field

class DictionaryGenerativeResponse(BaseModel):
    data: Dict[str, List[str]]