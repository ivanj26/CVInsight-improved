from pydantic import BaseModel
from typing import List, Optional

class AIMessageResponse(BaseModel):
    """ Model for store the output content from assistant """
    role: str # user, assistant, system
    content: str

class AIChoiceResponse(BaseModel):
    """ Model for store the choices from assistant """
    message: AIMessageResponse
    index: int
    finish_reason: Optional[str] = None

class AITokenUsageResponse(BaseModel):
    """ Model for store the token usage """
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class ContentGenerationResponse(BaseModel):
    """ Model for store the AI-generative response from LLM API """
    id: str
    model: str
    choices: List[AIChoiceResponse]
    created: Optional[int] = None
    usage: Optional[AITokenUsageResponse] = None