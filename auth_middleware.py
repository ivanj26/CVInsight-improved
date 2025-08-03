import os

from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

# Load API key from .env file if available
load_dotenv()

# Get the exact x-api-key
server_key = os.environ.get("X_API_KEY")

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("x-api-key")
        if api_key != server_key or api_key is None:
            return JSONResponse(status_code=401, content={"message": "Invalid API key"})
        return await call_next(request)