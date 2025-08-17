import os

from fastapi import FastAPI
from dotenv import load_dotenv

from api.v1 import router as api_v1_router
from middlewares.auth_middleware import APIKeyMiddleware

# Load API key from .env file if available
load_dotenv()
app_name = os.environ.get("APP_NAME")

# Initialize the FastAPI app
app = FastAPI(app_name=app_name)
app.add_middleware(APIKeyMiddleware)

# Register routers
app.include_router(api_v1_router, prefix="/api/v1")

@app.get("/")
def health_check():
    return { "message": "The CVParser service is healthy, running on port 9001!" }
