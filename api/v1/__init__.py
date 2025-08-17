from fastapi import APIRouter
from . import ai

router = APIRouter()
router.include_router(ai.router)
