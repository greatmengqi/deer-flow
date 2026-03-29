"""Models API — server-layer implementation using DeerFlowClient."""

from fastapi import APIRouter, HTTPException

from ..deps import get_client

router = APIRouter(prefix="/api")


@router.get("/models")
async def list_models():
    return get_client().list_models()


@router.get("/models/{model_name}")
async def get_model(model_name: str):
    result = get_client().get_model(model_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return result
