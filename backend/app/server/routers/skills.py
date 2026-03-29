"""Skills API — server-layer implementation using DeerFlowClient."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.gateway.path_utils import resolve_thread_virtual_path

from ..deps import get_client

router = APIRouter(prefix="/api")


class SkillUpdateRequest(BaseModel):
    enabled: bool


class SkillInstallRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID where the .skill file is located")
    path: str = Field(..., description="Virtual path to the .skill file")


@router.get("/skills")
async def list_skills():
    return get_client().list_skills()


@router.get("/skills/{skill_name}")
async def get_skill(skill_name: str):
    result = get_client().get_skill(skill_name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")
    return result


@router.put("/skills/{skill_name}")
async def update_skill(skill_name: str, request: SkillUpdateRequest):
    try:
        return get_client().update_skill(skill_name, enabled=request.enabled)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")


@router.post("/skills/install")
async def install_skill(request: SkillInstallRequest):
    from deerflow.skills.installer import SkillAlreadyExistsError

    try:
        path = resolve_thread_virtual_path(request.thread_id, request.path)
        return get_client().install_skill(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SkillAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
