from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List

from ..config import settings
from ..models.manager import model_manager

router = APIRouter()


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    size: str
    status: str  # available, downloading, ready, error
    progress: float = 0.0


class ModelList(BaseModel):
    """List of available models."""
    models: List[ModelInfo]


@router.get("/available", response_model=ModelList)
async def get_available_models():
    """Get list of available models."""
    models = []
    
    # Debug: Print available models
    print(f"Available models from settings: {list(settings.available_models.keys())}")
    
    for model_id in settings.available_models:
        try:
            info = await model_manager.get_model_info(model_id)
            models.append(ModelInfo(**info))
            print(f"Added model: {model_id}")
        except Exception as e:
            print(f"Error getting model info for {model_id}: {e}")
            # Still add the model with basic info even if there's an error
            config = settings.available_models[model_id]
            models.append(ModelInfo(
                id=model_id,
                name=config["repo_id"],
                size=model_manager.MODEL_SIZES.get(model_id, "Unknown"),
                status="available",
                progress=0.0
            ))
    
    print(f"Returning {len(models)} models")
    return ModelList(models=models)


@router.post("/download/{model_id}")
async def download_model(model_id: str, background_tasks: BackgroundTasks):
    """Start downloading a model."""
    if model_id not in settings.available_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Check if already downloading
    if model_id in model_manager.download_progress:
        status = model_manager.download_progress[model_id].get("status")
        if status == "downloading":
            return {"status": "already_downloading", "model_id": model_id}
    
    # Start download in background
    background_tasks.add_task(model_manager.download_model, model_id)
    
    return {"status": "started", "model_id": model_id}


@router.get("/status/{model_id}")
async def get_model_status(model_id: str):
    """Get download/load status of a model."""
    if model_id not in settings.available_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    info = await model_manager.get_model_info(model_id)
    
    # Add download progress if available
    if model_id in model_manager.download_progress:
        info.update(model_manager.download_progress[model_id])
    
    # Check if model is loaded
    if model_manager.get_loaded_model(model_id):
        info["loaded"] = True
    else:
        info["loaded"] = False
    
    return info


@router.post("/load/{model_id}")
async def load_model(model_id: str):
    """Load a model into memory."""
    if model_id not in settings.available_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        await model_manager.load_model(model_id)
        return {"status": "loaded", "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload/{model_id}")
async def unload_model(model_id: str):
    """Unload a model from memory."""
    if model_id not in settings.available_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    await model_manager.unload_model(model_id)
    return {"status": "unloaded", "model_id": model_id}


@router.get("/loaded")
async def get_loaded_models():
    """Get list of currently loaded models."""
    loaded = list(model_manager.loaded_models.keys())
    return {"loaded_models": loaded}