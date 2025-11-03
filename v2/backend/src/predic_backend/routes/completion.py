"""Code completion routes."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from ..models.inference import inference_engine

router = APIRouter()


class CompletionRequest(BaseModel):
    """Code completion request."""
    model_id: str
    prompt: str
    max_tokens: int = 150
    temperature: float = 0.2
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


class CompletionResponse(BaseModel):
    """Code completion response."""
    completion: str
    model_id: str


@router.post("/", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest):
    """Generate code completion."""
    try:
        if request.stream:
            # Return streaming response
            async def generate():
                async for token in inference_engine.generate_completion(
                    model_id=request.model_id,
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop_sequences=request.stop_sequences,
                    stream=True,
                ):
                    yield token
                    
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # Return complete response
            completion = await inference_engine.generate_completion(
                model_id=request.model_id,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop_sequences=request.stop_sequences,
                stream=False,
            )
            
            return CompletionResponse(
                completion=completion,
                model_id=request.model_id,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")