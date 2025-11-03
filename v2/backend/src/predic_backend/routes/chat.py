from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from ..models.inference import inference_engine

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message."""
    role: str  # user, assistant, system
    content: str


class ChatRequest(BaseModel):
    """Chat request."""
    model_id: str
    messages: List[ChatMessage]
    max_tokens: int = 500
    temperature: float = 0.7
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response."""
    message: ChatMessage
    model_id: str


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat conversation."""
    try:
        # Convert messages to dict format
        messages = [msg.dict() for msg in request.messages]
        
        if request.stream:
            # Return streaming response
            async def generate():
                content = ""
                async for token in inference_engine.chat(
                    model_id=request.model_id,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=True,
                ):
                    content += token
                    yield token
                    
            return StreamingResponse(generate(), media_type="text/plain")
        else:
            # Return complete response
            response_text = await inference_engine.chat(
                model_id=request.model_id,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=False,
            )
            
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=response_text,
                ),
                model_id=request.model_id,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")