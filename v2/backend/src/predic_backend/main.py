import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import models, completion, chat


# Configure logging to ignore health checks
class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return not (
            record.getMessage().find("/health") != -1
            and record.getMessage().find("200 OK") != -1
        )


# Apply filter to uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifecycle events."""
    # Startup
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    yield
    # Shutdown
    # Cleanup if needed


app = FastAPI(
    title="Predic Backend",
    description="Local LLM server for offline coding assistance",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["vscode-webview://*", "http://localhost:*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(completion.router, prefix="/api/completion", tags=["completion"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}