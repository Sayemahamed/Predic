import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
os.environ["HF_HOME"] = str(Path.home() / ".predic" / "cache")  # Set Hugging Face cache

import uvicorn
from predic_backend.config import settings
from predic_backend.logging_config import LOGGING_CONFIG

if __name__ == "__main__":
    print(f"Starting Predic backend server on {settings.host}:{settings.port}")
    print(f"Device: {settings.device}")
    print(f"Models directory: {settings.models_dir}")
    
    uvicorn.run(
        "predic_backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        reload_dirs=[str(Path(__file__).parent / "src")],
        log_config=LOGGING_CONFIG,
    )