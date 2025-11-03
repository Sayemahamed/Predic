import asyncio
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import torch
from huggingface_hub import snapshot_download, HfApi, hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

from ..config import settings


class ModelManager:
    """Manages model downloading, loading, and inference."""
    
    # Approximate model sizes in GB (for display purposes)
    MODEL_SIZES = {
        "deepseek-coder-1.3b": "2.5 GB",
        "stablecode-3b": "5.5 GB",
        "codegemma-2b": "5.0 GB",
        "phi-2": "5.0 GB",
        "deepseek-coder-6.7b": "13 GB",
        "codellama-7b": "13 GB",
        "mistral-7b-instruct": "14 GB",
        "codellama-13b": "26 GB",
        "deepseek-coder-33b": "65 GB",
        "phind-codellama-34b": "68 GB",
        "wizardcoder-15b": "30 GB",
    }
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.download_progress: Dict[str, Dict[str, Any]] = {}
        self.api = HfApi()
        
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a model."""
        config = settings.available_models.get(model_id)
        if not config:
            raise ValueError(f"Unknown model: {model_id}")
            
        model_path = settings.models_dir / model_id
        
        # Check if model is downloaded
        status = "available"
        if model_path.exists() and (model_path / "config.json").exists():
            status = "ready"
        elif model_id in self.download_progress:
            status = "downloading"
            
        # Get model size
        size = self.MODEL_SIZES.get(model_id, "Unknown")
            
        return {
            "id": model_id,
            "name": config["repo_id"],
            "size": size,
            "status": status,
            "progress": self.download_progress.get(model_id, {}).get("progress", 0),
            "description": config.get("description", ""),
            "size_category": config.get("size_category", "medium"),
        }
        
    async def download_model(self, model_id: str) -> None:
        """Download a model from Hugging Face."""
        config = settings.available_models.get(model_id)
        if not config:
            raise ValueError(f"Unknown model: {model_id}")
            
        model_path = settings.models_dir / model_id
        
        # Check if already downloaded
        if model_path.exists() and (model_path / "config.json").exists():
            return
            
        # Initialize progress tracking
        self.download_progress[model_id] = {
            "status": "downloading",
            "progress": 0,
            "started_at": datetime.now().isoformat(),
        }
        
        try:
            # Create model directory
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Download in a separate thread with progress callback
            def progress_callback(current, total):
                if total > 0:
                    progress = (current / total) * 100
                    self.download_progress[model_id]["progress"] = progress
                    self.download_progress[model_id]["current"] = current
                    self.download_progress[model_id]["total"] = total
            
            await asyncio.to_thread(
                snapshot_download,
                repo_id=config["repo_id"],
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True,
                cache_dir=str(settings.cache_dir),
            )
            
            # Update status
            self.download_progress[model_id]["status"] = "completed"
            self.download_progress[model_id]["progress"] = 100
            
        except Exception as e:
            self.download_progress[model_id]["status"] = "error"
            self.download_progress[model_id]["error"] = str(e)
            # Clean up partial download
            if model_path.exists():
                shutil.rmtree(model_path)
            raise
            
    async def load_model(self, model_id: str) -> None:
        """Load a model into memory."""
        if model_id in self.loaded_models:
            return  # Already loaded
            
        config = settings.available_models.get(model_id)
        if not config:
            raise ValueError(f"Unknown model: {model_id}")
            
        model_path = settings.models_dir / model_id
        if not model_path.exists():
            raise ValueError(f"Model not downloaded: {model_id}")
            
        # Configure quantization based on model size
        quantization_config = None
        size_category = config.get("size_category", "medium")
        
        if size_category == "large" or (size_category == "medium" and settings.device == "cpu"):
            # Use 4-bit quantization for large models or medium models on CPU
            if settings.load_in_4bit and settings.device != "cpu":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
        elif settings.load_in_8bit and settings.device != "cpu":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
        # Load tokenizer
        tokenizer = await asyncio.to_thread(
            AutoTokenizer.from_pretrained,
            str(model_path),
            trust_remote_code=True,
            padding_side="left",
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with appropriate settings
        device_map = "auto" if settings.device == "cuda" else settings.device
        torch_dtype = torch.float16 if settings.device in ["cuda", "mps"] else torch.float32
        
        model = await asyncio.to_thread(
            AutoModelForCausalLM.from_pretrained,
            str(model_path),
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        self.loaded_models[model_id] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
        }
        
    async def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    def get_loaded_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get a loaded model."""
        return self.loaded_models.get(model_id)
        
    def _get_device(self) -> str:
        """Get the appropriate device for model loading."""
        return settings.device
        
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable string."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


# Global model manager instance
model_manager = ModelManager()