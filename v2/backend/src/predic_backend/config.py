import torch
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    host: str = "127.0.0.1"
    port: int = 8000
    
    # Model settings
    models_dir: Path = Path.home() / ".predic" / "models"
    cache_dir: Path = Path.home() / ".predic" / "cache"
    
    # Model loading settings
    @property
    def device(self) -> str:
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    load_in_8bit: bool = False
    load_in_4bit: bool = True
    
    # Available models configuration - Updated with better models
    available_models: dict = {
        # Low resource models (1-3B parameters)
        "deepseek-coder-1.3b": {
            "repo_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
            "task": "text-generation",
            "context_length": 16384,
            "size_category": "small",
            "description": "Excellent 1.3B model for code completion and generation"
        },
        "stablecode-3b": {
            "repo_id": "stabilityai/stable-code-3b",
            "task": "text-generation", 
            "context_length": 16384,
            "size_category": "small",
            "description": "StabilityAI's 3B model trained on diverse programming languages"
        },
        "codegemma-2b": {
            "repo_id": "google/codegemma-2b",
            "task": "text-generation",
            "context_length": 8192,
            "size_category": "small",
            "description": "Google's 2B model optimized for code generation"
        },
        "phi-2": {
            "repo_id": "microsoft/phi-2",
            "task": "text-generation",
            "context_length": 2048,
            "size_category": "small",
            "description": "Microsoft's 2.7B model with strong coding capabilities"
        },
        
        # Medium resource models (6-7B parameters)
        "deepseek-coder-6.7b": {
            "repo_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
            "task": "text-generation",
            "context_length": 16384,
            "size_category": "medium",
            "description": "Powerful 6.7B model with excellent code understanding"
        },
        "codellama-7b": {
            "repo_id": "codellama/CodeLlama-7b-Instruct-hf",
            "task": "text-generation",
            "context_length": 16384,
            "size_category": "medium",
            "description": "Meta's CodeLlama 7B for code generation and completion"
        },
        "mistral-7b-instruct": {
            "repo_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text-generation",
            "context_length": 32768,
            "size_category": "medium",
            "description": "Mistral 7B with good coding capabilities and long context"
        },
        
        # High resource models (13B+ parameters)
        "codellama-13b": {
            "repo_id": "codellama/CodeLlama-13b-Instruct-hf",
            "task": "text-generation",
            "context_length": 16384,
            "size_category": "large",
            "description": "Meta's larger CodeLlama model for complex coding tasks"
        },
        "deepseek-coder-33b": {
            "repo_id": "deepseek-ai/deepseek-coder-33b-instruct",
            "task": "text-generation",
            "context_length": 16384,
            "size_category": "large",
            "description": "DeepSeek's flagship 33B model - best for complex code"
        },
        "phind-codellama-34b": {
            "repo_id": "Phind/Phind-CodeLlama-34B-v2",
            "task": "text-generation",
            "context_length": 16384,
            "size_category": "large",
            "description": "Phind's fine-tuned 34B model optimized for programming"
        },
        "wizardcoder-15b": {
            "repo_id": "WizardLM/WizardCoder-15B-V1.0",
            "task": "text-generation",
            "context_length": 8192,
            "size_category": "large",
            "description": "WizardLM's 15B model with strong code generation"
        },
    }
    
    class Config:
        env_prefix = "PREDIC_"


settings = Settings()