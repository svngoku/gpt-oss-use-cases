"""Configuration management utilities."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelConfig(BaseModel):
    """Configuration for model loading."""
    
    model_id: str = Field(..., description="HuggingFace model ID or local path")
    device: str = Field("auto", description="Device to use for model")
    dtype: str = Field("auto", description="Data type for model weights")
    quantization: Optional[Dict[str, Any]] = Field(None, description="Quantization config")
    cache_dir: Optional[str] = Field(None, description="Cache directory for models")


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    learning_rate: float = Field(5e-5, description="Learning rate")
    batch_size: int = Field(4, description="Training batch size")
    gradient_accumulation_steps: int = Field(1, description="Gradient accumulation steps")
    num_epochs: int = Field(3, description="Number of training epochs")
    warmup_steps: int = Field(500, description="Number of warmup steps")
    save_steps: int = Field(1000, description="Save checkpoint every N steps")
    eval_steps: int = Field(500, description="Evaluate every N steps")
    output_dir: str = Field("./output", description="Output directory for checkpoints")
    use_wandb: bool = Field(False, description="Whether to use Weights & Biases")


class DataConfig(BaseModel):
    """Configuration for data processing."""
    
    dataset_name: Optional[str] = Field(None, description="HuggingFace dataset name")
    train_file: Optional[str] = Field(None, description="Path to training data file")
    val_file: Optional[str] = Field(None, description="Path to validation data file")
    test_file: Optional[str] = Field(None, description="Path to test data file")
    max_length: int = Field(2048, description="Maximum sequence length")
    preprocessing_num_workers: int = Field(4, description="Number of preprocessing workers")


class Config(BaseModel):
    """Main configuration class."""
    
    model: ModelConfig
    training: Optional[TrainingConfig] = None
    data: Optional[DataConfig] = None
    seed: int = Field(42, description="Random seed for reproducibility")
    experiment_name: str = Field("default", description="Experiment name for tracking")


def load_config(config_path: Path) -> Config:
    """Load configuration from a file.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Loaded configuration
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load based on file extension
    if config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    elif config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    # Substitute environment variables
    config_dict = _substitute_env_vars(config_dict)
    
    return Config(**config_dict)


def save_config(config: Config, config_path: Path) -> None:
    """Save configuration to a file.
    
    Args:
        config: Configuration object
        config_path: Path to save configuration
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.model_dump(exclude_none=True)
    
    # Save based on file extension
    if config_path.suffix == ".json":
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    elif config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def _substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively substitute environment variables in config.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Config with environment variables substituted
    """
    for key, value in config_dict.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            config_dict[key] = os.getenv(env_var, value)
        elif isinstance(value, dict):
            config_dict[key] = _substitute_env_vars(value)
        elif isinstance(value, list):
            config_dict[key] = [
                _substitute_env_vars(item) if isinstance(item, dict) else item
                for item in value
            ]
    
    return config_dict
