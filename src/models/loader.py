"""Model loading utilities for GPT OSS models."""

import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig,
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility class for loading and managing GPT models."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the model loader.
        
        Args:
            model_id: HuggingFace model ID or local path
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Data type for model weights ("auto", "float16", "float32", "bfloat16")
            use_auth_token: HuggingFace API token for private models
            cache_dir: Directory to cache downloaded models
        """
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_auth_token = use_auth_token
        self.cache_dir = cache_dir
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def load_model(
        self,
        quantization_config: Optional[Dict[str, Any]] = None,
        device_map: Optional[Union[str, Dict]] = None,
        low_memory: bool = False,
    ) -> PreTrainedModel:
        """Load the model with specified configuration.
        
        Args:
            quantization_config: Configuration for model quantization
            device_map: Device mapping for model parallelism
            low_memory: Whether to use low memory loading
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model: {self.model_id}")
        
        # Parse torch dtype
        if self.torch_dtype == "auto":
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            dtype = getattr(torch, self.torch_dtype)
        
        # Setup quantization if requested
        bnb_config = None
        if quantization_config:
            bnb_config = BitsAndBytesConfig(**quantization_config)
        
        # Determine device map
        if device_map is None:
            device_map = self.device
        
        # Load model
        kwargs = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "use_auth_token": self.use_auth_token,
            "cache_dir": self.cache_dir,
        }
        
        if bnb_config:
            kwargs["quantization_config"] = bnb_config
        
        if low_memory:
            kwargs["low_cpu_mem_usage"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **kwargs
        )
        
        logger.info(f"Model loaded successfully on {device_map}")
        return self.model

    def load_tokenizer(self, **kwargs) -> PreTrainedTokenizer:
        """Load the tokenizer for the model.
        
        Args:
            **kwargs: Additional arguments for tokenizer
            
        Returns:
            Loaded tokenizer
        """
        logger.info(f"Loading tokenizer: {self.model_id}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_auth_token=self.use_auth_token,
            cache_dir=self.cache_dir,
            **kwargs
        )
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
        return self.tokenizer

    def load_quantized(
        self,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_use_double_quant: bool = True,
    ) -> PreTrainedModel:
        """Load model with quantization for reduced memory usage.
        
        Args:
            load_in_8bit: Whether to load in 8-bit precision
            load_in_4bit: Whether to load in 4-bit precision
            bnb_4bit_compute_dtype: Compute dtype for 4-bit quantization
            bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
            bnb_4bit_use_double_quant: Whether to use double quantization
            
        Returns:
            Quantized model
        """
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot load in both 8-bit and 4-bit simultaneously")
        
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
            quantization_config = {
                "load_in_8bit": load_in_8bit,
                "load_in_4bit": load_in_4bit,
                "bnb_4bit_compute_dtype": compute_dtype,
                "bnb_4bit_quant_type": bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": bnb_4bit_use_double_quant,
            }
        
        return self.load_model(quantization_config=quantization_config)

    def prepare_for_training(self, gradient_checkpointing: bool = True):
        """Prepare model for training/fine-tuning.
        
        Args:
            gradient_checkpointing: Whether to enable gradient checkpointing
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Enable gradient checkpointing
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Prepare model for k-bit training if quantized
        if hasattr(self.model, "prepare_for_kbit_training"):
            self.model = self.model.prepare_for_kbit_training()
            logger.info("Model prepared for k-bit training")
        
        return self.model
