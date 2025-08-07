"""PyTorch/HuggingFace provider implementation."""

import logging
from typing import List, Dict, Any, Optional, Generator, Union
from src.providers.base import (
    BaseProvider,
    GenerationConfig,
    CompletionResponse,
    Message,
    ProviderType
)
from src.models import ModelLoader, InferencePipeline
from src.models.inference import InferenceConfig

logger = logging.getLogger(__name__)


class PyTorchProvider(BaseProvider):
    """Provider for PyTorch/HuggingFace Transformers models.
    
    This provider loads models directly using HuggingFace Transformers
    and runs inference locally with PyTorch.
    """
    
    def _initialize(self, **kwargs):
        """Initialize PyTorch provider.
        
        Args:
            device: Device to use ("auto", "cuda", "cpu")
            torch_dtype: Data type for weights ("auto", "float16", "float32", "bfloat16")
            cache_dir: Directory to cache models
            use_auth_token: HuggingFace API token
            quantization: Quantization configuration
        """
        self.provider_type = ProviderType.PYTORCH
        
        # Initialize model loader
        self.loader = ModelLoader(
            model_id=self.model_id,
            device=kwargs.get("device", "auto"),
            torch_dtype=kwargs.get("torch_dtype", "auto"),
            use_auth_token=kwargs.get("use_auth_token"),
            cache_dir=kwargs.get("cache_dir"),
        )
        
        # Load model and tokenizer
        quantization = kwargs.get("quantization")
        if quantization:
            # Load with quantization
            if quantization.get("load_in_4bit"):
                self.model = self.loader.load_quantized(load_in_4bit=True)
            elif quantization.get("load_in_8bit"):
                self.model = self.loader.load_quantized(load_in_8bit=True)
            else:
                self.model = self.loader.load_model(quantization_config=quantization)
        else:
            self.model = self.loader.load_model()
        
        self.tokenizer = self.loader.load_tokenizer()
        
        # Initialize inference pipeline
        self.pipeline = InferencePipeline(self.model, self.tokenizer)
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate text using PyTorch model.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Completion response
        """
        if config is None:
            config = GenerationConfig()
        
        # Convert to InferenceConfig
        inf_config = InferenceConfig(
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            num_beams=config.num_beams,
        )
        
        # Generate
        generated_text = self.pipeline.generate(prompt, inf_config, **kwargs)
        
        # Calculate tokens (approximate)
        prompt_tokens = len(self.tokenizer.encode(prompt))
        completion_tokens = len(self.tokenizer.encode(generated_text))
        
        return CompletionResponse(
            text=generated_text,
            model=self.model_id,
            provider=self.provider_type,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
        )
    
    def chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate chat response using PyTorch model.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Completion response
        """
        if config is None:
            config = GenerationConfig()
        
        # Normalize messages
        normalized = self._normalize_messages(messages)
        
        # Convert to dict format for pipeline
        message_dicts = [msg.to_dict() for msg in normalized]
        
        # Convert config
        inf_config = InferenceConfig(
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            num_beams=config.num_beams,
        )
        
        # Generate
        generated_text = self.pipeline.chat(message_dicts, inf_config, **kwargs)
        
        # Calculate tokens (approximate)
        full_prompt = self.tokenizer.apply_chat_template(
            message_dicts,
            add_generation_prompt=True,
            tokenize=False,
        ) if hasattr(self.tokenizer, "apply_chat_template") else str(message_dicts)
        
        prompt_tokens = len(self.tokenizer.encode(full_prompt))
        completion_tokens = len(self.tokenizer.encode(generated_text))
        
        return CompletionResponse(
            text=generated_text,
            model=self.model_id,
            provider=self.provider_type,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason="stop",
        )
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generated text from PyTorch model.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        if config is None:
            config = GenerationConfig()
        
        # Convert config
        inf_config = InferenceConfig(
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
        )
        
        # Stream generation
        yield from self.pipeline.stream_generate(prompt, inf_config, **kwargs)
    
    def stream_chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response from PyTorch model.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Additional generation parameters
            
        Yields:
            Generated text chunks
        """
        # For simplicity, convert to prompt and use stream_generate
        normalized = self._normalize_messages(messages)
        message_dicts = [msg.to_dict() for msg in normalized]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                message_dicts,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # Fallback formatting
            prompt = "\n\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in message_dicts
            ]) + "\n\nAssistant:"
        
        yield from self.stream_generate(prompt, config, **kwargs)
    
    def list_models(self) -> List[str]:
        """List available models (returns current model).
        
        Returns:
            List containing the current model ID
        """
        # For local PyTorch, we just return the loaded model
        return [self.model_id]
    
    def unload_model(self):
        """Unload the model from memory."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info(f"Model {self.model_id} unloaded from memory")
