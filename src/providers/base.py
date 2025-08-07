"""Base provider interface for unified model access."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Union
from dataclasses import dataclass
from enum import Enum


class ProviderType(Enum):
    """Supported provider types."""
    PYTORCH = "pytorch"
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENROUTER = "openrouter"
    LANGCHAIN = "langchain"


@dataclass
class GenerationConfig:
    """Unified generation configuration across providers."""
    
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = 50
    repetition_penalty: float = 1.0
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None
    
    # Provider-specific settings
    frequency_penalty: float = 0.0  # OpenAI-style
    presence_penalty: float = 0.0   # OpenAI-style
    num_beams: int = 1              # HuggingFace-style
    do_sample: bool = True          # HuggingFace-style
    
    def to_ollama_params(self) -> Dict[str, Any]:
        """Convert to Ollama parameters."""
        return {
            "num_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repetition_penalty,
            "stop": self.stop_sequences or [],
            "seed": self.seed,
        }
    
    def to_vllm_params(self) -> Dict[str, Any]:
        """Convert to vLLM parameters."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stop": self.stop_sequences,
            "seed": self.seed,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
    
    def to_openrouter_params(self) -> Dict[str, Any]:
        """Convert to OpenRouter parameters."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop_sequences,
            "seed": self.seed,
        }
    
    def to_huggingface_params(self) -> Dict[str, Any]:
        """Convert to HuggingFace parameters."""
        return {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
        }


@dataclass
class Message:
    """Chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"role": self.role, "content": self.content}


@dataclass
class CompletionResponse:
    """Unified completion response."""
    
    text: str
    model: str
    provider: ProviderType
    usage: Optional[Dict[str, int]] = None  # tokens used
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseProvider(ABC):
    """Base interface for all model providers."""
    
    def __init__(self, model_id: str, **kwargs):
        """Initialize provider with model ID.
        
        Args:
            model_id: Model identifier (provider-specific)
            **kwargs: Provider-specific configuration
        """
        self.model_id = model_id
        self.provider_type = ProviderType.PYTORCH
        self._initialize(**kwargs)
    
    @abstractmethod
    def _initialize(self, **kwargs):
        """Provider-specific initialization."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Provider-specific parameters
            
        Returns:
            Completion response
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate response in chat format.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Provider-specific parameters
            
        Returns:
            Completion response
        """
        pass
    
    @abstractmethod
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generated text.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Provider-specific parameters
            
        Yields:
            Generated text chunks
        """
        pass
    
    @abstractmethod
    def stream_chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Provider-specific parameters
            
        Yields:
            Generated text chunks
        """
        pass
    
    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[CompletionResponse]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
            **kwargs: Provider-specific parameters
            
        Returns:
            List of completion responses
        """
        # Default implementation: sequential processing
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, config, **kwargs)
            responses.append(response)
        return responses
    
    def _normalize_messages(
        self, 
        messages: List[Union[Message, Dict[str, str]]]
    ) -> List[Message]:
        """Normalize messages to Message objects.
        
        Args:
            messages: List of messages (Message objects or dicts)
            
        Returns:
            List of Message objects
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(Message(**msg))
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        return normalized
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models for this provider.
        
        Returns:
            List of model IDs
        """
        pass
    
    def health_check(self) -> bool:
        """Check if the provider is healthy and accessible.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try a simple generation
            response = self.generate(
                "Hello",
                GenerationConfig(max_tokens=5)
            )
            return bool(response.text)
        except Exception:
            return False
