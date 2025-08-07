"""Factory for creating model providers."""

import logging
from typing import Dict, Any, Optional
from src.providers.base import BaseProvider, ProviderType

logger = logging.getLogger(__name__)


class ProviderFactory:
    """Factory for creating model providers."""
    
    _providers: Dict[ProviderType, type] = {}
    
    @classmethod
    def register_provider(cls, provider_type: ProviderType, provider_class: type):
        """Register a provider class.
        
        Args:
            provider_type: Type of provider
            provider_class: Provider class
        """
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def create(
        cls,
        provider_type: ProviderType,
        model_id: str,
        **kwargs
    ) -> BaseProvider:
        """Create a provider instance.
        
        Args:
            provider_type: Type of provider to create
            model_id: Model identifier
            **kwargs: Provider-specific configuration
            
        Returns:
            Provider instance
        """
        # Lazy import providers to avoid circular dependencies
        if provider_type not in cls._providers:
            cls._load_provider(provider_type)
        
        provider_class = cls._providers.get(provider_type)
        if not provider_class:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        logger.info(f"Creating {provider_type.value} provider for model: {model_id}")
        return provider_class(model_id, **kwargs)
    
    @classmethod
    def create_from_string(
        cls,
        provider_name: str,
        model_id: str,
        **kwargs
    ) -> BaseProvider:
        """Create a provider from string name.
        
        Args:
            provider_name: Name of provider (e.g., "ollama", "vllm")
            model_id: Model identifier
            **kwargs: Provider-specific configuration
            
        Returns:
            Provider instance
        """
        try:
            provider_type = ProviderType(provider_name.lower())
        except ValueError:
            raise ValueError(
                f"Unknown provider: {provider_name}. "
                f"Available: {[p.value for p in ProviderType]}"
            )
        
        return cls.create(provider_type, model_id, **kwargs)
    
    @classmethod
    def _load_provider(cls, provider_type: ProviderType):
        """Lazy load a provider module.
        
        Args:
            provider_type: Type of provider to load
        """
        if provider_type == ProviderType.PYTORCH:
            from src.providers.pytorch.provider import PyTorchProvider
            cls.register_provider(ProviderType.PYTORCH, PyTorchProvider)
            
        elif provider_type == ProviderType.OLLAMA:
            from src.providers.ollama.provider import OllamaProvider
            cls.register_provider(ProviderType.OLLAMA, OllamaProvider)
            
        elif provider_type == ProviderType.VLLM:
            from src.providers.vllm.provider import VLLMProvider
            cls.register_provider(ProviderType.VLLM, VLLMProvider)
            
        elif provider_type == ProviderType.OPENROUTER:
            from src.providers.openrouter.provider import OpenRouterProvider
            cls.register_provider(ProviderType.OPENROUTER, OpenRouterProvider)
            
        elif provider_type == ProviderType.LANGCHAIN:
            from src.providers.langchain.provider import LangChainProvider
            cls.register_provider(ProviderType.LANGCHAIN, LangChainProvider)
        else:
            raise ValueError(f"Provider {provider_type} not implemented")
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List available provider types.
        
        Returns:
            List of provider type names
        """
        return [p.value for p in ProviderType]
