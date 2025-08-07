"""OpenRouter provider implementation for accessing multiple models including gpt-oss.

Based on OpenRouter documentation: https://openrouter.ai/openai/gpt-oss-120b
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Generator, Union
import requests
from dataclasses import dataclass
from src.providers.base import (
    BaseProvider,
    GenerationConfig,
    CompletionResponse,
    Message,
    ProviderType
)

try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except ImportError:
    OPENAI_SDK_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ToolFunction:
    """Function definition for tool calling."""
    name: str
    description: str
    parameters: Dict[str, Any]


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API access with gpt-oss support.
    
    OpenRouter provides unified access to multiple LLM providers including:
    - openai/gpt-oss-20b: 20B parameter model (~$0.01/M input, ~$0.05/M output)
    - openai/gpt-oss-120b: 117B MoE model ($0.09/M input, $0.45/M output)
    
    Both models support:
    - 131,072 token context window
    - MXFP4 quantization
    - Tool calling and function calling
    - Structured outputs
    - Chain-of-thought reasoning
    
    See: https://openrouter.ai/
    """
    
    GPT_OSS_MODELS = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b"
    ]
    
    # Model-specific context limits
    MODEL_CONTEXT = {
        "openai/gpt-oss-20b": 131072,
        "openai/gpt-oss-120b": 131072
    }
    
    def _initialize(self, **kwargs):
        """Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (or from environment)
            base_url: API base URL (default: https://openrouter.ai/api/v1)
            site_url: Your site URL for OpenRouter tracking
            app_name: Your app name for OpenRouter tracking
            timeout: Request timeout in seconds
            use_openai_sdk: Whether to use OpenAI SDK (default: False)
            enable_fallback: Enable automatic fallback routing (default: True)
            provider_preferences: Provider routing preferences
        """
        self.provider_type = ProviderType.OPENROUTER
        self.api_key = kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        self.base_url = kwargs.get("base_url", "https://openrouter.ai/api/v1")
        self.site_url = kwargs.get("site_url", "")
        self.app_name = kwargs.get("app_name", "gpt-oss-use-cases")
        self.timeout = kwargs.get("timeout", 60)
        self.use_openai_sdk = kwargs.get("use_openai_sdk", False) and OPENAI_SDK_AVAILABLE
        self.enable_fallback = kwargs.get("enable_fallback", True)
        self.provider_preferences = kwargs.get("provider_preferences", {})
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Set headers
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        if self.site_url:
            self.headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            self.headers["X-Title"] = self.app_name
        
        # Initialize OpenAI client if requested
        if self.use_openai_sdk:
            self.openai_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": self.site_url or "",
                    "X-Title": self.app_name or ""
                }
            )
        else:
            self.openai_client = None
        
        # Check if model is gpt-oss
        self.is_gpt_oss = self._is_gpt_oss_model(self.model_id)
        
        # Log model info if gpt-oss
        if self.is_gpt_oss:
            context_limit = self.MODEL_CONTEXT.get(self.model_id, 131072)
            logger.info(
                f"Initialized OpenRouter with {self.model_id} "
                f"(context: {context_limit} tokens)"
            )
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate text using OpenRouter.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional OpenRouter parameters
            
        Returns:
            Completion response
        """
        if config is None:
            config = GenerationConfig()
        
        url = f"{self.base_url}/completions"
        
        params = config.to_openrouter_params()
        params.update(kwargs)
        
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            **params
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            choice = data["choices"][0]
            
            return CompletionResponse(
                text=choice["text"],
                model=data.get("model", self.model_id),
                provider=self.provider_type,
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason"),
                metadata={
                    "id": data.get("id"),
                    "provider": data.get("provider"),
                }
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise RuntimeError(f"Failed to generate with OpenRouter: {e}")
    
    def chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate chat response using OpenRouter.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Additional OpenRouter parameters
            
        Returns:
            Completion response
        """
        if config is None:
            config = GenerationConfig()
        
        # Normalize messages
        normalized = self._normalize_messages(messages)
        
        url = f"{self.base_url}/chat/completions"
        
        params = config.to_openrouter_params()
        params.update(kwargs)
        
        payload = {
            "model": self.model_id,
            "messages": [msg.to_dict() for msg in normalized],
            **params
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            choice = data["choices"][0]
            message = choice["message"]
            
            return CompletionResponse(
                text=message["content"],
                model=data.get("model", self.model_id),
                provider=self.provider_type,
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason"),
                metadata={
                    "id": data.get("id"),
                    "provider": data.get("provider"),
                }
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter chat request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise RuntimeError(f"Failed to chat with OpenRouter: {e}")
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generated text from OpenRouter.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional OpenRouter parameters
            
        Yields:
            Generated text chunks
        """
        if config is None:
            config = GenerationConfig()
        
        url = f"{self.base_url}/completions"
        
        params = config.to_openrouter_params()
        params.update(kwargs)
        params["stream"] = True
        
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            **params
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        line = line[6:]  # Remove "data: " prefix
                    if line == b"[DONE]":
                        break
                    try:
                        data = json.loads(line)
                        choice = data["choices"][0]
                        if "text" in choice:
                            yield choice["text"]
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter stream failed: {e}")
            raise RuntimeError(f"Failed to stream from OpenRouter: {e}")
    
    def stream_chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response from OpenRouter.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Additional OpenRouter parameters
            
        Yields:
            Generated text chunks
        """
        if config is None:
            config = GenerationConfig()
        
        # Normalize messages
        normalized = self._normalize_messages(messages)
        
        url = f"{self.base_url}/chat/completions"
        
        params = config.to_openrouter_params()
        params.update(kwargs)
        params["stream"] = True
        
        payload = {
            "model": self.model_id,
            "messages": [msg.to_dict() for msg in normalized],
            **params
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    if line.startswith(b"data: "):
                        line = line[6:]  # Remove "data: " prefix
                    if line == b"[DONE]":
                        break
                    try:
                        data = json.loads(line)
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter chat stream failed: {e}")
            raise RuntimeError(f"Failed to stream chat from OpenRouter: {e}")
    
    def list_models(self) -> List[str]:
        """List available models on OpenRouter.
        
        Returns:
            List of model IDs
        """
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            models = data.get("data", [])
            return [model["id"] for model in models]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list OpenRouter models: {e}")
            return []
    
    def get_model_info(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed information about a model.
        
        Args:
            model_id: Model ID to get info for (default: current model)
            
        Returns:
            Model information dictionary
        """
        model_id = model_id or self.model_id
        
        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            models = data.get("data", [])
            for model in models:
                if model["id"] == model_id:
                    return model
            
            return {}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def check_credits(self) -> Dict[str, Any]:
        """Check available credits on OpenRouter.
        
        Returns:
            Credit information
        """
        try:
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check credits: {e}")
            return {}
    
    def _is_gpt_oss_model(self, model_name: str) -> bool:
        """Check if model is a gpt-oss variant.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if it's a gpt-oss model
        """
        return model_name in self.GPT_OSS_MODELS
    
    def generate_with_openai_sdk(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate using OpenAI SDK.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        if not self.openai_client:
            raise RuntimeError("OpenAI SDK not initialized. Set use_openai_sdk=True")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare extra headers
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.app_name:
                extra_headers["X-Title"] = self.app_name
            
            # Prepare parameters
            params = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": config.max_tokens if config else 512,
                "temperature": config.temperature if config else 0.7,
            }
            
            if config:
                if config.top_p is not None:
                    params["top_p"] = config.top_p
                if config.frequency_penalty != 0:
                    params["frequency_penalty"] = config.frequency_penalty
                if config.presence_penalty != 0:
                    params["presence_penalty"] = config.presence_penalty
                if config.stop_sequences:
                    params["stop"] = config.stop_sequences
                if config.seed is not None:
                    params["seed"] = config.seed
            
            params.update(kwargs)
            
            response = self.openai_client.chat.completions.create(
                extra_headers=extra_headers,
                **params
            )
            
            return CompletionResponse(
                text=response.choices[0].message.content,
                model=response.model,
                provider=self.provider_type,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=response.choices[0].finish_reason,
                metadata={"id": response.id}
            )
            
        except Exception as e:
            logger.error(f"OpenAI SDK generation failed: {e}")
            raise RuntimeError(f"Failed to generate with OpenAI SDK: {e}")
    
    def chat_with_tools(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        tools: List[Union[ToolFunction, Dict[str, Any]]],
        config: Optional[GenerationConfig] = None,
        tool_choice: Optional[Union[str, Dict]] = "auto",
        **kwargs
    ) -> CompletionResponse:
        """Generate chat response with tool calling support.
        
        Args:
            messages: List of chat messages
            tools: List of available tools/functions
            config: Generation configuration
            tool_choice: Tool selection strategy ("none", "auto", or specific function)
            **kwargs: Additional parameters
            
        Returns:
            Completion response with potential tool calls
        """
        if self.use_openai_sdk and self.openai_client:
            # Use OpenAI SDK for tool calling
            normalized = self._normalize_messages(messages)
            
            # Format tools for OpenAI API
            formatted_tools = []
            for tool in tools:
                if isinstance(tool, ToolFunction):
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                    })
                elif isinstance(tool, dict):
                    formatted_tools.append(tool)
            
            try:
                # Prepare extra headers
                extra_headers = {}
                if self.site_url:
                    extra_headers["HTTP-Referer"] = self.site_url
                if self.app_name:
                    extra_headers["X-Title"] = self.app_name
                
                response = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=[msg.to_dict() for msg in normalized],
                    tools=formatted_tools if formatted_tools else None,
                    tool_choice=tool_choice,
                    max_tokens=config.max_tokens if config else 512,
                    temperature=config.temperature if config else 0.7,
                    extra_headers=extra_headers,
                    **kwargs
                )
                
                message = response.choices[0].message
                
                # Check for tool calls
                tool_calls = None
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls = [
                        {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                        for tc in message.tool_calls
                    ]
                
                return CompletionResponse(
                    text=message.content or "",
                    model=response.model,
                    provider=self.provider_type,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    },
                    finish_reason=response.choices[0].finish_reason,
                    metadata={
                        "tool_calls": tool_calls,
                        "id": response.id
                    }
                )
                
            except Exception as e:
                logger.error(f"OpenAI SDK tool calling failed: {e}")
                # Fall back to regular chat
                return self.chat(messages, config, **kwargs)
        else:
            # Use direct API with tool support
            normalized = self._normalize_messages(messages)
            
            url = f"{self.base_url}/chat/completions"
            
            params = config.to_openrouter_params() if config else {}
            params.update(kwargs)
            
            # Format tools
            formatted_tools = []
            for tool in tools:
                if isinstance(tool, ToolFunction):
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters
                        }
                    })
                elif isinstance(tool, dict):
                    formatted_tools.append(tool)
            
            payload = {
                "model": self.model_id,
                "messages": [msg.to_dict() for msg in normalized],
                "tools": formatted_tools if formatted_tools else None,
                "tool_choice": tool_choice,
                **params
            }
            
            # Add provider preferences if set
            if self.provider_preferences:
                payload["provider"] = self.provider_preferences
            
            # Enable fallback routing if requested
            if self.enable_fallback:
                payload["route"] = "fallback"
            
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                
                choice = data["choices"][0]
                message = choice["message"]
                
                # Check for tool calls
                tool_calls = None
                if "tool_calls" in message:
                    tool_calls = [
                        {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                        for tc in message["tool_calls"]
                    ]
                
                return CompletionResponse(
                    text=message.get("content", ""),
                    model=data.get("model", self.model_id),
                    provider=self.provider_type,
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason"),
                    metadata={
                        "tool_calls": tool_calls,
                        "id": data.get("id"),
                        "provider": data.get("provider")
                    }
                )
                
            except requests.exceptions.RequestException as e:
                logger.error(f"OpenRouter tool calling request failed: {e}")
                if hasattr(e.response, 'text'):
                    logger.error(f"Response: {e.response.text}")
                return self.chat(messages, config, **kwargs)
    
    def generate_with_fallback(
        self,
        prompt: str,
        models: List[str],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate with automatic model fallback.
        
        Args:
            prompt: Input prompt
            models: List of models to try in order
            config: Generation configuration
            **kwargs: Additional parameters
            
        Returns:
            Completion response from first successful model
        """
        url = f"{self.base_url}/completions"
        
        params = config.to_openrouter_params() if config else {}
        params.update(kwargs)
        
        payload = {
            "models": models,  # OpenRouter will try these in order
            "route": "fallback",
            "prompt": prompt,
            **params
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            choice = data["choices"][0]
            
            return CompletionResponse(
                text=choice["text"],
                model=data.get("model", models[0]),
                provider=self.provider_type,
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason"),
                metadata={
                    "id": data.get("id"),
                    "provider": data.get("provider"),
                    "models_tried": models
                }
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter fallback generation failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise RuntimeError(f"Failed to generate with fallback: {e}")
