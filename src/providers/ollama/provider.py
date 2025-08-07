"""Ollama provider implementation for local model inference including gpt-oss support.

Based on OpenAI cookbook: https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama
"""

import json
import logging
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


class OllamaProvider(BaseProvider):
    """Provider for Ollama local model inference with gpt-oss support.
    
    Supports both gpt-oss-20b and gpt-oss-120b models with:
    - MXFP4 quantization out of the box
    - Chat Completions-compatible API
    - Tool calling (function calling)
    - OpenAI SDK compatibility
    
    Model requirements:
    - gpt-oss-20b: ≥16GB VRAM or unified memory
    - gpt-oss-120b: ≥60GB VRAM or unified memory
    
    See: https://ollama.ai/
    """
    
    GPT_OSS_MODELS = ["gpt-oss:20b", "gpt-oss:120b", "gpt-oss"]
    
    def _initialize(self, **kwargs):
        """Initialize Ollama provider.
        
        Args:
            base_url: Ollama API base URL (default: http://localhost:11434)
            timeout: Request timeout in seconds
            use_openai_sdk: Whether to use OpenAI SDK for Chat Completions (default: False)
            auto_pull: Automatically pull gpt-oss models if not available (default: True)
        """
        self.provider_type = ProviderType.OLLAMA
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.timeout = kwargs.get("timeout", 60)
        self.use_openai_sdk = kwargs.get("use_openai_sdk", False) and OPENAI_SDK_AVAILABLE
        self.auto_pull = kwargs.get("auto_pull", True)
        
        # Initialize OpenAI client if requested
        if self.use_openai_sdk:
            self.openai_client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key="ollama"  # Dummy key for Ollama
            )
        else:
            self.openai_client = None
        
        # Check if Ollama is running
        if not self._check_ollama_status():
            logger.warning(
                f"Ollama server not responding at {self.base_url}. "
                "Please ensure Ollama is running."
            )
        
        # Auto-pull gpt-oss model if needed
        if self.auto_pull and self._is_gpt_oss_model(self.model_id):
            self._ensure_gpt_oss_model()
    
    def _check_ollama_status(self) -> bool:
        """Check if Ollama server is running.
        
        Returns:
            True if server is accessible
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional Ollama parameters
            
        Returns:
            Completion response
        """
        if config is None:
            config = GenerationConfig()
        
        # Prepare request
        params = config.to_ollama_params()
        params.update(kwargs)
        
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": params,
        }
        
        # Make request
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            return CompletionResponse(
                text=data.get("response", ""),
                model=self.model_id,
                provider=self.provider_type,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (
                        data.get("prompt_eval_count", 0) + 
                        data.get("eval_count", 0)
                    ),
                },
                finish_reason="stop" if data.get("done") else "length",
                metadata={
                    "eval_duration": data.get("eval_duration"),
                    "load_duration": data.get("load_duration"),
                }
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise RuntimeError(f"Failed to generate with Ollama: {e}")
    
    def chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate chat response using Ollama.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Additional Ollama parameters
            
        Returns:
            Completion response
        """
        if config is None:
            config = GenerationConfig()
        
        # Normalize messages
        normalized = self._normalize_messages(messages)
        
        # Prepare request
        params = config.to_ollama_params()
        params.update(kwargs)
        
        payload = {
            "model": self.model_id,
            "messages": [msg.to_dict() for msg in normalized],
            "stream": False,
            "options": params,
        }
        
        # Make request
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            message = data.get("message", {})
            
            return CompletionResponse(
                text=message.get("content", ""),
                model=self.model_id,
                provider=self.provider_type,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": (
                        data.get("prompt_eval_count", 0) + 
                        data.get("eval_count", 0)
                    ),
                },
                finish_reason="stop" if data.get("done") else "length",
                metadata={
                    "eval_duration": data.get("eval_duration"),
                    "load_duration": data.get("load_duration"),
                }
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama chat request failed: {e}")
            raise RuntimeError(f"Failed to chat with Ollama: {e}")
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generated text from Ollama.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional Ollama parameters
            
        Yields:
            Generated text chunks
        """
        if config is None:
            config = GenerationConfig()
        
        # Prepare request
        params = config.to_ollama_params()
        params.update(kwargs)
        
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": True,
            "options": params,
        }
        
        # Stream response
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama stream failed: {e}")
            raise RuntimeError(f"Failed to stream from Ollama: {e}")
    
    def stream_chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response from Ollama.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Additional Ollama parameters
            
        Yields:
            Generated text chunks
        """
        if config is None:
            config = GenerationConfig()
        
        # Normalize messages
        normalized = self._normalize_messages(messages)
        
        # Prepare request
        params = config.to_ollama_params()
        params.update(kwargs)
        
        payload = {
            "model": self.model_id,
            "messages": [msg.to_dict() for msg in normalized],
            "stream": True,
            "options": params,
        }
        
        # Stream response
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    message = data.get("message", {})
                    if "content" in message:
                        yield message["content"]
                    if data.get("done"):
                        break
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama chat stream failed: {e}")
            raise RuntimeError(f"Failed to stream chat from Ollama: {e}")
    
    def list_models(self) -> List[str]:
        """List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            models = data.get("models", [])
            return [model["name"] for model in models]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                stream=True,
                timeout=None  # No timeout for pulling
            )
            response.raise_for_status()
            
            # Stream progress
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if "status" in data:
                        logger.info(f"Pull status: {data['status']}")
                    if data.get("error"):
                        logger.error(f"Pull error: {data['error']}")
                        return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama.
        
        Args:
            model_name: Name of the model to delete
            
        Returns:
            True if successful
        """
        try:
            response = requests.delete(
                f"{self.base_url}/api/delete",
                json={"name": model_name},
                timeout=self.timeout
            )
            response.raise_for_status()
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def _is_gpt_oss_model(self, model_name: str) -> bool:
        """Check if model is a gpt-oss variant.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if it's a gpt-oss model
        """
        return any(model_name.startswith(prefix) for prefix in self.GPT_OSS_MODELS)
    
    def _ensure_gpt_oss_model(self):
        """Ensure gpt-oss model is available, pulling if necessary."""
        available_models = self.list_models()
        
        if self.model_id not in available_models:
            logger.info(f"Model {self.model_id} not found. Pulling from Ollama library...")
            if self.pull_model(self.model_id):
                logger.info(f"Successfully pulled {self.model_id}")
            else:
                logger.warning(f"Failed to pull {self.model_id}. Please run: ollama pull {self.model_id}")
    
    def chat_with_tools(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        tools: List[Union[ToolFunction, Dict[str, Any]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate chat response with tool calling support.
        
        Args:
            messages: List of chat messages
            tools: List of available tools/functions
            config: Generation configuration
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
                response = self.openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=[msg.to_dict() for msg in normalized],
                    tools=formatted_tools if formatted_tools else None,
                    max_tokens=config.max_tokens if config else 512,
                    temperature=config.temperature if config else 0.7,
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
            # Ollama native API doesn't support tools directly
            # Add tools to system message as a workaround
            tool_descriptions = self._format_tools_for_prompt(tools)
            
            enhanced_messages = messages.copy()
            if tool_descriptions:
                system_msg = {
                    "role": "system",
                    "content": f"You have access to the following tools:\n{tool_descriptions}\n\nWhen you need to use a tool, respond with a JSON object in this format: {{\"tool\": \"function_name\", \"arguments\": {{...}}}}"
                }
                enhanced_messages.insert(0, system_msg)
            
            return self.chat(enhanced_messages, config, **kwargs)
    
    def _format_tools_for_prompt(self, tools: List[Union[ToolFunction, Dict[str, Any]]]) -> str:
        """Format tools for inclusion in prompt.
        
        Args:
            tools: List of tools
            
        Returns:
            Formatted tool descriptions
        """
        descriptions = []
        for tool in tools:
            if isinstance(tool, ToolFunction):
                descriptions.append(f"- {tool.name}: {tool.description}")
            elif isinstance(tool, dict) and "function" in tool:
                func = tool["function"]
                descriptions.append(f"- {func['name']}: {func.get('description', '')}")
        
        return "\n".join(descriptions)
    
    def generate_with_openai_sdk(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate using OpenAI SDK compatibility.
        
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
            
            response = self.openai_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=config.max_tokens if config else 512,
                temperature=config.temperature if config else 0.7,
                top_p=config.top_p if config else 0.9,
                **kwargs
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
