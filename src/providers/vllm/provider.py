"""vLLM provider implementation for high-performance inference with gpt-oss support.

Based on OpenAI cookbook: https://cookbook.openai.com/articles/gpt-oss/run-vllm
"""

import logging
from typing import List, Dict, Any, Optional, Generator, Union
import requests
import json
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

try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message as HarmonyMessage,
        Role,
        SystemContent,
        DeveloperContent,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ToolFunction:
    """Function definition for tool calling."""
    name: str
    description: str
    parameters: Dict[str, Any]


class VLLMProvider(BaseProvider):
    """Provider for vLLM high-performance inference with gpt-oss support.
    
    Supports both gpt-oss-20b and gpt-oss-120b models with:
    - MXFP4 quantization out of the box
    - Chat Completions and Responses API compatibility
    - Tool calling (function calling)
    - Harmony format support for direct sampling
    - Multi-GPU support
    
    Model requirements:
    - openai/gpt-oss-20b: ~16GB VRAM
    - openai/gpt-oss-120b: ≥60GB VRAM (single H100 or multi-GPU)
    
    See: https://github.com/vllm-project/vllm
    """
    
    GPT_OSS_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    
    def _initialize(self, **kwargs):
        """Initialize vLLM provider.
        
        Args:
            base_url: vLLM server URL (default: http://localhost:8000)
            api_key: Optional API key for authentication (default: "EMPTY" for local)
            timeout: Request timeout in seconds
            use_openai_api: Whether to use OpenAI-compatible API (default: True)
            use_responses_api: Whether to use Responses API (default: False)
            use_openai_sdk: Whether to use OpenAI SDK (default: False)
            use_harmony: Whether to use harmony format for direct sampling (default: False)
        """
        self.provider_type = ProviderType.VLLM
        self.base_url = kwargs.get("base_url", "http://localhost:8000")
        self.api_key = kwargs.get("api_key", "EMPTY")
        self.timeout = kwargs.get("timeout", 60)
        self.use_openai_api = kwargs.get("use_openai_api", True)
        self.use_responses_api = kwargs.get("use_responses_api", False)
        self.use_openai_sdk = kwargs.get("use_openai_sdk", False) and OPENAI_SDK_AVAILABLE
        self.use_harmony = kwargs.get("use_harmony", False) and HARMONY_AVAILABLE
        
        # Set headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Initialize OpenAI client if requested
        if self.use_openai_sdk:
            self.openai_client = OpenAI(
                base_url=f"{self.base_url}/v1",
                api_key=self.api_key
            )
        else:
            self.openai_client = None
        
        # Initialize harmony encoding if requested
        if self.use_harmony:
            self.harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        else:
            self.harmony_encoding = None
        
        # Check if model is gpt-oss
        self.is_gpt_oss = self._is_gpt_oss_model(self.model_id)
    
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate text using vLLM.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional vLLM parameters
            
        Returns:
            Completion response
        """
        if config is None:
            config = GenerationConfig()
        
        if self.use_openai_api:
            # Use OpenAI-compatible completions endpoint
            url = f"{self.base_url}/v1/completions"
            
            params = config.to_vllm_params()
            params.update(kwargs)
            
            payload = {
                "model": self.model_id,
                "prompt": prompt,
                **params
            }
        else:
            # Use native vLLM API
            url = f"{self.base_url}/generate"
            
            params = config.to_vllm_params()
            params.update(kwargs)
            
            payload = {
                "prompt": prompt,
                "sampling_params": params
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
            
            if self.use_openai_api:
                # Parse OpenAI-style response
                choice = data["choices"][0]
                return CompletionResponse(
                    text=choice["text"],
                    model=data.get("model", self.model_id),
                    provider=self.provider_type,
                    usage=data.get("usage", {}),
                    finish_reason=choice.get("finish_reason"),
                    metadata={"id": data.get("id")}
                )
            else:
                # Parse native vLLM response
                output = data["outputs"][0]
                return CompletionResponse(
                    text=output["text"],
                    model=self.model_id,
                    provider=self.provider_type,
                    usage={
                        "prompt_tokens": output.get("prompt_tokens", 0),
                        "completion_tokens": output.get("completion_tokens", 0),
                        "total_tokens": (
                            output.get("prompt_tokens", 0) +
                            output.get("completion_tokens", 0)
                        )
                    },
                    finish_reason=output.get("finish_reason"),
                    metadata={"request_id": data.get("request_id")}
                )
                
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM request failed: {e}")
            raise RuntimeError(f"Failed to generate with vLLM: {e}")
    
    def chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate chat response using vLLM.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Additional vLLM parameters
            
        Returns:
            Completion response
        """
        if config is None:
            config = GenerationConfig()
        
        # Normalize messages
        normalized = self._normalize_messages(messages)
        
        if self.use_openai_api:
            # Use OpenAI-compatible chat endpoint
            url = f"{self.base_url}/v1/chat/completions"
            
            params = config.to_vllm_params()
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
                    metadata={"id": data.get("id")}
                )
                
            except requests.exceptions.RequestException as e:
                logger.error(f"vLLM chat request failed: {e}")
                raise RuntimeError(f"Failed to chat with vLLM: {e}")
        else:
            # Convert to prompt format for native API
            prompt = self._format_chat_prompt(normalized)
            return self.generate(prompt, config, **kwargs)
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generated text from vLLM.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            **kwargs: Additional vLLM parameters
            
        Yields:
            Generated text chunks
        """
        if config is None:
            config = GenerationConfig()
        
        if self.use_openai_api:
            # Use OpenAI-compatible streaming
            url = f"{self.base_url}/v1/completions"
            
            params = config.to_vllm_params()
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
                logger.error(f"vLLM stream failed: {e}")
                raise RuntimeError(f"Failed to stream from vLLM: {e}")
        else:
            # Native vLLM doesn't support streaming in the same way
            # Fall back to non-streaming
            response = self.generate(prompt, config, **kwargs)
            yield response.text
    
    def stream_chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream chat response from vLLM.
        
        Args:
            messages: List of chat messages
            config: Generation configuration
            **kwargs: Additional vLLM parameters
            
        Yields:
            Generated text chunks
        """
        if config is None:
            config = GenerationConfig()
        
        # Normalize messages
        normalized = self._normalize_messages(messages)
        
        if self.use_openai_api:
            # Use OpenAI-compatible chat streaming
            url = f"{self.base_url}/v1/chat/completions"
            
            params = config.to_vllm_params()
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
                logger.error(f"vLLM chat stream failed: {e}")
                raise RuntimeError(f"Failed to stream chat from vLLM: {e}")
        else:
            # Convert to prompt and use generate streaming
            prompt = self._format_chat_prompt(normalized)
            yield from self.stream_generate(prompt, config, **kwargs)
    
    def list_models(self) -> List[str]:
        """List available models in vLLM server.
        
        Returns:
            List of model IDs
        """
        if self.use_openai_api:
            try:
                response = requests.get(
                    f"{self.base_url}/v1/models",
                    headers=self.headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                data = response.json()
                return [model["id"] for model in data.get("data", [])]
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to list vLLM models: {e}")
                return []
        else:
            # Native vLLM API doesn't have a list models endpoint
            # Return the configured model
            return [self.model_id]
    
    def _format_chat_prompt(self, messages: List[Message]) -> str:
        """Format chat messages into a prompt string.
        
        Args:
            messages: List of messages
            
        Returns:
            Formatted prompt
        """
        # Simple format - can be customized based on model
        formatted = []
        for msg in messages:
            if msg.role == "system":
                formatted.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Assistant: {msg.content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)
    
    def _is_gpt_oss_model(self, model_name: str) -> bool:
        """Check if model is a gpt-oss variant.
        
        Args:
            model_name: Model name to check
            
        Returns:
            True if it's a gpt-oss model
        """
        return any(model_name == model or model_name.startswith(model) for model in self.GPT_OSS_MODELS)
    
    def responses_api_generate(
        self,
        instructions: str,
        input_text: str,
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate using Responses API.
        
        Args:
            instructions: System instructions
            input_text: User input
            config: Generation configuration
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        if self.use_openai_sdk and self.openai_client:
            try:
                response = self.openai_client.responses.create(
                    model=self.model_id,
                    instructions=instructions,
                    input=input_text,
                    max_tokens=config.max_tokens if config else 512,
                    temperature=config.temperature if config else 0.7,
                    **kwargs
                )
                
                return CompletionResponse(
                    text=response.output_text,
                    model=self.model_id,
                    provider=self.provider_type,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                        "completion_tokens": response.usage.completion_tokens if hasattr(response, 'usage') else 0,
                        "total_tokens": response.usage.total_tokens if hasattr(response, 'usage') else 0,
                    },
                    finish_reason="stop",
                    metadata={"id": response.id if hasattr(response, 'id') else None}
                )
            except Exception as e:
                logger.error(f"Responses API generation failed: {e}")
                # Fall back to chat API
                messages = [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text}
                ]
                return self.chat(messages, config, **kwargs)
        else:
            # Use direct Responses API endpoint
            url = f"{self.base_url}/v1/responses"
            
            params = config.to_vllm_params() if config else {}
            params.update(kwargs)
            
            payload = {
                "model": self.model_id,
                "instructions": instructions,
                "input": input_text,
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
                
                return CompletionResponse(
                    text=data.get("output_text", ""),
                    model=self.model_id,
                    provider=self.provider_type,
                    usage=data.get("usage", {}),
                    finish_reason="stop",
                    metadata={"id": data.get("id")}
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"Responses API request failed: {e}")
                # Fall back to chat API
                messages = [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input_text}
                ]
                return self.chat(messages, config, **kwargs)
    
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
            # Use direct API with tool support
            url = f"{self.base_url}/v1/chat/completions"
            
            normalized = self._normalize_messages(messages)
            params = config.to_vllm_params() if config else {}
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
                        "id": data.get("id")
                    }
                )
                
            except requests.exceptions.RequestException as e:
                logger.error(f"vLLM tool calling request failed: {e}")
                return self.chat(messages, config, **kwargs)
    
    def generate_with_harmony(
        self,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate using harmony format for direct sampling.
        
        Args:
            messages: List of message dictionaries
            config: Generation configuration
            **kwargs: Additional parameters
            
        Returns:
            Completion response with parsed harmony output
        """
        if not self.use_harmony or not self.harmony_encoding:
            raise RuntimeError("Harmony not available. Install openai-harmony and set use_harmony=True")
        
        # Build conversation using harmony format
        harmony_messages = []
        for msg in messages:
            role_map = {
                "system": Role.SYSTEM,
                "developer": Role.DEVELOPER,
                "user": Role.USER,
                "assistant": Role.ASSISTANT,
            }
            
            role = role_map.get(msg["role"].lower(), Role.USER)
            content = msg["content"]
            
            if role == Role.SYSTEM:
                harmony_messages.append(
                    HarmonyMessage.from_role_and_content(role, SystemContent.new())
                )
            elif role == Role.DEVELOPER:
                harmony_messages.append(
                    HarmonyMessage.from_role_and_content(
                        role,
                        DeveloperContent.new().with_instructions(content)
                    )
                )
            else:
                harmony_messages.append(
                    HarmonyMessage.from_role_and_content(role, content)
                )
        
        # Create conversation
        convo = Conversation.from_messages(harmony_messages)
        
        # Render prompt
        prefill_ids = self.harmony_encoding.render_conversation_for_completion(
            convo, Role.ASSISTANT
        )
        stop_token_ids = self.harmony_encoding.stop_tokens_for_assistant_actions()
        
        # Use vLLM's generate endpoint with token IDs
        url = f"{self.base_url}/generate"
        
        params = config.to_vllm_params() if config else {}
        params["stop_token_ids"] = stop_token_ids
        
        payload = {
            "prompt_token_ids": prefill_ids,
            "sampling_params": params
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
            
            output = data["outputs"][0]
            output_tokens = output.get("token_ids", [])
            
            # Parse completion tokens
            entries = self.harmony_encoding.parse_messages_from_completion_tokens(
                output_tokens, Role.ASSISTANT
            )
            
            # Extract text from parsed messages
            text_parts = []
            for entry in entries:
                entry_dict = entry.to_dict()
                if "content" in entry_dict:
                    text_parts.append(entry_dict["content"])
            
            return CompletionResponse(
                text="\n".join(text_parts),
                model=self.model_id,
                provider=self.provider_type,
                usage={
                    "prompt_tokens": len(prefill_ids),
                    "completion_tokens": len(output_tokens),
                    "total_tokens": len(prefill_ids) + len(output_tokens)
                },
                finish_reason=output.get("finish_reason", "stop"),
                metadata={
                    "harmony_entries": [entry.to_dict() for entry in entries],
                    "request_id": data.get("request_id")
                }
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Harmony generation failed: {e}")
            raise RuntimeError(f"Failed to generate with harmony: {e}")
