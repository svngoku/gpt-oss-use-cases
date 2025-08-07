"""Inference pipeline for GPT models including OpenAI gpt-oss support."""

import logging
from typing import List, Dict, Any, Optional, Union, Generator
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from dataclasses import dataclass
import warnings

# Optional imports for enhanced gpt-oss support
try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        SystemContent,
        DeveloperContent,
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    
try:
    from transformers.distributed import DistributedConfig
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for model inference."""
    
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[Union[int, List[int]]] = None


class InferencePipeline:
    """Pipeline for running inference with GPT models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        """Initialize the inference pipeline.
        
        Args:
            model: Pre-trained model for inference
            tokenizer: Tokenizer for the model
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        # Ensure model is in eval mode
        self.model.eval()
        
    def generate(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
        **kwargs
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt text
            config: Generation configuration
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        if config is None:
            config = InferenceConfig()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Setup generation config
        gen_config = GenerationConfig(
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            do_sample=config.do_sample,
            num_beams=config.num_beams,
            early_stopping=config.early_stopping,
            pad_token_id=config.pad_token_id or self.tokenizer.pad_token_id,
            eos_token_id=config.eos_token_id or self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        return generated_text
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        config: Optional[InferenceConfig] = None,
        add_generation_prompt: bool = True,
        **kwargs
    ) -> str:
        """Generate response in a chat format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            config: Generation configuration
            add_generation_prompt: Whether to add generation prompt
            **kwargs: Additional generation arguments
            
        Returns:
            Generated response
        """
        # Apply chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        else:
            # Fallback to simple formatting
            prompt = self._format_messages(messages)
        
        return self.generate(prompt, config, **kwargs)
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Formatted prompt
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(content)
        
        return "\n\n".join(formatted) + "\n\nAssistant:"
    
    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[InferenceConfig] = None,
        batch_size: int = 1,
        **kwargs
    ) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            config: Generation configuration
            batch_size: Batch size for generation
            **kwargs: Additional generation arguments
            
        Returns:
            List of generated texts
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Process batch
            for prompt in batch_prompts:
                result = self.generate(prompt, config, **kwargs)
                results.append(result)
        
        return results
    
    def stream_generate(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
        **kwargs
    ):
        """Stream generated text token by token.
        
        Args:
            prompt: Input prompt text
            config: Generation configuration
            **kwargs: Additional generation arguments
            
        Yields:
            Generated tokens
        """
        if config is None:
            config = InferenceConfig()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Setup for streaming
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate token by token
        past_key_values = None
        generated_tokens = []
        
        for _ in range(config.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Sample next token
            if config.do_sample:
                probs = torch.softmax(logits / config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Yield token
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            yield token_text
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Update inputs for next iteration
            input_ids = next_token
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=self.device)
            ], dim=1)
            generated_tokens.append(next_token.item())


class GPTOSSInferencePipeline:
    """Specialized pipeline for OpenAI gpt-oss models (gpt-oss-20b and gpt-oss-120b).
    
    This pipeline implements the recommended approaches from OpenAI's cookbook for
    running gpt-oss models with Transformers, including MXFP4 quantization support,
    harmony format, and multi-GPU capabilities.
    """
    
    SUPPORTED_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    
    def __init__(
        self,
        model_name: str = "openai/gpt-oss-20b",
        device_map: Optional[Union[str, Dict]] = "auto",
        torch_dtype: Optional[str] = "auto",
        use_harmony: bool = False,
        enable_expert_parallel: bool = False,
        enable_tensor_parallel: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        """Initialize the gpt-oss inference pipeline.
        
        Args:
            model_name: Model name from HuggingFace hub (openai/gpt-oss-20b or openai/gpt-oss-120b)
            device_map: Device mapping strategy ("auto" for automatic placement)
            torch_dtype: Data type for model weights ("auto" for MXFP4 when available)
            use_harmony: Whether to use openai-harmony for prompt formatting
            enable_expert_parallel: Enable expert parallelism for multi-GPU
            enable_tensor_parallel: Enable tensor parallelism for multi-GPU
            attn_implementation: Custom attention implementation (e.g., "flash-attn3")
        """
        if model_name not in self.SUPPORTED_MODELS:
            warnings.warn(
                f"Model {model_name} is not officially supported. "
                f"Supported models: {self.SUPPORTED_MODELS}"
            )
        
        self.model_name = model_name
        self.use_harmony = use_harmony and HARMONY_AVAILABLE
        
        # Setup device map for multi-GPU if needed
        if isinstance(device_map, str) and device_map == "auto":
            device_map_config = {"device_map": "auto"}
        else:
            device_map_config = {}
            
        # Add distributed config if available and requested
        if DISTRIBUTED_AVAILABLE and (enable_expert_parallel or enable_tensor_parallel):
            distributed_config = {}
            if enable_expert_parallel:
                distributed_config["enable_expert_parallel"] = 1
            device_map_config["distributed_config"] = DistributedConfig(**distributed_config)
            
        if enable_tensor_parallel:
            device_map_config["tp_plan"] = "auto"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimal settings
        model_kwargs = {
            "torch_dtype": torch_dtype,
            **device_map_config,
        }
        
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        logger.info(f"Loading model {model_name} with config: {model_kwargs}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Initialize harmony encoding if available
        if self.use_harmony:
            self.harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        else:
            self.harmony_encoding = None
            
        # Setup pipeline for quick inference
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device_map if isinstance(device_map, str) else None,
        )
        
        logger.info(f"GPT-OSS pipeline initialized for {model_name}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """Generate text using the high-level pipeline API.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text
        """
        result = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )
        
        return result[0]["generated_text"]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        add_generation_prompt: bool = True,
        **kwargs
    ) -> str:
        """Generate response using chat format with proper template.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            add_generation_prompt: Whether to add generation prompt
            **kwargs: Additional generation arguments
            
        Returns:
            Generated response
        """
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
            return_dict=True,
        )
        
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        # Decode only the generated tokens
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def chat_with_harmony(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response using harmony format for enhanced tool calling support.
        
        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Returns:
            Dictionary containing parsed response with potential tool calls
        """
        if not self.use_harmony or not self.harmony_encoding:
            raise RuntimeError(
                "Harmony format not available. Install openai-harmony: pip install openai-harmony"
            )
        
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
                    Message.from_role_and_content(role, SystemContent.new())
                )
            elif role == Role.DEVELOPER:
                harmony_messages.append(
                    Message.from_role_and_content(
                        role, 
                        DeveloperContent.new().with_instructions(content)
                    )
                )
            else:
                harmony_messages.append(
                    Message.from_role_and_content(role, content)
                )
        
        # Create conversation
        convo = Conversation.from_messages(harmony_messages)
        
        # Render prompt
        prefill_ids = self.harmony_encoding.render_conversation_for_completion(
            convo, Role.ASSISTANT
        )
        stop_token_ids = self.harmony_encoding.stop_tokens_for_assistant_action()
        
        # Generate
        input_tensor = torch.tensor([prefill_ids])
        if hasattr(self.model, "device"):
            input_tensor = input_tensor.to(self.model.device)
        
        outputs = self.model.generate(
            input_ids=input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token_id=stop_token_ids,
            **kwargs
        )
        
        # Parse completion
        completion_ids = outputs[0][len(prefill_ids):].tolist()
        entries = self.harmony_encoding.parse_messages_from_completion_tokens(
            completion_ids, Role.ASSISTANT
        )
        
        # Convert to dictionary format
        result = {
            "messages": [msg.to_dict() for msg in entries],
            "raw_completion": self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        }
        
        return result
    
    def advanced_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        return_full_text: bool = False,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Advanced generation with more control using the .generate() method.
        
        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            return_full_text: Whether to return the full text including prompt
            **kwargs: Additional generation arguments
            
        Returns:
            Generated text or dictionary with additional information
        """
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        
        # Generate with detailed config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            **kwargs
        )
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode response
        if return_full_text:
            full_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            generated_text = full_text
        else:
            generated_text = self.tokenizer.decode(
                outputs.sequences[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )
        
        return {
            "generated_text": generated_text,
            "num_tokens": outputs.sequences[0].shape[0] - inputs["input_ids"].shape[-1],
            "finish_reason": "eos" if outputs.sequences[0][-1] == self.tokenizer.eos_token_id else "length"
        }
    
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream generation token by token.
        
        Args:
            messages: List of message dictionaries
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation arguments
            
        Yields:
            Generated text chunks
        """
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        
        # Setup for streaming
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        past_key_values = None
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Decode and yield token
            token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
            yield token_text
            
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Update inputs for next iteration
            input_ids = next_token
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((1, 1), device=attention_mask.device)
            ], dim=1)
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "openai/gpt-oss-20b",
        **kwargs
    ) -> "GPTOSSInferencePipeline":
        """Create pipeline from pretrained model.
        
        Args:
            model_name: Model name from HuggingFace hub
            **kwargs: Additional arguments for pipeline initialization
            
        Returns:
            Initialized GPTOSSInferencePipeline
        """
        return cls(model_name=model_name, **kwargs)
