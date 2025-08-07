"""
Example usage of GPTOSSInferencePipeline for OpenAI gpt-oss models.

This example demonstrates various ways to use the gpt-oss-20b and gpt-oss-120b models
with the specialized inference pipeline based on OpenAI's cookbook recommendations.

Reference: https://cookbook.openai.com/articles/gpt-oss/run-transformers
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.inference import GPTOSSInferencePipeline


def basic_generation_example():
    """Basic text generation using the pipeline API."""
    print("=== Basic Generation Example ===\n")
    
    # Initialize pipeline with gpt-oss-20b (smaller model)
    pipeline = GPTOSSInferencePipeline(
        model_name="openai/gpt-oss-20b",
        device_map="auto",  # Automatically place on available GPUs
        torch_dtype="auto",  # Use MXFP4 quantization when available
    )
    
    # Simple generation
    prompt = "Explain what MXFP4 quantization is in simple terms:"
    response = pipeline.generate(
        prompt,
        max_new_tokens=200,
        temperature=1.0,
        top_p=0.9,
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}\n")


def chat_example():
    """Chat-based generation with proper template."""
    print("=== Chat Example ===\n")
    
    # Initialize pipeline
    pipeline = GPTOSSInferencePipeline(
        model_name="openai/gpt-oss-20b",
        device_map="auto",
    )
    
    # Chat messages
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are the benefits of using transformers for NLP?"},
    ]
    
    # Generate response
    response = pipeline.chat(
        messages,
        max_new_tokens=300,
        temperature=0.7,
    )
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print(f"Assistant: {response}\n")


def advanced_generation_example():
    """Advanced generation with detailed output."""
    print("=== Advanced Generation Example ===\n")
    
    # Initialize pipeline
    pipeline = GPTOSSInferencePipeline(
        model_name="openai/gpt-oss-20b",
        device_map="auto",
    )
    
    messages = [
        {"role": "user", "content": "Write a haiku about machine learning."},
    ]
    
    # Advanced generation with metadata
    result = pipeline.advanced_generate(
        messages,
        max_new_tokens=100,
        temperature=0.9,
        return_full_text=False,
    )
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    print(f"\nGenerated text: {result['generated_text']}")
    print(f"Number of tokens: {result['num_tokens']}")
    print(f"Finish reason: {result['finish_reason']}\n")


def streaming_example():
    """Streaming generation token by token."""
    print("=== Streaming Example ===\n")
    
    # Initialize pipeline
    pipeline = GPTOSSInferencePipeline(
        model_name="openai/gpt-oss-20b",
        device_map="auto",
    )
    
    messages = [
        {"role": "user", "content": "Count from 1 to 10 with explanations."},
    ]
    
    print("Streaming response:")
    print("User: Count from 1 to 10 with explanations.")
    print("Assistant: ", end="", flush=True)
    
    # Stream tokens
    for token in pipeline.stream_generate(
        messages,
        max_new_tokens=150,
        temperature=0.7,
    ):
        print(token, end="", flush=True)
    
    print("\n")


def harmony_format_example():
    """Example using harmony format for enhanced tool calling support."""
    print("=== Harmony Format Example (if openai-harmony is installed) ===\n")
    
    try:
        # Initialize pipeline with harmony support
        pipeline = GPTOSSInferencePipeline(
            model_name="openai/gpt-oss-20b",
            device_map="auto",
            use_harmony=True,  # Enable harmony format
        )
        
        messages = [
            {"role": "developer", "content": "You are a helpful assistant that can use tools."},
            {"role": "user", "content": "What's the weather like in San Francisco?"},
        ]
        
        # Generate with harmony format
        result = pipeline.chat_with_harmony(
            messages,
            max_new_tokens=200,
            temperature=0.7,
        )
        
        print("Messages:")
        for msg in messages:
            print(f"  {msg['role']}: {msg['content']}")
        print(f"\nParsed response:")
        for msg in result['messages']:
            print(f"  {msg}")
        print(f"\nRaw completion: {result['raw_completion']}\n")
        
    except RuntimeError as e:
        print(f"Harmony format not available: {e}")
        print("Install with: pip install openai-harmony\n")


def multi_gpu_example():
    """Example for multi-GPU inference with gpt-oss-120b."""
    print("=== Multi-GPU Example (for gpt-oss-120b) ===\n")
    
    # This example shows how to configure for multi-GPU setups
    # Requires multiple GPUs with sufficient VRAM
    
    print("Configuration for multi-GPU inference:")
    print("""
    pipeline = GPTOSSInferencePipeline(
        model_name="openai/gpt-oss-120b",
        device_map="auto",  # Automatic device placement
        torch_dtype="auto",  # MXFP4 quantization
        enable_expert_parallel=True,  # Expert parallelism
        enable_tensor_parallel=True,  # Tensor parallelism
        attn_implementation="flash-attn3",  # Fast attention
    )
    
    # Then run with torchrun for distributed setup:
    # torchrun --nproc_per_node=4 your_script.py
    """)
    
    print("\nNote: This requires ~60GB+ VRAM across multiple GPUs\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("GPT-OSS Inference Pipeline Examples")
    print("Based on: https://cookbook.openai.com/articles/gpt-oss/run-transformers")
    print("="*60 + "\n")
    
    # Comment out examples you don't want to run
    # Note: These require significant GPU memory
    
    try:
        # basic_generation_example()
        # chat_example()
        # advanced_generation_example()
        # streaming_example()
        # harmony_format_example()
        multi_gpu_example()  # Just shows configuration
        
        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nNote: These examples require:")
        print("- A GPU with sufficient VRAM (16GB+ for gpt-oss-20b)")
        print("- Installing dependencies: pip install transformers accelerate torch")
        print("- For MXFP4: pip install triton triton-kernels")
        print("- For harmony: pip install openai-harmony")


if __name__ == "__main__":
    main()
