"""
Example usage of Ollama provider with OpenAI gpt-oss models.

This example demonstrates how to run gpt-oss-20b and gpt-oss-120b locally
using Ollama based on OpenAI's cookbook recommendations.

Reference: https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama

Prerequisites:
1. Install Ollama: https://ollama.ai/
2. Pull the model: 
   - ollama pull gpt-oss:20b  (for 20B model)
   - ollama pull gpt-oss:120b (for 120B model)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.providers.ollama.provider import OllamaProvider, ToolFunction
from src.providers.base import GenerationConfig, Message


def basic_generation_example():
    """Basic text generation with Ollama."""
    print("=== Basic Generation with Ollama ===\n")
    
    # Initialize Ollama provider with gpt-oss-20b
    provider = OllamaProvider(
        model_id="gpt-oss:20b",  # or gpt-oss:120b for larger model
        base_url="http://localhost:11434",
        auto_pull=True,  # Automatically pull model if not available
    )
    
    # Generate text
    config = GenerationConfig(
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = provider.generate(
        "Explain what MXFP4 quantization is in simple terms:",
        config=config
    )
    
    print(f"Response: {response.text}")
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.usage}\n")


def chat_example():
    """Chat-based generation with Ollama."""
    print("=== Chat with Ollama ===\n")
    
    provider = OllamaProvider(
        model_id="gpt-oss:20b",
        base_url="http://localhost:11434",
    )
    
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What's the weather in Berlin right now?"),
    ]
    
    config = GenerationConfig(max_tokens=150, temperature=0.7)
    
    response = provider.chat(messages, config=config)
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg.role}: {msg.content}")
    print(f"\nAssistant: {response.text}\n")


def openai_sdk_example():
    """Using Ollama with OpenAI SDK compatibility."""
    print("=== OpenAI SDK Compatibility ===\n")
    
    # Initialize with OpenAI SDK support
    provider = OllamaProvider(
        model_id="gpt-oss:20b",
        base_url="http://localhost:11434",
        use_openai_sdk=True,  # Enable OpenAI SDK
    )
    
    # Generate using OpenAI SDK format
    response = provider.generate_with_openai_sdk(
        "Write a haiku about machine learning",
        config=GenerationConfig(max_tokens=100, temperature=0.9)
    )
    
    print(f"Haiku:\n{response.text}\n")


def tool_calling_example():
    """Tool calling (function calling) with Ollama."""
    print("=== Tool Calling Example ===\n")
    
    provider = OllamaProvider(
        model_id="gpt-oss:20b",
        base_url="http://localhost:11434",
        use_openai_sdk=True,  # Required for proper tool calling
    )
    
    # Define a tool/function
    get_weather_tool = ToolFunction(
        name="get_weather",
        description="Get current weather in a given city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"]
        }
    )
    
    messages = [
        Message(role="user", content="What's the weather in Tokyo right now?"),
    ]
    
    # Call with tools
    response = provider.chat_with_tools(
        messages=messages,
        tools=[get_weather_tool],
        config=GenerationConfig(max_tokens=200, temperature=0.7)
    )
    
    print(f"Response: {response.text}")
    
    # Check if tool was called
    if response.metadata and response.metadata.get("tool_calls"):
        print(f"\nTool calls detected:")
        for tool_call in response.metadata["tool_calls"]:
            print(f"  - Function: {tool_call['name']}")
            print(f"    Arguments: {tool_call['arguments']}")
    print()


def streaming_example():
    """Streaming generation with Ollama."""
    print("=== Streaming Example ===\n")
    
    provider = OllamaProvider(
        model_id="gpt-oss:20b",
        base_url="http://localhost:11434",
    )
    
    messages = [
        Message(role="user", content="Count from 1 to 5 with explanations."),
    ]
    
    config = GenerationConfig(max_tokens=200, temperature=0.7)
    
    print("Streaming response:")
    print("User: Count from 1 to 5 with explanations.")
    print("Assistant: ", end="", flush=True)
    
    # Stream the response
    for chunk in provider.stream_chat(messages, config=config):
        print(chunk, end="", flush=True)
    
    print("\n")


def model_management_example():
    """Model management operations with Ollama."""
    print("=== Model Management ===\n")
    
    provider = OllamaProvider(
        model_id="gpt-oss:20b",
        base_url="http://localhost:11434",
    )
    
    # List available models
    models = provider.list_models()
    print(f"Available models: {models}")
    
    # Check if gpt-oss models are available
    gpt_oss_models = [m for m in models if "gpt-oss" in m]
    if gpt_oss_models:
        print(f"GPT-OSS models found: {gpt_oss_models}")
    else:
        print("No GPT-OSS models found. Pulling gpt-oss:20b...")
        success = provider.pull_model("gpt-oss:20b")
        if success:
            print("Successfully pulled gpt-oss:20b")
        else:
            print("Failed to pull model. Please run: ollama pull gpt-oss:20b")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Ollama GPT-OSS Examples")
    print("Based on: https://cookbook.openai.com/articles/gpt-oss/run-locally-ollama")
    print("="*60 + "\n")
    
    print("Prerequisites:")
    print("1. Install Ollama from https://ollama.ai/")
    print("2. Start Ollama service")
    print("3. Pull model: ollama pull gpt-oss:20b")
    print("\n" + "="*60 + "\n")
    
    try:
        # Run examples (comment out ones you don't want to run)
        # basic_generation_example()
        # chat_example()
        # openai_sdk_example()
        # tool_calling_example()
        # streaming_example()
        model_management_example()
        
        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Ollama is running: Check http://localhost:11434")
        print("2. Pull the model: ollama pull gpt-oss:20b")
        print("3. For OpenAI SDK examples: pip install openai")
        print("4. Check system requirements:")
        print("   - gpt-oss:20b needs ≥16GB VRAM/RAM")
        print("   - gpt-oss:120b needs ≥60GB VRAM/RAM")


if __name__ == "__main__":
    main()
