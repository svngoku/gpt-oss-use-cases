"""
Example usage of OpenRouter provider with OpenAI gpt-oss models.

This example demonstrates how to use gpt-oss-20b and gpt-oss-120b models
through OpenRouter's unified API based on their documentation.

Reference: https://openrouter.ai/openai/gpt-oss-120b

Model Details:
- openai/gpt-oss-20b: 20B parameter model (context: 131,072 tokens)
  Cost: ~$0.01/M input tokens, ~$0.05/M output tokens
  
- openai/gpt-oss-120b: 117B MoE model (context: 131,072 tokens)
  Cost: $0.09/M input tokens, $0.45/M output tokens
  Features: 5.1B active parameters, MXFP4 quantization, runs on single H100

Prerequisites:
1. Sign up at https://openrouter.ai/
2. Get your API key from https://openrouter.ai/keys
3. Set environment variable: export OPENROUTER_API_KEY="your-key"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.providers.openrouter.provider import OpenRouterProvider, ToolFunction
from src.providers.base import GenerationConfig, Message


def basic_generation_example():
    """Basic text generation with OpenRouter."""
    print("=== Basic Generation with OpenRouter ===\n")
    
    # Initialize OpenRouter provider with gpt-oss-120b
    provider = OpenRouterProvider(
        model_id="openai/gpt-oss-120b",  # or "openai/gpt-oss-20b" for smaller model
        api_key=os.getenv("OPENROUTER_API_KEY"),  # Or pass directly
        site_url="https://your-site.com",  # Optional: for OpenRouter rankings
        app_name="GPT-OSS Examples",  # Optional: for OpenRouter rankings
    )
    
    # Generate text
    config = GenerationConfig(
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = provider.generate(
        "What is the meaning of life?",
        config=config
    )
    
    print(f"Response: {response.text}")
    print(f"Model used: {response.model}")
    print(f"Tokens: {response.usage}")
    print(f"Provider: {response.metadata.get('provider', 'Unknown')}\n")


def chat_example():
    """Chat-based generation with OpenRouter."""
    print("=== Chat with OpenRouter ===\n")
    
    provider = OpenRouterProvider(
        model_id="openai/gpt-oss-20b",  # Using smaller model for cost efficiency
        site_url="https://your-site.com",
        app_name="GPT-OSS Chat",
    )
    
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Explain MXFP4 quantization in simple terms."),
    ]
    
    config = GenerationConfig(
        max_tokens=300,
        temperature=0.7,
    )
    
    response = provider.chat(messages, config=config)
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg.role}: {msg.content}")
    print(f"\nAssistant: {response.text}\n")


def openai_sdk_example():
    """Using OpenRouter with OpenAI SDK compatibility."""
    print("=== OpenAI SDK Compatibility ===\n")
    
    # Initialize with OpenAI SDK support
    provider = OpenRouterProvider(
        model_id="openai/gpt-oss-120b",
        use_openai_sdk=True,  # Enable OpenAI SDK
        site_url="https://your-site.com",
        app_name="GPT-OSS SDK Example",
    )
    
    # Generate using OpenAI SDK format
    response = provider.generate_with_openai_sdk(
        "Write a haiku about artificial intelligence",
        config=GenerationConfig(max_tokens=100, temperature=0.9)
    )
    
    print(f"Haiku:\n{response.text}\n")


def tool_calling_example():
    """Tool calling with OpenRouter and gpt-oss models."""
    print("=== Tool Calling Example ===\n")
    
    provider = OpenRouterProvider(
        model_id="openai/gpt-oss-120b",
        use_openai_sdk=True,  # Better tool support with SDK
        site_url="https://your-site.com",
        app_name="GPT-OSS Tools",
    )
    
    # Define tools
    get_weather_tool = ToolFunction(
        name="get_weather",
        description="Get current weather in a given city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city"]
        }
    )
    
    search_tool = ToolFunction(
        name="web_search",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    )
    
    messages = [
        Message(role="user", content="What's the weather in Tokyo and search for best sushi restaurants there?"),
    ]
    
    # Call with tools
    response = provider.chat_with_tools(
        messages=messages,
        tools=[get_weather_tool, search_tool],
        config=GenerationConfig(max_tokens=300, temperature=0.7),
        tool_choice="auto"  # Let model decide which tools to use
    )
    
    print(f"Response: {response.text}")
    
    # Check for tool calls
    if response.metadata and response.metadata.get("tool_calls"):
        print(f"\nTool calls detected:")
        for tool_call in response.metadata["tool_calls"]:
            print(f"  - Function: {tool_call['name']}")
            print(f"    Arguments: {tool_call['arguments']}")
    print()


def model_fallback_example():
    """Automatic model fallback with OpenRouter."""
    print("=== Model Fallback Example ===\n")
    
    provider = OpenRouterProvider(
        model_id="openai/gpt-oss-120b",  # Primary model
        enable_fallback=True,  # Enable automatic fallback
    )
    
    # Use fallback with multiple models
    models = [
        "openai/gpt-oss-120b",  # Try this first
        "openai/gpt-oss-20b",   # Fall back to smaller model
        "openai/gpt-4o",        # Fall back to GPT-4o if needed
    ]
    
    response = provider.generate_with_fallback(
        prompt="Explain quantum computing in one paragraph",
        models=models,
        config=GenerationConfig(max_tokens=200, temperature=0.7)
    )
    
    print(f"Response: {response.text}")
    print(f"Model used: {response.model}")
    print(f"Models tried: {response.metadata.get('models_tried', [])}\n")


def streaming_example():
    """Streaming generation with OpenRouter."""
    print("=== Streaming Example ===\n")
    
    provider = OpenRouterProvider(
        model_id="openai/gpt-oss-20b",  # Using smaller model for faster streaming
        site_url="https://your-site.com",
        app_name="GPT-OSS Streaming",
    )
    
    messages = [
        Message(role="user", content="Count from 1 to 5 with brief explanations."),
    ]
    
    config = GenerationConfig(max_tokens=200, temperature=0.7)
    
    print("Streaming response:")
    print("User: Count from 1 to 5 with brief explanations.")
    print("Assistant: ", end="", flush=True)
    
    # Stream the response
    for chunk in provider.stream_chat(messages, config=config):
        print(chunk, end="", flush=True)
    
    print("\n")


def model_comparison_example():
    """Compare gpt-oss-20b vs gpt-oss-120b."""
    print("=== Model Comparison ===\n")
    
    prompt = "Solve this step by step: If a train travels 120 km in 2 hours, what is its average speed?"
    
    # Test with gpt-oss-20b
    provider_20b = OpenRouterProvider(
        model_id="openai/gpt-oss-20b",
        site_url="https://your-site.com",
        app_name="Model Comparison",
    )
    
    print("Testing gpt-oss-20b:")
    response_20b = provider_20b.generate(
        prompt,
        config=GenerationConfig(max_tokens=150, temperature=0.7)
    )
    print(f"Response: {response_20b.text}")
    print(f"Tokens used: {response_20b.usage}\n")
    
    # Test with gpt-oss-120b
    provider_120b = OpenRouterProvider(
        model_id="openai/gpt-oss-120b",
        site_url="https://your-site.com",
        app_name="Model Comparison",
    )
    
    print("Testing gpt-oss-120b:")
    response_120b = provider_120b.generate(
        prompt,
        config=GenerationConfig(max_tokens=150, temperature=0.7)
    )
    print(f"Response: {response_120b.text}")
    print(f"Tokens used: {response_120b.usage}\n")
    
    # Cost comparison (approximate)
    if response_20b.usage and response_120b.usage:
        cost_20b = (response_20b.usage.get("prompt_tokens", 0) * 0.01 + 
                   response_20b.usage.get("completion_tokens", 0) * 0.05) / 1_000_000
        cost_120b = (response_120b.usage.get("prompt_tokens", 0) * 0.09 + 
                    response_120b.usage.get("completion_tokens", 0) * 0.45) / 1_000_000
        
        print(f"Estimated costs:")
        print(f"  gpt-oss-20b: ${cost_20b:.6f}")
        print(f"  gpt-oss-120b: ${cost_120b:.6f}")
        print(f"  Difference: ${abs(cost_120b - cost_20b):.6f}\n")


def check_account_info():
    """Check OpenRouter account and available models."""
    print("=== Account Information ===\n")
    
    provider = OpenRouterProvider(
        model_id="openai/gpt-oss-20b",
    )
    
    # Check credits
    credits = provider.check_credits()
    if credits:
        print(f"Account info: {credits}\n")
    
    # List available models
    models = provider.list_models()
    
    # Filter for gpt-oss models
    gpt_oss_models = [m for m in models if "gpt-oss" in m]
    if gpt_oss_models:
        print(f"Available GPT-OSS models: {gpt_oss_models}")
    
    # Get model info
    for model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        info = provider.get_model_info(model)
        if info:
            print(f"\n{model} info:")
            print(f"  Context length: {info.get('context_length', 'N/A')}")
            print(f"  Pricing: {info.get('pricing', 'N/A')}")
    print()


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("OpenRouter GPT-OSS Examples")
    print("Based on: https://openrouter.ai/openai/gpt-oss-120b")
    print("="*60 + "\n")
    
    print("Prerequisites:")
    print("1. Sign up at https://openrouter.ai/")
    print("2. Get API key from https://openrouter.ai/keys")
    print("3. Set OPENROUTER_API_KEY environment variable")
    print("\nModel Information:")
    print("- gpt-oss-20b: 20B params, ~$0.01/M input, ~$0.05/M output")
    print("- gpt-oss-120b: 117B MoE, $0.09/M input, $0.45/M output")
    print("- Both: 131,072 token context, MXFP4 quantization")
    print("\n" + "="*60 + "\n")
    
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY not set!")
        print("Please set your API key:")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        return
    
    try:
        # Run examples (comment out ones you don't want to run)
        # basic_generation_example()
        # chat_example()
        # openai_sdk_example()
        # tool_calling_example()
        # model_fallback_example()
        # streaming_example()
        # model_comparison_example()
        check_account_info()
        
        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Check API key is valid")
        print("2. Check you have credits at https://openrouter.ai/")
        print("3. For OpenAI SDK: pip install openai")


if __name__ == "__main__":
    main()
