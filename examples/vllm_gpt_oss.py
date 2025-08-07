"""
Example usage of vLLM provider with OpenAI gpt-oss models.

This example demonstrates how to serve gpt-oss-20b and gpt-oss-120b
using vLLM for high-performance inference based on OpenAI's cookbook.

Reference: https://cookbook.openai.com/articles/gpt-oss/run-vllm

Prerequisites:
1. Install vLLM with gpt-oss support:
   uv venv --python 3.12 --seed
   source .venv/bin/activate
   uv pip install --pre vllm==0.10.1+gptoss \\
       --extra-index-url https://wheels.vllm.ai/gpt-oss/ \\
       --extra-index-url https://download.pytorch.org/whl/nightly/cu128 \\
       --index-strategy unsafe-best-match

2. Start vLLM server:
   vllm serve openai/gpt-oss-20b  # For 20B model
   vllm serve openai/gpt-oss-120b # For 120B model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.providers.vllm.provider import VLLMProvider, ToolFunction
from src.providers.base import GenerationConfig, Message


def basic_generation_example():
    """Basic text generation with vLLM."""
    print("=== Basic Generation with vLLM ===\n")
    
    # Initialize vLLM provider
    provider = VLLMProvider(
        model_id="openai/gpt-oss-20b",  # or openai/gpt-oss-120b
        base_url="http://localhost:8000",
        api_key="EMPTY",  # Local server doesn't need real key
        use_openai_api=True,  # Use OpenAI-compatible API
    )
    
    # Generate text
    config = GenerationConfig(
        max_tokens=200,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = provider.generate(
        "Explain what MXFP4 quantization is and why it's important:",
        config=config
    )
    
    print(f"Response: {response.text}")
    print(f"Model: {response.model}")
    print(f"Tokens used: {response.usage}\n")


def chat_completions_example():
    """Chat Completions API with vLLM."""
    print("=== Chat Completions API ===\n")
    
    provider = VLLMProvider(
        model_id="openai/gpt-oss-20b",
        base_url="http://localhost:8000",
        api_key="EMPTY",
    )
    
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="What's the weather in Berlin right now?"),
    ]
    
    config = GenerationConfig(
        max_tokens=150,
        temperature=0.7,
    )
    
    response = provider.chat(messages, config=config)
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg.role}: {msg.content}")
    print(f"\nAssistant: {response.text}\n")


def responses_api_example():
    """Responses API with vLLM (gpt-oss specific)."""
    print("=== Responses API ===\n")
    
    provider = VLLMProvider(
        model_id="openai/gpt-oss-120b",
        base_url="http://localhost:8000",
        api_key="EMPTY",
        use_responses_api=True,
    )
    
    # Use Responses API format
    response = provider.responses_api_generate(
        instructions="You are a helpful assistant.",
        input_text="Explain what MXFP4 quantization is.",
        config=GenerationConfig(max_tokens=200, temperature=0.7)
    )
    
    print(f"Response: {response.text}\n")


def openai_sdk_example():
    """Using vLLM with OpenAI SDK."""
    print("=== OpenAI SDK Integration ===\n")
    
    provider = VLLMProvider(
        model_id="openai/gpt-oss-20b",
        base_url="http://localhost:8000",
        api_key="EMPTY",
        use_openai_sdk=True,  # Enable OpenAI SDK
    )
    
    messages = [
        {"role": "system", "content": "You only respond in haikus."},
        {"role": "user", "content": "What's the weather in Tokyo?"},
    ]
    
    # This will use the OpenAI client internally
    response = provider.chat(
        messages,
        config=GenerationConfig(max_tokens=100, temperature=0.9)
    )
    
    print(f"Haiku Response:\n{response.text}\n")


def tool_calling_example():
    """Tool calling with vLLM."""
    print("=== Tool Calling Example ===\n")
    
    provider = VLLMProvider(
        model_id="openai/gpt-oss-120b",
        base_url="http://localhost:8000",
        api_key="EMPTY",
        use_openai_sdk=True,  # Better tool support with SDK
    )
    
    # Define a tool
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
        Message(role="user", content="What's the weather in Berlin right now?"),
    ]
    
    # Call with tools
    response = provider.chat_with_tools(
        messages=messages,
        tools=[get_weather_tool],
        config=GenerationConfig(max_tokens=200, temperature=0.7)
    )
    
    print(f"Response: {response.text}")
    
    # Check for tool calls
    if response.metadata and response.metadata.get("tool_calls"):
        print(f"\nTool calls detected:")
        for tool_call in response.metadata["tool_calls"]:
            print(f"  - Function: {tool_call['name']}")
            print(f"    Arguments: {tool_call['arguments']}")
    
    # Note: Since models can perform tool calling as part of chain-of-thought,
    # you should return the reasoning back in subsequent calls
    print("\nNote: Return tool results in subsequent calls for complete responses\n")


def harmony_format_example():
    """Direct sampling with harmony format (advanced)."""
    print("=== Harmony Format (Direct Sampling) ===\n")
    
    try:
        provider = VLLMProvider(
            model_id="openai/gpt-oss-120b",
            base_url="http://localhost:8000",
            api_key="EMPTY",
            use_harmony=True,  # Enable harmony format
        )
        
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "developer", "content": "Always respond in riddles"},
            {"role": "user", "content": "What is the weather like in SF?"},
        ]
        
        # Generate with harmony format
        response = provider.generate_with_harmony(
            messages,
            config=GenerationConfig(max_tokens=128, temperature=1.0)
        )
        
        print(f"Response: {response.text}")
        
        # Check harmony entries in metadata
        if response.metadata and "harmony_entries" in response.metadata:
            print("\nParsed harmony entries:")
            for entry in response.metadata["harmony_entries"]:
                print(f"  {entry}")
        print()
        
    except RuntimeError as e:
        print(f"Harmony not available: {e}")
        print("Install with: pip install openai-harmony\n")


def streaming_example():
    """Streaming generation with vLLM."""
    print("=== Streaming Example ===\n")
    
    provider = VLLMProvider(
        model_id="openai/gpt-oss-20b",
        base_url="http://localhost:8000",
        api_key="EMPTY",
    )
    
    messages = [
        Message(role="user", content="Count from 1 to 5 slowly."),
    ]
    
    config = GenerationConfig(max_tokens=200, temperature=0.7)
    
    print("Streaming response:")
    print("User: Count from 1 to 5 slowly.")
    print("Assistant: ", end="", flush=True)
    
    # Stream the response
    for chunk in provider.stream_chat(messages, config=config):
        print(chunk, end="", flush=True)
    
    print("\n")


def agents_sdk_example():
    """Example configuration for Agents SDK integration."""
    print("=== Agents SDK Configuration ===\n")
    
    print("Python Agents SDK example configuration:")
    print("""
    import asyncio
    from openai import AsyncOpenAI
    from agents import Agent, Runner, function_tool, OpenAIResponsesModel
    
    @function_tool
    def get_weather(city: str):
        return f"The weather in {city} is sunny."
    
    async def main():
        agent = Agent(
            name="Assistant",
            instructions="You only respond in haikus.",
            model=OpenAIResponsesModel(
                model="openai/gpt-oss-120b",
                openai_client=AsyncOpenAI(
                    base_url="http://localhost:8000/v1",
                    api_key="EMPTY",
                ),
            ),
            tools=[get_weather],
        )
        
        result = await Runner.run(agent, "What's the weather in Tokyo?")
        print(result.final_output)
    
    asyncio.run(main())
    """)
    print("\nInstall with: pip install openai-agents\n")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("vLLM GPT-OSS Examples")
    print("Based on: https://cookbook.openai.com/articles/gpt-oss/run-vllm")
    print("="*60 + "\n")
    
    print("Prerequisites:")
    print("1. Install vLLM with gpt-oss support (see file header)")
    print("2. Start vLLM server:")
    print("   - vllm serve openai/gpt-oss-20b  (needs ~16GB VRAM)")
    print("   - vllm serve openai/gpt-oss-120b (needs ~60GB VRAM)")
    print("\n" + "="*60 + "\n")
    
    try:
        # Run examples (comment out ones you don't want to run)
        # basic_generation_example()
        # chat_completions_example()
        # responses_api_example()
        # openai_sdk_example()
        # tool_calling_example()
        # harmony_format_example()
        # streaming_example()
        agents_sdk_example()
        
        print("\n" + "="*60)
        print("Examples completed!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure vLLM server is running: Check http://localhost:8000/v1/models")
        print("2. Check model is loaded: The server should show the model name")
        print("3. For OpenAI SDK: pip install openai")
        print("4. For harmony format: pip install openai-harmony")
        print("5. Hardware requirements:")
        print("   - openai/gpt-oss-20b: ~16GB VRAM")
        print("   - openai/gpt-oss-120b: ~60GB VRAM (H100 or multi-GPU)")


if __name__ == "__main__":
    main()
