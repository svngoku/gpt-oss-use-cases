#!/usr/bin/env python3
"""Example demonstrating multiple provider backends."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.providers import (
    ProviderFactory,
    GenerationConfig,
    Message,
    ProviderType
)
from src.utils import setup_logging
import argparse


def test_provider(provider_name: str, model_id: str, **kwargs):
    """Test a specific provider.
    
    Args:
        provider_name: Name of the provider
        model_id: Model ID to use
        **kwargs: Provider-specific configuration
    """
    print(f"\n{'='*60}")
    print(f"Testing {provider_name.upper()} Provider")
    print(f"Model: {model_id}")
    print("="*60)
    
    try:
        # Create provider
        provider = ProviderFactory.create_from_string(
            provider_name,
            model_id,
            **kwargs
        )
        
        # Test simple generation
        print("\n--- Text Generation ---")
        config = GenerationConfig(
            max_tokens=50,
            temperature=0.7,
        )
        
        prompt = "The benefits of open-source AI models include"
        response = provider.generate(prompt, config)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response.text}")
        print(f"Provider: {response.provider.value}")
        print(f"Tokens used: {response.usage}")
        
        # Test chat
        print("\n--- Chat Completion ---")
        messages = [
            Message(role="system", content="You are a helpful AI assistant."),
            Message(role="user", content="What is machine learning in one sentence?")
        ]
        
        chat_response = provider.chat(messages, config)
        print(f"Assistant: {chat_response.text}")
        
        # Test streaming (if supported)
        print("\n--- Streaming Generation ---")
        print("Streaming response: ", end="", flush=True)
        for chunk in provider.stream_generate("Once upon a time", config):
            print(chunk, end="", flush=True)
        print()
        
        print(f"\n✅ {provider_name.upper()} provider test successful!")
        
    except Exception as e:
        print(f"\n❌ {provider_name.upper()} provider test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run multi-provider examples."""
    parser = argparse.ArgumentParser(description="Test multiple LLM providers")
    parser.add_argument(
        "--provider",
        choices=ProviderFactory.list_providers(),
        help="Specific provider to test"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all available providers"
    )
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    print("="*60)
    print("Multi-Provider LLM Example")
    print("="*60)
    print(f"\nAvailable providers: {ProviderFactory.list_providers()}")
    
    # Provider configurations
    provider_configs = {
        "ollama": {
            "model_id": "gpt-oss:20b",  # or any model you have in Ollama
            "base_url": "http://localhost:11434",
        },
        "vllm": {
            "model_id": "openai/gpt-oss-20b",  # or your vLLM model
            "base_url": "http://localhost:8000",
            "use_openai_api": True,
        },
        "openrouter": {
            "model_id": "openai/gpt-oss-120b",  # or any OpenRouter model
            # API key will be read from OPENROUTER_API_KEY env var
        },
        "pytorch": {
            "model_id": "gpt2",  # Small model for testing
            "device": "cpu",  # Use CPU for testing
        },
    }
    
    if args.all:
        # Test all providers
        for provider_name, config in provider_configs.items():
            if provider_name == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
                print(f"\n⚠️  Skipping OpenRouter (no API key found)")
                continue
            
            model_id = config.pop("model_id")
            test_provider(provider_name, model_id, **config)
    
    elif args.provider:
        # Test specific provider
        config = provider_configs.get(args.provider, {})
        if args.provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
            print("Error: OPENROUTER_API_KEY environment variable not set")
            return
        
        model_id = config.pop("model_id", "gpt2")
        test_provider(args.provider, model_id, **config)
    
    else:
        # Default: test with a simple PyTorch model
        print("\nRunning default example with PyTorch provider...")
        print("Use --provider <name> or --all to test other providers")
        
        test_provider("pytorch", "gpt2", device="cpu")
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)


if __name__ == "__main__":
    main()
