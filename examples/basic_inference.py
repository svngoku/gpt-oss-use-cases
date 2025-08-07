#!/usr/bin/env python3
"""Basic inference example with GPT OSS models."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import ModelLoader, InferencePipeline
from src.models.inference import InferenceConfig
from src.utils import setup_logging


def main():
    """Run basic inference example."""
    # Setup logging
    setup_logging(level="INFO")
    
    # Model configuration
    model_id = "openai/gpt-oss-20b"  # You can change this to any HuggingFace model
    
    # Initialize model loader
    loader = ModelLoader(
        model_id=model_id,
        device="auto",
        torch_dtype="auto",
    )
    
    # Load model and tokenizer
    print(f"Loading model: {model_id}")
    model = loader.load_model()
    tokenizer = loader.load_tokenizer()
    
    # Initialize inference pipeline
    pipeline = InferencePipeline(model, tokenizer)
    
    # Configure generation
    config = InferenceConfig(
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
    )
    
    # Example 1: Simple text generation
    print("\n" + "="*50)
    print("Example 1: Text Generation")
    print("="*50)
    
    prompt = "The future of artificial intelligence is"
    print(f"Prompt: {prompt}")
    print(f"Response: {pipeline.generate(prompt, config)}")
    
    # Example 2: Chat format
    print("\n" + "="*50)
    print("Example 2: Chat Format")
    print("="*50)
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What are the benefits of open-source AI models?"}
    ]
    
    print("Messages:")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")
    
    response = pipeline.chat(messages, config)
    print(f"Assistant: {response}")
    
    # Example 3: Batch generation
    print("\n" + "="*50)
    print("Example 3: Batch Generation")
    print("="*50)
    
    prompts = [
        "Python is a programming language that",
        "Machine learning models can be used to",
        "The most important aspect of AI safety is",
    ]
    
    results = pipeline.batch_generate(prompts, config)
    for prompt, result in zip(prompts, results):
        print(f"\nPrompt: {prompt}")
        print(f"Response: {result}")


if __name__ == "__main__":
    main()
