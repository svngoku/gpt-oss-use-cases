"""Tests for model loader functionality."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.loader import ModelLoader


def test_model_loader_initialization():
    """Test ModelLoader initialization."""
    loader = ModelLoader(
        model_id="gpt2",
        device="cpu",
        torch_dtype="float32",
    )
    
    assert loader.model_id == "gpt2"
    assert loader.device == "cpu"
    assert loader.torch_dtype == "float32"
    assert loader.model is None
    assert loader.tokenizer is None


def test_invalid_quantization_config():
    """Test that invalid quantization raises error."""
    loader = ModelLoader(model_id="gpt2")
    
    with pytest.raises(ValueError, match="Cannot load in both 8-bit and 4-bit"):
        loader.load_quantized(load_in_8bit=True, load_in_4bit=True)


if __name__ == "__main__":
    # Run basic tests
    test_model_loader_initialization()
    print("✓ Model loader initialization test passed")
    
    test_invalid_quantization_config()
    print("✓ Invalid quantization config test passed")
    
    print("\nAll tests passed!")
