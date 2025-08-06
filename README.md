# gpt-oss-use-cases
## Overview
This project explores real-world use cases for open GPT models, featuring example scripts, workflows, and guidance for customization. Our goal is to support the community in adapting, training, and integrating generative AI for diverse needs.
## Getting Started
[
Getting Started content would be here
]
## Using `uv` for Fast Dependency Management
We recommend using [`uv`](https://github.com/astral-sh/uv) for lightning-fast Python dependency installation, script running, and environment management.
### Install `uv`
First, install `uv` if you haven't already:
```bash
pip install uv
```
### Install dependencies with `uv`
Instead of pip, use:
```bash
uv pip install -r requirements.txt
```
This is significantly faster and more reliable for large dependency trees.
### Run scripts with `uv`
To run Python scripts with the correct environment isolation:
```bash
uv venv
source .venv/bin/activate
python scripts/your_script.py
```
You can also use `uv` to manage virtual environments and direct script execution, similar to `pipx` or `venv`.

## Inference Example: Run gpt-oss-20b on Google Colab

Below is a step-by-step guide to run the gpt-oss-20b model on Google Colab, even on resource-constrained environments, thanks to MXFP4 quantization.

### 1. Setup Environment
Make sure you are using a recent version of PyTorch and CUDA, and install transformers from source for MXFP4 and triton compatibility:

```python
!pip install -q --upgrade torch
!pip install -q git+https://github.com/huggingface/transformers triton==3.4 git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
!pip uninstall -q torchvision torchaudio -y
```

### 2. Load the Model from Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda",
)
```

*If the model is public, authentication is optional but can help with access speed and reliability.*

### 3. Run Inference with Chat Template

You can use messages with roles for chat-like inference:

```python
messages = [
    {"role": "system", "content": "Always respond in riddles"},
    {"role": "user", "content": "What is the weather like in Madrid?"},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

generated = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))
```

*You can change the system prompt, the user question, or set other generation parameters as needed.*
