# Public Model - GlobalHealthAtlas Inference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Public inference model for GlobalHealthAtlas. This model generates responses for public health questions with Chain of Thought reasoning.

## Model

- **Model**: [aerovane0/GlobalHealthAtlas_Public_Model](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Model)
- **Base Model**: Qwen3-8B

## Requirements

```bash
pip install vllm torch transformers
```

## Setup

1. Update the model paths in the script:

```python
base_model_path = "/path/to/Qwen3-8B"
lora_path = "/path/to/Lora/Qwen3-8B/sft100/"
```

2. Configure input/output paths:

```python
INPUT_JSON_PATH = "./data/input.json"
OUTPUT_JSON_PATH = "./data/output.json"
```

3. Configure LoRA settings (optional):

```python
ENABLE_LORA = True  # Set to False to use base model only
MAX_LORA_RANK = 64
```

4. (Optional) Merge base model with LoRA weights:

If you want to merge the base model with fine-tuned LoRA weights into a single model:

```bash
python merge_weights.py
```

Update the paths in `merge_weights.py` before running:
- `base_model_path`: Path to Qwen3-8B base model
- `lora_path`: Path to LoRA fine-tuned weights
- `output_path`: Path for the merged model

## Usage

### Batch Processing

Process multiple items in batches for faster inference:

```bash
python answer_batch.py
```

### Single Item Processing

Process items one by one (useful for debugging or smaller datasets):

```bash
python answer_single.py
```

## Input Format

Each item in the input JSON file should contain:

```json
{
  "id": "unique_id",
  "question": "Your question here"
}
```

## Output Format

The output includes the model's reasoning and final answer:

```json
{
  "id": "unique_id",
  "question": "Your question here",
  "llm_complexCOT": "Model's chain of thought reasoning",
  "llm_answer": "Model's final answer"
}
```

## Features

- **LoRA Support**: Optional LoRA adapter loading for fine-tuned models
- **Tensor Parallelism**: Multi-GPU support via vLLM
- **Cross-Lingual**: Handles both English and Chinese inputs
- **Chain of Thought**: Generates structured reasoning before final answer
- **Checkpoint Support**: Resume from last checkpoint if interrupted
- **Length Validation**: Skips overly long prompts

## Response Format

The model follows this output structure:

```


Answer: [Final answer]
```

For Single-Choice questions, the answer is a single letter (A, B, C, or D).
For Open-Ended questions, the answer is a direct conclusion or solution.

## Configuration Parameters

- `MAX_MODEL_LEN`: Maximum model length (default: 40960 tokens)
- `DTYPE`: Model data type (default: "bfloat16")
- `GPU_MEMORY_UTILIZATION`: GPU memory usage (default: 0.9)
- `ENABLE_LORA`: Enable LoRA adapter (default: True)
- `TEMPERATURE`: Sampling temperature (default: 0.1)
- `TOP_P`: Nucleus sampling parameter (default: 1.0)
- `MAX_TOKENS`: Maximum output tokens (default: 2500)
- `REPETITION_PENALTY`: Repetition penalty (default: 1.1)

## Notes

- The model responds in the same language as the input question
- Reasoning should be kept under 512 tokens
- Checkpoint files are saved as `{output}_checkpoint.json`
- For batch processing, default batch size is 5000 items
- For single processing, checkpoints are saved every 10 items
