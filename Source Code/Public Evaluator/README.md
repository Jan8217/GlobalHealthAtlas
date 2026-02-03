# Scorer Usage Guide

This scorer evaluates model-generated responses against standard reference answers using a strict 1-10 scale across six dimensions.

## Models

- **Base Model**: Qwen3-8B-Merged
- **Fine-tuned Weights**: [aerovane0/GlobalHealthAtlas_Public_Evaluator](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Evaluator)

## Requirements

```bash
pip install vllm torch transformers
```

## Setup

1. Update the model paths in the script:

```python
model_path = "/path/to/your/Qwen3-8B-Merged/"
base_model_path = "/path/to/your/Qwen3-8B-Merged/"
```

2. Configure input/output file pairs:

```python
INPUT_OUTPUT_PAIRS = [
    {"input": "./path/to/input.json", "output": "./path/to/output.json"}
]
```

## Usage

### Batch Processing

Process multiple items in batches for faster inference:

```bash
python scorer_batch.py
```

### Single Item Processing

Process items one by one (useful for debugging or smaller datasets):

```bash
python scorer_single.py
```

## Input Format

Each item in the input JSON file should contain:

```json
{
  "id": "unique_id",
  "domain": "public_health",
  "label": "Question-Answer",
  "question": "Your question here",
  "answer": "Reference answer",
  "complexCOT": "Reference reasoning",
  "llm_complexCOT": "Model's chain of thought",
  "llm_answer": "Model's final answer"
}
```

## Output Format

The output includes scores and descriptions for six dimensions:

```json
{
  "id": "unique_id",
  "scores": {
    "Accuracy": {
      "score": 10,
      "description": "Perfect match with reference"
    },
    "Reasoning": {
      "score": 8,
      "description": "Strong logical flow with minor gaps"
    },
    "Completeness": {
      "score": 9,
      "description": "Comprehensive coverage"
    },
    "Consensus Alignment": {
      "score": 10,
      "description": "Perfect alignment with scientific consensus"
    },
    "Terminology Norms": {
      "score": 8,
      "description": "Professional terminology usage"
    },
    "Insightfulness": {
      "score": 7,
      "description": "Good explanation of underlying mechanisms"
    }
  }
}
```

## Scoring Dimensions

1. **Accuracy**: Factual correctness compared to reference
2. **Reasoning**: Logical validity of the reasoning process
3. **Completeness**: Coverage of all key information points
4. **Consensus Alignment**: Adherence to scientific consensus
5. **Terminology Norms**: Professional terminology usage
6. **Insightfulness**: Depth of mechanism explanation

## Features

- **Checkpoint Support**: Resume from last checkpoint if interrupted
- **Tensor Parallelism**: Multi-GPU support via vLLM
- **Cross-Lingual**: Handles both English and Chinese inputs
- **Length Validation**: Skips overly long prompts

## Notes

- For Single-Choice tasks, Accuracy is binary (0 or 10)
- If no COT is provided, Reasoning score is 0
- All scores are integers from 0 to 10
- Checkpoint files are saved as `{output}_checkpoint.json`
