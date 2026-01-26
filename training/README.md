# Training Module

LoRA fine-tuning scripts for GlobalHealthAtlas evaluator models.

## Files

- `train_lora.sh` - LoRA fine-tuning script using LLaMA-Factory

## Model Configuration

The training uses:
- **Base Model**: Qwen3-8B
- **Fine-tuning Method**: LoRA (Rank=16)
- **Target Modules**: Attention layers (q_proj, k_proj, v_proj, o_proj)
- **Precision**: bfloat16

## Training Parameters

| Parameter | Value |
|-----------|--------|
| Learning Rate | 5e-5 |
| Batch Size | 2 (per device) |
| Gradient Accumulation | 8 |
| Effective Batch Size | 16 |
| Epochs | 2 |
| Warmup Ratio | 0.1 |
| Max Sequence Length | 2048 |

## Usage

```bash
# Install LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

# Run training
cd GlobalHealthAtlas/training
bash train_lora.sh
```

## Dataset

- **Dataset**: distill_psychology-10k-r1
- **Location**: /root/LLaMA-Factory/data
- **Max Samples**: 5400

## Output

Trained LoRA adapters are saved to `/root/autodl-tmp/Lora-LLM/saves/Qwen3-8B/lora/sft2/`
