# Public Model - GlobalHealthAtlas 推理

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

GlobalHealthAtlas 公共推理模型。该模型为公共卫生问题生成响应，包含思维链推理。

## 模型

- **模型**: [aerovane0/GlobalHealthAtlas_Public_Model](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Model)
- **基础模型**: Qwen3-8B

## 环境要求

```bash
pip install vllm torch transformers
```

## 配置

1. 更新脚本中的模型路径：

```python
base_model_path = "/path/to/Qwen3-8B"
lora_path = "/path/to/Lora/Qwen3-8B/sft100/"
```

2. 配置输入/输出路径：

```python
INPUT_JSON_PATH = "./data/input.json"
OUTPUT_JSON_PATH = "./data/output.json"
```

3. 配置 LoRA 设置（可选）：

```python
ENABLE_LORA = True  # 设置为 False 仅使用基础模型
MAX_LORA_RANK = 64
```

4. （可选）合并基础模型和 LoRA 权重：

如果要将基础模型与微调的 LoRA 权重合并为单个模型：

```bash
python merge_weights.py
```

运行前请更新 `merge_weights.py` 中的路径：
- `base_model_path`: Qwen3-8B 基础模型路径
- `lora_path`: LoRA 微调权重路径
- `output_path`: 合并后模型的输出路径

## 使用方法

### 批量处理

批量处理多条数据以实现更快推理：

```bash
python answer_batch.py
```

### 单条处理

逐条处理数据（适用于调试或小数据集）：

```bash
python answer_single.py
```

## 输入格式

输入 JSON 文件中的每条数据应包含：

```json
{
  "id": "unique_id",
  "question": "您的问题"
}
```

## 输出格式

输出包含模型的推理和最终答案：

```json
{
  "id": "unique_id",
  "question": "您的问题",
  "llm_complexCOT": "模型的思维链推理",
  "llm_answer": "模型的最终答案"
}
```

## 功能特性

- **LoRA 支持**: 可选 LoRA 适配器加载，用于微调模型
- **张量并行**: 通过 vLLM 支持多 GPU
- **跨语言**: 支持中英文输入
- **思维链**: 在最终答案之前生成结构化推理
- **断点续传**: 中断后可从上次检查点继续
- **长度验证**: 自动跳过过长的提示

## 响应格式

模型遵循以下输出结构：

```
<think>
[推理过程]
</think>

Answer: [最终答案]
```

对于单选题，答案是单个字母（A、B、C 或 D）。
对于开放式问题，答案是直接的结论或解决方案。

## 配置参数

- `MAX_MODEL_LEN`: 最大模型长度（默认：40960 令牌）
- `DTYPE`: 模型数据类型（默认："bfloat16"）
- `GPU_MEMORY_UTILIZATION`: GPU 内存使用率（默认：0.9）
- `ENABLE_LORA`: 启用 LoRA 适配器（默认：True）
- `TEMPERATURE`: 采样温度（默认：0.1）
- `TOP_P`: 核采样参数（默认：1.0）
- `MAX_TOKENS`: 最大输出令牌数（默认：2500）
- `REPETITION_PENALTY`: 重复惩罚（默认：1.1）

## 注意事项

- 模型使用与输入问题相同的语言回答
- 推理过程应保持在 512 令牌以下
- 检查点文件保存为 `{output}_checkpoint.json`
- 批处理模式下，默认批次大小为 5000 条数据
- 单条处理模式下，每处理 10 条数据保存一次检查点
