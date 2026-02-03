# 评分器使用指南

此评分器使用严格的 1-10 分制对模型生成的回答与标准参考答案进行评估，涵盖六个维度。

## 模型

- **基础模型**: Qwen3-8B-Merged
- **微调权重**: [aerovane0/GlobalHealthAtlas_Public_Evaluator](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Evaluator)

## 环境要求

```bash
pip install vllm torch transformers
```

## 配置

1. 更新脚本中的模型路径：

```python
model_path = "/path/to/your/Qwen3-8B-Merged/"
base_model_path = "/path/to/your/Qwen3-8B-Merged/"
```

2. 配置输入/输出文件对：

```python
INPUT_OUTPUT_PAIRS = [
    {"input": "./path/to/input.json", "output": "./path/to/output.json"}
]
```

## 使用方法

### 批量处理

批量处理多条数据以实现更快推理：

```bash
python scorer_batch.py
```

### 单条处理

逐条处理数据（适用于调试或小数据集）：

```bash
python scorer_single.py
```

## 输入格式

输入 JSON 文件中的每条数据应包含：

```json
{
  "id": "unique_id",
  "domain": "public_health",
  "label": "Question-Answer",
  "question": "您的问题",
  "answer": "参考答案",
  "complexCOT": "参考推理过程",
  "llm_complexCOT": "模型的思维链",
  "llm_answer": "模型的最终答案"
}
```

## 输出格式

输出包含六个维度的分数和描述：

```json
{
  "id": "unique_id",
  "scores": {
    "Accuracy": {
      "score": 10,
      "description": "与参考答案完美匹配"
    },
    "Reasoning": {
      "score": 8,
      "description": "逻辑流程强，存在微小差距"
    },
    "Completeness": {
      "score": 9,
      "description": "涵盖全面"
    },
    "Consensus Alignment": {
      "score": 10,
      "description": "与科学共识完美对齐"
    },
    "Terminology Norms": {
      "score": 8,
      "description": "专业术语使用恰当"
    },
    "Insightfulness": {
      "score": 7,
      "description": "对底层机制有良好解释"
    }
  }
}
```

## 评分维度

1. **准确性**: 与参考答案相比的事实正确性
2. **推理性**: 推理过程的逻辑有效性
3. **完整性**: 所有关键信息点的覆盖程度
4. **共识一致性**: 对科学共识的遵循程度
5. **术语规范性**: 专业术语使用的规范性
6. **洞察力**: 机制解释的深度

## 功能特性

- **断点续传**: 中断后可从上次检查点继续
- **张量并行**: 通过 vLLM 支持多 GPU
- **跨语言**: 支持中英文输入
- **长度验证**: 自动跳过过长的提示

## 注意事项

- 对于单选题任务，准确性评分是二元的（0 或 10）
- 如果未提供思维链，推理得分为 0
- 所有分数为 0 到 10 的整数
- 检查点文件保存为 `{output}_checkpoint.json`

## 权重合并

使用 `merge_weights.py` 合并基础模型和微调权重：

```python
# 修改以下路径
base_model_path = "/path/to/Qwen/Qwen3-8B"
output_path = "/path/to/output/Qwen3-8B-Merged"
```

运行：

```bash
python merge_weights.py
```
