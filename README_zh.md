# GlobalHealthAtlas（全球健康图谱）

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

一个大规模、多语言的公共卫生推理数据集及其评估框架。

## 🤖 模型资源

- **Public Evaluator**: [aerovane0/GlobalHealthAtlas_Public_Evaluator](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Evaluator) - 专用评估模型
- **Public Model**: [aerovane0/GlobalHealthAtlas_Public_Model](https://huggingface.co/aerovane0/GlobalHealthAtlas_Public_Model) - 公共推理模型

## 🎓 论文链接和资源

### 论文摘要
我们的研究论文介绍了 GlobalHealthAtlas，这是一个大规模结构化健康推理数据集，包含 280,210 个精选实例，涵盖 15 个公共卫生领域和 17 种语言。该论文解决了结构化机器学习问题在公共卫生推理方面的关键缺口，在这一安全关键领域中，LLM 缺乏合适的训练信号或可靠的基准。

### 论文的关键创新：
1. **以证据为中心的数据工程流水线**：一种新颖的流水线，将异构的公共卫生 PDF 转换为结构化 Markdown，分割为证据块，并合成具有多维元数据的问题-答案对。

2. **领域对齐评估器**：一个专门的评估模型，经过训练以沿六个互补维度评估输出：准确性、推理、完整性、共识对齐、术语规范和洞察力。

3. **LLM 支持的质量控制**：一种创新方法，利用 LLM 进行多阶段过滤、验证和细化，以确保大规模数据一致性。

4. **跨语言能力**：支持 17 种语言，具有针对多语言环境调整的领域特定评估标准。

### 实验结果
该论文展示了即使最先进的 LLM 在鲁棒公共卫生推理方面也存在显著局限性，特别是在现实扰动和跨语言环境中。我们的评估框架显示出与参考判断的卓越一致性（ICC = 0.9735）和相比通用评估器的稳定性。

### 研究影响
这项工作有助于在安全关键领域中为现实世界决策开发可靠的大型语言模型，为推进领域对齐推理数据集和评估方法建立了一个原则性基础。

## 📊 数据集概览

GlobalHealthAtlas 是一个综合性的数据集，包含 280,210 个实例，涵盖 15 个公共卫生领域和 17 种语言。它涵盖了两种任务格式（问答和单选）和三个难度级别（A/B/C）。

### 关键统计数据

- **总实例数**: 280,210
- **领域**: 15 个（传染病预防、卫生政策、疫苗接种等）
- **语言**: 17 种（包括英语、中文、西班牙语等）
- **问题类型**: 问答（138,267）和单选（141,943）
- **难度级别**: A (26.26%), B (69.33%), C (4.41%)

### 研究论文

此仓库包含了我们研究论文《From Knowledge to Inference: Scaling Laws of Specialized Reasoning on GlobalHealthAtlas》中描述的实现和评估框架。该论文展示了：

- 一个包含 280,210 个精选实例的大规模结构化健康推理数据集
- 一个支持 LLM 的数据构建和质量控制流水线
- 一个用于沿六个维度评估输出的领域对齐评估器
- 显示当前 LLM 在推理盲点方面的全面实验

论文强调，与个体诊断或封闭式科学推理不同，公共卫生为 LLM 提出了独特的挑战性推理场景，需要基于科学证据、专家共识和安全约束的人群层面推理。然而，在结构化机器学习问题方面，这一领域仍待探索，使 LLM 在这一安全关键领域缺乏合适的训练信号或可靠的基准。

## 🏗️ 项目结构

此项目已重构为模块化组件，以提高可维护性：

```
GlobalHealthAtlas/
├── src/                           # 源代码模块
│   ├── __init__.py               # 包初始化器
│   ├── config/                   # 配置模块
│   │   ├── __init__.py           # 配置包初始化器 - 初始化配置包
│   │   ├── paths.py              # 路径配置和文件位置 - 管理模型路径和文件 I/O 位置
│   │   ├── model_config.py       # 模型参数、批次大小和 JSON 模式 - 定义 MAX_MODEL_LEN、BATCH_SIZE、JSON_SCHEMA 用于引导解码
│   │   └── prompts.py            # 综合评估提示模板 - 包含论文中描述的详细 6 维评估提示
│   ├── core/                     # 核心功能
│   │   ├── __init__.py           # 核心包初始化器
│   │   ├── prompt_builder.py     # 从模板动态构建提示 - 从模板中获取数据项并填充个性化评估提示
│   │   ├── model_initializer.py  # 模型加载、分词器设置和参数配置 - 用适当的张量并行、内存管理和引导解码参数初始化 vLLM 引擎
│   │   └── batch_processor.py    # 批处理逻辑（含错误处理）- 使用 vLLM 进行评估，处理批次数据
│   │   └── inference.py          # 未经训练的模型推理脚本，为多语言基准测试设计.
│   │   └── leakage_inference.py  # 数据泄露检测推理方法.
│   ├── handlers/                 # 处理器
│   │   ├── __init__.py           # 处理器包初始化器
│   │   └── file_processor.py     # 文件级处理编排（含批处理）- 使用检查点管理单个文件的处理
│   ├── utils/                    # 实用函数
│   │   ├── __init__.py           # 实用工具包初始化器
│   │   ├── data_handler.py       # 数据加载/保存（带原子写入）- 使用原子写入保护处理 JSON I/O 操作
│   │   └── checkpoint_manager.py # 检查点保存/加载（用于中断恢复）- 管理检查点持久化以启用恢复功能
│   └── main.py                   # 主应用程序入口点 - 协调整个评估流水线
├── experiments/                  # 实验脚本
│   ├── __init__.py               # 实验包初始化器
│   ├── experiment_runner.py      # 实验执行接口 - 为运行实验提供命令行界面
│   └── result_analyzer.py        # 结果分析和 Excel 导出 - 分析评分结果并以统计分解导出到 Excel
├── scoring/                      # 评分功能
│   ├── __init__.py               # 评分包初始化器
│   └── scorer.py                 # 评分接口（带命令行支持）- 运行评分过程的主要接口
├── training/                     # 训练脚本
│   ├── __init__.py               # 训练包初始化器
│   ├── train_lora.sh             # LoRA 微调脚本 - 在 GlobalHealthAtlas 数据上训练专用模型
│   └── README.md                 # 训练文档
├── data/                         # 数据目录
│   └── Benchmark_evaluation_results/ # 评估结果
│       ├── Qwen2.5-72Binstruct.csv
│       └── Qwen2.5-7Binstruct.csv
├── config.yaml                   # 配置文件
├── requirements.txt              # 依赖项
├── setup.py                      # 设置脚本
├── pyproject.toml                # 项目配置
└── LICENSE                       # 许可证信息
```

## 🔬 评估框架 - 六维评估

基于我们的研究论文，GlobalHealthAtlas 评估器沿着六个维度评估模型输出：

### 1. 准确性（Accuracy）
此维度评估模型响应相对于标准参考答案的绝对事实正确性。对于单选题，应用二元评分（仅 0 或 10）。对于问答题，模型必须准确捕获所有特定关键事实（数字、实体）。这解决了公共卫生中事实核查的基本挑战，因为错误信息可能会产生严重后果。

### 2. 推理（Reasoning）
此维度评估模型逐步推理过程（链式思维/COT）的逻辑有效性和连贯性。它检验推导是否符合既定的公共卫生指南，将干预措施与健康结果联系起来，而不存在逻辑漏洞、循环推理或因果谬误。这对于公共卫生至关重要，因为推理链必须可追溯且科学可靠。

### 3. 完整性（Completeness）
此维度衡量模型检索和包含标准参考答案中存在的所有关键信息点的程度。它需要整体比较，以确保不会遗漏任何关键组成部分。在公共卫生背景下，遗漏信息可能导致对复杂干预措施的理解不完整。

### 4. 共识对齐（Consensus Alignment）
此维度评估模型对来自 CDC、WHO 或 ECDC 等机构的既定科学共识和权威指南的遵守程度。它审查响应中是否存在与公认医学科学或公共卫生协议相矛盾的任何声明。这确保响应与官方建议保持一致。

### 5. 术语规范（Terminology Norms）
此维度评估所用语言的词汇精确性和专业密度。它要求正确使用领域特定的术语，而不是外行或口语等价物。这确保了健康背景下的专业沟通标准。

### 6. 洞察力（Insightfulness）
此维度评估机制解释的深度，区分简单的事实检索和专家级理解。它询问模型是否解释了现象背后的"为什么"和"如何"，而不只是陈述结果。这对于高级公共卫生推理至关重要。

每个维度都按照 0-10 分制进行评分，并附有基于论文方法论的详细标准。

## 🚀 快速开始

### 先决条件

- Python 3.8 或更高版本
- 支持 CUDA 的 GPU（推荐用于最佳性能）
- 至少 80GB VRAM 用于完整模型加载（或使用张量并行化）

### 安装

```bash
git clone https://github.com/globalhealthatlas/GlobalHealthAtlas.git
cd GlobalHealthAtlas
pip install -r requirements.txt
```

### 模型设置

1. 从 ModelScope 或 Hugging Face 下载所需模型
2. 更新 `src/config/paths.py` 中的路径以指向您的模型位置：
   - `MODEL_PATH`: 合并的 Qwen3-8B 模型路径
   - `BASE_MODEL_PATH`: 基础 Qwen/Qwen3-8B 模型路径

### 评分评估

#### 使用主入口点（运行完整流水线）：
```bash
cd src
python main.py
```

这将使用检查点恢复功能处理所有配置的文件对。主模块协调整个过程：
- 初始化模型组件
- 创建批处理器
- 创建文件处理器
- 加载全局检查点（如果可用）
- 使用恢复功能处理文件

#### 使用评分模块：
```bash
cd scoring
python scorer.py --input-file ../data/input.json --output-file ../data/output.json
```

评分模块提供简化的接口，可以运行完整流水线或处理自定义文件。

### 运行实验

#### 运行评分实验：
```bash
cd experiments
python experiment_runner.py --experiment-type scoring
```

实验运行器为不同类型实验提供统一接口，包括评分和分析。

#### 分析结果：
```bash
cd experiments
python result_analyzer.py --input-file ../data/scores.json --output-file ../results/analysis.xlsx
```

结果分析器生成按标签、领域和语言分类的详细统计数据，包括所有六个评估维度的平均分数。它聚合分数并以格式化表格导出到 Excel。

### 模型训练

```bash
cd training
bash train_lora.sh
```

此脚本将在 GlobalHealthAtlas 数据集上使用 LoRA 对模型进行微调，创建用于公共卫生推理的专用模型。

## 📦 Source Code

项目包含两个主要的源代码模块，分别用于推理和评估：

### Public Model
用于生成公共健康问题的推理响应，支持批量处理和单条处理。

**目录**: `Source Code/Public Model/`

**文件**:
- `answer_batch.py`: 批量处理脚本
- `answer_single.py`: 单条处理脚本
- `merge_weights.py`: 合并基础模型和 LoRA 权重
- `README.md`: 英文使用文档
- `README_ZH.md`: 中文使用文档

**使用方法**:
```bash
cd "Source Code/Public Model"

# 批量处理
python answer_batch.py

# 单条处理
python answer_single.py

# 合并权重（可选）
python merge_weights.py
```

### Public Evaluator
用于评估模型生成的回答，基于六个维度进行评分。

**目录**: `Source Code/Public Evaluator/`

**文件**:
- `scorer_batch.py`: 批量评分脚本
- `scorer_single.py`: 单条评分脚本
- `merge_weights.py`: 合并基础模型和微调权重
- `README.md`: 英文使用文档
- `README_ZH.md`: 中文使用文档

**使用方法**:
```bash
cd "Source Code/Public Evaluator"

# 批量评分
python scorer_batch.py

# 单条评分
python scorer_single.py

# 合并权重（可选）
python merge_weights.py
```

## 🔧 配置选项

### 路径配置 (`src/config/paths.py`)
- `MODEL_PATH`: 推理用合并模型路径
- `BASE_MODEL_PATH`: 分词器用基础模型路径
- `GLOBAL_CHECKPOINT_FILE`: 全局检查点文件位置
- `FILE_PAIRS`: 用于处理的默认输入/输出文件对

### 模型配置 (`src/config/model_config.py`)
- `MAX_MODEL_LEN`: 最大模型长度（默认：40960 令牌）
- `BATCH_SIZE`: 批处理大小（默认：4000 项目以防止内存崩溃）
- `SAFE_INPUT_LIMIT`: 输入的安全令牌限制（MAX_MODEL_LEN - 1024）
- `STOP_TOKEN_IDS`: 使用的其他停止令牌
- `JSON_SCHEMA`: 用于引导解码验证的模式

### 提示自定义 (`src/config/prompts.py`)
包含基于论文方法论的综合 6 维评估提示模板。

## 📊 模块特定功能

### 核心模块分解：

**`src/config/paths.py`**: 管理所有文件系统路径和模型位置。包含可以使用环境变量覆盖的硬编码路径。定义将处理的输入/输出文件对。

**`src/config/model_config.py`**: 定义所有模型特定参数，包括令牌限制、批次大小和用于引导解码的 JSON 模式。JSON 模式确保模型以预期的 6 维格式输出结构化数据。

**`src/config/prompts.py`**: 包含全面的评估提示模板，带有所有 6 个维度的详细标准。这是论文中描述的评估方法的核心。

**`src/core/prompt_builder.py`**: 获取数据项并将其填充到 prompts.py 中的模板。为每个数据点创建个性化的评估提示，包含所有必需信息。

**`src/core/model_initializer.py`**: 使用适当的张量并行、内存管理和引导解码参数处理 vLLM 引擎的复杂初始化。设置分词器并确保所有组件正确配置。

**`src/core/batch_processor.py`**: 实现核心批处理逻辑。处理令牌长度验证，过滤过长输入，通过模型处理有效输入，解析 JSON 响应，并为每个批次管理错误处理。

**`src/handlers/file_processor.py`**: 管理带检查点的文件级处理。读取输入文件，将数据划分为批次，为每个批次调用批处理器，保存中间结果，并更新全局检查点。

**`src/utils/data_handler.py`**: 提供安全的数据 I/O 操作，使用原子写入防止中断时文件损坏。处理 JSON 数据的加载和保存。

**`src/utils/checkpoint_manager.py`**: 管理检查点持久化以启用恢复功能。保存和加载检查点信息以启用从中断中恢复。

**`src/main.py`**: 协调整个流水线。初始化所有组件，加载检查点（如果可用），并处理所有配置的文件。

**`experiments/result_analyzer.py`**: 分析评分结果并创建详细的 Excel 报告，按领域、语言和难度级别分解。

## 📈 处理功能

### 批处理
系统以 4000 个项目批次处理数据，以高效处理大文件，同时防止内存崩溃。每个批次独立处理，具有错误隔离。

### 检查点恢复
系统在每个批次后保存检查点并维护全局检查点文件。如果中断，处理将从最后一个完成的批次恢复，防止进度损失。

### 长度过滤
超出令牌限制的样本将被过滤并在结果中标记为错误，确保稳定处理。

### 错误处理
全面的错误处理确保单个失败的样本不会中断整个处理流水线。

### 原子写入
结果使用原子写入操作保存，以防止中断期间的文件损坏。

## 🤖 模型集成

系统使用 vLLM 进行高效批处理，具有：
- 跨所有可用 GPU 的张量并行化
- 引导解码以确保 JSON 模式合规性
- 使用前缀缓存优化内存使用
- 高吞吐量推理

## 📁 数据格式

输入文件应为包含对象的 JSON 数组，其中包含：
- `id`: 唯一标识符
- `domain`: 公共卫生领域（来自 15 个定义领域）
- `label`: 任务类型（问答或单选）
- `question`: 问题文本
- `answer`: 标准参考答案
- `complexCOT`: 标准参考推理（来自论文的方法）
- `llm_complexCOT`: 模型的链式思维
- `llm_answer`: 模型的最终响应

输出文件将包含相同的字段以及：
- `scores`: 包含六个维度分数的对象
- `raw_response`: 原始模型输出（如果解析失败）
- `error`: 错误消息（如果处理失败）

## 📄 引用

```bibtex
@article{globalhealthatlas2026,
  title={From Knowledge to Inference: Scaling Laws of Specialized Reasoning on GlobalHealthAtlas},
  author={GlobalHealthAtlas Team},
  year={2026}
}
```

## 📄 许可证

该项目根据 MIT 许可证授权 - 请参阅 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

我们感谢公共卫生界提供的宝贵数据源和评估标准。这项工作得到了我们研究团队致力于推进 AI 在公共卫生领域的社会公益承诺的支持。

GlobalHealthAtlas 数据集代表了 AI 辅助公共卫生推理领域的重大贡献，为评估和改进 LLM 在安全关键健康领域的能力提供了强大框架。基于我们的研究发现，即使是最先进的 LLM 在鲁棒公共卫生推理方面也表现出显著局限性，特别是在现实扰动和跨语言设置下，突出了这一领域当前的差距和 LLM 的未开发潜力。