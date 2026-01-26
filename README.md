# GlobalHealthAtlas

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[中文文档](./README_zh.md)

A large-scale, multilingual dataset for public health reasoning and its evaluation framework.

## 📊 Dataset Overview

GlobalHealthAtlas is a comprehensive dataset of 280,210 instances spanning 15 public health domains and 17 languages. It covers two task formats (Question-Answer and Single-Choice) and three difficulty levels (A/B/C).

### Key Statistics

- **Total Instances**: 280,210
- **Domains**: 15 (Infectious Disease Prevention, Health Policy, Vaccination, etc.)
- **Languages**: 17 (including English, Chinese, Spanish, etc.)
- **Question Types**: QA (138,267) and Single-Choice (141,943)
- **Difficulty Levels**: A (26.26%), B (69.33%), C (4.41%)

### Research Paper

This repository contains the implementation and evaluation framework described in our research paper: "From Knowledge to Inference: Scaling Laws of Specialized Reasoning on GlobalHealthAtlas". The paper presents:

- A large-scale, structured health reasoning dataset with 280,210 curated instances
- An LLM-supported data construction and quality control pipeline
- A domain-aligned evaluator for assessing outputs along six dimensions
- Comprehensive experiments showing reasoning blind spots in current LLMs

The paper highlights that unlike individual diagnosis or closed-form scientific reasoning, public health poses uniquely challenging reasoning settings for LLMs, requiring population-level inference grounded in scientific evidence, expert consensus, and safety constraints. Yet, it remains underexplored as a structured machine learning problem, leaving LLMs without suitable training signals or reliable benchmarks in this safety-critical domain.

## 🏗️ Project Structure

This project has been restructured into modular components for improved maintainability:

```
GlobalHealthAtlas/
├── src/                           # Source code modules
│   ├── __init__.py               # Package initializer
│   ├── config/                   # Configuration modules
│   │   ├── __init__.py           # Config package initializer - initializes the configuration package
│   │   ├── paths.py              # Path configurations and file locations - manages model paths and file I/O locations
│   │   ├── model_config.py       # Model parameters, batch sizes, and JSON schemas - defines MAX_MODEL_LEN, BATCH_SIZE, JSON_SCHEMA for guided decoding
│   │   └── prompts.py            # Comprehensive evaluation prompt template - contains the detailed 6-dimensional evaluation prompt from the paper
│   ├── core/                     # Core functionality
│   │   ├── __init__.py           # Core package initializer
│   │   ├── prompt_builder.py     # Dynamic prompt construction from templates - builds evaluation prompts from the template with actual data
│   │   ├── model_initializer.py  # Model loading, tokenizer setup, and parameter configuration - initializes vLLM engine with proper parameters
│   │   └── batch_processor.py    # Batch processing logic with error handling - processes data in batches using vLLM for evaluation
│   │   └── inference.py          # Model inference script without training, designed for benchmarking on multiple languages.
│   │   └── leakage_inference.py  # data leakage inference module for n-gram continuation detection using DashScope (OpenAI compatible) API.
│   ├── handlers/                 # Processing handlers
│   │   ├── __init__.py           # Handlers package initializer
│   │   └── file_processor.py     # File-level processing orchestration with batching - manages processing of individual files with checkpointing
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py           # Utils package initializer
│   │   ├── data_handler.py       # Data loading/saving with atomic writes - handles JSON I/O operations with atomic write protection
│   │   └── checkpoint_manager.py # Checkpoint saving/loading for interruption recovery - manages checkpoint persistence for resume capability
│   └── main.py                   # Main application entry point - orchestrates the entire evaluation pipeline
├── experiments/                  # Experiment scripts
│   ├── __init__.py               # Experiments package initializer
│   ├── experiment_runner.py      # Experiment execution interface - provides command-line interface for running experiments
│   └── result_analyzer.py        # Results analysis and Excel export - analyzes scoring results and exports to Excel with statistical breakdowns
├── scoring/                      # Scoring functionality
│   ├── __init__.py               # Scoring package initializer
│   └── scorer.py                 # Scoring interface with command-line support - main interface for running the scoring process
├── training/                     # Training scripts
│   ├── __init__.py               # Training package initializer
│   ├── train_lora.sh             # LoRA fine-tuning script - trains specialized models on GlobalHealthAtlas data
│   └── README.md                 # Training documentation
├── data/                         # Data directory
│   └── Benchmark_evaluation_results/ # Evaluation results
│       ├── Qwen2.5-72Binstruct.csv
│       └── Qwen2.5-7Binstruct.csv
├── config.yaml                   # Configuration file
├── requirements.txt              # Dependencies
├── setup.py                      # Setup script
├── pyproject.toml                # Project configuration
└── LICENSE                       # License information
```

## 🔬 Evaluation Framework - Six-Dimensional Assessment

Based on our research paper, the GlobalHealthAtlas evaluator assesses model outputs along six dimensions:

### 1. Accuracy
This dimension evaluates the absolute factual correctness of the model's response compared to the Standard Reference Answer. For Single-Choice tasks, binary scoring applies (0 or 10 only). For Question-Answer tasks, the model must accurately capture all specific key facts (numbers, entities). This addresses the fundamental challenge of fact-checking in public health where misinformation can have serious consequences.

### 2. Reasoning
This dimension assesses the logical validity and coherence of the model's step-by-step reasoning process (Chain of Thought/COT). It examines whether the derivation adheres to established public health guidelines, linking interventions to health outcomes without logical gaps, circular reasoning, or causal fallacies. This is critical in public health where reasoning chains must be traceable and scientifically sound.

### 3. Completeness
This dimension measures the extent to which the model retrieves and includes all Key Information Points (KIPs) present in the Standard Reference Answer. It requires a holistic comparison to ensure no critical components are omitted. In public health contexts, missing information can lead to incomplete understanding of complex interventions.

### 4. Consensus Alignment
This dimension evaluates the model's adherence to established scientific consensus and authoritative guidelines from bodies such as the CDC, WHO, or ECDC. It scrutinizes the response for any claims that contradict accepted medical science or public health protocols. This ensures responses align with official recommendations.

### 5. Terminology Norms
This dimension assesses the lexical precision and professional density of the language used. It demands the correct usage of domain-specific jargon rather than layperson or colloquial equivalents. This ensures professional communication standards in health contexts.

### 6. Insightfulness
This dimension evaluates the depth of the mechanism explanation, distinguishing between mere fact retrieval and expert-level understanding. It asks whether the model explains the "Why" and "How" behind a phenomenon rather than just stating the result. This is essential for advanced public health reasoning.

Each dimension is scored on a 0-10 scale with detailed rubrics based on the paper's methodology.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU(s) for optimal performance
- At least 80GB VRAM for full model loading (or use tensor parallelism)

### Installation

```bash
git clone https://github.com/globalhealthatlas/GlobalHealthAtlas.git
cd GlobalHealthAtlas
pip install -r requirements.txt
```

### Model Setup

1. Download the required models from ModelScope or Hugging Face
2. Update paths in `src/config/paths.py` to point to your model locations:
   - `MODEL_PATH`: Path to the merged Qwen3-8B model
   - `BASE_MODEL_PATH`: Path to the base Qwen/Qwen3-8B model

### Scoring Evaluation

#### Using the main entry point (runs the complete pipeline):
```bash
cd src
python main.py
```

This will process all configured file pairs with checkpoint resume capability. The main module orchestrates the entire process:
- Initializes model components
- Creates batch processor
- Creates file processor
- Loads global checkpoint if available
- Processes files with resume capability

#### Using the scoring module:
```bash
cd scoring
python scorer.py --input-file ../data/input.json --output-file ../data/output.json
```

The scorer module provides a simplified interface that can either run the full pipeline or process custom files.

### Running Experiments

#### Run scoring experiments:
```bash
cd experiments
python experiment_runner.py --experiment-type scoring
```

The experiment runner provides a unified interface for different types of experiments, including scoring and analysis.

#### Analyze results:
```bash
cd experiments
python result_analyzer.py --input-file ../data/scores.json --output-file ../results/analysis.xlsx
```

The result analyzer produces detailed statistics by label, domain, and language, including average scores across all six evaluation dimensions. It aggregates scores and exports to Excel with formatted tables.

### Model Training

```bash
cd training
bash train_lora.sh
```

This script will fine-tune the model on the GlobalHealthAtlas dataset using LoRA, creating specialized models for public health reasoning.

## 🔧 Configuration Options

### Path Configuration (`src/config/paths.py`)
- `MODEL_PATH`: Path to the merged model for inference
- `BASE_MODEL_PATH`: Path to the base model for tokenizer
- `GLOBAL_CHECKPOINT_FILE`: Location of global checkpoint file
- `FILE_PAIRS`: Default input/output file pairs for processing

### Model Configuration (`src/config/model_config.py`)
- `MAX_MODEL_LEN`: Maximum model length (default: 40960 tokens)
- `BATCH_SIZE`: Batch processing size (default: 4000 items to prevent memory crashes)
- `SAFE_INPUT_LIMIT`: Safe token limit for inputs (MAX_MODEL_LEN - 1024)
- `STOP_TOKEN_IDS`: Additional stop tokens to use
- `JSON_SCHEMA`: Schema for guided decoding validation

### Prompt Customization (`src/config/prompts.py`)
Contains the comprehensive 6-dimensional evaluation prompt template based on the paper methodology.

## 📊 Module-Specific Functionality

### Core Modules Breakdown:

**`src/config/paths.py`**: Manages all file system paths and model locations. Contains hardcoded paths that can be overridden with environment variables. Defines the input/output file pairs that will be processed.

**`src/config/model_config.py`**: Defines all model-specific parameters including token limits, batch sizes, and the JSON schema used for guided decoding. The JSON schema ensures the model outputs structured data in the expected 6-dimensional format.

**`src/config/prompts.py`**: Contains the comprehensive evaluation prompt template with detailed rubrics for all 6 dimensions. This prompt is the core of the evaluation methodology described in the paper.

**`src/core/prompt_builder.py`**: Takes data items and fills them into the template from prompts.py. Creates personalized evaluation prompts for each data point with all required information.

**`src/core/model_initializer.py`**: Handles the complex initialization of the vLLM engine with proper tensor parallelism, memory management, and guided decoding parameters. Sets up the tokenizer and ensures all components are properly configured.

**`src/core/batch_processor.py`**: Implements the core batch processing logic. Handles token length validation, filters out too-long inputs, processes valid inputs through the model, parses JSON responses, and manages error handling for each batch.

**`src/handlers/file_processor.py`**: Manages file-level processing with checkpointing. Reads input files, divides data into batches, calls the batch processor for each batch, saves intermediate results, and updates global checkpoints.

**`src/utils/data_handler.py`**: Provides safe data I/O operations with atomic writes to prevent file corruption during interruption. Handles both loading and saving of JSON data.

**`src/utils/checkpoint_manager.py`**: Manages checkpoint persistence for resume capability. Saves and loads checkpoint information to enable resuming from interruptions.

**`src/main.py`**: Orchestrates the entire pipeline. Initializes all components, loads checkpoints if available, and processes all configured files.

**`experiments/result_analyzer.py`**: Analyzes scoring results and creates detailed Excel reports with breakdowns by domain, language, and difficulty level.

## 📈 Processing Features

### Batch Processing
The system processes data in batches of 4000 items to handle large files efficiently while preventing memory crashes. Each batch is processed independently with error isolation.

### Checkpoint Resume
The system saves checkpoints after each batch and maintains a global checkpoint file. If interrupted, processing will resume from the last completed batch, preventing loss of progress.

### Length Filtering
Samples that exceed the token limit are filtered out and marked with an error in the results, ensuring stable processing.

### Error Handling
Comprehensive error handling ensures that individual failed samples don't halt the entire processing pipeline.

### Atomic Writes
Results are saved using atomic write operations to prevent file corruption during interruption.

## 🤖 Model Integration

The system uses vLLM for efficient batch processing with:
- Tensor parallelism across all available GPUs
- Guided decoding to ensure JSON schema compliance
- Optimized memory usage with prefix caching
- High throughput inference

## 📁 Data Format

Input files should be JSON arrays with objects containing:
- `id`: Unique identifier
- `domain`: Public health domain (from 15 defined domains)
- `label`: Task type (Question-Answer or Single-Choice)
- `question`: The question text
- `answer`: Standard reference answer
- `complexCOT`: Standard reference reasoning (from the paper's methodology)
- `llm_complexCOT`: Model's Chain of Thought
- `llm_answer`: Model's final response

Output files will contain the same fields plus:
- `scores`: Object with scores for each of the 6 dimensions
- `raw_response`: Raw model output (if parsing fails)
- `error`: Error message (if processing failed)

## 📄 Citation

```bibtex
@article{globalhealthatlas2026,
  title={From Knowledge to Inference: Scaling Laws of Specialized Reasoning on GlobalHealthAtlas},
  author={GlobalHealthAtlas Team},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the public health community for providing valuable data sources and evaluation standards. This work was supported by our research team's commitment to advancing AI for social good in public health.

The GlobalHealthAtlas dataset represents a significant contribution to the field of AI-assisted public health reasoning, providing a robust framework for evaluating and improving LLM capabilities in safety-critical health domains. Based on our research findings, even state-of-the-art LLMs exhibit substantial limitations in robust public health reasoning, particularly under realistic perturbations and cross-lingual settings, highlighting both the current gaps and the untapped potential of LLMs in this domain.

## 🎓 Paper Links and Resources

### Paper Abstract
Our research paper introduces GlobalHealthAtlas, a large-scale, structured health reasoning dataset with 280,210 curated instances spanning 15 public health domains and 17 languages. The paper addresses the critical gap in structured machine learning problems for public health reasoning, where LLMs lack suitable training signals or reliable benchmarks in this safety-critical domain.

### Key Innovations from the Paper:
1. **Evidence-Centric Data Engineering Pipeline**: A novel pipeline that converts heterogeneous public health PDFs into structured Markdown, segments into evidence chunks, and synthesizes question-answer pairs with multi-dimensional metadata.

2. **Domain-Aligned Evaluator**: A specialized evaluation model trained to assess outputs along six complementary dimensions: Accuracy, Reasoning, Completeness, Consensus Alignment, Terminology Norms, and Insightfulness.

3. **LLM-Supported Quality Control**: An innovative approach that leverages LLMs for multi-stage filtering, validation, and refinement to ensure data consistency at scale.

4. **Cross-Lingual Capability**: Support for 17 languages with domain-specific evaluation criteria adapted for multilingual settings.

### Experimental Results
The paper demonstrates that even state-of-the-art LLMs exhibit substantial limitations in robust public health reasoning, particularly under realistic perturbations and cross-lingual settings. Our evaluation framework shows superior agreement with reference judgments (ICC = 0.9735) and stability compared to general-purpose evaluators.

### Research Impact
This work contributes to the broader development of reliable large language models for real-world decision-making in safety-critical domains, establishing a principled foundation for advancing domain-aligned reasoning datasets and evaluation methodologies.