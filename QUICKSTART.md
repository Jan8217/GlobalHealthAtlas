# Quick Start Guide

This guide will help you get started with GlobalHealthAtlas quickly.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt


```

## First Test

Score a sample file:

```bash
python -m globalhealthatlas score \
  -i data/samples/sample_input.json \
  -o data/test_scores.json \
  -m /path/to/your/model
```

## Common Commands

```bash
# Score model outputs
python -m globalhealthatlas score -i input.json -o output.json -m /model

# Generate answers
python -m globalhealthatlas infer -i input.json -o output.json -b /base -l /lora

# Analyze scores
python -m globalhealthatlas analyze -i scored.json -o analysis.xlsx
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [data/README.md](data/README.md) for data management
- Review example data in `data/samples/`
