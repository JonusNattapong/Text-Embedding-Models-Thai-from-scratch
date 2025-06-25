# Thai Text Embedding Model from Scratch

This project implements a Thai text embedding model built from scratch, including data preprocessing, model architecture, training pipeline, and evaluation framework.

## Overview

This project creates a specialized embedding model for Thai text that can:
- Generate meaningful vector representations of Thai sentences and documents
- Handle Thai language-specific characteristics (no spaces between words, complex script)
- Support various downstream tasks like semantic similarity, clustering, and retrieval

## Project Structure

```
├── data/                    # Dataset storage
│   ├── raw/                # Raw datasets
│   ├── processed/          # Preprocessed datasets
│   └── embeddings/         # Generated embeddings
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # Model architectures
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation metrics and scripts
│   └── utils/             # Utility functions
├── configs/               # Configuration files
├── scripts/               # Training and evaluation scripts
├── notebooks/             # Jupyter notebooks for experimentation
├── tests/                 # Unit tests
└── checkpoints/           # Model checkpoints
```

## Features

- **Thai-specific preprocessing**: Handles Thai word segmentation, normalization
- **Multiple model architectures**: Transformer-based and traditional approaches
- **Comprehensive evaluation**: Semantic similarity, clustering, retrieval tasks
- **Flexible training**: Support for different loss functions and training strategies
- **Visualization tools**: t-SNE, PCA for embedding visualization

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Prepare data**:
```bash
python scripts/prepare_data.py
```

3. **Train model**:
```bash
python scripts/train_model.py --config configs/base_config.yaml
```

4. **Evaluate model**:
```bash
python scripts/evaluate_model.py --model_path checkpoints/best_model.pt
```

## Model Architecture

The model uses a Transformer-based architecture optimized for Thai text:
- Custom tokenizer trained on Thai corpus
- Bi-directional attention mechanism
- Mean pooling for sentence-level embeddings
- Contrastive learning for training

## Datasets

The model is trained on various Thai datasets:
- Thai Wikipedia
- Thai news articles
- Thai social media posts
- Thai academic papers

## Evaluation

The model is evaluated on:
- Semantic similarity tasks
- Text classification
- Information retrieval
- Clustering quality

## Contributing

Please read the contribution guidelines before submitting pull requests.

## License

This project is licensed under the MIT License.