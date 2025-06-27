# Thai Text Embedding Model from Scratch

A modern, extensible codebase for building Thai text embedding models from scratch. This project is designed for both researchers and practitioners who want to understand, train, and experiment with Thai language embeddings using real-world datasets.

## Overview

This repository provides:
- A Transformer-based embedding model tailored for Thai text
- Thai-specific preprocessing and tokenization
- Training and evaluation pipelines
- Support for real Thai datasets from Hugging Face
- Clear documentation and quick-start scripts

## Why Thai Text Embedding?

Thai language has unique characteristics (no spaces between words, complex script) that require specialized models and preprocessing. This project demonstrates how to build such a model from the ground up, with flexibility to use both sample and large-scale real datasets.

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

## Key Features

- **Thai-specific preprocessing**: Word segmentation, normalization, and cleaning
- **Custom tokenizer**: Trained on Thai corpus, supports SentencePiece/BPE
- **Transformer-based model**: Bi-directional attention, mean pooling, contrastive learning
- **Flexible training**: Easily switch between sample data and real datasets
- **Comprehensive evaluation**: Semantic similarity, clustering, retrieval, and visualization
- **Ready for Hugging Face datasets**: Use large-scale Thai datasets with minimal code changes

## Quick Start

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Train a model (sample data)**:
```bash
python quick_start.py
```
แล้วเลือกข้อ 1 (Train a new model)

3. **Evaluate a model**:
```bash
python quick_start.py
```
แล้วเลือกข้อ 2 (Evaluate existing model)

4. **Experiment in Jupyter Notebook**:
```bash
python quick_start.py
```
แล้วเลือกข้อ 3

## Using Real Thai Datasets from Hugging Face

You can easily switch from sample data to real datasets. Example for loading a Hugging Face dataset:

```python
from datasets import load_dataset
# Load Thai Wikipedia dataset
wiki = load_dataset("ZombitX64/Thailand-wikipedia-2025", split="train")
texts = wiki["text"]
```
Replace the sample data loading part in `scripts/train_model.py` with your Hugging Face dataset as shown above.

**Recommended Thai datasets:**
- ZombitX64/Thailand-wikipedia-2025
- mOSCAR_Thai
- OpenSubtitles-Thai
- Sentence-Transformers-Thai

## Model Architecture

- Transformer encoder (multi-head attention, feed-forward, layer norm)
- Custom tokenizer for Thai
- Mean/CLS/Max pooling for sentence embeddings
- Contrastive loss for training

## Datasets

The model supports both sample data and real Thai datasets:
- Thai Wikipedia (Hugging Face)
- mOSCAR_Thai (Hugging Face)
- OpenSubtitles-Thai (Hugging Face)
- Thai news, social media, academic, and more

## Evaluation

- Semantic similarity
- Text classification
- Information retrieval
- Clustering quality
- Embedding visualization (t-SNE, PCA)

## How to Extend
- Swap in any Hugging Face dataset for training/evaluation
- Modify model/configs for your research
- Add new evaluation metrics or downstream tasks

## Contributing

Contributions are welcome! Please read the contribution guidelines before submitting pull requests.

## License

This project is licensed under the MIT License.

## Deep Dive: Technical Details

### 1. Data Preprocessing
- **Cleaning**: Remove HTML, URLs, emails, phone numbers, and normalize Unicode.
- **Segmentation**: Use PyThaiNLP for word and sentence segmentation, with optional stopword removal.
- **Custom pipeline**: All preprocessing steps are modular and can be extended in `src/data/thai_preprocessor.py`.

### 2. Tokenizer
- **Custom BPE/WordPiece**: Trained on Thai corpus using the `tokenizers` library.
- **Configurable vocab size**: Set in `configs/base_config.yaml`.
- **Training**: Tokenizer is (re)trained automatically before model training, and saved to `tokenizer/thai_tokenizer.json`.
- **Integration**: Tokenizer is used for both training and inference, and can be swapped for SentencePiece or other methods.

### 3. Model Architecture
- **Transformer Encoder**:
  - Multi-head self-attention, feed-forward, layer normalization, dropout
  - Configurable depth, width, and number of heads
- **Pooling**: Mean, CLS, or Max pooling for sentence-level embedding
- **Contrastive Loss**: For learning semantically meaningful embeddings
- **No pre-trained weights**: Model is trained from scratch for full control
- **Implementation**: See `src/models/thai_embedding_model.py`

### 4. Training Pipeline
- **Configurable**: All hyperparameters in `configs/base_config.yaml`
- **Trainer**: Handles batching, optimizer, scheduler, early stopping, checkpointing
- **Validation**: Supports validation every N steps or after each epoch
- **Logging**: Console logging, optional Weights & Biases integration
- **Resume**: Training can be resumed from any checkpoint
- **Script**: Main entry point is `scripts/train_model.py`

### 5. Evaluation
- **Similarity**: Computes cosine similarity, accuracy, F1, Pearson/Spearman correlation
- **Clustering**: KMeans, silhouette score, ARI, NMI
- **Retrieval**: Precision@k, Recall@k, MAP
- **Visualization**: t-SNE, PCA plots for embedding inspection
- **Script**: Main entry point is `scripts/evaluate_model.py`

### 6. Using Hugging Face Datasets
- **Plug-and-play**: Replace sample data loading in `train_model.py` with Hugging Face datasets
- **Example**:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("ZombitX64/Thailand-wikipedia-2025", split="train")
  texts = dataset["text"]
  # Use texts for tokenizer training and model input
  ```
- **Large-scale**: Can handle millions of sentences if hardware permits
- **Preprocessing**: All Hugging Face datasets can be passed through the same preprocessing pipeline

### 7. Extending the Project
- **Add new datasets**: Just change the data loading logic
- **Try new architectures**: Modify or add new models in `src/models/`
- **Experiment with losses**: Add triplet, margin, or other losses in `src/models/` and `src/training/`
- **Custom evaluation**: Add new metrics or downstream tasks in `src/utils/metrics.py`

### 8. Example: Full Training with Hugging Face Dataset
```python
from datasets import load_dataset
from src.data.thai_preprocessor import ThaiTokenizer

# Load dataset
wiki = load_dataset("ZombitX64/Thailand-wikipedia-2025", split="train")
texts = wiki["text"]

# Preprocess and train tokenizer
tokenizer = ThaiTokenizer(vocab_size=30000)
tokenizer.train_tokenizer(texts, "tokenizer/thai_tokenizer.json")

# Use texts for model training as in scripts/train_model.py
```

---

## FAQ

**Q: Can I use my own Thai dataset?**
A: Yes! Just load your data as a list of sentences and use it in place of the sample data.

**Q: How do I change model size or architecture?**
A: Edit `configs/base_config.yaml` for hyperparameters, or modify `src/models/thai_embedding_model.py` for architecture.

**Q: Can I use GPU?**
A: Yes, the code will use CUDA if available. For large datasets/models, GPU is highly recommended.

**Q: How do I visualize embeddings?**
A: Use the provided evaluation scripts or Jupyter notebook for t-SNE/PCA plots.

**Q: Is this production-ready?**
A: The code is research-grade and modular, suitable for prototyping, research, and education. For production, further optimization and testing are recommended.

---