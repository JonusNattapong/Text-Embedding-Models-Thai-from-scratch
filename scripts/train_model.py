#!/usr/bin/env python3
"""
Main Training Script for Thai Embedding Model

This script handles the complete training pipeline including:
- Data loading and preprocessing
- Tokenizer training
- Model training
- Evaluation and validation
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.thai_preprocessor import ThaiTextPreprocessor, ThaiTokenizer, create_thai_dataset
from training.trainer import TrainingConfig, ThaiEmbeddingTrainer, create_data_loaders
from utils.metrics import EmbeddingEvaluator


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


def prepare_sample_data() -> Tuple[List[str], List[str], List[int]]:
    """
    Create sample Thai text data for demonstration.
    In practice, you would load your actual Thai dataset here.
    """
    # Sample Thai sentences (in practice, load from your dataset)
    thai_sentences = [
        "สวัสดีครับ ผมชื่อสมชาย",
        "วันนี้อากาศดีมากเลย",
        "ขอบคุณสำหรับความช่วยเหลือ",
        "ผมอยากเรียนภาษาไทย",
        "เมืองไทยมีวัฒนธรรมที่สวยงาม",
        "อาหารไทยอร่อยมาก",
        "ผมรักประเทศไทย",
        "การศึกษาสำคัญมาก",
        "เทคโนโลยีเปลี่ยนแปลงโลก",
        "ธรรมชาติต้องการการอนุรักษ์",
        "ครอบครัวคือสิ่งสำคัญ",
        "มิตรภาพมีค่ามาก",
        "การทำงานหนักนำไปสู่ความสำเร็จ",
        "ความสุขอยู่ในใจเรา",
        "การเรียนรู้ไม่มีวันสิ้นสุด",
        "สุขภาพดีคือความมั่งคั่งที่แท้จริง",
        "เวลาคือสิ่งมีค่า",
        "ความหวังให้กำลังใจ",
        "การให้เป็นสุข",
        "ฝันใหญ่แล้วทำให้เป็นจริง"
    ]
    
    # Create pairs and labels
    texts1, texts2, labels = [], [], []
    
    # Positive pairs (similar sentences)
    for i in range(len(thai_sentences)):
        for j in range(i + 1, min(i + 3, len(thai_sentences))):
            texts1.append(thai_sentences[i])
            texts2.append(thai_sentences[j])
            labels.append(1)  # Similar
    
    # Negative pairs (dissimilar sentences)
    for i in range(0, len(thai_sentences), 4):
        for j in range(len(thai_sentences) - 1, max(len(thai_sentences) - 5, 0), -1):
            if abs(i - j) > 3:  # Ensure they're far apart
                texts1.append(thai_sentences[i])
                texts2.append(thai_sentences[j])
                labels.append(0)  # Dissimilar
    
    return texts1, texts2, labels


def prepare_tokenizer(texts: List[str], config: Dict) -> ThaiTokenizer:
    """Prepare and train the Thai tokenizer."""
    logging.info("Training Thai tokenizer...")
    
    tokenizer = ThaiTokenizer(vocab_size=config['tokenizer']['vocab_size'])
    
    # Combine all texts for tokenizer training
    all_texts = texts
    
    # Train tokenizer
    tokenizer_path = config['paths']['tokenizer_path']
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
    tokenizer.train_tokenizer(all_texts, tokenizer_path)
    
    logging.info(f"Tokenizer trained and saved to {tokenizer_path}")
    logging.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train Thai Embedding Model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--eval_only", 
        action="store_true",
        help="Only run evaluation"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config['monitoring']['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Thai Embedding Model Training")
    logger.info(f"Configuration: {args.config}")
    
    # Prepare data
    logger.info("Preparing data...")
    texts1, texts2, labels = prepare_sample_data()
    
    # Combine all texts for preprocessing
    all_texts = texts1 + texts2
    
    # Split data
    train_texts1, val_texts1, train_texts2, val_texts2, train_labels, val_labels = train_test_split(
        texts1, texts2, labels,
        test_size=config['data']['val_split'] + config['data']['test_split'],
        random_state=42,
        stratify=labels
    )
    
    logger.info(f"Training samples: {len(train_texts1)}")
    logger.info(f"Validation samples: {len(val_texts1)}")
    
    # Prepare tokenizer
    tokenizer = prepare_tokenizer(all_texts, config)
    
    # Create training configuration
    training_config = TrainingConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        pooling_strategy=config['model']['pooling_strategy'],
        
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        num_epochs=config['training']['num_epochs'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        gradient_clipping=config['training']['gradient_clipping'],
        
        loss_function=config['training']['loss_function'],
        temperature=config['training']['temperature'],
        margin=config['training']['margin'],
        
        validation_steps=config['training']['validation_steps'],
        save_steps=config['training']['save_steps'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        
        output_dir=config['paths']['output_dir'],
        log_dir=config['paths']['log_dir'],
        
        use_wandb=config['monitoring']['use_wandb'],
        wandb_project=config['monitoring']['wandb_project'],
        log_level=config['monitoring']['log_level']
    )
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        train_texts1, train_texts2,
        val_texts1, val_texts2,
        train_labels, val_labels,
        tokenizer=tokenizer,
        batch_size=training_config.batch_size,
        max_length=config['data']['max_length']
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = ThaiEmbeddingTrainer(training_config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    if not args.eval_only:
        # Train the model
        logger.info("Starting training...")
        trainer.train(train_loader, val_loader)
        logger.info("Training completed!")
    
    # Evaluation
    logger.info("Running evaluation...")
    
    # Load best model for evaluation
    best_model_path = os.path.join(training_config.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        trainer.load_checkpoint(best_model_path)
        logger.info("Loaded best model for evaluation")
    
    # Create evaluator
    evaluator = EmbeddingEvaluator(
        model=trainer.model,
        tokenizer=tokenizer,
        device=trainer.device
    )
    
    # Evaluate on validation set
    similarity_metrics = evaluator.evaluate_similarity_task(
        val_texts1, val_texts2, val_labels
    )
    
    logger.info("Evaluation Results:")
    for metric, value in similarity_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save evaluation results
    results_path = os.path.join(training_config.output_dir, "evaluation_results.json")
    import json
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(similarity_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to {results_path}")
    
    # Example: Encode some text
    logger.info("\nExample embeddings:")
    sample_texts = [
        "สวัสดีครับ",
        "วันนี้อากาศดี",
        "ขอบคุณมาก"
    ]
    
    sample_embeddings = evaluator.encode_texts(sample_texts)
    for i, (text, embedding) in enumerate(zip(sample_texts, sample_embeddings)):
        logger.info(f"Text: '{text}'")
        logger.info(f"Embedding shape: {embedding.shape}")
        logger.info(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        if i < len(sample_embeddings) - 1:
            logger.info("---")
    
    logger.info("Script completed successfully!")


if __name__ == "__main__":
    main()
