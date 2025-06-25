#!/usr/bin/env python3
"""
Evaluation Script for Thai Embedding Model

This script provides comprehensive evaluation of trained Thai embedding models
including similarity, clustering, and retrieval tasks.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.thai_preprocessor import ThaiTokenizer
from models.thai_embedding_model import ThaiEmbeddingModel
from utils.metrics import (
    EmbeddingEvaluator, 
    compute_similarity_metrics,
    compute_clustering_metrics,
    compute_retrieval_metrics,
    visualize_embeddings,
    compute_embedding_quality_metrics
)


def load_model_and_tokenizer(
    model_path: str, 
    tokenizer_path: str,
    device: str = 'cpu'
) -> Tuple[ThaiEmbeddingModel, ThaiTokenizer]:
    """Load trained model and tokenizer."""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    
    # Create model
    from models.thai_embedding_model import create_thai_embedding_model
    model = create_thai_embedding_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = ThaiTokenizer()
    tokenizer.load_tokenizer(tokenizer_path)
    
    return model, tokenizer


def prepare_evaluation_data() -> Dict[str, any]:
    """Prepare various evaluation datasets."""
    
    # Similarity task data
    similarity_texts1 = [
        "สวัสดีครับ ยินดีที่ได้รู้จัก",
        "วันนี้อากาศดีมาก",
        "ขอบคุณสำหรับความช่วยเหลือ",
        "ผมชอบอาหารไทยมาก",
        "การศึกษามีความสำคัญ",
    ]
    
    similarity_texts2 = [
        "สวัสดีค่ะ ดีใจที่ได้พบ",
        "อากาศวันนี้สดใส",
        "ขอบใจที่ช่วยเหลือ",
        "อาหารไทยรสชาติดี",
        "การเรียนรู้สำคัญมาก",
    ]
    
    similarity_labels = [1, 1, 1, 1, 1]  # All similar pairs
    
    # Add some dissimilar pairs
    similarity_texts1.extend([
        "สวัสดีครับ",
        "วันนี้ฝนตก",
        "ผมหิวข้าว"
    ])
    similarity_texts2.extend([
        "เทคโนโลยีก้าวหน้า",
        "รักการอ่านหนังสือ",
        "ดนตรีไทยไพเราะ"
    ])
    similarity_labels.extend([0, 0, 0])  # Dissimilar pairs
    
    # Clustering task data
    clustering_texts = [
        # Technology cluster
        "เทคโนโลยีก้าวหน้าอย่างรวดเร็ว",
        "คอมพิวเตอร์เปลี่ยนชีวิตเรา",
        "อินเทอร์เน็ตเชื่อมต่อโลก",
        "ปัญญาประดิษฐ์พัฒนาขึ้น",
        
        # Food cluster
        "อาหารไทยมีรสชาติจัดจ้าน",
        "ส้มตำเป็นอาหารยอดนิยม",
        "ต้มยำกุ้งเป็นอาหารไทยที่มีชื่อ",
        "ผัดไทยอร่อยมาก",
        
        # Education cluster
        "การศึกษาพัฒนาคน",
        "การเรียนรู้ไม่มีวันสิ้นสุด",
        "ครูเป็นผู้ถ่ายทอดความรู้",
        "โรงเรียนคือสถานที่เรียนรู้",
    ]
    
    clustering_labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    
    # Retrieval task data
    queries = [
        "อาหารไทยอร่อย",
        "เทคโนโลยีใหม่",
        "การศึกษาสำคัญ"
    ]
    
    documents = [
        "ส้มตำเป็นอาหารไทยที่อร่อย",
        "ผัดไทยมีรสชาติดี",
        "คอมพิวเตอร์เป็นเทคโนโลยีทันสมัย",
        "อินเทอร์เน็ตเป็นนวัตกรรมสำคัญ",
        "การเรียนรู้พัฒนาตนเอง",
        "โรงเรียนสอนความรู้",
        "ดนตรีไทยไพเราะ",
        "ธรรมชาติสวยงาม"
    ]
    
    # Relevance matrix (queries x documents)
    relevance_matrix = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0],  # Query 0: food related
        [0, 0, 1, 1, 0, 0, 0, 0],  # Query 1: technology related
        [0, 0, 0, 0, 1, 1, 0, 0],  # Query 2: education related
    ])
    
    return {
        'similarity': {
            'texts1': similarity_texts1,
            'texts2': similarity_texts2,
            'labels': similarity_labels
        },
        'clustering': {
            'texts': clustering_texts,
            'labels': clustering_labels
        },
        'retrieval': {
            'queries': queries,
            'documents': documents,
            'relevance_matrix': relevance_matrix
        }
    }


def run_similarity_evaluation(
    evaluator: EmbeddingEvaluator,
    data: Dict[str, any],
    output_dir: str
) -> Dict[str, float]:
    """Run similarity evaluation."""
    logging.info("Running similarity evaluation...")
    
    metrics = evaluator.evaluate_similarity_task(
        data['texts1'],
        data['texts2'],
        data['labels']
    )
    
    # Save detailed results
    results_path = os.path.join(output_dir, 'similarity_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics


def run_clustering_evaluation(
    evaluator: EmbeddingEvaluator,
    data: Dict[str, any],
    output_dir: str
) -> Dict[str, float]:
    """Run clustering evaluation."""
    logging.info("Running clustering evaluation...")
    
    metrics = evaluator.evaluate_clustering_task(
        data['texts'],
        data['labels']
    )
    
    # Visualize embeddings
    embeddings = evaluator.encode_texts(data['texts'])
    
    # t-SNE visualization
    fig_tsne = visualize_embeddings(
        embeddings,
        np.array(data['labels']),
        method='tsne',
        title='Thai Text Embeddings (t-SNE)',
        save_path=os.path.join(output_dir, 'embeddings_tsne.png')
    )
    plt.close(fig_tsne)
    
    # PCA visualization
    fig_pca = visualize_embeddings(
        embeddings,
        np.array(data['labels']),
        method='pca',
        title='Thai Text Embeddings (PCA)',
        save_path=os.path.join(output_dir, 'embeddings_pca.png')
    )
    plt.close(fig_pca)
    
    # Save results
    results_path = os.path.join(output_dir, 'clustering_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics


def run_retrieval_evaluation(
    evaluator: EmbeddingEvaluator,
    data: Dict[str, any],
    output_dir: str
) -> Dict[str, float]:
    """Run retrieval evaluation."""
    logging.info("Running retrieval evaluation...")
    
    metrics = evaluator.evaluate_retrieval_task(
        data['queries'],
        data['documents'],
        data['relevance_matrix']
    )
    
    # Save results
    results_path = os.path.join(output_dir, 'retrieval_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics


def run_quality_analysis(
    evaluator: EmbeddingEvaluator,
    sample_texts: List[str],
    output_dir: str
) -> Dict[str, float]:
    """Run embedding quality analysis."""
    logging.info("Running embedding quality analysis...")
    
    embeddings = evaluator.encode_texts(sample_texts)
    quality_metrics = compute_embedding_quality_metrics(embeddings)
    
    # Save results
    results_path = os.path.join(output_dir, 'quality_analysis.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(quality_metrics, f, indent=2, ensure_ascii=False)
    
    return quality_metrics


def generate_evaluation_report(
    similarity_metrics: Dict[str, float],
    clustering_metrics: Dict[str, float],
    retrieval_metrics: Dict[str, float],
    quality_metrics: Dict[str, float],
    output_dir: str
):
    """Generate comprehensive evaluation report."""
    
    report = f"""
# Thai Embedding Model Evaluation Report

## Model Performance Summary

### Similarity Task
- **Cosine Similarity**: {similarity_metrics.get('cosine_similarity', 0):.4f}
- **Accuracy**: {similarity_metrics.get('accuracy', 0):.4f}
- **F1 Score**: {similarity_metrics.get('f1', 0):.4f}
- **Pearson Correlation**: {similarity_metrics.get('pearson_correlation', 0):.4f}

### Clustering Task
- **Silhouette Score**: {clustering_metrics.get('silhouette_score', 0):.4f}
- **Adjusted Rand Index**: {clustering_metrics.get('adjusted_rand_index', 0):.4f}
- **Normalized Mutual Info**: {clustering_metrics.get('normalized_mutual_info', 0):.4f}

### Retrieval Task
- **Precision@1**: {retrieval_metrics.get('precision_at_1', 0):.4f}
- **Precision@5**: {retrieval_metrics.get('precision_at_5', 0):.4f}
- **Recall@5**: {retrieval_metrics.get('recall_at_5', 0):.4f}
- **Mean Average Precision**: {retrieval_metrics.get('mean_average_precision', 0):.4f}

### Embedding Quality
- **Embedding Dimension**: {quality_metrics.get('embedding_dim', 0)}
- **Effective Dimensionality**: {quality_metrics.get('effective_dimensionality', 0)}
- **Mean Norm**: {quality_metrics.get('mean_norm', 0):.4f}
- **Sparsity**: {quality_metrics.get('sparsity', 0):.4f}

## Analysis

### Strengths
- The model shows good performance on similarity tasks with cosine similarity of {similarity_metrics.get('cosine_similarity', 0):.4f}
- Clustering performance indicates the model can capture semantic groups
- Retrieval metrics suggest the model is useful for information retrieval tasks

### Areas for Improvement
- Consider increasing model capacity if underfitting
- Data augmentation could improve robustness
- Fine-tuning on domain-specific data might boost performance

## Recommendations
1. Evaluate on larger, more diverse Thai datasets
2. Compare with existing Thai language models
3. Consider task-specific fine-tuning for production use
4. Implement more sophisticated evaluation metrics

Generated by Thai Embedding Model Evaluation Script
"""
    
    report_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logging.info(f"Evaluation report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Thai Embedding Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to trained tokenizer"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on (cpu/cuda)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("Starting Thai Embedding Model Evaluation")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Tokenizer: {args.tokenizer_path}")
    logger.info(f"Output: {args.output_dir}")
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        args.tokenizer_path,
        args.device
    )
    
    # Create evaluator
    evaluator = EmbeddingEvaluator(model, tokenizer, args.device)
    
    # Prepare evaluation data
    logger.info("Preparing evaluation data...")
    eval_data = prepare_evaluation_data()
    
    # Run evaluations
    similarity_metrics = run_similarity_evaluation(
        evaluator, eval_data['similarity'], args.output_dir
    )
    
    clustering_metrics = run_clustering_evaluation(
        evaluator, eval_data['clustering'], args.output_dir
    )
    
    retrieval_metrics = run_retrieval_evaluation(
        evaluator, eval_data['retrieval'], args.output_dir
    )
    
    # Run quality analysis
    all_texts = (eval_data['similarity']['texts1'] + 
                eval_data['similarity']['texts2'] + 
                eval_data['clustering']['texts'])
    
    quality_metrics = run_quality_analysis(
        evaluator, all_texts, args.output_dir
    )
    
    # Generate comprehensive report
    generate_evaluation_report(
        similarity_metrics,
        clustering_metrics,
        retrieval_metrics,
        quality_metrics,
        args.output_dir
    )
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    logger.info(f"Cosine Similarity: {similarity_metrics.get('cosine_similarity', 0):.4f}")
    logger.info(f"Clustering ARI: {clustering_metrics.get('adjusted_rand_index', 0):.4f}")
    logger.info(f"Retrieval MAP: {retrieval_metrics.get('mean_average_precision', 0):.4f}")
    logger.info(f"Embedding Dim: {quality_metrics.get('embedding_dim', 0)}")
    logger.info("="*50)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
