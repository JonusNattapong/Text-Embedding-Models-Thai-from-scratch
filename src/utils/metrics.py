"""
Evaluation Metrics for Thai Embedding Model

This module provides various evaluation metrics for assessing the quality
of Thai text embeddings including similarity, clustering, and retrieval metrics.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch


def compute_similarity_metrics(
    embeddings1: np.ndarray, 
    embeddings2: np.ndarray,
    labels: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute various similarity metrics between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings [n_samples, embedding_dim]
        embeddings2: Second set of embeddings [n_samples, embedding_dim]
        labels: Optional similarity labels (1 for similar, 0 for dissimilar)
        
    Returns:
        Dictionary of similarity metrics
    """
    assert embeddings1.shape == embeddings2.shape, "Embeddings must have same shape"
    
    metrics = {}
    
    # Cosine similarity
    cosine_sims = []
    for i in range(len(embeddings1)):
        sim = 1 - cosine(embeddings1[i], embeddings2[i])
        cosine_sims.append(sim)
    
    cosine_sims = np.array(cosine_sims)
    metrics['cosine_similarity'] = np.mean(cosine_sims)
    metrics['cosine_similarity_std'] = np.std(cosine_sims)
    
    # Euclidean distance
    euclidean_dists = []
    for i in range(len(embeddings1)):
        dist = euclidean(embeddings1[i], embeddings2[i])
        euclidean_dists.append(dist)
    
    euclidean_dists = np.array(euclidean_dists)
    metrics['euclidean_distance'] = np.mean(euclidean_dists)
    metrics['euclidean_distance_std'] = np.std(euclidean_dists)
    
    # Dot product similarity
    dot_products = np.sum(embeddings1 * embeddings2, axis=1)
    metrics['dot_product'] = np.mean(dot_products)
    metrics['dot_product_std'] = np.std(dot_products)
    
    # If labels are provided, compute classification metrics
    if labels is not None:
        # Use cosine similarity as predictions (threshold at 0.5)
        predictions = (cosine_sims > 0.5).astype(int)
        
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['f1'] = f1_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions, zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, zero_division=0)
        
        # Correlation with labels
        if len(set(labels)) > 1:  # Only if there's variation in labels
            corr_pearson, _ = pearsonr(cosine_sims, labels)
            corr_spearman, _ = spearmanr(cosine_sims, labels)
            
            metrics['pearson_correlation'] = corr_pearson
            metrics['spearman_correlation'] = corr_spearman
    
    return metrics


def compute_clustering_metrics(
    embeddings: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        embeddings: Input embeddings [n_samples, embedding_dim]
        true_labels: True cluster labels (if available)
        n_clusters: Number of clusters for k-means
        
    Returns:
        Dictionary of clustering metrics
    """
    metrics = {}
    
    # If n_clusters not provided, try to infer from true_labels
    if n_clusters is None and true_labels is not None:
        n_clusters = len(np.unique(true_labels))
    elif n_clusters is None:
        n_clusters = min(8, max(2, int(np.sqrt(len(embeddings)))))
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    predicted_labels = kmeans.fit_predict(embeddings)
    
    # Silhouette score (higher is better)
    if len(embeddings) > n_clusters:
        silhouette = silhouette_score(embeddings, predicted_labels)
        metrics['silhouette_score'] = silhouette
    
    # Inertia (lower is better)
    metrics['kmeans_inertia'] = kmeans.inertia_
    
    # If true labels are available, compute external metrics
    if true_labels is not None:
        # Adjusted Rand Index (higher is better, max=1)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        metrics['adjusted_rand_index'] = ari
        
        # Normalized Mutual Information (higher is better, max=1)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        metrics['normalized_mutual_info'] = nmi
    
    return metrics


def compute_retrieval_metrics(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    relevance_labels: np.ndarray,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute information retrieval metrics.
    
    Args:
        query_embeddings: Query embeddings [n_queries, embedding_dim]
        document_embeddings: Document embeddings [n_docs, embedding_dim]
        relevance_labels: Relevance matrix [n_queries, n_docs] (1=relevant, 0=not relevant)
        k_values: List of k values for top-k metrics
        
    Returns:
        Dictionary of retrieval metrics
    """
    metrics = {}
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(query_embeddings, document_embeddings)
    
    # For each query, get top-k most similar documents
    precision_at_k = {k: [] for k in k_values}
    recall_at_k = {k: [] for k in k_values}
    
    for query_idx in range(len(query_embeddings)):
        # Get similarity scores for this query
        query_similarities = similarity_matrix[query_idx]
        
        # Get top-k document indices
        top_k_indices = np.argsort(query_similarities)[::-1]
        
        # Get relevance labels for this query
        query_relevance = relevance_labels[query_idx]
        n_relevant = np.sum(query_relevance)
        
        if n_relevant > 0:  # Only evaluate if there are relevant documents
            for k in k_values:
                if k <= len(top_k_indices):
                    # Get top-k indices
                    top_k = top_k_indices[:k]
                    
                    # Count relevant documents in top-k
                    relevant_in_top_k = np.sum(query_relevance[top_k])
                    
                    # Precision@k
                    precision = relevant_in_top_k / k
                    precision_at_k[k].append(precision)
                    
                    # Recall@k
                    recall = relevant_in_top_k / n_relevant
                    recall_at_k[k].append(recall)
    
    # Average metrics across queries
    for k in k_values:
        if precision_at_k[k]:
            metrics[f'precision_at_{k}'] = np.mean(precision_at_k[k])
            metrics[f'recall_at_{k}'] = np.mean(recall_at_k[k])
    
    # Mean Average Precision (MAP)
    average_precisions = []
    for query_idx in range(len(query_embeddings)):
        query_similarities = similarity_matrix[query_idx]
        top_indices = np.argsort(query_similarities)[::-1]
        query_relevance = relevance_labels[query_idx]
        
        if np.sum(query_relevance) > 0:
            precisions = []
            relevant_count = 0
            
            for i, doc_idx in enumerate(top_indices):
                if query_relevance[doc_idx] == 1:
                    relevant_count += 1
                    precision = relevant_count / (i + 1)
                    precisions.append(precision)
            
            if precisions:
                average_precisions.append(np.mean(precisions))
    
    if average_precisions:
        metrics['mean_average_precision'] = np.mean(average_precisions)
    
    return metrics


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    title: str = "Embedding Visualization",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings: Input embeddings [n_samples, embedding_dim]
        labels: Optional labels for coloring points
        method: Dimensionality reduction method ("tsne", "pca")
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    # Dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        # Color points by labels
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f"Class {label}",
                alpha=0.7,
                s=50
            )
        ax.legend()
    else:
        # Single color for all points
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=0.7,
            s=50
        )
    
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_embedding_quality_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute general quality metrics for embeddings.
    
    Args:
        embeddings: Input embeddings [n_samples, embedding_dim]
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {}
    
    # Dimensionality
    metrics['n_samples'] = embeddings.shape[0]
    metrics['embedding_dim'] = embeddings.shape[1]
    
    # Statistical properties
    metrics['mean_norm'] = np.mean(np.linalg.norm(embeddings, axis=1))
    metrics['std_norm'] = np.std(np.linalg.norm(embeddings, axis=1))
    
    # Component-wise statistics
    metrics['mean_value'] = np.mean(embeddings)
    metrics['std_value'] = np.std(embeddings)
    metrics['min_value'] = np.min(embeddings)
    metrics['max_value'] = np.max(embeddings)
    
    # Sparsity
    zero_threshold = 1e-6
    sparsity = np.mean(np.abs(embeddings) < zero_threshold)
    metrics['sparsity'] = sparsity
    
    # Effective dimensionality (based on PCA)
    if embeddings.shape[1] > 1:
        pca = PCA()
        pca.fit(embeddings)
        
        # Cumulative explained variance
        cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # Effective dimensionality (dimensions needed for 95% variance)
        effective_dim = np.argmax(cumsum_variance >= 0.95) + 1
        metrics['effective_dimensionality'] = effective_dim
        
        # Concentration (how much variance is in top components)
        metrics['top_10_variance_ratio'] = np.sum(pca.explained_variance_ratio_[:10])
    
    return metrics


class EmbeddingEvaluator:
    """Comprehensive evaluator for embedding models."""
    
    def __init__(self, model, tokenizer=None, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode a list of texts to embeddings."""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch_texts:
                if self.tokenizer:
                    encoding = self.tokenizer.encode(text)
                    input_ids = torch.tensor([encoding['input_ids']], device=self.device)
                    attention_mask = torch.tensor([encoding['attention_mask']], device=self.device)
                else:
                    # Fallback encoding
                    words = text.split()
                    input_ids = torch.tensor([list(range(len(words)))], device=self.device)
                    attention_mask = torch.ones_like(input_ids)
                
                with torch.no_grad():
                    embedding = self.model(input_ids, attention_mask)
                    batch_embeddings.append(embedding.cpu().numpy())
            
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def evaluate_similarity_task(
        self,
        texts1: List[str],
        texts2: List[str],
        labels: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Evaluate on text similarity task."""
        embeddings1 = self.encode_texts(texts1)
        embeddings2 = self.encode_texts(texts2)
        
        return compute_similarity_metrics(embeddings1, embeddings2, labels)
    
    def evaluate_clustering_task(
        self,
        texts: List[str],
        true_labels: Optional[List[int]] = None,
        n_clusters: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate on text clustering task."""
        embeddings = self.encode_texts(texts)
        
        return compute_clustering_metrics(
            embeddings,
            np.array(true_labels) if true_labels else None,
            n_clusters
        )
    
    def evaluate_retrieval_task(
        self,
        queries: List[str],
        documents: List[str],
        relevance_matrix: np.ndarray,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """Evaluate on information retrieval task."""
        query_embeddings = self.encode_texts(queries)
        doc_embeddings = self.encode_texts(documents)
        
        return compute_retrieval_metrics(
            query_embeddings,
            doc_embeddings,
            relevance_matrix,
            k_values
        )
