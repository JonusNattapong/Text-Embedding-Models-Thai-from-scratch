"""
Training Module for Thai Embedding Model

This module handles the training process including:
- Training loop with various loss functions
- Validation and early stopping
- Learning rate scheduling
- Model checkpointing
- Logging and monitoring
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import wandb

from ..models.thai_embedding_model import ThaiEmbeddingModel, ContrastiveLoss
from ..utils.metrics import compute_similarity_metrics, compute_clustering_metrics


@dataclass
class TrainingConfig:
    """Configuration for training the Thai embedding model."""
    
    # Model parameters
    vocab_size: int = 30000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    pooling_strategy: str = "mean"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    
    # Loss function
    loss_function: str = "contrastive"  # "contrastive", "triplet", "cosine"
    temperature: float = 0.07
    margin: float = 0.2
    
    # Validation and checkpointing
    validation_steps: int = 500
    save_steps: int = 1000
    early_stopping_patience: int = 3
    
    # Paths
    output_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Monitoring
    use_wandb: bool = False
    wandb_project: str = "thai-embedding"
    log_level: str = "INFO"


class ThaiTextDataset(Dataset):
    """Dataset class for Thai text pairs."""
    
    def __init__(
        self,
        texts1: List[str],
        texts2: List[str],
        labels: Optional[List[int]] = None,
        tokenizer=None,
        max_length: int = 512
    ):
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        assert len(texts1) == len(texts2), "texts1 and texts2 must have same length"
        if labels is not None:
            assert len(texts1) == len(labels), "texts and labels must have same length"
    
    def __len__(self):
        return len(self.texts1)
    
    def __getitem__(self, idx):
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        
        # Tokenize texts
        if self.tokenizer:
            encoding1 = self.tokenizer.encode(text1, max_length=self.max_length)
            encoding2 = self.tokenizer.encode(text2, max_length=self.max_length)
        else:
            # Fallback to simple word splitting
            words1 = text1.split()[:self.max_length]
            words2 = text2.split()[:self.max_length]
            encoding1 = {
                'input_ids': list(range(len(words1))),
                'attention_mask': [1] * len(words1)
            }
            encoding2 = {
                'input_ids': list(range(len(words2))),
                'attention_mask': [1] * len(words2)
            }
        
        item = {
            'input_ids1': torch.tensor(encoding1['input_ids'], dtype=torch.long),
            'attention_mask1': torch.tensor(encoding1['attention_mask'], dtype=torch.long),
            'input_ids2': torch.tensor(encoding2['input_ids'], dtype=torch.long),
            'attention_mask2': torch.tensor(encoding2['attention_mask'], dtype=torch.long),
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return item


class ThaiEmbeddingTrainer:
    """Trainer class for Thai embedding model."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Training state
        self.global_step = 0
        self.best_score = -float('inf')
        self.epochs_without_improvement = 0
        
        # Initialize wandb if requested
        if config.use_wandb:
            wandb.init(project=config.wandb_project, config=config.__dict__)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_model(self) -> ThaiEmbeddingModel:
        """Create the Thai embedding model."""
        model_config = {
            'vocab_size': self.config.vocab_size,
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_layers': self.config.n_layers,
            'd_ff': self.config.d_ff,
            'max_seq_len': self.config.max_seq_len,
            'dropout': self.config.dropout,
            'pooling_strategy': self.config.pooling_strategy
        }
        
        from ..models.thai_embedding_model import create_thai_embedding_model
        return create_thai_embedding_model(model_config)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=self.config.num_epochs * 1000,  # Approximate
            pct_start=0.1
        )
    
    def _create_loss_function(self):
        """Create loss function based on configuration."""
        if self.config.loss_function == "contrastive":
            return ContrastiveLoss(temperature=self.config.temperature)
        elif self.config.loss_function == "triplet":
            return nn.TripletMarginLoss(margin=self.config.margin)
        elif self.config.loss_function == "cosine":
            return nn.CosineEmbeddingLoss(margin=self.config.margin)
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_function}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        embeddings1 = self.model(
            batch['input_ids1'], 
            batch.get('attention_mask1')
        )
        embeddings2 = self.model(
            batch['input_ids2'], 
            batch.get('attention_mask2')
        )
        
        # Compute loss
        if self.config.loss_function == "contrastive":
            loss = self.criterion(embeddings1, embeddings2)
        elif self.config.loss_function == "cosine":
            # For cosine loss, we need labels
            labels = batch.get('labels', torch.ones(embeddings1.size(0)).to(self.device))
            loss = self.criterion(embeddings1, embeddings2, labels)
        else:
            # For triplet loss, we would need negative samples
            # This is a simplified version
            loss = self.criterion(embeddings1, embeddings2, embeddings1)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.gradient_clipping
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        return {'loss': loss.item()}
    
    def validation_step(self, val_loader: DataLoader) -> Dict[str, float]:
        """Perform validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_embeddings1 = []
        all_embeddings2 = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                embeddings1 = self.model(
                    batch['input_ids1'], 
                    batch.get('attention_mask1')
                )
                embeddings2 = self.model(
                    batch['input_ids2'], 
                    batch.get('attention_mask2')
                )
                
                # Compute loss
                if self.config.loss_function == "contrastive":
                    loss = self.criterion(embeddings1, embeddings2)
                elif self.config.loss_function == "cosine":
                    labels = batch.get('labels', torch.ones(embeddings1.size(0)).to(self.device))
                    loss = self.criterion(embeddings1, embeddings2, labels)
                else:
                    loss = self.criterion(embeddings1, embeddings2, embeddings1)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect embeddings for evaluation metrics
                all_embeddings1.append(embeddings1.cpu())
                all_embeddings2.append(embeddings2.cpu())
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu())
        
        # Compute validation metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Compute similarity metrics
        all_embeddings1 = torch.cat(all_embeddings1, dim=0)
        all_embeddings2 = torch.cat(all_embeddings2, dim=0)
        
        similarity_metrics = compute_similarity_metrics(
            all_embeddings1.numpy(), 
            all_embeddings2.numpy()
        )
        
        metrics = {
            'val_loss': avg_loss,
            **similarity_metrics
        }
        
        return metrics
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None
    ):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            num_batches = 0
            
            # Training loop
            progress_bar = tqdm(
                train_loader, 
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}"
            )
            
            for batch in progress_bar:
                # Training step
                metrics = self.train_step(batch)
                
                epoch_loss += metrics['loss']
                num_batches += 1
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Log to wandb
                if self.config.use_wandb:
                    wandb.log({
                        'train_loss': metrics['loss'],
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
                
                # Validation
                if (val_loader is not None and 
                    self.global_step % self.config.validation_steps == 0):
                    
                    val_metrics = self.validation_step(val_loader)
                    
                    self.logger.info(
                        f"Step {self.global_step} - "
                        f"Val Loss: {val_metrics['val_loss']:.4f}, "
                        f"Cosine Sim: {val_metrics.get('cosine_similarity', 0):.4f}"
                    )
                    
                    # Log validation metrics
                    if self.config.use_wandb:
                        wandb.log({
                            f'val_{k}': v for k, v in val_metrics.items()
                        })
                    
                    # Check for improvement
                    current_score = val_metrics.get('cosine_similarity', val_metrics['val_loss'])
                    if current_score > self.best_score:
                        self.best_score = current_score
                        self.epochs_without_improvement = 0
                        self.save_checkpoint("best_model.pt")
                    else:
                        self.epochs_without_improvement += 1
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_time = time.time() - epoch_start_time
            
            self.logger.info(
                f"Epoch {epoch+1} completed - "
                f"Avg Loss: {avg_epoch_loss:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Early stopping check
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                self.logger.info(
                    f"Early stopping triggered after {self.config.early_stopping_patience} "
                    f"epochs without improvement"
                )
                break
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_score': self.best_score,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_score = checkpoint['best_score']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


def create_data_loaders(
    train_texts1: List[str],
    train_texts2: List[str],
    val_texts1: Optional[List[str]] = None,
    val_texts2: Optional[List[str]] = None,
    train_labels: Optional[List[int]] = None,
    val_labels: Optional[List[int]] = None,
    tokenizer=None,
    batch_size: int = 32,
    max_length: int = 512
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for training and validation."""
    
    # Create datasets
    train_dataset = ThaiTextDataset(
        train_texts1, train_texts2, train_labels, tokenizer, max_length
    )
    
    val_dataset = None
    if val_texts1 is not None and val_texts2 is not None:
        val_dataset = ThaiTextDataset(
            val_texts1, val_texts2, val_labels, tokenizer, max_length
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    return train_loader, val_loader
