# Thai Text Embedding Models - Source Package

from .models.thai_embedding_model import ThaiEmbeddingModel, create_thai_embedding_model
from .data.thai_preprocessor import ThaiTextPreprocessor, ThaiTokenizer
from .training.trainer import ThaiEmbeddingTrainer, TrainingConfig
from .utils.metrics import EmbeddingEvaluator

__version__ = "0.1.0"
__author__ = "Thai Embedding Team"
__description__ = "Thai text embedding models built from scratch"

__all__ = [
    "ThaiEmbeddingModel",
    "create_thai_embedding_model", 
    "ThaiTextPreprocessor",
    "ThaiTokenizer",
    "ThaiEmbeddingTrainer",
    "TrainingConfig",
    "EmbeddingEvaluator"
]
