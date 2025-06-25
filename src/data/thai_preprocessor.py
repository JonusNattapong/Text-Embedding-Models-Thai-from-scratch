"""
Thai Text Preprocessing Module

This module handles Thai-specific text preprocessing including:
- Text cleaning and normalization
- Word segmentation using PyThaiNLP
- Custom tokenization for Thai text
- Data augmentation techniques
"""

import re
import string
from typing import List, Dict, Optional, Tuple
import unicodedata

import pandas as pd
import numpy as np
from pythainlp import word_tokenize, sent_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.util import normalize
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.processors import BertProcessing


class ThaiTextPreprocessor:
    """Thai text preprocessing pipeline."""
    
    def __init__(self, remove_stopwords: bool = False, normalize_text: bool = True):
        self.remove_stopwords = remove_stopwords
        self.normalize_text = normalize_text
        self.stopwords = set(thai_stopwords()) if remove_stopwords else set()
        
        # Common Thai punctuation and special characters
        self.thai_punctuation = "๏๎๑๒๓๔๕๖๗๘๙๐ฯๆฯฯ"
        
    def clean_text(self, text: str) -> str:
        """
        Clean Thai text by removing unwanted characters and normalizing.
        
        Args:
            text: Input Thai text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Normalize Thai text
        if self.normalize_text:
            text = normalize(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (Thai format)
        text = re.sub(r'0\d{1,2}-?\d{3}-?\d{4}', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def segment_words(self, text: str, engine: str = "newmm") -> List[str]:
        """
        Segment Thai text into words.
        
        Args:
            text: Input Thai text
            engine: Word segmentation engine ("newmm", "longest", "deepcut")
            
        Returns:
            List of segmented words
        """
        if not text:
            return []
        
        words = word_tokenize(text, engine=engine, keep_whitespace=False)
        
        # Filter out stopwords if requested
        if self.remove_stopwords:
            words = [word for word in words if word not in self.stopwords]
        
        # Filter out single characters and punctuation
        words = [word for word in words if len(word) > 1 or word.isalnum()]
        
        return words
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment Thai text into sentences.
        
        Args:
            text: Input Thai text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        sentences = sent_tokenize(text)
        
        # Clean each sentence
        cleaned_sentences = []
        for sentence in sentences:
            cleaned_sentence = self.clean_text(sentence)
            if cleaned_sentence and len(cleaned_sentence.split()) > 2:
                cleaned_sentences.append(cleaned_sentence)
        
        return cleaned_sentences
    
    def preprocess_text(self, text: str) -> Dict[str, any]:
        """
        Complete preprocessing pipeline for Thai text.
        
        Args:
            text: Input Thai text
            
        Returns:
            Dictionary containing processed text information
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Segment into sentences
        sentences = self.segment_sentences(cleaned_text)
        
        # Segment into words
        words = self.segment_words(cleaned_text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'words': words,
            'word_count': len(words),
            'sentence_count': len(sentences)
        }


class ThaiTokenizer:
    """Custom tokenizer for Thai text using SentencePiece-like approach."""
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.preprocessor = ThaiTextPreprocessor()
        
    def train_tokenizer(self, texts: List[str], save_path: Optional[str] = None) -> None:
        """
        Train a custom tokenizer on Thai text corpus.
        
        Args:
            texts: List of Thai texts for training
            save_path: Path to save the trained tokenizer
        """
        # Initialize tokenizer
        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        
        # Set pre-tokenizer (split on whitespace and punctuation)
        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation()
        ])
        
        # Set decoder
        self.tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
        
        # Initialize trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
            show_progress=True
        )
        
        # Preprocess texts
        processed_texts = []
        for text in texts:
            processed = self.preprocessor.clean_text(text)
            if processed:
                # Segment words and join with spaces
                words = self.preprocessor.segment_words(processed)
                processed_texts.append(" ".join(words))
        
        # Train tokenizer
        self.tokenizer.train_from_iterator(processed_texts, trainer)
        
        # Add post-processor for BERT-like format
        self.tokenizer.post_processor = BertProcessing(
            sep=("[SEP]", self.tokenizer.token_to_id("[SEP]")),
            cls=("[CLS]", self.tokenizer.token_to_id("[CLS]"))
        )
        
        # Save tokenizer if path provided
        if save_path:
            self.tokenizer.save(save_path)
    
    def load_tokenizer(self, path: str) -> None:
        """Load a pre-trained tokenizer."""
        self.tokenizer = Tokenizer.from_file(path)
    
    def encode(self, text: str, max_length: int = 512) -> Dict[str, List[int]]:
        """
        Encode Thai text to token IDs.
        
        Args:
            text: Input Thai text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained or loaded. Call train_tokenizer() or load_tokenizer() first.")
        
        # Preprocess text
        processed_text = self.preprocessor.clean_text(text)
        words = self.preprocessor.segment_words(processed_text)
        processed_text = " ".join(words)
        
        # Encode
        encoding = self.tokenizer.encode(processed_text)
        
        # Truncate or pad to max_length
        input_ids = encoding.ids[:max_length]
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            pad_token_id = self.tokenizer.token_to_id("[PAD]")
            input_ids.extend([pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not trained or loaded.")
        
        return self.tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self.tokenizer:
            return 0
        return self.tokenizer.get_vocab_size()


class ThaiDataAugmentation:
    """Data augmentation techniques for Thai text."""
    
    def __init__(self):
        self.preprocessor = ThaiTextPreprocessor()
    
    def synonym_replacement(self, text: str, num_replacements: int = 1) -> str:
        """
        Replace words with synonyms (placeholder - would need Thai synonym dictionary).
        
        Args:
            text: Input text
            num_replacements: Number of words to replace
            
        Returns:
            Augmented text
        """
        # This is a placeholder. In practice, you would need:
        # - Thai synonym dictionary
        # - Word sense disambiguation
        # - Context-aware replacement
        
        words = self.preprocessor.segment_words(text)
        if len(words) <= num_replacements:
            return text
        
        # For now, just return original text
        # TODO: Implement with Thai synonym resources
        return text
    
    def random_deletion(self, text: str, deletion_rate: float = 0.1) -> str:
        """
        Randomly delete words from text.
        
        Args:
            text: Input text
            deletion_rate: Proportion of words to delete
            
        Returns:
            Augmented text with some words deleted
        """
        words = self.preprocessor.segment_words(text)
        if len(words) <= 1:
            return text
        
        num_deletions = max(1, int(len(words) * deletion_rate))
        indices_to_delete = np.random.choice(
            len(words), size=num_deletions, replace=False
        )
        
        augmented_words = [
            word for i, word in enumerate(words) 
            if i not in indices_to_delete
        ]
        
        return " ".join(augmented_words)
    
    def random_insertion(self, text: str, insertion_rate: float = 0.1) -> str:
        """
        Randomly insert words into text.
        
        Args:
            text: Input text
            insertion_rate: Proportion of words to insert
            
        Returns:
            Augmented text with randomly inserted words
        """
        words = self.preprocessor.segment_words(text)
        if len(words) <= 1:
            return text
        
        num_insertions = max(1, int(len(words) * insertion_rate))
        
        for _ in range(num_insertions):
            # Insert a random word from the existing words
            random_word = np.random.choice(words)
            random_position = np.random.randint(0, len(words) + 1)
            words.insert(random_position, random_word)
        
        return " ".join(words)
    
    def random_swap(self, text: str, num_swaps: int = 1) -> str:
        """
        Randomly swap positions of words.
        
        Args:
            text: Input text
            num_swaps: Number of swaps to perform
            
        Returns:
            Augmented text with swapped words
        """
        words = self.preprocessor.segment_words(text)
        if len(words) <= 1:
            return text
        
        for _ in range(num_swaps):
            if len(words) >= 2:
                idx1, idx2 = np.random.choice(len(words), size=2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return " ".join(words)


def create_thai_dataset(
    texts: List[str],
    labels: Optional[List[any]] = None,
    augment: bool = False,
    augmentation_factor: int = 2
) -> pd.DataFrame:
    """
    Create a processed dataset from Thai texts.
    
    Args:
        texts: List of Thai texts
        labels: Optional labels for supervised tasks
        augment: Whether to apply data augmentation
        augmentation_factor: How many augmented versions to create per text
        
    Returns:
        Processed dataset as DataFrame
    """
    preprocessor = ThaiTextPreprocessor()
    augmenter = ThaiDataAugmentation() if augment else None
    
    processed_data = []
    
    for i, text in enumerate(texts):
        # Process original text
        processed = preprocessor.preprocess_text(text)
        data_point = {
            'text': processed['cleaned_text'],
            'word_count': processed['word_count'],
            'sentence_count': processed['sentence_count'],
            'is_augmented': False
        }
        
        if labels is not None:
            data_point['label'] = labels[i]
        
        processed_data.append(data_point)
        
        # Add augmented versions if requested
        if augment and augmenter:
            for _ in range(augmentation_factor):
                # Apply random augmentation
                aug_method = np.random.choice([
                    augmenter.random_deletion,
                    augmenter.random_insertion,
                    augmenter.random_swap
                ])
                
                augmented_text = aug_method(processed['cleaned_text'])
                
                aug_data_point = {
                    'text': augmented_text,
                    'word_count': len(augmented_text.split()),
                    'sentence_count': len(augmented_text.split('.')),
                    'is_augmented': True
                }
                
                if labels is not None:
                    aug_data_point['label'] = labels[i]
                
                processed_data.append(aug_data_point)
    
    return pd.DataFrame(processed_data)
