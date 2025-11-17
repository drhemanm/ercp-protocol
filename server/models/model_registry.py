"""
Model Registry for ERCP Protocol
Manages loading, caching, and accessing ML models.
"""

import os
import hashlib
from typing import Dict, Any, Optional
from functools import lru_cache
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from sentence_transformers import SentenceTransformer
import spacy


class ModelRegistry:
    """Singleton registry for managing ML models."""

    _instance = None
    _models: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the model registry."""
        self.device = self._get_device()
        self.cache_dir = os.getenv("GENERATE_MODEL_CACHE_DIR", "./models/cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_device(self) -> str:
        """Determine the compute device (CPU or CUDA)."""
        device_env = os.getenv("DEVICE", "cpu").lower()
        if device_env.startswith("cuda") and torch.cuda.is_available():
            return device_env
        return "cpu"

    @lru_cache(maxsize=10)
    def get_generation_model(
        self, model_name: Optional[str] = None
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load and cache a text generation model.

        Args:
            model_name: HuggingFace model name (default from env)

        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or os.getenv("GENERATE_MODEL_NAME", "gpt2")
        cache_key = f"gen_{model_name}"

        if cache_key not in self._models:
            print(f"Loading generation model: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=self.cache_dir
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=self.cache_dir
            )
            model = model.to(self.device)
            model.eval()

            self._models[cache_key] = (model, tokenizer)

        return self._models[cache_key]

    @lru_cache(maxsize=5)
    def get_nli_pipeline(self, model_name: Optional[str] = None):
        """
        Load and cache an NLI model for verification.

        Args:
            model_name: HuggingFace NLI model name

        Returns:
            HuggingFace pipeline for text classification
        """
        model_name = model_name or os.getenv(
            "NLI_MODEL_NAME", "microsoft/deberta-v3-base-mnli"
        )
        cache_key = f"nli_{model_name}"

        if cache_key not in self._models:
            print(f"Loading NLI model: {model_name}")
            device_id = 0 if self.device.startswith("cuda") else -1
            nli_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=device_id,
                cache_dir=self.cache_dir,
            )
            self._models[cache_key] = nli_pipeline

        return self._models[cache_key]

    @lru_cache(maxsize=5)
    def get_sentence_transformer(self, model_name: Optional[str] = None):
        """
        Load and cache a sentence transformer for semantic similarity.

        Args:
            model_name: Sentence transformer model name

        Returns:
            SentenceTransformer instance
        """
        model_name = model_name or os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
        )
        cache_key = f"st_{model_name}"

        if cache_key not in self._models:
            print(f"Loading sentence transformer: {model_name}")
            st_model = SentenceTransformer(
                model_name, cache_folder=self.cache_dir, device=self.device
            )
            self._models[cache_key] = st_model

        return self._models[cache_key]

    @lru_cache(maxsize=3)
    def get_spacy_nlp(self, model_name: Optional[str] = None):
        """
        Load and cache a spaCy NLP model.

        Args:
            model_name: spaCy model name

        Returns:
            spaCy Language object
        """
        model_name = model_name or os.getenv("SPACY_MODEL", "en_core_web_sm")
        cache_key = f"spacy_{model_name}"

        if cache_key not in self._models:
            print(f"Loading spaCy model: {model_name}")
            try:
                nlp = spacy.load(model_name)
            except OSError:
                # Model not found, attempt to download
                print(f"Downloading spaCy model: {model_name}")
                os.system(f"python -m spacy download {model_name}")
                nlp = spacy.load(model_name)

            self._models[cache_key] = nlp

        return self._models[cache_key]

    def get_model_fingerprint(self, model_name: str) -> str:
        """
        Generate a fingerprint for a model.

        Args:
            model_name: Model identifier

        Returns:
            SHA256 hash fingerprint
        """
        # In production, this should hash actual model weights
        # For now, use model name and version as proxy
        fingerprint_str = f"{model_name}:{self.device}"
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    def clear_cache(self):
        """Clear all cached models (useful for memory management)."""
        self._models.clear()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("Model cache cleared")


# Global singleton instance
model_registry = ModelRegistry()
