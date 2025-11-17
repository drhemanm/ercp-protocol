"""
Model Registry - Centralized Model Loading and Caching
Author: ERCP Protocol Implementation
License: Apache-2.0

Implements singleton pattern for loading and caching ML models in memory.
Supports: LLM models, NLI models, and sentence-transformers.
"""

import logging
import os
from typing import Optional, Any, Dict
import threading


logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry for loading and caching ML models.

    This class ensures that models are loaded only once and cached in memory
    for efficient reuse across multiple requests.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the model registry."""
        if self._initialized:
            return

        self._models: Dict[str, Any] = {}
        self._loading_locks: Dict[str, threading.Lock] = {}
        self._initialized = True
        logger.info("ModelRegistry initialized")

    def _get_lock(self, model_key: str) -> threading.Lock:
        """Get or create a lock for a specific model."""
        if model_key not in self._loading_locks:
            with self._lock:
                if model_key not in self._loading_locks:
                    self._loading_locks[model_key] = threading.Lock()
        return self._loading_locks[model_key]

    def get_generate_model(self, model_name: str = "gpt2") -> Any:
        """
        Get or load a text generation model.

        Args:
            model_name: Name of the model to load (e.g., "gpt2", "facebook/opt-125m")

        Returns:
            Loaded model and tokenizer as a tuple (model, tokenizer)

        Raises:
            Exception: If model loading fails
        """
        model_key = f"generate_{model_name}"

        if model_key in self._models:
            logger.debug(f"Returning cached model: {model_key}")
            return self._models[model_key]

        # Use per-model lock to prevent concurrent loading of the same model
        lock = self._get_lock(model_key)
        with lock:
            # Double-check after acquiring lock
            if model_key in self._models:
                return self._models[model_key]

            logger.info(f"Loading generation model: {model_name}")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                # Determine device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Using device: {device}")

                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                model.to(device)
                model.eval()

                # Set padding token if not set
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                self._models[model_key] = {
                    "model": model,
                    "tokenizer": tokenizer,
                    "device": device
                }

                logger.info(f"Successfully loaded model: {model_name}")
                return self._models[model_key]

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                raise

    def get_nli_model(self, model_name: str = "microsoft/deberta-v3-base-mnli-fever-anli") -> Any:
        """
        Get or load an NLI (Natural Language Inference) model.

        Args:
            model_name: Name of the NLI model to load

        Returns:
            Transformers pipeline for text classification

        Raises:
            Exception: If model loading fails
        """
        model_key = f"nli_{model_name}"

        if model_key in self._models:
            logger.debug(f"Returning cached NLI model: {model_key}")
            return self._models[model_key]

        lock = self._get_lock(model_key)
        with lock:
            if model_key in self._models:
                return self._models[model_key]

            logger.info(f"Loading NLI model: {model_name}")
            try:
                from transformers import pipeline
                import torch

                device = 0 if torch.cuda.is_available() else -1

                # Create NLI pipeline
                nli_pipeline = pipeline(
                    "text-classification",
                    model=model_name,
                    device=device
                )

                self._models[model_key] = nli_pipeline
                logger.info(f"Successfully loaded NLI model: {model_name}")
                return self._models[model_key]

            except Exception as e:
                logger.error(f"Failed to load NLI model {model_name}: {str(e)}")
                raise

    def get_embedding_model(self, model_name: str = "all-MiniLM-L6-v2") -> Any:
        """
        Get or load a sentence embedding model.

        Args:
            model_name: Name of the sentence-transformers model to load

        Returns:
            SentenceTransformer model

        Raises:
            Exception: If model loading fails
        """
        model_key = f"embedding_{model_name}"

        if model_key in self._models:
            logger.debug(f"Returning cached embedding model: {model_key}")
            return self._models[model_key]

        lock = self._get_lock(model_key)
        with lock:
            if model_key in self._models:
                return self._models[model_key]

            logger.info(f"Loading embedding model: {model_name}")
            try:
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(model_name)
                self._models[model_key] = model
                logger.info(f"Successfully loaded embedding model: {model_name}")
                return self._models[model_key]

            except Exception as e:
                logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
                raise

    def warm_up(self, models_to_load: Optional[list] = None):
        """
        Pre-load models on server startup.

        Args:
            models_to_load: List of model types to pre-load
                           Options: ["generate", "nli", "embedding"]
                           If None, loads all models.
        """
        if models_to_load is None:
            models_to_load = ["nli", "embedding"]  # Don't load heavy LLM by default

        logger.info(f"Warming up models: {models_to_load}")

        for model_type in models_to_load:
            try:
                if model_type == "generate":
                    # Use lightweight model for generation
                    self.get_generate_model("gpt2")
                elif model_type == "nli":
                    self.get_nli_model()
                elif model_type == "embedding":
                    self.get_embedding_model()
                else:
                    logger.warning(f"Unknown model type for warm-up: {model_type}")
            except Exception as e:
                logger.error(f"Failed to warm up {model_type} model: {str(e)}")

        logger.info("Model warm-up complete")

    def clear_cache(self):
        """Clear all cached models from memory."""
        with self._lock:
            logger.info(f"Clearing {len(self._models)} cached models")
            self._models.clear()
            logger.info("Model cache cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about currently loaded models.

        Returns:
            Dictionary with model information
        """
        return {
            "loaded_models": list(self._models.keys()),
            "count": len(self._models)
        }


# Global singleton instance
_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global ModelRegistry instance."""
    return _registry
