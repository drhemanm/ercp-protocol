"""
Model Registry for ERCP Protocol
Manages loading, caching, and accessing ML models.
"""

import os
import time
import hashlib
from typing import Dict, Any, Optional
from functools import lru_cache
import torch
import psutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)
from sentence_transformers import SentenceTransformer
import spacy


class ModelRegistry:
    """Singleton registry for managing ML models with memory-aware LRU eviction."""

    _instance = None
    _models: Dict[str, Any] = {}
    _last_used: Dict[str, float] = {}
    _model_sizes: Dict[str, float] = {}

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

        # Memory management settings
        self.max_memory_gb = float(os.getenv("MODEL_REGISTRY_MAX_MEMORY_GB", "8.0"))
        self.model_ttl_seconds = float(os.getenv("MODEL_TTL_SECONDS", "3600"))  # 1 hour default

    def _get_device(self) -> str:
        """Determine the compute device (CPU or CUDA)."""
        device_env = os.getenv("DEVICE", "cpu").lower()
        if device_env.startswith("cuda") and torch.cuda.is_available():
            return device_env
        return "cpu"

    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in GB.

        Returns:
            Memory usage in GB
        """
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024**3)
        return memory_gb

    def _estimate_model_size(self, model) -> float:
        """
        Estimate model size in GB.

        Args:
            model: PyTorch model

        Returns:
            Estimated size in GB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_gb = (param_size + buffer_size) / (1024**3)
        return size_gb

    def _evict_lru_model(self):
        """
        Evict the least recently used model to free memory.
        """
        if not self._last_used:
            return

        # Find LRU model
        lru_key = min(self._last_used.keys(), key=lambda k: self._last_used[k])
        lru_time = self._last_used[lru_key]

        print(f"Evicting LRU model: {lru_key} (last used {time.time() - lru_time:.1f}s ago)")

        # Remove from cache
        if lru_key in self._models:
            del self._models[lru_key]
        del self._last_used[lru_key]
        if lru_key in self._model_sizes:
            del self._model_sizes[lru_key]

        # Clear CUDA cache if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _check_and_evict_stale_models(self):
        """
        Check for stale models (beyond TTL) and evict them.
        """
        current_time = time.time()
        stale_keys = []

        for key, last_used_time in self._last_used.items():
            if current_time - last_used_time > self.model_ttl_seconds:
                stale_keys.append(key)

        for key in stale_keys:
            print(f"Evicting stale model: {key} (TTL exceeded)")
            if key in self._models:
                del self._models[key]
            del self._last_used[key]
            if key in self._model_sizes:
                del self._model_sizes[key]

        if stale_keys and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_memory_available(self, estimated_size_gb: float = 2.0):
        """
        Ensure sufficient memory is available for loading a new model.

        Args:
            estimated_size_gb: Estimated size of model to load
        """
        # Check for stale models first
        self._check_and_evict_stale_models()

        # Check memory pressure
        current_memory = self._get_memory_usage()

        while current_memory + estimated_size_gb > self.max_memory_gb and self._models:
            print(f"Memory pressure detected: {current_memory:.2f}GB + {estimated_size_gb:.2f}GB > {self.max_memory_gb}GB")
            self._evict_lru_model()
            current_memory = self._get_memory_usage()

    def get_generation_model(
        self, model_name: Optional[str] = None
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load and cache a text generation model with memory management.

        Args:
            model_name: HuggingFace model name (default from env)

        Returns:
            Tuple of (model, tokenizer)
        """
        model_name = model_name or os.getenv("GENERATE_MODEL_NAME", "gpt2")
        cache_key = f"gen_{model_name}"

        # Update last used timestamp if model is cached
        if cache_key in self._models:
            self._last_used[cache_key] = time.time()
            return self._models[cache_key]

        # Ensure memory is available before loading
        self._ensure_memory_available(estimated_size_gb=2.0)

        print(f"Loading generation model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=self.cache_dir
        )
        model = model.to(self.device)
        model.eval()

        # Track model size and last used time
        self._models[cache_key] = (model, tokenizer)
        self._model_sizes[cache_key] = self._estimate_model_size(model)
        self._last_used[cache_key] = time.time()

        print(f"Loaded model {model_name}: {self._model_sizes[cache_key]:.2f}GB")

        return self._models[cache_key]

    def get_nli_pipeline(self, model_name: Optional[str] = None):
        """
        Load and cache an NLI model for verification with memory management.

        Args:
            model_name: HuggingFace NLI model name

        Returns:
            HuggingFace pipeline for text classification
        """
        model_name = model_name or os.getenv(
            "NLI_MODEL_NAME", "microsoft/deberta-v3-base-mnli"
        )
        cache_key = f"nli_{model_name}"

        # Update last used timestamp if model is cached
        if cache_key in self._models:
            self._last_used[cache_key] = time.time()
            return self._models[cache_key]

        # Ensure memory is available before loading
        self._ensure_memory_available(estimated_size_gb=1.5)

        print(f"Loading NLI model: {model_name}")
        device_id = 0 if self.device.startswith("cuda") else -1
        nli_pipeline = pipeline(
            "text-classification",
            model=model_name,
            device=device_id,
            cache_dir=self.cache_dir,
        )

        self._models[cache_key] = nli_pipeline
        self._last_used[cache_key] = time.time()
        # Estimate size for pipeline models
        self._model_sizes[cache_key] = 1.0  # Rough estimate

        return self._models[cache_key]

    def get_sentence_transformer(self, model_name: Optional[str] = None):
        """
        Load and cache a sentence transformer for semantic similarity with memory management.

        Args:
            model_name: Sentence transformer model name

        Returns:
            SentenceTransformer instance
        """
        model_name = model_name or os.getenv(
            "SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2"
        )
        cache_key = f"st_{model_name}"

        # Update last used timestamp if model is cached
        if cache_key in self._models:
            self._last_used[cache_key] = time.time()
            return self._models[cache_key]

        # Ensure memory is available before loading
        self._ensure_memory_available(estimated_size_gb=0.5)

        print(f"Loading sentence transformer: {model_name}")
        st_model = SentenceTransformer(
            model_name, cache_folder=self.cache_dir, device=self.device
        )

        self._models[cache_key] = st_model
        self._last_used[cache_key] = time.time()
        self._model_sizes[cache_key] = 0.4  # Rough estimate for MiniLM

        return self._models[cache_key]

    def get_spacy_nlp(self, model_name: Optional[str] = None):
        """
        Load and cache a spaCy NLP model with memory management.

        Args:
            model_name: spaCy model name

        Returns:
            spaCy Language object
        """
        model_name = model_name or os.getenv("SPACY_MODEL", "en_core_web_sm")
        cache_key = f"spacy_{model_name}"

        # Update last used timestamp if model is cached
        if cache_key in self._models:
            self._last_used[cache_key] = time.time()
            return self._models[cache_key]

        # Ensure memory is available before loading
        self._ensure_memory_available(estimated_size_gb=0.3)

        print(f"Loading spaCy model: {model_name}")
        try:
            nlp = spacy.load(model_name)
        except OSError:
            # Model not found, attempt to download
            print(f"Downloading spaCy model: {model_name}")
            os.system(f"python -m spacy download {model_name}")
            nlp = spacy.load(model_name)

        self._models[cache_key] = nlp
        self._last_used[cache_key] = time.time()
        self._model_sizes[cache_key] = 0.2  # Rough estimate for small model

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
