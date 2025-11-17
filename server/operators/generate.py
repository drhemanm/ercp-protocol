"""
Generate Operator (G) - Real ML Implementation
Uses transformer models for reasoning generation with constraint injection.
"""

import uuid
import re
import asyncio
from typing import List, Dict, Any, Optional
import torch
import psutil
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from opentelemetry import trace, metrics
from opentelemetry.trace import Status, StatusCode

from .base import BaseOperator
from server.models.model_registry import model_registry
from server.logging import logger


# ============================================
# Custom Exceptions
# ============================================

class GenerationError(Exception):
    """Base exception for generation errors."""
    pass


class ModelLoadError(GenerationError):
    """Failed to load model."""
    pass


class GenerationTimeoutError(GenerationError):
    """Generation exceeded timeout."""
    pass


class OutOfMemoryError(GenerationError):
    """Insufficient memory for generation."""
    pass


class InvalidInputError(GenerationError):
    """Invalid input provided."""
    pass


# ============================================
# OpenTelemetry Setup
# ============================================

# Initialize OpenTelemetry tracer and meter
tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Create custom metrics for ML operations
token_generation_counter = meter.create_counter(
    name="ml.token_generation.total",
    description="Total number of tokens generated",
    unit="tokens"
)

inference_latency_histogram = meter.create_histogram(
    name="ml.inference.latency",
    description="Distribution of model inference latency",
    unit="ms"
)

gpu_memory_gauge = meter.create_observable_gauge(
    name="ml.gpu.memory_used",
    description="Current GPU memory usage",
    unit="bytes",
    callbacks=[lambda options: _get_gpu_memory_usage()]
)

cache_hit_counter = meter.create_counter(
    name="ml.cache.hits",
    description="Number of cache hits",
    unit="hits"
)

cache_miss_counter = meter.create_counter(
    name="ml.cache.misses",
    description="Number of cache misses",
    unit="misses"
)

error_counter = meter.create_counter(
    name="ml.errors.total",
    description="Total number of ML operation errors by model",
    unit="errors"
)

def _get_gpu_memory_usage():
    """Get current GPU memory usage for observability."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            cached = torch.cuda.memory_reserved(i)
            yield metrics.Observation(
                value=allocated,
                attributes={
                    "device_id": i,
                    "metric_type": "allocated"
                }
            )
            yield metrics.Observation(
                value=cached,
                attributes={
                    "device_id": i,
                    "metric_type": "cached"
                }
            )


class GenerateOperator(BaseOperator):
    """
    Generate reasoning text using LLM with constraint injection.

    Features:
    - Deterministic generation (temperature=0, fixed seed)
    - Constraint injection into prompts
    - Token budget management
    - Structured output parsing
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_name = config.get("model", "gpt2") if config else "gpt2"
        self.model, self.tokenizer = model_registry.get_generation_model(
            self.model_name
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def execute(
        self,
        problem: str,
        constraints: List[Dict[str, Any]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate reasoning for a problem with constraints.

        Args:
            problem: Problem description
            constraints: List of constraint dictionaries
            config: Generation configuration

        Returns:
            Dictionary with reasoning_id, reasoning_text, sentences, claims
            
        Raises:
            InvalidInputError: If input validation fails
            GenerationError: If generation fails
        """
        # ============================================
        # INPUT VALIDATION
        # ============================================
        
        if not problem or not isinstance(problem, str):
            raise InvalidInputError("Problem must be a non-empty string")
        
        if len(problem.strip()) == 0:
            raise InvalidInputError("Problem cannot be empty or whitespace only")
        
        if len(problem) > 10000:  # Reasonable limit
            raise InvalidInputError(f"Problem too long: {len(problem)} chars (max 10000)")
        
        if not isinstance(constraints, list):
            raise InvalidInputError("Constraints must be a list")
        
        if not isinstance(config, dict):
            raise InvalidInputError("Config must be a dictionary")
        
        max_tokens = config.get("max_tokens", 500)
        if not isinstance(max_tokens, int) or max_tokens < 10 or max_tokens > 4000:
            raise InvalidInputError(f"max_tokens must be between 10 and 4000, got {max_tokens}")

        # ============================================
        # GENERATION WITH PROPER ERROR HANDLING
        # ============================================
        
        with tracer.start_as_current_span(
            "generate_operator.execute",
            attributes={
                "model.name": self.model_name,
                "problem.length": len(problem),
                "constraints.count": len(constraints),
                "config.max_tokens": max_tokens,
                "config.temperature": config.get("temperature", 0.0),
            }
        ) as span:
            try:
                # Build prompt with constraints
                prompt = self._build_prompt(problem, constraints)
                span.set_attribute("prompt.length", len(prompt))

                # Generate text
                reasoning_text = self._generate_text(prompt, config)
                span.set_attribute("output.length", len(reasoning_text))

                # Parse output
                sentences = self._segment_sentences(reasoning_text)
                claims = self._extract_claims(reasoning_text, sentences)

                span.set_attribute("output.sentences", len(sentences))
                span.set_attribute("output.claims", len(claims))
                span.set_status(Status(StatusCode.OK))

                return {
                    "reasoning_id": self._generate_id(),
                    "reasoning_text": reasoning_text,
                    "sentences": sentences,
                    "claims": claims,
                    "model_name": self.model_name,
                }
                
            except InvalidInputError:
                # Re-raise validation errors as-is
                raise
                
            except GenerationError:
                # Re-raise our custom errors as-is
                raise
                
            except Exception as e:
                # Wrap unexpected errors
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                error_counter.add(
                    1,
                    attributes={
                        "model": self.model_name,
                        "error_type": type(e).__name__,
                        "operation": "execute"
                    }
                )
                
                logger.error(
                    "generate.unexpected_error",
                    prompt_length=len(prompt) if 'prompt' in locals() else 0,
                    error_type=type(e).__name__,
                    error=str(e)
                )
                
                # Wrap in our exception type with context
                raise GenerationError(
                    f"Unexpected error during generation: {type(e).__name__}: {str(e)}"
                ) from e

    def _build_prompt(
        self, problem: str, constraints: List[Dict[str, Any]]
    ) -> str:
        """
        Build a prompt that incorporates problem and constraints.

        Args:
            problem: Problem description
            constraints: List of constraints

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "Task: Provide clear, logical reasoning to answer the following question.",
            "",
            f"Question: {problem}",
            "",
        ]

        if constraints:
            prompt_parts.append("Requirements:")
            for i, constraint in enumerate(constraints, 1):
                nl_text = constraint.get("nl_text", str(constraint))
                prompt_parts.append(f"{i}. {nl_text}")
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "Instructions:",
                "- Provide step-by-step reasoning",
                "- Be concise and factual",
                "- Address all requirements above",
                "- Use clear, simple language",
                "",
                "Answer:",
            ]
        )

        return "\n".join(prompt_parts)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RuntimeError, ConnectionError, GenerationTimeoutError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def _generate_text(self, prompt: str, config: Dict[str, Any]) -> str:
        """
        Generate text using the LLM with retries and error handling.

        Args:
            prompt: Input prompt
            config: Generation config (max_tokens, temperature, etc.)

        Returns:
            Generated text

        Raises:
            torch.cuda.OutOfMemoryError: If GPU runs out of memory
            TimeoutError: If generation exceeds timeout
            RuntimeError: If model generation fails
        """
        with tracer.start_as_current_span(
            "generate_operator.inference",
            attributes={
                "model.name": self.model_name,
                "prompt.length": len(prompt),
            }
        ) as span:
            max_tokens = config.get("max_tokens", 500)
            temperature = config.get("temperature", 0.0)
            deterministic = config.get("deterministic", True)
            timeout_seconds = config.get("generation_timeout", 60.0)

            span.set_attribute("config.max_tokens", max_tokens)
            span.set_attribute("config.temperature", temperature)
            span.set_attribute("config.deterministic", deterministic)

            # Set seed for deterministic generation
            if deterministic:
                torch.manual_seed(42)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(42)

            try:
                # Record GPU memory before inference
                if torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    span.set_attribute("gpu.memory_before_mb", gpu_memory_before)

                # Tokenize input
                inputs = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                input_token_count = inputs['input_ids'].shape[1]
                span.set_attribute("tokens.input", input_token_count)

                # Generate with timeout protection
                import time
                inference_start = time.perf_counter()

                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                if start_time:
                    start_time.record()

                with torch.no_grad():
                    if temperature == 0.0:
                        # Greedy decoding
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=False,
                            num_beams=1,
                            pad_token_id=self.tokenizer.eos_token_id,
                            early_stopping=True,
                        )
                    else:
                        # Sampling with temperature
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=config.get("top_p", 0.9),
                            pad_token_id=self.tokenizer.eos_token_id,
                        )

                inference_end = time.perf_counter()
                elapsed_ms = (inference_end - inference_start) * 1000

                if end_time:
                    end_time.record()
                    torch.cuda.synchronize()
                    gpu_elapsed = start_time.elapsed_time(end_time)
                    span.set_attribute("inference.gpu_time_ms", gpu_elapsed)

                    if gpu_elapsed / 1000.0 > timeout_seconds:
                        logger.warning(
                            "generate.timeout_warning",
                            elapsed_seconds=gpu_elapsed / 1000.0,
                            timeout=timeout_seconds
                        )

                # Calculate tokens generated
                output_token_count = outputs.shape[1]
                tokens_generated = output_token_count - input_token_count

                # Record metrics
                span.set_attribute("tokens.output", output_token_count)
                span.set_attribute("tokens.generated", tokens_generated)
                span.set_attribute("inference.latency_ms", elapsed_ms)

                # Calculate token generation rate (tokens/second)
                if elapsed_ms > 0:
                    tokens_per_second = (tokens_generated * 1000) / elapsed_ms
                    span.set_attribute("tokens.per_second", tokens_per_second)

                # Record metrics
                token_generation_counter.add(tokens_generated, attributes={"model": self.model_name})
                inference_latency_histogram.record(elapsed_ms, attributes={"model": self.model_name})

                # Record GPU memory after inference
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
                    span.set_attribute("gpu.memory_after_mb", gpu_memory_after)
                    span.set_attribute("gpu.memory_peak_mb", gpu_memory_peak)
                    span.set_attribute("gpu.memory_delta_mb", gpu_memory_after - gpu_memory_before)

                # Decode
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract only the answer part (after "Answer:")
                if "Answer:" in generated_text:
                    answer_text = generated_text.split("Answer:")[-1].strip()
                else:
                    answer_text = generated_text[len(prompt) :].strip()

                span.set_attribute("output.text_length", len(answer_text))
                span.set_status(Status(StatusCode.OK))

                logger.info(
                    "generate.success",
                    prompt_length=len(prompt),
                    output_length=len(answer_text),
                    tokens_generated=tokens_generated,
                    inference_latency_ms=elapsed_ms
                )

                return answer_text

            except torch.cuda.OutOfMemoryError as e:
                span.set_status(Status(StatusCode.ERROR, "CUDA OOM"))
                span.record_exception(e)
                error_counter.add(1, attributes={"model": self.model_name, "error_type": "cuda_oom"})

                logger.error(
                    "generate.cuda_oom",
                    prompt_length=len(prompt),
                    max_tokens=max_tokens,
                    error=str(e)
                )
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Wrap in our exception type
                raise OutOfMemoryError(
                    f"CUDA out of memory. Try reducing max_tokens (current: {max_tokens}) "
                    f"or use a smaller model."
                ) from e

            except RuntimeError as e:
                span.set_status(Status(StatusCode.ERROR, "Runtime error"))
                span.record_exception(e)
                error_counter.add(1, attributes={"model": self.model_name, "error_type": "runtime_error"})

                logger.error(
                    "generate.runtime_error",
                    prompt_length=len(prompt),
                    error=str(e)
                )
                
                # Check if it's a timeout or other runtime error
                error_msg = str(e).lower()
                if "timeout" in error_msg or "killed" in error_msg:
                    raise GenerationTimeoutError(
                        f"Generation exceeded timeout. Error: {str(e)}"
                    ) from e
                else:
                    raise GenerationError(
                        f"Model generation failed: {str(e)}"
                    ) from e

            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                error_counter.add(1, attributes={"model": self.model_name, "error_type": type(e).__name__})

                logger.error(
                    "generate.unexpected_error",
                    prompt_length=len(prompt),
                    error_type=type(e).__name__,
                    error=str(e)
                )
                
                # Don't return a string - raise an exception
                raise GenerationError(
                    f"Unexpected generation error: {type(e).__name__}: {str(e)}"
                ) from e

    def _segment_sentences(self, text: str) -> List[str]:
        """
        Segment text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence segmentation (can be improved with spaCy)
        # Split on . ! ? followed by space or end of string
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _extract_claims(
        self, text: str, sentences: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Extract claims from reasoning text.

        Args:
            text: Full reasoning text
            sentences: Segmented sentences

        Returns:
            List of claim dictionaries
        """
        claims = []

        # Simple claim extraction: treat each sentence as a potential claim
        # In production, use more sophisticated claim detection
        for i, sentence in enumerate(sentences):
            # Skip very short sentences
            if len(sentence.split()) < 3:
                continue

            # Check if sentence contains factual content
            # (heuristic: contains verbs and nouns)
            if self._is_factual_claim(sentence):
                claims.append(
                    {
                        "claim_id": str(uuid.uuid4()),
                        "claim": sentence,
                        "source": "llm",
                        "sentence_index": i,
                    }
                )

        return claims

    def _is_factual_claim(self, sentence: str) -> bool:
        """
        Check if a sentence appears to be a factual claim.

        Args:
            sentence: Input sentence

        Returns:
            True if likely a factual claim
        """
        # Heuristic: contains certain keywords or patterns
        factual_indicators = [
            "is",
            "are",
            "was",
            "were",
            "occurs",
            "happens",
            "due to",
            "because",
            "results in",
            "causes",
            "temperature",
            "pressure",
            "altitude",
        ]

        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in factual_indicators)
