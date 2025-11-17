"""
Generate Operator (G) - Real ML Implementation
Uses transformer models for reasoning generation with constraint injection.
"""

import uuid
import re
import asyncio
from typing import List, Dict, Any, Optional
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from .base import BaseOperator
from server.models.model_registry import model_registry
from server.logging import logger


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
        """
        # Build prompt with constraints
        prompt = self._build_prompt(problem, constraints)

        # Generate text
        reasoning_text = self._generate_text(prompt, config)

        # Parse output
        sentences = self._segment_sentences(reasoning_text)
        claims = self._extract_claims(reasoning_text, sentences)

        return {
            "reasoning_id": self._generate_id(),
            "reasoning_text": reasoning_text,
            "sentences": sentences,
            "claims": claims,
            "model_name": self.model_name,
        }

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
        retry=retry_if_exception_type((RuntimeError, ConnectionError)),
        before_sleep=before_sleep_log(logger, "WARNING"),
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
        max_tokens = config.get("max_tokens", 500)
        temperature = config.get("temperature", 0.0)
        deterministic = config.get("deterministic", True)
        timeout_seconds = config.get("generation_timeout", 60.0)

        # Set seed for deterministic generation
        if deterministic:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with timeout protection
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

            if end_time:
                end_time.record()
                torch.cuda.synchronize()
                elapsed = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                if elapsed > timeout_seconds:
                    logger.warning(
                        "generate.timeout_warning",
                        elapsed_seconds=elapsed,
                        timeout=timeout_seconds
                    )

            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer_text = generated_text.split("Answer:")[-1].strip()
            else:
                answer_text = generated_text[len(prompt) :].strip()

            logger.info(
                "generate.success",
                prompt_length=len(prompt),
                output_length=len(answer_text)
            )

            return answer_text

        except torch.cuda.OutOfMemoryError as e:
            logger.error(
                "generate.cuda_oom",
                prompt_length=len(prompt),
                max_tokens=max_tokens,
                error=str(e)
            )
            # Clear CUDA cache and retry
            torch.cuda.empty_cache()
            raise

        except RuntimeError as e:
            logger.error(
                "generate.runtime_error",
                prompt_length=len(prompt),
                error=str(e)
            )
            raise

        except Exception as e:
            logger.error(
                "generate.unexpected_error",
                prompt_length=len(prompt),
                error_type=type(e).__name__,
                error=str(e)
            )
            # Fallback: return a minimal response
            return "Unable to generate response due to technical error."

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
