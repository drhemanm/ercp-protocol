"""
Generate Operator - LLM-based Reasoning Generation
Author: ERCP Protocol Implementation
License: Apache-2.0

Implements the Generate operator (G) from the ERCP protocol.
Generates reasoning using an LLM with constraint-aware prompting.
"""

import re
import uuid
import hashlib
from typing import List, Dict, Any, Optional
import torch

from .base import BaseOperator
from ..models.model_registry import get_model_registry


class GenerateOperator(BaseOperator):
    """
    Generate reasoning using constraint-aware LLM prompting.

    This operator:
    1. Accepts a problem description and list of constraints
    2. Builds a prompt that incorporates constraints into the system message
    3. Uses deterministic sampling (temperature=0) for reproducibility
    4. Parses output into sentences
    5. Extracts claims from reasoning
    6. Returns structured response with reasoning_id, text, sentences, and claims
    """

    def __init__(self, model_name: str = "gpt2", **kwargs):
        """
        Initialize the Generate operator.

        Args:
            model_name: Name of the LLM model to use for generation
            **kwargs: Additional arguments passed to BaseOperator
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.registry = get_model_registry()

    def _build_constraint_prompt(self, constraints: List[Dict[str, Any]]) -> str:
        """
        Build constraint text for the system prompt.

        Args:
            constraints: List of constraint dictionaries

        Returns:
            Formatted constraint text
        """
        if not constraints:
            return ""

        constraint_lines = []
        for i, constraint in enumerate(constraints, 1):
            nl_text = constraint.get("nl_text", "")
            if nl_text:
                constraint_lines.append(f"{i}. {nl_text}")

        if not constraint_lines:
            return ""

        return "You must follow these constraints:\n" + "\n".join(constraint_lines)

    def _build_prompt(
        self,
        problem_description: str,
        constraints: List[Dict[str, Any]]
    ) -> str:
        """
        Build the complete prompt for the LLM.

        Args:
            problem_description: The problem to reason about
            constraints: List of constraints to incorporate

        Returns:
            Complete prompt string
        """
        constraint_text = self._build_constraint_prompt(constraints)

        if constraint_text:
            prompt = f"System: You are a helpful AI assistant that provides clear, logical reasoning.\n\n{constraint_text}\n\nUser: {problem_description}\n\nAssistant: Let me think through this step by step."
        else:
            prompt = f"User: {problem_description}\n\nAssistant: Let me think through this step by step."

        return prompt

    def _generate_text(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: Optional[int] = 42
    ) -> str:
        """
        Generate text using the LLM.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 for deterministic)
            seed: Random seed for reproducibility

        Returns:
            Generated text
        """
        try:
            model_info = self.registry.get_generate_model(self.model_name)
            model = model_info["model"]
            tokenizer = model_info["tokenizer"]
            device = model_info["device"]

            # Set random seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    top_p=1.0,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()

            return generated_text

        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            raise

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple regex-based approach.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Simple sentence splitting using regex
        # This is a basic approach - in production, use spaCy or nltk for better accuracy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_claims(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Extract claims from sentences.

        Args:
            sentences: List of sentences

        Returns:
            List of claim dictionaries
        """
        claims = []

        # Simple heuristic: treat each declarative sentence as a claim
        for i, sentence in enumerate(sentences):
            # Skip questions and very short sentences
            if '?' in sentence or len(sentence.split()) < 3:
                continue

            claims.append({
                "claim": sentence,
                "source": "llm",
                "sentence_index": i
            })

        return claims

    def execute(
        self,
        problem_description: str,
        constraints: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the Generate operator.

        Args:
            problem_description: The problem to reason about
            constraints: Optional list of constraints to incorporate
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            seed: Random seed for reproducibility
            **kwargs: Additional parameters

        Returns:
            Dictionary containing:
                - reasoning_id: Unique ID for this reasoning
                - reasoning_text: Generated reasoning text
                - sentences: List of sentences
                - claims: List of extracted claims
        """
        if constraints is None:
            constraints = []

        # Build prompt
        prompt = self._build_prompt(problem_description, constraints)

        self.logger.info(
            f"Generating reasoning for problem (constraints: {len(constraints)})"
        )

        # Generate text
        reasoning_text = self._generate_text(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed
        )

        # Split into sentences
        sentences = self._split_into_sentences(reasoning_text)

        # Extract claims
        claims = self._extract_claims(sentences)

        # Generate reasoning ID
        reasoning_id = str(uuid.uuid4())

        self.logger.info(
            f"Generated reasoning: {len(sentences)} sentences, {len(claims)} claims"
        )

        return {
            "reasoning_id": reasoning_id,
            "reasoning_text": reasoning_text,
            "sentences": sentences,
            "claims": claims
        }
