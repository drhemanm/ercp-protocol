"""
Mutate Operator (M) - Real ML Implementation
Problem mutation for non-convergent cases.
"""

import uuid
from typing import Dict, Any, Optional, List
from .base import BaseOperator
from server.models.model_registry import model_registry


class MutateOperator(BaseOperator):
    """
    Mutate problem when reasoning fails to converge.

    Strategies:
    - Decompose: Break problem into sub-problems
    - Relax: Remove or weaken constraints
    - Reframe: Rephrase problem from different angle
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model, self.tokenizer = model_registry.get_generation_model()

    def execute(
        self,
        problem: Dict[str, Any],
        reasoning: str,
        strategy: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Mutate a problem based on strategy.

        Args:
            problem: Problem dictionary
            reasoning: Current reasoning text
            strategy: Mutation strategy (decompose, relax, reframe)
            config: Configuration

        Returns:
            Dictionary with new_problem, new_constraints, mutation_notes
        """
        strategy = strategy.lower()

        if strategy == "decompose":
            return self._decompose(problem, reasoning, config)
        elif strategy == "relax":
            return self._relax_constraints(problem, reasoning, config)
        elif strategy == "reframe":
            return self._reframe_problem(problem, reasoning, config)
        else:
            raise ValueError(f"Unknown mutation strategy: {strategy}")

    def _decompose(
        self, problem: Dict[str, Any], reasoning: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Decompose problem into sub-problems.

        Args:
            problem: Original problem
            reasoning: Current reasoning
            config: Configuration

        Returns:
            Mutation result with sub-problems
        """
        problem_desc = problem.get("description", "")

        # Build decomposition prompt
        prompt = f"""
Problem: {problem_desc}

Current reasoning (which failed to converge):
{reasoning}

Task: Break this problem down into 2-3 simpler, more focused sub-problems.
Each sub-problem should be:
1. More specific than the original
2. Independently answerable
3. Contribute to solving the original problem

Sub-problems:
1."""

        # Generate decomposition
        decomposition_text = self._generate_with_llm(prompt, max_tokens=200)

        # Parse sub-problems (simple parsing)
        sub_problems = self._parse_sub_problems(decomposition_text)

        return {
            "new_problem": {
                "id": f"{problem.get('id', 'unknown')}_decomposed",
                "description": f"Sub-problem 1: {sub_problems[0] if sub_problems else problem_desc}",
                "metadata": {
                    **problem.get("metadata", {}),
                    "original_problem": problem_desc,
                    "mutation": "decompose",
                    "sub_problems": sub_problems,
                },
            },
            "new_constraints": [],
            "mutation_notes": f"Decomposed into {len(sub_problems)} sub-problems",
        }

    def _relax_constraints(
        self, problem: Dict[str, Any], reasoning: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Relax constraints that may be blocking convergence.

        Args:
            problem: Original problem
            reasoning: Current reasoning
            config: Configuration

        Returns:
            Mutation result with relaxed constraints
        """
        problem_desc = problem.get("description", "")

        # For now, just add a note to be less strict
        # In production, analyze which constraints are problematic
        return {
            "new_problem": {
                **problem,
                "description": f"{problem_desc} (Note: Focus on core requirements)",
                "metadata": {
                    **problem.get("metadata", {}),
                    "mutation": "relax",
                },
            },
            "new_constraints": [],
            "mutation_notes": "Relaxed constraints to focus on core requirements",
        }

    def _reframe_problem(
        self, problem: Dict[str, Any], reasoning: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Reframe problem from different perspective.

        Args:
            problem: Original problem
            reasoning: Current reasoning
            config: Configuration

        Returns:
            Mutation result with reframed problem
        """
        problem_desc = problem.get("description", "")

        # Build reframing prompt
        prompt = f"""
Original problem: {problem_desc}

Current reasoning (which failed to converge):
{reasoning}

Task: Rephrase the problem from a different angle or perspective while keeping the core question the same.
Make it more concrete or provide a different framing that might be easier to reason about.

Reframed problem:"""

        # Generate reframing
        reframed_text = self._generate_with_llm(prompt, max_tokens=100)

        return {
            "new_problem": {
                **problem,
                "description": reframed_text.strip(),
                "metadata": {
                    **problem.get("metadata", {}),
                    "original_description": problem_desc,
                    "mutation": "reframe",
                },
            },
            "new_constraints": [],
            "mutation_notes": "Reframed problem from different perspective",
        }

    def _generate_with_llm(self, prompt: str, max_tokens: int = 200) -> str:
        """
        Generate text with LLM.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only new content
        if prompt in generated:
            generated = generated.split(prompt)[-1].strip()

        return generated

    def _parse_sub_problems(self, text: str) -> List[str]:
        """
        Parse sub-problems from generated text.

        Args:
            text: Generated text with sub-problems

        Returns:
            List of sub-problems
        """
        import re

        # Look for numbered items
        pattern = r'\d+\.\s*(.+?)(?=\d+\.|$)'
        matches = re.findall(pattern, text, re.DOTALL)

        sub_problems = [m.strip() for m in matches if m.strip()]

        # Fallback: if no matches, split by newlines
        if not sub_problems:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            sub_problems = lines[:3]  # Take first 3 lines

        return sub_problems
