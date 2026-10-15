"""Identity-labeled critic for evaluating solutions."""

import json
import re
from dataclasses import dataclass

from identity_bias.critic.prompts import (
    build_critic_messages,
    build_verdict_only_messages,
    strip_thinking,
)
from identity_bias.data.base import Problem, Solution
from identity_bias.llm.base import BaseLLM
from identity_bias.types import IdentityCondition


@dataclass
class CritiqueResult:
    """Result of a critic's evaluation."""

    problem_id: str
    identity_condition: str
    critic_model: str
    solver_model: str
    # Critique outputs (full call: verdict + repair)
    is_correct: bool  # Critic's judgment
    error_description: str | None
    corrected_answer: str | None
    confidence: float
    # Ground truth
    actually_correct: bool  # Whether the solution was actually correct
    # Raw data
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Optional first-stage verdict (two-stage / verdict-vs-repair diagnostic).
    # Populated only when IdentityCritic(two_stage=True). Lets us separate
    # the critic's initial accept/reject decision from the quality of the
    # repair attempt in the full call.
    initial_is_correct: bool | None = None
    initial_error_description: str | None = None
    initial_confidence: float | None = None
    initial_raw_response: str | None = None
    initial_prompt_tokens: int = 0
    initial_completion_tokens: int = 0


class IdentityCritic:
    """Critic that evaluates solutions with identity-conditioned prompts."""

    def __init__(
        self, llm: BaseLLM, model_name: str = "unknown", two_stage: bool = False
    ):
        self.llm = llm
        self.model_name = model_name
        self.two_stage = two_stage

    def critique(
        self,
        problem: Problem,
        solution: Solution,
        identity_condition: IdentityCondition,
    ) -> CritiqueResult:
        """Critique a solution under the given identity condition.

        When two_stage=True, first issues a verdict-only call (no repair
        requested) and records its is_correct/error_description/confidence
        as initial_*. Then issues the standard verdict+repair call. The two
        calls are independent (no shared conversation context) so the
        second-stage judgment is not anchored by the first.
        """
        initial_is_correct = None
        initial_error_description = None
        initial_confidence = None
        initial_raw_response = None
        initial_prompt_tokens = 0
        initial_completion_tokens = 0

        if self.two_stage:
            verdict_messages = build_verdict_only_messages(
                question=problem.question,
                chain_of_thought=solution.chain_of_thought,
                final_answer=solution.final_answer,
                identity_condition=identity_condition,
            )
            verdict_response = self.llm.generate(verdict_messages)
            verdict_parsed = self._parse_response(verdict_response.text)
            initial_is_correct = verdict_parsed.get("is_correct", True)
            initial_error_description = verdict_parsed.get("error_description")
            initial_confidence = verdict_parsed.get("confidence", 0.5)
            initial_raw_response = verdict_response.text
            initial_prompt_tokens = verdict_response.prompt_tokens
            initial_completion_tokens = verdict_response.completion_tokens

        messages = build_critic_messages(
            question=problem.question,
            chain_of_thought=solution.chain_of_thought,
            final_answer=solution.final_answer,
            identity_condition=identity_condition,
        )

        response = self.llm.generate(messages)
        parsed = self._parse_response(response.text)

        return CritiqueResult(
            problem_id=problem.id,
            identity_condition=identity_condition.value,
            critic_model=self.model_name,
            solver_model=solution.solver_model,
            is_correct=parsed.get("is_correct", True),
            error_description=parsed.get("error_description"),
            corrected_answer=parsed.get("corrected_answer"),
            confidence=parsed.get("confidence", 0.5),
            actually_correct=solution.is_correct,
            raw_response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            initial_is_correct=initial_is_correct,
            initial_error_description=initial_error_description,
            initial_confidence=initial_confidence,
            initial_raw_response=initial_raw_response,
            initial_prompt_tokens=initial_prompt_tokens,
            initial_completion_tokens=initial_completion_tokens,
        )

    def critique_all_conditions(
        self,
        problem: Problem,
        solution: Solution,
        conditions: list[IdentityCondition] | None = None,
    ) -> list[CritiqueResult]:
        """Critique a solution under all identity conditions."""
        if conditions is None:
            conditions = list(IdentityCondition)
        return [self.critique(problem, solution, c) for c in conditions]

    def _parse_response(self, text: str) -> dict:
        """Parse the JSON response from the critic."""
        # Strip <think>...</think> blocks from reasoning models
        text = strip_thinking(text)

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON block in markdown code fence
        match = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object in the text (allowing nested braces)
        match = re.search(
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\"is_correct\"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
            text,
            re.DOTALL,
        )
        if not match:
            match = re.search(r"\{.*\"is_correct\".*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Fallback
        return {
            "is_correct": True,
            "error_description": None,
            "corrected_answer": None,
            "confidence": 0.5,
            "_parse_error": True,
        }
