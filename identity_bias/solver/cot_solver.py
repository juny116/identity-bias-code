"""Chain-of-thought solver for generating solutions."""

import re

from identity_bias.config import Dataset
from identity_bias.data.base import Problem, Solution
from identity_bias.data import check_answer
from identity_bias.llm.base import BaseLLM


SOLVER_SYSTEM_PROMPT = r"""You are a careful problem solver. Solve the given problem step by step, showing your complete chain of thought.

Format your response as follows:
1. Show your step-by-step reasoning
2. End with your final answer enclosed in \boxed{}, for example: \boxed{42}

For multiple choice questions, put the letter choice in the box, e.g. \boxed{A}"""


SOLVER_USER_PROMPT = """Solve the following problem step by step.

Problem: {question}"""


class CoTSolver:
    """Chain-of-thought solver that generates step-by-step solutions."""

    def __init__(self, llm: BaseLLM, model_name: str = "unknown"):
        self.llm = llm
        self.model_name = model_name

    def solve(self, problem: Problem, dataset: Dataset) -> Solution:
        """Generate a chain-of-thought solution for the given problem."""
        messages = [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": SOLVER_USER_PROMPT.format(question=problem.question)},
        ]

        response = self.llm.generate(messages)
        cot = response.text

        # Extract final answer from \boxed{}
        final_answer = self._extract_answer(cot)

        # Check correctness
        is_correct = check_answer(dataset, final_answer, problem.ground_truth)

        return Solution(
            problem_id=problem.id,
            chain_of_thought=cot,
            final_answer=final_answer,
            is_correct=is_correct,
            solver_model=self.model_name,
            metadata={
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            },
        )

    @staticmethod
    def _extract_boxed(text: str) -> str | None:
        """Extract content from the last \\boxed{...}, handling nested braces."""
        # Find all \boxed{ positions, take the last one
        idx = text.rfind("\\boxed{")
        if idx == -1:
            return None
        start = idx + len("\\boxed{")
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start:i - 1].strip()
        return None

    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from \boxed{} in the CoT response."""
        boxed = self._extract_boxed(text)
        if boxed is not None:
            return boxed

        # Fallback: look for "Final Answer:" pattern
        match = re.search(r"Final Answer:\s*(.+)", text)
        if match:
            return match.group(1).strip().strip("*").strip()

        # Last resort: return the last line
        lines = text.strip().split("\n")
        return lines[-1].strip() if lines else ""

    def solve_batch(self, problems: list[Problem], dataset: Dataset) -> list[Solution]:
        """Solve a batch of problems sequentially."""
        return [self.solve(problem, dataset) for problem in problems]
