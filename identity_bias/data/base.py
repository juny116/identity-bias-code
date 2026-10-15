"""Base data structures for problems and solutions."""

from dataclasses import dataclass, field


@dataclass
class Problem:
    """A reasoning problem from a benchmark dataset."""
    id: str
    question: str
    ground_truth: str
    dataset: str
    difficulty: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Solution:
    """A generated solution to a problem."""
    problem_id: str
    chain_of_thought: str
    final_answer: str
    is_correct: bool
    solver_model: str
    metadata: dict = field(default_factory=dict)
