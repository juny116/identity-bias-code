"""MMLU-Pro dataset loader (TIGER-Lab/MMLU-Pro)."""

import random
from datasets import load_dataset

from identity_bias.data.base import Problem

OPTION_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


def load_mmlu_pro(
    n_samples: int | None = 600,
    seed: int = 42,
    categories: list[str] | None = None,
) -> list[Problem]:
    """Load problems from MMLU-Pro.

    Args:
        n_samples: Number to sample (default 600). None = all 12K.
        categories: Filter by category. None = all.
    """
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    problems = []
    for item in ds:
        if categories and item["category"] not in categories:
            continue

        options = item["options"]  # list of strings
        choices_text = "\n".join(
            f"({OPTION_LABELS[i]}) {opt}"
            for i, opt in enumerate(options)
        )
        question = f"{item['question']}\n\n{choices_text}"

        problems.append(Problem(
            id=f"mmlu_pro_{item['question_id']}",
            question=question,
            ground_truth=item["answer"],  # letter like 'A'-'J'
            dataset="mmlu_pro",
            metadata={
                "category": item["category"],
                "answer_index": item["answer_index"],
                "options": options,
            },
        ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def check_mmlu_pro_answer(predicted: str, ground_truth: str) -> bool:
    """Check MMLU-Pro answer — both should be a letter A-J."""
    import re
    pred = predicted.strip().upper()
    # Extract first letter A-J if surrounded by parentheses or standalone
    match = re.search(r"\b([A-J])\b", pred)
    if match:
        pred = match.group(1)
    else:
        pred = pred.strip("()")[:1]
    return pred == ground_truth.strip().upper()
