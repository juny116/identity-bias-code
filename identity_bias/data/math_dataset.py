"""MATH-500 dataset loader."""

import random

from datasets import load_dataset

from identity_bias.data.base import Problem


def load_math(split: str = "test", n_samples: int | None = None, seed: int = 42) -> list[Problem]:
    """Load problems from MATH-500 dataset."""
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)

    problems = []
    for i, item in enumerate(ds):
        problems.append(Problem(
            id=f"math500_{i}",
            question=item["problem"],
            ground_truth=item["answer"],
            dataset="math",
            difficulty=item.get("level"),
            metadata={
                "subject": item.get("subject", ""),
                "unique_id": item.get("unique_id", ""),
                "solution": item.get("solution", ""),
            },
        ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def check_math_answer(predicted: str, ground_truth: str) -> bool:
    """Check using math-verify."""
    from math_verify import parse, verify
    try:
        gold = parse(ground_truth)
        pred = parse(predicted)
        return verify(gold, pred)
    except Exception:
        # Fallback to string comparison
        return predicted.strip() == ground_truth.strip()
