"""Minerva Math dataset loader (math-ai/minervamath)."""

import random

from datasets import load_dataset

from identity_bias.data.base import Problem


def load_minerva(
    n_samples: int | None = None,
    seed: int = 42,
) -> list[Problem]:
    """Load all 272 problems from Minerva Math benchmark."""
    ds = load_dataset("math-ai/minervamath", split="test")

    problems = []
    for i, item in enumerate(ds):
        problems.append(Problem(
            id=f"minerva_{i}",
            question=item["question"],
            ground_truth=str(item["answer"]).strip(),
            dataset="minerva",
        ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def check_minerva_answer(predicted: str, ground_truth: str) -> bool:
    """Check Minerva answer using math_verify, fallback to string."""
    from math_verify import parse, verify
    try:
        gold = parse(ground_truth)
        pred = parse(predicted)
        return verify(gold, pred)
    except Exception:
        pass
    return predicted.strip().lower() == ground_truth.strip().lower()
