"""AIME dataset loader (aime24 + aime25 + aime26 from math-ai collection)."""

import re
import random

from datasets import load_dataset

from identity_bias.data.base import Problem


def _extract_boxed(text: str) -> str | None:
    """Extract answer from \\boxed{...} notation."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def load_aime(
    years: list[str] | None = None,
    n_samples: int | None = None,
    seed: int = 42,
) -> list[Problem]:
    """Load AIME problems from math-ai collection.

    Combines aime24, aime25, aime26 by default.
    All AIME answers are integers 0-999.
    """
    if years is None:
        years = ["aime24", "aime25", "aime26"]

    problems = []

    for year in years:
        ds = load_dataset(f"math-ai/{year}", split="test")
        for item in ds:
            # aime24 stores answer in solution as \boxed{N}; aime25/26 have answer field
            if "answer" in item and item["answer"] is not None:
                answer = str(item["answer"]).strip()
            elif "solution" in item and item["solution"]:
                answer = _extract_boxed(item["solution"]) or item["solution"].strip()
            else:
                continue

            pid = f"aime_{year}_{item.get('id', len(problems))}"
            problems.append(Problem(
                id=pid,
                question=item["problem"],
                ground_truth=answer,
                dataset="aime",
                metadata={"year": year, "url": item.get("url", "")},
            ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def check_aime_answer(predicted: str, ground_truth: str) -> bool:
    """AIME answers are integers 0-999; compare numerically."""
    def extract_int(s: str) -> int | None:
        s = s.strip()
        # Try direct int
        try:
            return int(s)
        except ValueError:
            pass
        # Try extracting last integer from string
        nums = re.findall(r"\b(\d{1,3})\b", s)
        if nums:
            return int(nums[-1])
        return None

    pred = extract_int(predicted)
    truth = extract_int(ground_truth)
    if pred is not None and truth is not None:
        return pred == truth
    return predicted.strip() == ground_truth.strip()
