"""GPQA dataset loader."""

import random

from datasets import load_dataset

from identity_bias.data.base import Problem


def load_gpqa(
    subset: str = "gpqa_diamond",
    n_samples: int | None = None,
    seed: int = 42,
) -> list[Problem]:
    """Load problems from GPQA dataset.

    Shuffles choice order per-problem to avoid position bias.
    Ground truth is stored as the letter (A/B/C/D).
    """
    ds = load_dataset("Idavidrein/gpqa", subset, split="train")
    rng = random.Random(seed)

    problems = []
    for i, item in enumerate(ds):
        choices = [
            item["Correct Answer"],
            item["Incorrect Answer 1"],
            item["Incorrect Answer 2"],
            item["Incorrect Answer 3"],
        ]
        # Shuffle to avoid position bias; track correct answer
        indices = list(range(4))
        rng.shuffle(indices)
        shuffled = [choices[j] for j in indices]
        correct_letter = chr(ord("A") + indices.index(0))  # 0 = Correct Answer

        labels = ["A", "B", "C", "D"]
        choices_text = "\n".join(f"({l}) {c}" for l, c in zip(labels, shuffled))
        question = f"{item['Question']}\n\n{choices_text}"

        problems.append(Problem(
            id=f"gpqa_{subset}_{i}",
            question=question,
            ground_truth=correct_letter,
            dataset="gpqa",
            metadata={
                "domain": item.get("Subdomain", ""),
                "choices": {l: c for l, c in zip(labels, shuffled)},
            },
        ))

    if n_samples is not None and n_samples < len(problems):
        problems = rng.sample(problems, n_samples)

    return problems


def check_gpqa_answer(predicted: str, ground_truth: str) -> bool:
    """Check GPQA answer — both should be a letter A/B/C/D."""
    pred = predicted.strip().upper().strip("()")
    truth = ground_truth.strip().upper()
    return pred == truth
