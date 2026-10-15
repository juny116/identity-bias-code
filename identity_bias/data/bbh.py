"""BigBench Hard (BBH) dataset loader."""

import random

from datasets import load_dataset

from identity_bias.data.base import Problem


# Tasks suitable for chain-of-thought critique experiments
BBH_TASKS = [
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "multistep_arithmetic_two",
    "date_understanding",
    "navigate",
]


def load_bbh(
    tasks: list[str] | None = None,
    n_samples: int | None = None,
    seed: int = 42,
) -> list[Problem]:
    """Load problems from BigBench Hard dataset."""
    if tasks is None:
        tasks = BBH_TASKS

    problems = []
    for task in tasks:
        ds = load_dataset("lukaemon/bbh", task, split="test")

        for i, item in enumerate(ds):
            problems.append(Problem(
                id=f"bbh_{task}_{i}",
                question=item["input"],
                ground_truth=item["target"],
                dataset="bbh",
                metadata={"task": task},
            ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def check_bbh_answer(predicted: str, ground_truth: str) -> bool:
    """Check BBH answer — normalize and compare strings."""
    def normalize(s: str) -> str:
        s = s.strip().lower()
        s = s.strip("()")
        return s.strip()

    pred = normalize(predicted)
    truth = normalize(ground_truth)

    if pred == truth:
        return True

    # For True/False, Yes/No type answers
    if truth in pred.split():
        return True

    return False
