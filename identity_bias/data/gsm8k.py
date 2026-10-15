"""GSM8K dataset loader."""

import re
import random

from datasets import load_dataset

from identity_bias.data.base import Problem


def load_gsm8k(split: str = "test", n_samples: int | None = None, seed: int = 42) -> list[Problem]:
    """Load problems from GSM8K dataset.

    Args:
        split: Dataset split ("train" or "test").
        n_samples: Number of samples to load. None for all.
        seed: Random seed for sampling.

    Returns:
        List of Problem instances.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)

    problems = []
    for i, item in enumerate(ds):
        # Extract numeric answer from "#### <number>" format
        answer_text = item["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        ground_truth = match.group(1).strip() if match else answer_text

        problems.append(Problem(
            id=f"gsm8k_{split}_{i}",
            question=item["question"],
            ground_truth=ground_truth,
            dataset="gsm8k",
        ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def check_gsm8k_answer(predicted: str, ground_truth: str) -> bool:
    """Check if the predicted answer matches ground truth for GSM8K.

    Handles numeric comparison with tolerance for formatting differences.
    """
    def extract_number(text: str) -> float | None:
        # Remove commas and whitespace
        text = text.strip().replace(",", "").replace("$", "")
        try:
            return float(text)
        except ValueError:
            # Try to find the last number in the text
            numbers = re.findall(r"-?\d+\.?\d*", text)
            if numbers:
                return float(numbers[-1])
            return None

    pred_num = extract_number(predicted)
    truth_num = extract_number(ground_truth)

    if pred_num is not None and truth_num is not None:
        return abs(pred_num - truth_num) < 1e-6

    return predicted.strip().lower() == ground_truth.strip().lower()
