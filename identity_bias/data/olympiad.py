"""OlympiadBench dataset loader (math-ai/olympiadbench)."""

import ast
import random

from datasets import load_dataset

from identity_bias.data.base import Problem


def _parse_final_answer(raw) -> str:
    """Parse final_answer field — may be a list or string repr of a list."""
    if isinstance(raw, list):
        return ", ".join(str(x) for x in raw)
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return ", ".join(str(x) for x in parsed)
            return str(parsed)
        except Exception:
            return raw.strip("[]'\"")
    return str(raw)


def load_olympiad(
    n_samples: int | None = None,
    seed: int = 42,
    answer_types: list[str] | None = None,
) -> list[Problem]:
    """Load problems from OlympiadBench (text-only English math subset).

    Args:
        n_samples: Number to sample. None = all 674.
        answer_types: Filter by answer type. None = all.
            Options: 'Numerical', 'Expression', 'Tuple', 'Interval'
    """
    ds = load_dataset("math-ai/olympiadbench", split="test")

    problems = []
    for item in ds:
        # Keep text-only English problems
        if item["modality"] != "Text-only" or item["language"] != "English":
            continue
        if answer_types and item["answer_type"] not in answer_types:
            continue

        answer = _parse_final_answer(item["final_answer"])

        # Append unit if present
        unit = item.get("unit")
        if unit and unit not in ("None", None):
            answer = f"{answer} {unit}"

        problems.append(Problem(
            id=f"olympiad_{item['id']}",
            question=item["question"],
            ground_truth=answer,
            dataset="olympiad",
            metadata={
                "subfield": item["subfield"],
                "difficulty": item["difficulty"],
                "answer_type": item["answer_type"],
                "is_multiple_answer": item["is_multiple_answer"],
            },
        ))

    if n_samples is not None and n_samples < len(problems):
        rng = random.Random(seed)
        problems = rng.sample(problems, n_samples)

    return problems


def check_olympiad_answer(predicted: str, ground_truth: str) -> bool:
    """Check OlympiadBench answer using math_verify, fallback to string."""
    from math_verify import parse, verify
    try:
        gold = parse(ground_truth)
        pred = parse(predicted)
        return verify(gold, pred)
    except Exception:
        pass
    # Fallback: normalize and compare
    pred = predicted.strip().lower().rstrip(".")
    truth = ground_truth.strip().lower().rstrip(".")
    return pred == truth
