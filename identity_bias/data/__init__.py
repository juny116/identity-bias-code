"""Dataset loaders."""

from identity_bias.config import Dataset
from identity_bias.data.base import Problem, Solution


def load_dataset_problems(
    dataset: Dataset,
    n_samples: int | None = None,
    seed: int = 42,
    **kwargs,
) -> list[Problem]:
    """Load problems from the specified dataset."""
    if dataset == Dataset.MATH:
        from identity_bias.data.math_dataset import load_math
        return load_math(n_samples=n_samples, seed=seed, **kwargs)
    elif dataset == Dataset.GPQA:
        from identity_bias.data.gpqa import load_gpqa
        return load_gpqa(n_samples=n_samples, seed=seed)
    elif dataset == Dataset.BBH:
        from identity_bias.data.bbh import load_bbh
        return load_bbh(n_samples=n_samples, seed=seed)
    elif dataset == Dataset.AIME:
        from identity_bias.data.aime import load_aime
        return load_aime(n_samples=n_samples, seed=seed, **kwargs)
    elif dataset == Dataset.MMLU_PRO:
        from identity_bias.data.mmlu_pro import load_mmlu_pro
        return load_mmlu_pro(n_samples=n_samples, seed=seed, **kwargs)
    elif dataset == Dataset.OLYMPIAD:
        from identity_bias.data.olympiad import load_olympiad
        return load_olympiad(n_samples=n_samples, seed=seed, **kwargs)
    elif dataset == Dataset.MINERVA:
        from identity_bias.data.minerva import load_minerva
        return load_minerva(n_samples=n_samples, seed=seed, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def check_answer(dataset: Dataset, predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth for the given dataset."""
    if dataset == Dataset.MATH:
        from identity_bias.data.math_dataset import check_math_answer
        return check_math_answer(predicted, ground_truth)
    elif dataset == Dataset.GPQA:
        from identity_bias.data.gpqa import check_gpqa_answer
        return check_gpqa_answer(predicted, ground_truth)
    elif dataset == Dataset.BBH:
        from identity_bias.data.bbh import check_bbh_answer
        return check_bbh_answer(predicted, ground_truth)
    elif dataset == Dataset.AIME:
        from identity_bias.data.aime import check_aime_answer
        return check_aime_answer(predicted, ground_truth)
    elif dataset == Dataset.MMLU_PRO:
        from identity_bias.data.mmlu_pro import check_mmlu_pro_answer
        return check_mmlu_pro_answer(predicted, ground_truth)
    elif dataset == Dataset.OLYMPIAD:
        from identity_bias.data.olympiad import check_olympiad_answer
        return check_olympiad_answer(predicted, ground_truth)
    elif dataset == Dataset.MINERVA:
        from identity_bias.data.minerva import check_minerva_answer
        return check_minerva_answer(predicted, ground_truth)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


__all__ = ["Problem", "Solution", "load_dataset_problems", "check_answer"]
