"""Metrics computation for identity bias experiments."""

import numpy as np
from dataclasses import dataclass, field

from identity_bias.critic.identity_critic import CritiqueResult


@dataclass
class ConditionMetrics:
    """Aggregated metrics for a single experimental condition."""
    condition: str
    n_samples: int
    n_correct_solutions: int
    n_wrong_solutions: int
    # Core metrics
    detection_accuracy: float      # Overall: critic correctly identifies correct/wrong
    correction_success_rate: float  # Among wrong solutions: wrong→correct after critique
    harmful_correction_rate: float  # Among correct solutions: correct→wrong after critique
    false_error_rate: float         # Correct solutions labeled as wrong
    # Confidence
    mean_confidence: float
    confidence_calibration: float   # ECE


@dataclass
class SelfCrossGap:
    """Self vs Cross critique performance gap."""
    self_detection_accuracy: float
    cross_detection_accuracy: float
    detection_gap: float  # cross - self (positive = cross is better)
    self_correction_success: float
    cross_correction_success: float
    correction_gap: float
    self_harmful_rate: float
    cross_harmful_rate: float
    harmful_gap: float  # self - cross (positive = self is worse)


def compute_condition_metrics(
    critiques: list[CritiqueResult],
    check_correction_fn=None,
    ground_truths: dict[str, str] | None = None,
) -> ConditionMetrics:
    """Compute metrics for a set of critiques under one condition.

    Args:
        critiques: List of CritiqueResult for one condition.
        check_correction_fn: Optional fn(predicted, ground_truth) -> bool.
        ground_truths: Dict mapping problem_id to ground truth answer.
    """
    if not critiques:
        return ConditionMetrics(
            condition="", n_samples=0, n_correct_solutions=0, n_wrong_solutions=0,
            detection_accuracy=0, correction_success_rate=0,
            harmful_correction_rate=0, false_error_rate=0,
            mean_confidence=0, confidence_calibration=0,
        )

    condition = critiques[0].identity_condition
    n = len(critiques)

    correct_detections = 0
    # For wrong solutions
    wrong_solutions = [c for c in critiques if not c.actually_correct]
    correct_solutions = [c for c in critiques if c.actually_correct]

    # Detection accuracy: critic says "wrong" for wrong solutions, "correct" for correct
    for c in critiques:
        critic_says_correct = c.is_correct
        if c.actually_correct == critic_says_correct:
            correct_detections += 1

    detection_accuracy = correct_detections / n if n > 0 else 0

    # False error rate: correct solutions that critic labels as wrong
    false_errors = sum(1 for c in correct_solutions if not c.is_correct)
    false_error_rate = false_errors / len(correct_solutions) if correct_solutions else 0

    # Correction success rate: among wrong solutions, does the corrected answer match ground truth?
    correction_successes = 0
    if check_correction_fn and ground_truths and wrong_solutions:
        for c in wrong_solutions:
            if not c.is_correct and c.corrected_answer and c.problem_id in ground_truths:
                if check_correction_fn(c.corrected_answer, ground_truths[c.problem_id]):
                    correction_successes += 1
    correction_success_rate = correction_successes / len(wrong_solutions) if wrong_solutions else 0

    # Harmful correction rate: correct solutions where critic "corrects" to a wrong answer
    harmful_corrections = 0
    if check_correction_fn and ground_truths and correct_solutions:
        for c in correct_solutions:
            if not c.is_correct and c.corrected_answer and c.problem_id in ground_truths:
                if not check_correction_fn(c.corrected_answer, ground_truths[c.problem_id]):
                    harmful_corrections += 1
    harmful_correction_rate = harmful_corrections / len(correct_solutions) if correct_solutions else 0

    # Confidence
    confidences = [c.confidence for c in critiques]
    accuracies = [int(c.actually_correct == c.is_correct) for c in critiques]

    return ConditionMetrics(
        condition=condition,
        n_samples=n,
        n_correct_solutions=len(correct_solutions),
        n_wrong_solutions=len(wrong_solutions),
        detection_accuracy=detection_accuracy,
        correction_success_rate=correction_success_rate,
        harmful_correction_rate=harmful_correction_rate,
        false_error_rate=false_error_rate,
        mean_confidence=float(np.mean(confidences)) if confidences else 0,
        confidence_calibration=_compute_ece(confidences, accuracies),
    )


def compute_self_cross_gap(
    self_metrics: ConditionMetrics,
    cross_metrics: ConditionMetrics,
) -> SelfCrossGap:
    """Compute the gap between self-critique and cross-critique performance."""
    return SelfCrossGap(
        self_detection_accuracy=self_metrics.detection_accuracy,
        cross_detection_accuracy=cross_metrics.detection_accuracy,
        detection_gap=cross_metrics.detection_accuracy - self_metrics.detection_accuracy,
        self_correction_success=self_metrics.correction_success_rate,
        cross_correction_success=cross_metrics.correction_success_rate,
        correction_gap=cross_metrics.correction_success_rate - self_metrics.correction_success_rate,
        self_harmful_rate=self_metrics.harmful_correction_rate,
        cross_harmful_rate=cross_metrics.harmful_correction_rate,
        harmful_gap=self_metrics.harmful_correction_rate - cross_metrics.harmful_correction_rate,
    )


def _compute_ece(confidences: list[float], accuracies: list[int], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    if not confidences:
        return 0.0

    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = accuracies[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_accuracy - bin_confidence)

    return float(ece)
