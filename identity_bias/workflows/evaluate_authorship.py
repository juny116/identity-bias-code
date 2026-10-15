"""Evaluate the same solution under different authorship labels."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from identity_bias.critic.identity_critic import IdentityCritic
from identity_bias.data import check_answer
from identity_bias.evaluation.metrics import (
    compute_condition_metrics,
    compute_self_cross_gap,
)
from identity_bias.io.result_logger import ResultLogger
from identity_bias.llm import create_llm
from identity_bias.runtime.models import MODEL_PRESETS
from identity_bias.types import Dataset, IdentityCondition

CONDITION_MAP = {c.value: c for c in IdentityCondition}


def main():
    parser = argparse.ArgumentParser(
        description="Run identity swap critique experiment"
    )
    parser.add_argument(
        "--critic-model", type=str, required=True, choices=list(MODEL_PRESETS.keys())
    )
    parser.add_argument("--solutions-file", type=str, required=True)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["self", "other_model", "weak_model", "anonymous", "human"],
        choices=list(CONDITION_MAP.keys()),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math",
        choices=[d.value for d in Dataset],
        help="Dataset name for answer checking",
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--fresh", action="store_true", help="Start fresh (delete existing log)"
    )
    parser.add_argument(
        "--two-stage",
        action="store_true",
        help="Issue a verdict-only call before the full critique "
        "and record both. Doubles per-item cost.",
    )
    args = parser.parse_args()

    # Load solutions
    print(f"Loading solutions from {args.solutions_file}...")
    problems, solutions = ResultLogger.load_solutions(args.solutions_file)
    n_correct = sum(s.is_correct for s in solutions)
    n_wrong = len(solutions) - n_correct
    print(f"Loaded {len(problems)} problems ({n_correct} correct, {n_wrong} wrong)")

    # Create critic
    config = MODEL_PRESETS[args.critic_model](base_url=args.base_url)
    config.temperature = args.temperature
    llm = create_llm(config)
    critic = IdentityCritic(llm, model_name=args.critic_model, two_stage=args.two_stage)

    conditions = [CONDITION_MAP[c] for c in args.conditions]
    dataset = Dataset(args.dataset)
    ground_truths = {p.id: p.ground_truth for p in problems}
    check_fn = lambda pred, gt: check_answer(dataset, pred, gt)

    # Set up logging (resume by default)
    solver_model = solutions[0].solver_model if solutions else "unknown"
    exp_name = f"identity_swap_{args.critic_model}_on_{solver_model}_{args.dataset}"
    logger = ResultLogger(args.log_dir, exp_name, resume=not args.fresh)

    completed = logger.get_completed_ids("critique")

    # Run critique for each condition
    all_critiques = {c.value: [] for c in conditions}
    all_metrics = {}

    for condition in conditions:
        print(f"\n--- Condition: {condition.value} ---")

        # Filter out already completed
        pairs = []
        for p, s in zip(problems, solutions):
            key = f"{p.id}_{condition.value}"
            if key not in completed:
                pairs.append((p, s))

        print(f"  Already done: {len(problems) - len(pairs)}, remaining: {len(pairs)}")

        def critique_one(ps, cond=condition):
            p, s = ps
            return critic.critique(p, s, cond)

        errors = 0
        if pairs:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(critique_one, pair): pair for pair in pairs}
                for future in tqdm(
                    as_completed(futures), total=len(pairs), desc=condition.value
                ):
                    try:
                        result = future.result()
                        logger.log_critique(result)
                    except Exception as e:
                        errors += 1
                        tqdm.write(f"[ERROR] {e}")
        if errors:
            print(f"  Skipped {errors} critiques due to errors")

        # Load all critiques for this condition (including previously completed)
        all_logged = ResultLogger.load_critiques(logger.log_file)
        all_critiques[condition.value] = [
            c for c in all_logged if c.identity_condition == condition.value
        ]

        metrics = compute_condition_metrics(
            all_critiques[condition.value], check_fn, ground_truths
        )
        all_metrics[condition.value] = metrics

    # Display results
    print("\n" + "=" * 90)
    print("IDENTITY SWAP RESULTS")
    print("=" * 90)
    print(
        f"{'Condition':<15} {'Det.Acc':>8} {'Corr.Succ':>10} {'Harmful':>8} "
        f"{'FalseErr':>9} {'Confidence':>11}"
    )
    print("-" * 90)

    for condition in conditions:
        m = all_metrics[condition.value]
        print(
            f"{condition.value:<15} {m.detection_accuracy:>8.3f} "
            f"{m.correction_success_rate:>10.3f} {m.harmful_correction_rate:>8.3f} "
            f"{m.false_error_rate:>9.3f} {m.mean_confidence:>11.3f}"
        )

    # Self vs Cross gap
    if "self" in all_metrics and "anonymous" in all_metrics:
        gap = compute_self_cross_gap(all_metrics["self"], all_metrics["anonymous"])
        print(f"\n--- Self vs Anonymous Gap ---")
        print(
            f"  Detection gap:  {gap.detection_gap:+.3f} (positive = anonymous better)"
        )
        print(f"  Correction gap: {gap.correction_gap:+.3f}")
        print(f"  Harmful gap:    {gap.harmful_gap:+.3f} (positive = self worse)")

    print(f"\nResults saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
