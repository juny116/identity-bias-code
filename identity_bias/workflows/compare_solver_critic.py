"""Compare critique metrics across solver and critic model pairs."""

import argparse
import glob
from dataclasses import asdict

from tqdm import tqdm

from identity_bias.critic.identity_critic import IdentityCritic
from identity_bias.data import check_answer
from identity_bias.evaluation.metrics import compute_condition_metrics
from identity_bias.io.result_logger import ResultLogger
from identity_bias.llm import create_llm
from identity_bias.runtime.models import MODEL_PRESETS
from identity_bias.types import Dataset, IdentityCondition


def main():
    parser = argparse.ArgumentParser(description="Run solver-critic matrix experiment")
    parser.add_argument(
        "--solutions-dir",
        type=str,
        default="logs",
        help="Directory containing solution JSONL files",
    )
    parser.add_argument(
        "--critic-models", nargs="+", required=True, choices=list(MODEL_PRESETS.keys())
    )
    parser.add_argument(
        "--solver-models",
        nargs="+",
        default=None,
        choices=list(MODEL_PRESETS.keys()),
        help="Models whose solutions to evaluate. Defaults to critic-models.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=[d.value for d in Dataset]
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    check_fn = lambda pred, gt: check_answer(dataset, pred, gt)

    solver_model_names = (
        args.solver_models if args.solver_models else args.critic_models
    )

    # Discover solution files for each solver model
    solver_files = {}
    for model in solver_model_names:
        # Try exact name first, then wildcard with timestamp suffix
        exact = f"{args.solutions_dir}/solutions_{model}_{args.dataset}.jsonl"
        pattern = f"{args.solutions_dir}/solutions_{model}_{args.dataset}_*.jsonl"
        matches = sorted(glob.glob(exact)) or sorted(glob.glob(pattern))
        if matches:
            solver_files[model] = matches[-1]
            print(f"Found solutions for {model}: {solver_files[model]}")
        else:
            print(f"WARNING: No solution file found for {model} ({exact})")

    if not solver_files:
        print("ERROR: No solution files found. Run run_solver.py first.")
        return

    # Build the matrix: for each (solver, critic) pair, run anonymous critique
    results_matrix = {}  # (solver, critic) -> ConditionMetrics

    for critic_name in args.critic_models:
        config = MODEL_PRESETS[critic_name](base_url=args.base_url)
        llm = create_llm(config)
        critic = IdentityCritic(llm, model_name=critic_name)

        for solver_name, sol_file in solver_files.items():
            print(f"\n--- Critic: {critic_name} | Solver: {solver_name} ---")
            problems, solutions = ResultLogger.load_solutions(sol_file)
            ground_truths = {p.id: p.ground_truth for p in problems}

            # Use SELF if critic==solver, ANONYMOUS otherwise
            if critic_name == solver_name:
                condition = IdentityCondition.SELF
            else:
                condition = IdentityCondition.ANONYMOUS

            exp_name = f"matrix_{critic_name}_on_{solver_name}_{args.dataset}"
            logger = ResultLogger(args.log_dir, exp_name)
            completed_ids = logger.get_completed_ids("critique")

            critiques = []
            skipped = 0
            for problem, solution in tqdm(
                zip(problems, solutions),
                total=len(problems),
                desc=f"{critic_name}→{solver_name}",
            ):
                key = f"{problem.id}_{condition.value}"
                if key in completed_ids:
                    skipped += 1
                    continue
                result = critic.critique(problem, solution, condition)
                logger.log_critique(result)
                critiques.append(result)

            if skipped:
                print(f"  (skipped {skipped} already-completed critiques)")

            metrics = compute_condition_metrics(critiques, check_fn, ground_truths)
            logger.log_metrics(asdict(metrics))
            results_matrix[(solver_name, critic_name)] = metrics

    # Display matrix
    solver_names = list(solver_files.keys())
    critic_names = args.critic_models

    col_w = 18

    def print_matrix(title, metric_fn):
        print("\n" + "=" * (18 + col_w * len(critic_names)))
        print(f"SOLVER-CRITIC MATRIX: {title}")
        print("=" * (18 + col_w * len(critic_names)))
        label = "Solver \\ Critic"
        header = f"{label:<18}" + "".join(f"{c:>{col_w}}" for c in critic_names)
        print(header)
        print("-" * (18 + col_w * len(critic_names)))
        for solver in solver_names:
            row = f"{solver:<18}"
            for critic in critic_names:
                key = (solver, critic)
                if key in results_matrix:
                    m = results_matrix[key]
                    marker = "*" if solver == critic else " "
                    row += f"{metric_fn(m):>{col_w - 1}.3f}{marker}"
                else:
                    row += f"{'N/A':>{col_w}}"
            print(row)
        print("* = self-critique (diagonal)")

    print_matrix("Detection Accuracy", lambda m: m.detection_accuracy)
    print_matrix("Correction Success", lambda m: m.correction_success_rate)
    print_matrix("Harmful Correction", lambda m: m.harmful_correction_rate)
    print_matrix("False Error Rate", lambda m: m.false_error_rate)
    print_matrix("Mean Confidence", lambda m: m.mean_confidence)


if __name__ == "__main__":
    main()
