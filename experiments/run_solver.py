"""Step 1: Generate solutions for problems using solver models.

Usage:
    python experiments/run_solver.py --model qwen3 --dataset math
    python experiments/run_solver.py --model qwen3 --dataset math --fresh  # start over
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from identity_bias.config import Dataset, MODEL_PRESETS
from identity_bias.llm import create_llm
from identity_bias.data import load_dataset_problems
from identity_bias.solver.cot_solver import CoTSolver
from identity_bias.logging.result_logger import ResultLogger


def main():
    parser = argparse.ArgumentParser(description="Generate solutions for reasoning problems")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--dataset", type=str, required=True,
                        choices=[d.value for d in Dataset])
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (delete existing log)")
    args = parser.parse_args()

    dataset = Dataset(args.dataset)
    preset_fn = MODEL_PRESETS[args.model]
    config = preset_fn(base_url=args.base_url)
    config.temperature = args.temperature

    # Load problems
    print(f"Loading problems from {args.dataset}...")
    problems = load_dataset_problems(dataset, n_samples=args.n_samples, seed=args.seed)
    print(f"Loaded {len(problems)} problems")

    # Create solver
    llm = create_llm(config)
    solver = CoTSolver(llm, model_name=args.model)

    # Set up logging (resume by default)
    exp_name = f"solutions_{args.model}_{args.dataset}"
    logger = ResultLogger(args.log_dir, exp_name, resume=not args.fresh)

    # Skip already completed
    completed = logger.get_completed_ids("solution")
    remaining = [p for p in problems if p.id not in completed]
    print(f"Already completed: {len(completed)}, remaining: {len(remaining)}")

    if not remaining:
        print("All problems already solved. Use --fresh to re-run.")
        return

    # Solve problems concurrently
    correct = 0
    total = 0

    errors = 0

    def solve_one(problem):
        return problem, solver.solve(problem, dataset)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(solve_one, p): p for p in remaining}
        for future in tqdm(as_completed(futures), total=len(remaining),
                           desc=f"Solving ({args.model})"):
            try:
                problem, solution = future.result()
                logger.log_solution(problem, solution)
                if solution.is_correct:
                    correct += 1
                total += 1
            except Exception as e:
                errors += 1
                tqdm.write(f"[ERROR] {e}")

    if errors:
        print(f"\nSkipped {errors} problems due to errors")

    # Report overall stats (including previously completed)
    all_problems, all_solutions = ResultLogger.load_solutions(logger.log_file)
    total_correct = sum(s.is_correct for s in all_solutions)
    print(f"\nTotal: {total_correct}/{len(all_solutions)} correct "
          f"({total_correct/len(all_solutions)*100:.1f}%)")
    print(f"Solutions saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
