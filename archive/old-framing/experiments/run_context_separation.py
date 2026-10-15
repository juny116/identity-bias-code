"""Step 4 (Optional): Run Context/Anchoring Ablation (Experiment 3).

Tests whether session context affects critique quality independently of identity.

Usage:
    python experiments/run_context_separation.py \
        --critic-model qwen3 \
        --solutions-file logs/solutions_qwen3_math_*.jsonl \
        --dataset math
"""

import argparse
from tqdm import tqdm
from dataclasses import asdict

from identity_bias.config import (
    ContextCondition, IdentityCondition, Dataset, MODEL_PRESETS,
)
from identity_bias.llm import create_llm
from identity_bias.data import check_answer
from identity_bias.critic.identity_critic import IdentityCritic, CritiqueResult
from identity_bias.critic.prompts import CRITIC_SYSTEM_PROMPT
from identity_bias.evaluation.metrics import compute_condition_metrics
from identity_bias.logging.result_logger import ResultLogger


CONTEXT_MAP = {c.value: c for c in ContextCondition}


def build_same_session_messages(question, solution):
    """Simulate same-session: solution as assistant turn, then review request."""
    return [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve the following problem step by step.\n\nProblem: {question}"},
        {"role": "assistant", "content": solution.chain_of_thought},
        {"role": "user", "content": (
            "Please review your solution above. Check each step carefully and determine "
            "if there are any errors. Respond with your assessment in the required JSON format."
        )},
    ]


def main():
    parser = argparse.ArgumentParser(description="Run context separation experiment")
    parser.add_argument("--critic-model", type=str, required=True,
                        choices=["qwen3", "gpt-oss", "gemini-flash", "claude-sonnet"])
    parser.add_argument("--solutions-file", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True,
                        choices=[d.value for d in Dataset])
    parser.add_argument("--conditions", nargs="+",
                        default=["same_session", "new_session", "paraphrased"],
                        choices=list(CONTEXT_MAP.keys()))
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--log-dir", type=str, default="logs")
    args = parser.parse_args()

    # Load solutions
    print(f"Loading solutions from {args.solutions_file}...")
    problems, solutions = ResultLogger.load_solutions(args.solutions_file)
    print(f"Loaded {len(problems)} problems")

    config = MODEL_PRESETS[args.critic_model](base_url=args.base_url)
    llm = create_llm(config)
    critic = IdentityCritic(llm, model_name=args.critic_model)

    dataset = Dataset(args.dataset)
    ground_truths = {p.id: p.ground_truth for p in problems}
    check_fn = lambda pred, gt: check_answer(dataset, pred, gt)

    conditions = [CONTEXT_MAP[c] for c in args.conditions]
    exp_name = f"context_sep_{args.critic_model}_{args.dataset}"
    logger = ResultLogger(args.log_dir, exp_name)

    all_metrics = {}

    for condition in conditions:
        print(f"\n--- Condition: {condition.value} ---")
        critiques = []

        for problem, solution in tqdm(zip(problems, solutions), total=len(problems),
                                       desc=condition.value):
            if condition == ContextCondition.SAME_SESSION:
                messages = build_same_session_messages(problem.question, solution)
                response = llm.generate(messages)
                parsed = critic._parse_response(response.text)
                critique = CritiqueResult(
                    problem_id=problem.id,
                    identity_condition=condition.value,
                    critic_model=args.critic_model,
                    solver_model=solution.solver_model,
                    is_correct=parsed.get("is_correct", True),
                    error_description=parsed.get("error_description"),
                    corrected_answer=parsed.get("corrected_answer"),
                    confidence=parsed.get("confidence", 0.5),
                    actually_correct=solution.is_correct,
                    raw_response=response.text,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )
            elif condition == ContextCondition.NEW_SESSION:
                critique = critic.critique(problem, solution, IdentityCondition.SELF)
                critique.identity_condition = condition.value
            else:
                # Paraphrased: use anonymous (paraphrasing TODO for future)
                critique = critic.critique(problem, solution, IdentityCondition.ANONYMOUS)
                critique.identity_condition = condition.value

            logger.log_critique(critique)
            critiques.append(critique)

        metrics = compute_condition_metrics(critiques, check_fn, ground_truths)
        all_metrics[condition.value] = metrics
        logger.log_metrics(asdict(metrics))

    # Results
    print("\n" + "=" * 90)
    print("CONTEXT SEPARATION RESULTS")
    print("=" * 90)
    print(f"{'Condition':<20} {'Det.Acc':>8} {'Corr.Succ':>10} {'Harmful':>8} {'Confidence':>11}")
    print("-" * 60)

    for condition in conditions:
        m = all_metrics[condition.value]
        print(f"{condition.value:<20} {m.detection_accuracy:>8.3f} "
              f"{m.correction_success_rate:>10.3f} {m.harmful_correction_rate:>8.3f} "
              f"{m.mean_confidence:>11.3f}")

    print(f"\nResults saved to: {logger.log_file}")


if __name__ == "__main__":
    main()
