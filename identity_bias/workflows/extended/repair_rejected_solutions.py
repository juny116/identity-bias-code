"""Repair rejected solutions."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from identity_bias.critic.identity_critic import IdentityCritic
from identity_bias.critic.prompts import build_repair_only_messages
from identity_bias.llm import create_llm
from identity_bias.runtime.models import MODEL_PRESETS
from identity_bias.types import Dataset, IdentityCondition

CONDITION_MAP = {c.value: c for c in IdentityCondition}


def load_stage1_rejects(verdict_only_neutral_log, conditions):
    """Return list of (problem_id, identity_condition, ...) for records
    where stage-1 verdict rejected."""
    rejects = []
    with open(verdict_only_neutral_log) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") != "critique":
                continue
            c = rec["critique"]
            if c["identity_condition"] not in conditions:
                continue
            if c["is_correct"]:  # stage-1 accepted, no repair needed
                continue
            rejects.append(
                {
                    "problem_id": c["problem_id"],
                    "identity_condition": c["identity_condition"],
                    "stage1_is_correct": c["is_correct"],
                    "stage1_error_description": c.get("error_description"),
                    "actually_correct": c["actually_correct"],
                    "solver_model": c["solver_model"],
                }
            )
    return rejects


def main():
    parser = argparse.ArgumentParser(
        description="Gated mitigation: stage-2 repair-only"
    )
    parser.add_argument(
        "--critic-model", required=True, choices=list(MODEL_PRESETS.keys())
    )
    parser.add_argument(
        "--solutions-file",
        required=True,
        help="Same solutions file used by Caveat A stage 1.",
    )
    parser.add_argument("--dataset", required=True, choices=[d.value for d in Dataset])
    parser.add_argument(
        "--verdict-only-log",
        required=True,
        help="Path to the Caveat A verdict_only_neutral log for this "
        "critic+solver+dataset (stage-1 verdicts).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["anonymous", "self"],
        choices=list(CONDITION_MAP.keys()),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--log-dir", default="logs/gated_run_1")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    rejects = load_stage1_rejects(args.verdict_only_log, set(args.conditions))
    print(
        f"loaded {len(rejects)} stage-1 reject records "
        f"(critic={args.critic_model}, dataset={args.dataset}, conditions={args.conditions})"
    )
    if not rejects:
        print("no rejects to repair; exiting.")
        return

    # Load solutions for content
    from identity_bias.io.result_logger import ResultLogger

    problems_all, solutions_all = ResultLogger.load_solutions(args.solutions_file)
    sol_by_pid = {p.id: (p, s) for p, s in zip(problems_all, solutions_all)}

    # Set up LLM
    config = MODEL_PRESETS[args.critic_model](base_url=args.base_url)
    config.temperature = args.temperature
    llm = create_llm(config)

    # Group by condition for cleaner output files
    for cond in args.conditions:
        cond_records = [r for r in rejects if r["identity_condition"] == cond]
        if not cond_records:
            continue

        solver_model = cond_records[0]["solver_model"]
        out_path = log_dir / (
            f"gated_stage2_{args.critic_model}_{solver_model}"
            f"_{args.dataset}__{cond}.jsonl"
        )
        # Skip already-done records (resume)
        done_keys = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        done_keys.add(rec["problem_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass

        todo = [r for r in cond_records if r["problem_id"] not in done_keys]
        print(
            f"\n--- condition={cond} already={len(cond_records) - len(todo)} remaining={len(todo)} ---"
        )
        if not todo:
            continue

        def repair_one(rec):
            pid = rec["problem_id"]
            if pid not in sol_by_pid:
                raise KeyError(f"problem_id {pid} not in solutions file")
            problem, solution = sol_by_pid[pid]
            messages = build_repair_only_messages(
                question=problem.question,
                chain_of_thought=solution.chain_of_thought,
                final_answer=solution.final_answer,
                identity_condition=CONDITION_MAP[cond],
            )
            response = llm.generate(messages)
            # parse with the same JSON parser as critic
            critic = IdentityCritic(llm, model_name=args.critic_model, two_stage=False)
            parsed = critic._parse_response(response.text)
            corrected = parsed.get("corrected_answer")
            return {
                "problem_id": pid,
                "identity_condition": cond,
                "critic_model": args.critic_model,
                "solver_model": rec["solver_model"],
                "stage1_is_correct": rec["stage1_is_correct"],
                "stage1_error_description": rec["stage1_error_description"],
                "actually_correct": rec["actually_correct"],
                "corrected_answer": corrected,
                "raw_response": response.text,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
            }

        errors = 0
        with (
            ThreadPoolExecutor(max_workers=args.workers) as pool,
            open(out_path, "a") as fp,
        ):
            futures = {pool.submit(repair_one, rec): rec for rec in todo}
            for future in tqdm(
                as_completed(futures), total=len(todo), desc=f"repair/{cond}"
            ):
                try:
                    out_rec = future.result()
                    fp.write(json.dumps(out_rec) + "\n")
                    fp.flush()
                except Exception as e:
                    errors += 1
                    tqdm.write(f"[ERROR] {e}")
        if errors:
            print(f"  skipped {errors} items due to errors")

    print(f"\nDone. Outputs in {log_dir}/")


if __name__ == "__main__":
    main()
