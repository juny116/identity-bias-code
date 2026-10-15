"""N-of-K majority-voting baseline for the gated-pipeline mitigation comparison.

Calls verdict_only_neutral K times per item with non-zero temperature so the
critic produces different verdicts across runs, then aggregates by majority
vote. Compares against the single-shot gated baseline (already in
caveat_a_run_1 / gated_run_1).

Output: logs/n_of_k_run_1/n_of_k_<critic>_<solver>_<dataset>__<condition>.jsonl
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from identity_bias.config import IdentityCondition, Dataset, MODEL_PRESETS, LLMConfig
from identity_bias.critic.prompts import build_caveat_a_messages
from identity_bias.critic.identity_critic import IdentityCritic
from identity_bias.llm import create_llm


CONDITION_MAP = {c.value: c for c in IdentityCondition}


def main():
    parser = argparse.ArgumentParser(description="N-of-K majority voting baseline")
    parser.add_argument("--critic-model", required=True,
                        choices=list(MODEL_PRESETS.keys()))
    parser.add_argument("--solutions-file", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=[d.value for d in Dataset])
    parser.add_argument("--sample-cache", required=True,
                        help="Path to caveat_a_samples cache file for matching items")
    parser.add_argument("--conditions", nargs="+",
                        default=["anonymous", "self"],
                        choices=list(CONDITION_MAP.keys()))
    parser.add_argument("--K", type=int, default=5,
                        help="Number of verdict-only samples per item")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (must be > 0 for diversity)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--log-dir", default="logs/n_of_k_run_1")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.sample_cache) as f:
        sample = json.load(f)
    keep_ids = set(sample["wrong_ids"]) | set(sample["correct_ids"])

    from identity_bias.logging.result_logger import ResultLogger
    problems_all, solutions_all = ResultLogger.load_solutions(args.solutions_file)
    pairs = [(p, s) for p, s in zip(problems_all, solutions_all) if p.id in keep_ids]
    print(f"loaded {len(pairs)} sample items for K={args.K} voting")
    if not pairs:
        raise SystemExit("no items selected")

    # Build LLM with sampling-temperature for diversity
    cfg = MODEL_PRESETS[args.critic_model](base_url=args.base_url)
    cfg.temperature = args.temperature
    # Some presets fix top_p/top_k; ensure they allow diversity
    if cfg.top_p is None: cfg.top_p = 0.95
    if cfg.top_k is None: cfg.top_k = 50
    llm = create_llm(cfg)
    critic_helper = IdentityCritic(llm, model_name=args.critic_model, two_stage=False)

    solver_model = pairs[0][1].solver_model
    for cond in args.conditions:
        condc = CONDITION_MAP[cond]
        out_path = out_dir / (f"n_of_k_{args.critic_model}_{solver_model}"
                              f"_{args.dataset}__{cond}.jsonl")
        done = set()
        if out_path.exists():
            with open(out_path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        done.add(r["problem_id"])
                    except Exception:
                        pass
        todo = [(p, s) for (p, s) in pairs if p.id not in done]
        print(f"\n--- cond={cond} K={args.K} remaining={len(todo)} ---")
        if not todo:
            continue

        def vote_one(ps):
            p, s = ps
            messages = build_caveat_a_messages(
                question=p.question,
                chain_of_thought=s.chain_of_thought,
                final_answer=s.final_answer,
                identity_condition=condc,
                arm="verdict_only_neutral",
            )
            verdicts = []
            for _ in range(args.K):
                resp = llm.generate(messages)
                parsed = critic_helper._parse_response(resp.text)
                verdicts.append(bool(parsed.get("is_correct", True)))
            n_accept = sum(verdicts)
            majority_accept = n_accept > args.K / 2
            return {
                "problem_id": p.id,
                "identity_condition": cond,
                "critic_model": args.critic_model,
                "solver_model": s.solver_model,
                "K": args.K,
                "n_accept": n_accept,
                "majority_accept": majority_accept,
                "actually_correct": bool(s.is_correct),
            }

        errors = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool, open(out_path, "a") as fp:
            futures = {pool.submit(vote_one, ps): ps for ps in todo}
            for future in tqdm(as_completed(futures), total=len(todo),
                               desc=f"vote/{cond}"):
                try:
                    rec = future.result()
                    fp.write(json.dumps(rec) + "\n")
                    fp.flush()
                except Exception as e:
                    errors += 1
                    tqdm.write(f"[ERROR] {e}")
        if errors:
            print(f"  skipped {errors} items due to errors")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
