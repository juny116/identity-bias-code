"""Compare structured verdict interfaces on a fixed pool of solutions."""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from identity_bias.runtime.paths import workspace_root
from identity_bias.runtime.token_budget import available_tokens

REPO = workspace_root()

MAX_CONTEXT = 32768

from identity_bias.critic.verdict_interfaces import (
    INTERFACES,
    build_user_message,
    with_reasoning_field,
)
from identity_bias.io.result_logger import ResultLogger
from identity_bias.runtime.models import MODEL_PRESETS

SOLVERS = ["qwen3", "qwen3-no-think", "gpt-oss-think", "gpt-oss-no-think"]
SOLUTION_GLOBS = [
    "logs/run_1",
    "logs/run_2",
    "logs/run_3",
    "logs/run_4",
    "logs/run_5",
    "logs/no_think_run_1",
    "logs/no_think_run_2",
    "logs/no_think_run_3",
    "logs/oss_think_run_1",
    "logs/oss_no-think_run_1",
]
SAMPLE_CACHE = REPO / "logs" / "interface_confound_samples"


def collect_candidates(dataset: str):
    """Collect all cached (problem, solution) pairs for a dataset across solvers,
    deduplicated by (solver, problem_id). Returns list of dicts."""
    cands = []
    seen = set()
    for sol_path in REPO.glob(f"logs/**/solutions_*_{dataset}.jsonl"):
        stem = sol_path.stem.replace("solutions_", "")
        solver = stem[: -(len(dataset) + 1)] if stem.endswith("_" + dataset) else None
        if solver not in SOLVERS:
            continue
        try:
            problems, solutions = ResultLogger.load_solutions(str(sol_path))
        except Exception:
            continue
        pmap = {p.id: p for p in problems}
        for s in solutions:
            key = (solver, s.problem_id)
            if key in seen:
                continue
            seen.add(key)
            p = pmap.get(s.problem_id)
            if p is None:
                continue
            cands.append(
                {
                    "dataset": dataset,
                    "problem_id": s.problem_id,
                    "solver": solver,
                    "question": p.question,
                    "ground_truth": p.ground_truth,
                    "chain_of_thought": s.chain_of_thought,
                    "final_answer": s.final_answer,
                    "is_correct": bool(s.is_correct),
                }
            )
    return cands


def lock_pool(dataset: str, n_wrong: int, n_correct: int, seed: int):
    """Pick a fixed, balanced pool of candidates for the dataset, one per
    problem_id where possible, persisted so every judge sees identical content."""
    SAMPLE_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = SAMPLE_CACHE / f"{dataset}_w{n_wrong}_c{n_correct}_seed{seed}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    cands = collect_candidates(dataset)
    rng = random.Random(seed)
    rng.shuffle(cands)

    # prefer one candidate per problem_id for diversity
    by_problem_wrong, by_problem_correct = {}, {}
    extra_wrong, extra_correct = [], []
    for c in cands:
        if not c["is_correct"]:
            if c["problem_id"] not in by_problem_wrong:
                by_problem_wrong[c["problem_id"]] = c
            else:
                extra_wrong.append(c)
        else:
            if c["problem_id"] not in by_problem_correct:
                by_problem_correct[c["problem_id"]] = c
            else:
                extra_correct.append(c)

    wrong = list(by_problem_wrong.values())
    correct = list(by_problem_correct.values())
    rng.shuffle(wrong)
    rng.shuffle(correct)
    # top up from extras if we don't have enough unique-problem items
    wrong = (wrong + extra_wrong)[:n_wrong]
    correct = (correct + extra_correct)[:n_correct]

    pool = wrong + correct
    rng.shuffle(pool)
    meta = {
        "dataset": dataset,
        "seed": seed,
        "n_wrong_requested": n_wrong,
        "n_correct_requested": n_correct,
        "n_wrong": len(wrong),
        "n_correct": len(correct),
        "pool": pool,
    }
    with open(cache_path, "w") as f:
        json.dump(meta, f)
    return meta


def parse_verdict(text: str):
    """Parse a constrained-JSON judge response. Returns (parsed_dict, parse_ok)."""
    if not text:
        return None, False
    try:
        d = json.loads(text)
        if isinstance(d, dict) and "is_correct" in d:
            return d, True
    except json.JSONDecodeError:
        pass
    return None, False


REFERENCE_PREAMBLE = (
    "You are checking a candidate solution. You are GIVEN the correct final "
    "answer to the problem (shown at the end as 'CORRECT FINAL ANSWER "
    "(reference)'). Your task is ONLY to decide whether the candidate's final "
    "answer matches that provided correct answer. Treat the provided correct "
    "answer as authoritative; do not re-derive it or second-guess it.\n\n"
)


def run_one(client, model_name, iface, cand, extra_body, max_tokens, reference=False):
    user_msg = build_user_message(
        cand["question"], cand["chain_of_thought"], cand["final_answer"]
    )
    system = iface.system_prompt
    if reference:
        system = REFERENCE_PREAMBLE + system
        user_msg = (
            user_msg + f"\n\nCORRECT FINAL ANSWER (reference):\n{cand['ground_truth']}"
        )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    # clamp max_tokens to fit context window (long solver traces can be huge).
    # tiktoken undercounts vs the Qwen tokenizer, so apply a 1.25x safety factor.
    available = available_tokens(messages, MAX_CONTEXT)
    prompt_tok = MAX_CONTEXT - available - 128
    if available < 256:
        # judged content alone overflows context: log as non-verdict
        return {
            "parse_ok": False,
            "is_correct": None,
            "confidence": None,
            "corrected_answer": None,
            "quality_score": None,
            "finish_reason": "context_overflow",
            "completion_tokens": None,
            "raw": f"[SKIP] prompt {prompt_tok} tok exceeds context",
        }
    eff_max_tokens = min(max_tokens, available)
    create_kwargs = dict(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=eff_max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verdict", "schema": iface.schema, "strict": True},
        },
    )
    if extra_body:
        create_kwargs["extra_body"] = extra_body
    try:
        resp = client.chat.completions.create(**create_kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        finish = choice.finish_reason
        parsed, ok = parse_verdict(text)
        # truncation or invalid -> non-verdict
        if finish == "length":
            ok = False
        return {
            "parse_ok": ok,
            "is_correct": (parsed.get("is_correct") if ok else None),
            "confidence": (parsed.get("confidence") if ok else None),
            "corrected_answer": (parsed.get("corrected_answer") if ok else None),
            "quality_score": (parsed.get("quality_score") if ok else None),
            "finish_reason": finish,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
            "raw": text,
        }
    except Exception as e:
        return {
            "parse_ok": False,
            "is_correct": None,
            "confidence": None,
            "corrected_answer": None,
            "quality_score": None,
            "finish_reason": "error",
            "completion_tokens": None,
            "raw": f"[ERROR] {e}",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, help="MODEL_PRESETS key")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--datasets", nargs="+", default=["aime"])
    ap.add_argument("--n-wrong", type=int, default=50)
    ap.add_argument("--n-correct", type=int, default=25)
    ap.add_argument("--interfaces", nargs="+", default=list(INTERFACES.keys()))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--log-dir", default="logs/interface_confound_pilot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--think-schema",
        action="store_true",
        help="deliberation-in-schema: prepend a free-text reasoning "
        "field to each interface schema (for the think ablation)",
    )
    ap.add_argument(
        "--reference",
        action="store_true",
        help="reference-provided control: inject the gold final answer "
        "and ask only whether the candidate matches it",
    )
    args = ap.parse_args()

    config = MODEL_PRESETS[args.judge](base_url=args.base_url)
    model_name = config.model_name
    extra_body = {}
    if not config.enable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if config.reasoning_effort is not None:
        extra_body["reasoning_effort"] = config.reasoning_effort

    client = OpenAI(api_key="EMPTY", base_url=config.base_url, timeout=1800.0)

    log_dir = REPO / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        meta = lock_pool(dataset, args.n_wrong, args.n_correct, args.seed)
        pool = meta["pool"]
        print(
            f"[{dataset}] pool: {meta['n_wrong']} wrong + {meta['n_correct']} correct "
            f"= {len(pool)} items"
        )

        for iface_name in args.interfaces:
            iface = INTERFACES[iface_name]
            if args.think_schema:
                iface = with_reasoning_field(iface)
            out_path = log_dir / f"{args.judge}_{dataset}_{iface_name}.jsonl"
            # resume: skip done problem_ids
            done = set()
            if out_path.exists():
                with open(out_path) as f:
                    for line in f:
                        try:
                            done.add(json.loads(line)["problem_id"])
                        except Exception:
                            pass
            todo = [c for c in pool if c["problem_id"] not in done]
            if not todo:
                print(f"  {iface_name}: all done")
                continue

            with (
                open(out_path, "a") as out_f,
                ThreadPoolExecutor(max_workers=args.workers) as ex,
            ):
                futs = {
                    ex.submit(
                        run_one,
                        client,
                        model_name,
                        iface,
                        c,
                        extra_body,
                        args.max_tokens,
                        args.reference,
                    ): c
                    for c in todo
                }
                for fut in tqdm(
                    as_completed(futs),
                    total=len(futs),
                    desc=f"{args.judge}/{dataset}/{iface_name}",
                ):
                    c = futs[fut]
                    res = fut.result()
                    rec = {
                        "judge": args.judge,
                        "dataset": dataset,
                        "interface": iface_name,
                        "problem_id": c["problem_id"],
                        "solver": c["solver"],
                        "actually_correct": c["is_correct"],
                        **res,
                    }
                    out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()

    print("done")


if __name__ == "__main__":
    main()
