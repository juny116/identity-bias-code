"""Compare pairwise response preferences across judge interfaces and response orders."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from identity_bias.runtime.paths import workspace_root
from identity_bias.runtime.token_budget import available_tokens

REPO = workspace_root()
MAX_CONTEXT = 32768

from identity_bias.critic.pairwise_interfaces import (
    PAIRWISE_INTERFACES,
    build_pairwise_message,
)
from identity_bias.runtime.models import MODEL_PRESETS

SAMPLE = REPO / "logs" / "subjective_samples" / "rewardbench_150.json"


def parse_choice(text):
    if not text:
        return None, False
    try:
        d = json.loads(text)
        if isinstance(d, dict) and d.get("choice") in ("A", "B"):
            return d, True
    except json.JSONDecodeError:
        pass
    return None, False


def run_one(client, model_name, iface, pair, extra_body, max_tokens, swap):
    a = pair["response_B"] if swap else pair["response_A"]
    b = pair["response_A"] if swap else pair["response_B"]
    user = build_pairwise_message(pair["prompt"], a, b)
    messages = [
        {"role": "system", "content": iface.system_prompt},
        {"role": "user", "content": user},
    ]
    avail = available_tokens(messages, MAX_CONTEXT)
    if avail < 256:
        return {
            "parse_ok": False,
            "raw_choice": None,
            "picked_orig": None,
            "picked_chosen": None,
            "finish_reason": "context_overflow",
            "raw": "[SKIP]",
        }
    kwargs = dict(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=min(max_tokens, avail),
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "pref", "schema": iface.schema, "strict": True},
        },
    )
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        resp = client.chat.completions.create(**kwargs)
        ch = resp.choices[0]
        text = ch.message.content or ""
        parsed, ok = parse_choice(text)
        if ch.finish_reason == "length":
            ok = False
        raw_choice = parsed.get("choice") if ok else None
        # normalize to ORIGINAL frame: presented-A is orig-B when swapped
        picked_orig = picked_chosen = None
        if ok:
            picked_orig = "A" if raw_choice == "A" else "B"
            if swap:
                picked_orig = "B" if raw_choice == "A" else "A"
            picked_chosen = picked_orig == pair["chosen_side"]
        return {
            "parse_ok": ok,
            "raw_choice": raw_choice,
            "picked_orig": picked_orig,
            "picked_chosen": picked_chosen,
            "finish_reason": ch.finish_reason,
            "confidence": (parsed.get("confidence") if ok else None),
            "raw": text,
        }
    except Exception as e:
        return {
            "parse_ok": False,
            "raw_choice": None,
            "picked_orig": None,
            "picked_chosen": None,
            "finish_reason": "error",
            "raw": f"[ERROR] {e}",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--interfaces", nargs="+", default=list(PAIRWISE_INTERFACES))
    ap.add_argument("--swap-order", action="store_true")
    ap.add_argument("--log-dir", default="logs/subjective_confound")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8000)
    args = ap.parse_args()

    cfg = MODEL_PRESETS[args.judge](base_url=args.base_url)
    extra_body = {}
    if not cfg.enable_thinking:
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if cfg.reasoning_effort is not None:
        extra_body["reasoning_effort"] = cfg.reasoning_effort
    client = OpenAI(api_key="EMPTY", base_url=cfg.base_url, timeout=1800.0)

    pairs = json.load(open(SAMPLE))["pairs"]
    log_dir = REPO / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_swap" if args.swap_order else ""
    for iname in args.interfaces:
        iface = PAIRWISE_INTERFACES[iname]
        out_path = log_dir / f"{args.judge}_{iname}{suffix}.jsonl"
        done = set()
        if out_path.exists():
            for line in open(out_path):
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
        todo = [p for p in pairs if p["id"] not in done]
        if not todo:
            print(f"  {iname}{suffix}: all done")
            continue
        with (
            open(out_path, "a") as out_f,
            ThreadPoolExecutor(max_workers=args.workers) as ex,
        ):
            futs = {
                ex.submit(
                    run_one,
                    client,
                    cfg.model_name,
                    iface,
                    p,
                    extra_body,
                    args.max_tokens,
                    args.swap_order,
                ): p
                for p in todo
            }
            for fut in tqdm(
                as_completed(futs),
                total=len(futs),
                desc=f"{args.judge}/{iname}{suffix}",
            ):
                p = futs[fut]
                rec = {
                    "judge": args.judge,
                    "interface": iname,
                    "swap": args.swap_order,
                    "id": p["id"],
                    "category": p["category"],
                    "chosen_side": p["chosen_side"],
                    **fut.result(),
                }
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
    print("done")


if __name__ == "__main__":
    main()
