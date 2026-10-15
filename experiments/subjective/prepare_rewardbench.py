"""Sample a balanced subjective pairwise-preference set from RewardBench
(allenai/reward-bench) for the interface-confound subjective extension.

Keeps chat / chat-hard / safety categories (excludes code hep-* and math-prm,
which drift toward objective correctness). Randomizes which response is A vs B
per pair (fixed seed) and records the curated 'chosen' side as a SECONDARY label.
Writes logs/subjective_samples/rewardbench_150.json.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

REPO = Path(__file__).parent.parent.parent
OUT = REPO / "logs" / "subjective_samples" / "rewardbench_150.json"

CATEGORY = {
    "chat": ["alpacaeval-easy", "alpacaeval-hard", "alpacaeval-length",
             "mt-bench-easy", "mt-bench-med", "mt-bench-hard"],
    "chat-hard": ["llmbar-natural", "llmbar-adver-neighbor", "llmbar-adver-GPTInst",
                  "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-respond",
               "xstest-should-refuse", "donotanswer"],
}
SUB2CAT = {s: c for c, subs in CATEGORY.items() for s in subs}
PER_CAT = 50
SEED = 42


def main():
    ds = load_dataset("allenai/reward-bench", split="filtered")
    by_cat = defaultdict(list)
    for r in ds:
        c = SUB2CAT.get(r["subset"])
        if c:
            by_cat[c].append(r)
    rng = random.Random(SEED)
    out = []
    for cat, rows in by_cat.items():
        rng.shuffle(rows)
        for r in rows[:PER_CAT]:
            a_is_chosen = rng.random() < 0.5   # randomize A/B
            resp_a = r["chosen"] if a_is_chosen else r["rejected"]
            resp_b = r["rejected"] if a_is_chosen else r["chosen"]
            out.append({
                "id": f"{r['subset']}/{r['id']}",
                "subset": r["subset"], "category": cat,
                "prompt": r["prompt"],
                "response_A": resp_a, "response_B": resp_b,
                "chosen_side": "A" if a_is_chosen else "B",   # secondary label
                "chosen_model": r["chosen_model"], "rejected_model": r["rejected_model"],
            })
    rng.shuffle(out)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": SEED, "n": len(out), "pairs": out}, open(OUT, "w"))
    from collections import Counter
    print(f"wrote {OUT}: {len(out)} pairs | by category: {dict(Counter(p['category'] for p in out))}")
    print(f"A/B balance (chosen on A): {sum(1 for p in out if p['chosen_side']=='A')}/{len(out)}")


if __name__ == "__main__":
    main()
