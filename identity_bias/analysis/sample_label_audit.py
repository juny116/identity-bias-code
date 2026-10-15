"""Sample candidate solutions for manual correctness-label auditing."""

import argparse
import json
import re

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
SAMPLES = REPO / "logs" / "interface_confound_samples"
OUT = REPO / "paper" / "tables" / "label_audit_sheet.csv"
DATASETS = ["aime", "gpqa", "olympiad", "mmlu_pro"]


def edge_flags(dataset, cot, final):
    flags = []
    boxed = re.findall(r"\\boxed\{", cot or "")
    if len(boxed) == 0:
        flags.append("no_boxed")
    elif len(boxed) > 1:
        flags.append("multi_boxed")
    f = (final or "").strip()
    if dataset == "aime":
        m = re.fullmatch(r"\d{1,3}", f)
        if not m or not (0 <= int(f) <= 999 if f.isdigit() else False):
            flags.append("aime_not_0_999")
    if dataset in ("gpqa", "mmlu_pro"):
        opts = "A-D" if dataset == "gpqa" else "A-J"
        if not re.fullmatch(rf"[{opts}]", f.upper().strip("()")):
            flags.append("mcq_not_clean_letter")
    if f == "":
        flags.append("empty_final")
    return "|".join(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-dataset",
        type=int,
        default=15,
        help="target rows per dataset (split across correct/wrong)",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    import random

    rng = random.Random(args.seed)

    rows = []
    for ds in DATASETS:
        f = list(SAMPLES.glob(f"{ds}_w*_c*_seed42.json"))
        if not f:
            print(f"{ds}: no pool cache")
            continue
        pool = json.load(open(f[0]))["pool"]
        for c in pool:
            c["_edge"] = edge_flags(
                ds, c.get("chain_of_thought"), c.get("final_answer")
            )
        wrong = [c for c in pool if not c["is_correct"]]
        correct = [c for c in pool if c["is_correct"]]

        # oversample edge cases: put them first
        def pick(lst, k):
            edges = [c for c in lst if c["_edge"]]
            rest = [c for c in lst if not c["_edge"]]
            rng.shuffle(edges)
            rng.shuffle(rest)
            return (edges + rest)[:k]

        half = args.per_dataset // 2
        chosen = pick(wrong, args.per_dataset - half) + pick(correct, half)
        for c in chosen:
            rows.append(
                {
                    "dataset": ds,
                    "problem_id": c["problem_id"],
                    "solver": c["solver"],
                    "ground_truth": str(c.get("ground_truth"))[:60],
                    "extracted_final_answer": str(c.get("final_answer"))[:60],
                    "auto_label_is_correct": c["is_correct"],
                    "edge_flags": c["_edge"],
                    "question_head": " ".join(str(c.get("question", "")).split())[:140],
                    "cot_tail": " ".join(str(c.get("chain_of_thought", "")).split())[
                        -200:
                    ],
                    "human_verdict": "",  # ok | label_wrong | ambiguous | cannot_verify
                }
            )
    import csv

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    nedge = sum(1 for r in rows if r["edge_flags"])
    print(
        f"wrote {OUT} : {len(rows)} candidates "
        f"({sum(1 for r in rows if r['auto_label_is_correct'])} correct / "
        f"{sum(1 for r in rows if not r['auto_label_is_correct'])} wrong; "
        f"{nedge} with extraction edge flags)"
    )
    # per-dataset edge summary on the FULL pool (not just sample)
    print("\nfull-pool extraction edge-flag rates (potential label-risk):")
    for ds in DATASETS:
        f = list(SAMPLES.glob(f"{ds}_w*_c*_seed42.json"))
        if not f:
            continue
        pool = json.load(open(f[0]))["pool"]
        flagged = sum(
            1
            for c in pool
            if edge_flags(ds, c.get("chain_of_thought"), c.get("final_answer"))
        )
        print(f"  {ds:9}: {flagged}/{len(pool)} candidates have extraction edge flags")


if __name__ == "__main__":
    main()
