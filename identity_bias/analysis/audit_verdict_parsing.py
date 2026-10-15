"""Audit parse failures and token-limit confounds in judge responses."""

import json
from collections import defaultdict

import pandas as pd

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
PILOT = REPO / "logs" / "interface_confound_pilot"
TABLES = REPO / "paper" / "tables"
OVERFLOW_REASON = "candidate_solution_exceeds_32768_context_before_judging"


def category(rec):
    if rec.get("parse_ok"):
        return "parse_ok"
    fr = rec.get("finish_reason")
    if fr == "context_overflow":
        return "context_overflow"
    if fr == "length":
        return "length"
    if fr == "error":
        return "error"
    return "other"


def main():
    rows = []
    overflow = defaultdict(set)  # dataset -> {problem_id}
    for path in sorted(PILOT.glob("*.jsonl")):
        recs = [json.loads(l) for l in open(path)]
        if not recs:
            continue
        ds = recs[0]["dataset"]
        judge = recs[0]["judge"]
        iface = recs[0]["interface"]
        counts = defaultdict(int)
        for r in recs:
            c = category(r)
            counts[c] += 1
            if c == "context_overflow":
                overflow[ds].add(r["problem_id"])
        n = len(recs)
        analyzable = n - counts["context_overflow"]  # uniform-exclusion denominator
        rows.append(
            {
                "dataset": ds,
                "judge": judge,
                "interface": iface,
                "n": n,
                "parse_ok": counts["parse_ok"],
                "ctx_overflow": counts["context_overflow"],
                "length_trunc": counts["length"],
                "error": counts["error"],
                "other": counts["other"],
                "analyzable_n": analyzable,
                "valid_rate_on_analyzable": round(
                    100 * counts["parse_ok"] / analyzable, 1
                )
                if analyzable
                else None,
                "trunc_rate_on_analyzable": round(
                    100 * counts["length"] / analyzable, 1
                )
                if analyzable
                else None,
            }
        )
    df = pd.DataFrame(rows).sort_values(["dataset", "judge", "interface"])
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "ifc_parse_audit.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    # overflow manifest + uniformity check across judge/interface
    man = []
    for ds, pids in sorted(overflow.items()):
        for pid in sorted(pids):
            man.append({"dataset": ds, "problem_id": pid, "reason": OVERFLOW_REASON})
    mdf = pd.DataFrame(man)
    mdf.to_csv(TABLES / "ifc_overflow_manifest.csv", index=False)
    print("\n=== context-overflow manifest (uniform exclusion) ===")
    for ds, pids in sorted(overflow.items()):
        print(f"  {ds}: {len(pids)} excluded problem_ids")

    # uniformity: every (judge,interface) cell within a dataset should exclude the
    # SAME overflow problem_ids (content-driven, judge-independent)
    print("\n=== overflow uniformity check (same excluded ids across cells?) ===")
    by_cell = defaultdict(set)
    for path in sorted(PILOT.glob("*.jsonl")):
        recs = [json.loads(l) for l in open(path)]
        if not recs:
            continue
        ds = recs[0]["dataset"]
        cell = (recs[0]["judge"], recs[0]["interface"])
        for r in recs:
            if r.get("finish_reason") == "context_overflow":
                by_cell[(ds, cell)].add(r["problem_id"])
    for ds in sorted(overflow):
        cells = {k: v for k, v in by_cell.items() if k[0] == ds}
        sets = list(cells.values())
        uniform = all(s == sets[0] for s in sets) if sets else True
        print(
            f"  {ds}: {'UNIFORM' if uniform else 'NOT uniform'} across {len(sets)} cells"
        )


if __name__ == "__main__":
    main()
