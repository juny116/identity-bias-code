"""Subjective (no-gold) interface-confound analysis.

Primary (label-free): does the judge's preferred response change across the 5
pairwise interfaces for the same pair? (interface preference-flip rate).
Secondary: agreement with RewardBench 'chosen' (curated preference), per
interface; interface-induced change in that agreement.
Position-bias control: for choice_only and feedback_then_choice, compare fixed
vs order-swapped runs -> is interface instability explained by plain A/B position
bias, or does it persist beyond it?
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
DIR = REPO / "logs" / "subjective_confound"
IFACES = ["choice_only", "rubric_then_choice", "score_then_choice",
          "choice_then_feedback", "feedback_then_choice"]
SWAP_IFACES = ["choice_only", "feedback_then_choice"]


def load():
    rows = []
    for p in DIR.glob("*.jsonl"):
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        print("no subjective records yet"); return
    df = df[df["parse_ok"]].copy()
    fixed = df[~df["swap"]]
    print(f"records={len(df)} judges={sorted(df.judge.unique())}")

    # 1) interface preference-flip rate (fixed order, 5 interfaces)
    print("\n[1] interface preference-FLIP rate (same judge+pair, picked_orig across 5 interfaces)")
    for judge in sorted(fixed.judge.unique()):
        g = fixed[(fixed.judge == judge) & (fixed.interface.isin(IFACES))]
        byp = defaultdict(dict)
        for _, r in g.iterrows():
            byp[r["id"]][r["interface"]] = r["picked_orig"]
        multi = [d for d in byp.values() if len(d) >= 2]
        flip = sum(1 for d in multi if len(set(d.values())) > 1) / len(multi) * 100 if multi else float("nan")
        print(f"  {judge:18}: {flip:4.1f}%  ({len(multi)} pairs)")

    # 2) agreement-with-chosen per interface (secondary)
    print("\n[2] agreement with curated 'chosen' per interface (secondary, NOT accuracy)")
    piv = (fixed[fixed.interface.isin(IFACES)]
           .groupby(["judge", "interface"])["picked_chosen"].mean().mul(100).round(1)
           .unstack().reindex(columns=IFACES))
    print(piv.to_string())
    rng = (piv.max(axis=1) - piv.min(axis=1)).round(1)
    print("  interface range of agreement per judge:")
    print(rng.to_string())

    # 3) position-bias control (no-swap vs swap on the 2 control interfaces)
    print("\n[3] position-bias control (fixed vs order-swapped; same picked_orig => content-driven)")
    for judge in sorted(df.judge.unique()):
        for iface in SWAP_IFACES:
            a = df[(df.judge == judge) & (df.interface == iface) & (~df.swap)].set_index("id")["picked_orig"]
            b = df[(df.judge == judge) & (df.interface == iface) & (df.swap)].set_index("id")["picked_orig"]
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            consistent = (a.loc[common] == b.loc[common]).mean() * 100
            print(f"  {judge:18} {iface:22}: order-consistency={consistent:4.1f}%  "
                  f"(position-bias-driven flips={100-consistent:4.1f}%)  n={len(common)}")
    print("\nInterpretation: if [1] interface-flip >> [3] position-bias flips, the "
          "output interface confounds subjective judging beyond plain A/B position bias.")


if __name__ == "__main__":
    main()
