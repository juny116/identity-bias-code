"""Reference-provided control vs no-think, on the matched subset
(interfaces {verdict_only, rubric_verdict, repair_then_verdict} x {aime, gpqa}).

Question (codex): when the gold final answer is PROVIDED to the judge, does the
interface confound collapse? Reports per condition x judge x dataset:
  - accuracy vs (corrected) label, balanced accuracy, FAR, FRR
  - interface RANGE of balanced accuracy (does it shrink under reference?)
  - verdict-flip rate: % of candidates whose verdict changes across the 3
    interfaces (LABEL-FREE; the core "does the contract still perturb" metric)
  - rank stability across interfaces
Interpretation: ceiling + flat + no flips -> confound concentrated in
reference-FREE verification; residual flips -> contract perturbs even when the
answer is given.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
REF = REPO / "logs" / "interface_confound_reference"
NOTHINK = REPO / "logs" / "interface_confound_pilot"
IFACES = ["verdict_only", "rubric_verdict", "repair_then_verdict"]
DATASETS = ["aime", "gpqa"]


def load(dir_):
    rows = []
    for p in dir_.glob("*.jsonl"):
        for line in open(p):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    df = pd.DataFrame(rows)
    try:
        from analysis.label_fix import apply as _alf
        df, _, _ = _alf(df)
    except Exception:
        try:
            from label_fix import apply as _alf
            df, _, _ = _alf(df)
        except Exception:
            pass
    return df[df["interface"].isin(IFACES) & df["dataset"].isin(DATASETS)
              & df["parse_ok"]].copy()


def metrics(g):
    w = g[~g["actually_correct"].astype(bool)]
    c = g[g["actually_correct"].astype(bool)]
    far = w["is_correct"].mean() if len(w) else np.nan
    frr = (1 - c["is_correct"].mean()) if len(c) else np.nan
    acc = (g["is_correct"].astype(bool) == g["actually_correct"].astype(bool)).mean()
    return 100 * np.nanmean([1 - far, 1 - frr]), 100 * acc


def flip_rate(g):
    """% of candidates whose verdict is not constant across the 3 interfaces."""
    cand = defaultdict(dict)
    for _, r in g.iterrows():
        cand[(r["problem_id"], r["solver"])][r["interface"]] = bool(r["is_correct"])
    multi = [d for d in cand.values() if len(d) >= 2]
    if not multi:
        return np.nan
    return 100 * sum(1 for d in multi if len(set(d.values())) > 1) / len(multi)


def summarize(df, label):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    for ds in DATASETS:
        for judge in sorted(df["judge"].unique()):
            g = df[(df["dataset"] == ds) & (df["judge"] == judge)]
            if g.empty:
                continue
            baccs = {}
            for i in IFACES:
                gi = g[g["interface"] == i]
                if len(gi):
                    baccs[i] = metrics(gi)[0]
            if len(baccs) < 2:
                continue
            rng = max(baccs.values()) - min(baccs.values())
            fr = flip_rate(g)
            order = sorted(baccs, key=baccs.get, reverse=True)
            print(f"  {ds:5} {judge:18} bAcc[{'/'.join(f'{baccs[i]:.0f}' for i in IFACES if i in baccs)}]"
                  f"  range={rng:4.1f}pp  flip={fr:4.1f}%  best={order[0].split('_')[0]}")


def main():
    nt = load(NOTHINK)
    rf = load(REF)
    summarize(nt, "NO-THINK (reference-free, matched subset)")
    summarize(rf, "REFERENCE-PROVIDED (gold answer injected)")
    print(f"\n{'='*60}\nHEADLINE\n{'='*60}")
    for cond, df in [("no-think(ref-free)", nt), ("reference-provided", rf)]:
        ranges, flips = [], []
        for ds in DATASETS:
            for judge in df["judge"].unique():
                g = df[(df["dataset"] == ds) & (df["judge"] == judge)]
                baccs = [metrics(g[g["interface"] == i])[0]
                         for i in IFACES if len(g[g["interface"] == i])]
                if len(baccs) >= 2:
                    ranges.append(max(baccs) - min(baccs))
                fr = flip_rate(g)
                if fr == fr:
                    flips.append(fr)
        print(f"  {cond:20}: mean interface range = {np.mean(ranges):4.1f}pp | "
              f"mean verdict-flip = {np.mean(flips):4.1f}%")


if __name__ == "__main__":
    main()
