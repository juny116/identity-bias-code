"""Compare no-think (interface_confound_pilot) vs deliberation-in-schema
(interface_confound_think) on the SAME subset: interfaces
{verdict_only, rubric_verdict, repair_then_verdict} x datasets {aime, gpqa}.

Question (codex framing): does an explicit reasoning field reduce the
judge x interface instability (smaller interface RANGE per judge, more stable
judge rankings) under fixed judged content?

Reports, per condition x dataset:
  - balanced-acc pivot (judge x interface)
  - interface RANGE per judge (max-min bAcc over the 3 interfaces; lower=robust)
  - judge ranking per interface + number of distinct rankings (rank instability)
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
PILOT = REPO / "logs" / "interface_confound_pilot"
THINK = REPO / "logs" / "interface_confound_think"
import os
# Full think matrix now available (3 judges x 4 datasets x 5 interfaces). Default
# to the full comparison; set IFC_SUBSET=1 for the original matched 3-iface x
# {aime,gpqa} subset that codex's first deliberation tables used.
if os.environ.get("IFC_SUBSET"):
    IFACES = ["verdict_only", "rubric_verdict", "repair_then_verdict"]
    DATASETS = ["aime", "gpqa"]
else:
    IFACES = ["verdict_only", "rubric_verdict", "score_then_verdict",
              "verdict_plus_repair", "repair_then_verdict"]
    DATASETS = ["aime", "gpqa", "olympiad", "mmlu_pro"]


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
        try:
            from analysis.label_fix import apply as _alf
        except ImportError:
            from label_fix import apply as _alf
        df, _, _ = _alf(df)
    except Exception:
        pass
    return df[df["interface"].isin(IFACES) & df["dataset"].isin(DATASETS)].copy()


def bacc_pivot(df):
    rows = []
    for (j, i), g in df.groupby(["judge", "interface"]):
        v = g[g["parse_ok"]]
        w = v[~v["actually_correct"].astype(bool)]
        c = v[v["actually_correct"].astype(bool)]
        far = w["is_correct"].mean() if len(w) else np.nan
        frr = (1 - c["is_correct"].mean()) if len(c) else np.nan
        rows.append({"judge": j, "interface": i,
                     "bacc": 100 * np.nanmean([1 - far, 1 - frr])})
    p = pd.DataFrame(rows).pivot(index="judge", columns="interface", values="bacc")
    return p.reindex(columns=[c for c in IFACES if c in p.columns])


def coverage_and_reasoning(sub):
    """valid-rate on analyzable (excl context_overflow) + mean reasoning-field
    chars + truncation count, per the subset."""
    analyzable = sub[sub["finish_reason"] != "context_overflow"]
    valid = analyzable["parse_ok"].mean() * 100 if len(analyzable) else float("nan")
    trunc = int((sub["finish_reason"] == "length").sum())
    rlens = []
    for _, r in sub[sub["parse_ok"]].iterrows():
        try:
            rlens.append(len(json.loads(r["raw"]).get("reasoning", "")))
        except Exception:
            pass
    import numpy as _np
    rmean = _np.mean(rlens) if rlens else float("nan")
    return valid, trunc, rmean


def summarize(df, condition):
    print(f"\n{'#'*64}\n# {condition}\n{'#'*64}")
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            print(f"\n[{ds}] no data"); continue
        vr, tr, rl = coverage_and_reasoning(sub)
        print(f"\n[{ds}] valid-rate(analyzable)={vr:.1f}%  truncations={tr}"
              + (f"  mean reasoning chars={rl:.0f}" if rl == rl else ""))
        piv = bacc_pivot(sub)
        rng = (piv.max(axis=1) - piv.min(axis=1)).round(1)
        # rankings per interface
        rankings = {i: list(piv[i].sort_values(ascending=False).index)
                    for i in piv.columns}
        n_distinct = len({tuple(r) for r in rankings.values()})
        print(f"\n[{ds}] balanced-acc (judge x interface):")
        print(piv.round(1).to_string())
        print(f"  interface RANGE per judge (lower=more robust):")
        for j in piv.index:
            print(f"    {j:18} {rng[j]:.1f}pp")
        print(f"  mean interface range = {rng.mean():.1f}pp; "
              f"distinct judge-rankings across the 3 interfaces = {n_distinct} "
              f"({'rank-stable' if n_distinct==1 else 'rank UNSTABLE'})")
        for i, r in rankings.items():
            print(f"    {i:22}: {' > '.join(r)}")


def main():
    nt = load(PILOT)
    th = load(THINK)
    print(f"no-think rows={len(nt)}; think rows={len(th)} "
          f"(subset: {IFACES} x {DATASETS})")
    summarize(nt, "NO-THINK (no reasoning field)")
    summarize(th, "DELIBERATION-IN-SCHEMA (leading reasoning field)")
    # headline: mean interface range per condition
    print(f"\n{'='*64}\nHEADLINE: mean interface range (lower = more robust to interface)\n{'='*64}")
    for cond, df in [("no-think", nt), ("deliberation", th)]:
        ranges = []
        for ds in DATASETS:
            sub = df[df["dataset"] == ds]
            if sub.empty:
                continue
            piv = bacc_pivot(sub)
            ranges.extend((piv.max(axis=1) - piv.min(axis=1)).tolist())
        if ranges:
            print(f"  {cond:14}: mean range across judge x dataset = "
                  f"{np.nanmean(ranges):.1f}pp (n={len(ranges)})")


if __name__ == "__main__":
    main()
