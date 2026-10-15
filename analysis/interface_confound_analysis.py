"""Analysis for the judge-interface-confound pilot.

Loads logs/interface_confound_pilot/*.jsonl (one file per judge x dataset x
interface) and tests the core thesis: do benign output-interface choices
interact differently across judge models, changing model-comparison conclusions?

Outputs:
  1. Per (judge, interface): FAR, FRR, balanced accuracy, parse-fail rate.
  2. Per-interface judge ranking (by balanced accuracy) + Kendall tau across
     interfaces (ranking stability).
  3. Pairwise judge balanced-accuracy gaps per interface with paired bootstrap
     CIs (clustered by problem_id) — does any adjacent pair flip or change gap?
  4. GEE logistic: correct_verdict ~ judge * interface, clustered by problem_id
     — tests judge x interface interaction directly.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPO = Path(__file__).parent.parent
PILOT_DIR = REPO / "logs" / "interface_confound_pilot"
PAPER_TABLES = REPO / "paper" / "tables"


def load_records(pilot_dir=PILOT_DIR):
    rows = []
    for path in sorted(pilot_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    # apply robust MCQ label corrections (GPQA/MMLU LaTeX-answer mislabels)
    try:
        try:
            from analysis.label_fix import apply as _apply_label_fix
        except ImportError:
            from label_fix import apply as _apply_label_fix
        df, _n, _flip = _apply_label_fix(df)
        print(f"[label_fix] overrode {_n} MCQ labels ({_flip} flipped from stored)")
    except Exception as _e:
        print(f"[label_fix] skipped: {_e}")
    return df


def per_judge_interface(df, by=("judge", "interface")):
    """FAR, FRR, balanced accuracy, parse-fail rate per group (default
    judge x interface; pass by=('dataset','judge','interface') for per-dataset)."""
    by = list(by)
    rows = []
    for keys, g in df.groupby(by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keymap = dict(zip(by, keys))
        judge = keymap.get("judge")
        iface = keymap.get("interface")
        n = len(g)
        n_parse_fail = int((~g["parse_ok"]).sum())
        valid = g[g["parse_ok"]]
        wrong = valid[~valid["actually_correct"]]
        correct = valid[valid["actually_correct"]]
        far = wrong["is_correct"].mean() if len(wrong) else np.nan   # accept wrong
        frr = (1 - correct["is_correct"].mean()) if len(correct) else np.nan  # reject correct
        # balanced accuracy of the verdict as a wrongness detector
        tpr = 1 - far  # correctly reject wrong
        tnr = 1 - frr  # correctly accept correct
        bacc = np.nanmean([tpr, tnr])
        rows.append({
            **keymap, "n": n,
            "n_parse_fail": n_parse_fail,
            "parse_fail_pct": round(100 * n_parse_fail / n, 1) if n else np.nan,
            "n_wrong": len(wrong), "n_correct": len(correct),
            "FAR_pct": round(100 * far, 1) if not np.isnan(far) else np.nan,
            "FRR_pct": round(100 * frr, 1) if not np.isnan(frr) else np.nan,
            "balanced_acc_pct": round(100 * bacc, 1) if not np.isnan(bacc) else np.nan,
        })
    return pd.DataFrame(rows)


def rankings_and_tau(pji):
    """Per-interface judge ranking by balanced accuracy + Kendall tau matrix
    across interfaces."""
    pivot = pji.pivot(index="judge", columns="interface", values="balanced_acc_pct")
    interfaces = list(pivot.columns)
    # ranking per interface (higher balanced acc = better judge = rank 1)
    rankings = {}
    for iface in interfaces:
        order = pivot[iface].sort_values(ascending=False)
        rankings[iface] = list(order.index)
    # Kendall tau between every interface pair (on the judge ordering)
    tau_rows = []
    judges = list(pivot.index)
    for a, b in combinations(interfaces, 2):
        # rank vectors aligned by judge
        ra = pivot[a].rank(ascending=False)
        rb = pivot[b].rank(ascending=False)
        mask = ra.notna() & rb.notna()
        if mask.sum() < 3:
            continue
        tau, p = kendalltau(ra[mask], rb[mask])
        tau_rows.append({"interface_a": a, "interface_b": b,
                         "kendall_tau": round(tau, 3), "p": round(p, 3)})
    return pivot, rankings, pd.DataFrame(tau_rows)


def correct_verdict_frame(df):
    """Long frame with per-item correct_verdict (1 if verdict matches truth)."""
    valid = df[df["parse_ok"]].copy()
    valid["correct_verdict"] = (
        valid["is_correct"].astype(bool) == valid["actually_correct"].astype(bool)
    ).astype(int)
    return valid


def _cluster_stats(sub):
    """Per problem_id, accumulate (wrong_accepts, wrong_total, correct_rejects,
    correct_total) so a balanced-acc can be recomputed by summing clusters.
    A problem_id may carry multiple candidate rows (e.g. a wrong-candidate and a
    correct-candidate from different solvers); all rows in the cluster are kept.
    Returns dict: pid -> np.array([wa, wt, cr, ct])."""
    out = {}
    for pid, g in sub.groupby("problem_id"):
        wrong = g[~g["actually_correct"].astype(bool)]
        correct = g[g["actually_correct"].astype(bool)]
        wa = int(wrong["is_correct"].astype(bool).sum())          # accept-wrong (FAR num)
        wt = int(len(wrong))
        cr = int((~correct["is_correct"].astype(bool)).sum())     # reject-correct (FRR num)
        ct = int(len(correct))
        out[pid] = np.array([wa, wt, cr, ct], dtype=float)
    return out


def _bacc_from_counts(arr):
    """arr = [wrong_accepts, wrong_total, correct_rejects, correct_total]."""
    wa, wt, cr, ct = arr
    tpr = 1 - (wa / wt) if wt else np.nan   # correctly reject wrong
    tnr = 1 - (cr / ct) if ct else np.nan   # correctly accept correct
    return np.nanmean([tpr, tnr])


def pairwise_gaps(df, n_boot=2000, seed=42):
    """Per interface, pairwise judge balanced-accuracy gap with cluster bootstrap
    CI by problem_id (codex spec): the resampled unit is the problem_id cluster;
    ALL candidate rows in a sampled cluster are kept (a problem can appear as both
    a wrong and a correct candidate). Duplicate problem_ids after resampling are
    expected — we sum precomputed per-cluster counts rather than reindex."""
    valid = correct_verdict_frame(df)
    rows = []
    for iface, g in valid.groupby("interface"):
        judges = sorted(g["judge"].unique())
        for a, b in combinations(judges, 2):
            ga = g[g["judge"] == a]
            gb = g[g["judge"] == b]
            ca = _cluster_stats(ga)
            cb = _cluster_stats(gb)
            # cluster bootstrap over problem_ids present in BOTH judges (paired)
            pids = sorted(set(ca) & set(cb))
            if not pids:
                continue
            obs = (_bacc_from_counts(sum(ca[p] for p in pids))
                   - _bacc_from_counts(sum(cb[p] for p in pids))) * 100
            mat_a = np.array([ca[p] for p in pids])  # (n_pid, 4)
            mat_b = np.array([cb[p] for p in pids])
            rng = np.random.default_rng(seed)
            idx = np.arange(len(pids))
            boots = []
            for _ in range(n_boot):
                samp = rng.choice(idx, len(idx), replace=True)
                ba = _bacc_from_counts(mat_a[samp].sum(axis=0))
                bb = _bacc_from_counts(mat_b[samp].sum(axis=0))
                boots.append((ba - bb) * 100)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({
                "interface": iface, "judge_a": a, "judge_b": b,
                "n_pairs": len(pids),
                "bacc_gap_pp": round(obs, 1),
                "ci_lo": round(lo, 1), "ci_hi": round(hi, 1),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })
    return pd.DataFrame(rows)


def gee_interaction(df):
    """GEE logistic: correct_verdict ~ C(judge) * C(interface), clustered by
    problem_id. Returns the fit summary text and whether interaction terms are
    jointly notable (by |coef| and p)."""
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception as e:
        return f"[statsmodels unavailable: {e}]", None
    valid = correct_verdict_frame(df)
    if valid["judge"].nunique() < 2 or valid["interface"].nunique() < 2:
        return "[need >=2 judges and >=2 interfaces]", None
    valid = valid.copy()
    valid["problem_id"] = valid["problem_id"].astype("category")
    try:
        model = smf.gee(
            "correct_verdict ~ C(judge) * C(interface)",
            groups="problem_id", data=valid,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        )
        res = model.fit()
        # collect interaction terms
        inter = res.params.filter(like=":")
        inter_p = res.pvalues.filter(like=":")
        summary = res.summary().as_text()
        sig = [(name, round(inter[name], 3), round(inter_p[name], 4))
               for name in inter.index if inter_p[name] < 0.05]
        return summary, sig
    except Exception as e:
        return f"[GEE fit failed: {e}]", None


def gee_interaction_pooled(df):
    """Pooled GEE logistic: correct_verdict ~ C(judge)*C(interface) + C(dataset),
    clustered by problem_id nested in dataset (composite group)."""
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception as e:
        return f"[statsmodels unavailable: {e}]", None
    valid = correct_verdict_frame(df).copy()
    if valid["dataset"].nunique() < 2:
        return "[pooled model needs >=2 datasets]", None
    valid["grp"] = (valid["dataset"].astype(str) + "::" +
                    valid["problem_id"].astype(str)).astype("category")
    try:
        model = smf.gee(
            "correct_verdict ~ C(judge) * C(interface) + C(dataset)",
            groups="grp", data=valid,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        )
        res = model.fit()
        inter_p = res.pvalues.filter(like=":")
        inter = res.params.filter(like=":")
        sig = [(n, round(inter[n], 3), round(inter_p[n], 4))
               for n in inter.index if inter_p[n] < 0.05]
        return res.summary().as_text(), sig
    except Exception as e:
        return f"[pooled GEE failed: {e}]", None


def gee_3way(df):
    """Three-way GEE: correct_verdict ~ C(judge)*C(interface)*C(dataset),
    clustered by dataset:problem_id. Returns (summary_text, joint_wald) where
    joint_wald is a robust Wald joint test on ALL three-way interaction terms
    (the judge x interface x dataset block) — the proper test of whether the
    judge x interface effect is heterogeneous across datasets (codex spec)."""
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm
    except Exception as e:
        return f"[statsmodels unavailable: {e}]", None
    valid = correct_verdict_frame(df).copy()
    if valid["dataset"].nunique() < 2:
        return "[need >=2 datasets]", None
    valid["grp"] = (valid["dataset"].astype(str) + "::" +
                    valid["problem_id"].astype(str)).astype("category")
    try:
        model = smf.gee(
            "correct_verdict ~ C(judge)*C(interface)*C(dataset)",
            groups="grp", data=valid,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Independence(),
        )
        res = model.fit()
        # 3-way terms have exactly two ':' in the name
        threeway = [n for n in res.params.index if n.count(":") == 2]
        if not threeway:
            return res.summary().as_text(), None
        import numpy as _np
        R = _np.zeros((len(threeway), len(res.params)))
        idx = {n: i for i, n in enumerate(res.params.index)}
        for r, n in enumerate(threeway):
            R[r, idx[n]] = 1.0
        wald = res.wald_test(R, scalar=True)
        return res.summary().as_text(), {
            "n_threeway_terms": len(threeway),
            "wald_stat": float(wald.statistic),
            "df": int(getattr(wald, "df_denom", len(threeway)) or len(threeway)),
            "p_value": float(wald.pvalue),
        }
    except Exception as e:
        return f"[3-way GEE failed: {e}]", None


def analyze_subset(df, label, tag):
    """Per judge x interface table, rankings + Kendall tau, pairwise bootstrap
    gaps for one slice of the data (a single dataset, or all pooled)."""
    print(f"\n{'='*70}\n### {label}  (n={len(df)})\n{'='*70}")
    pji = per_judge_interface(df)
    pji.to_csv(PAPER_TABLES / f"ifc_{tag}_per_judge_interface.csv", index=False)
    print(pji.to_string(index=False))
    if df["judge"].nunique() < 2:
        print("(need >=2 judges for rankings/gaps)")
        return
    pivot, rankings, tau = rankings_and_tau(pji)
    print("\nbalanced-acc pivot (judge x interface):")
    print(pivot.round(1).to_string())
    print("\nrankings per interface (best->worst):")
    for iface, order in rankings.items():
        print(f"  {iface}: {' > '.join(order)}")
    # interface range per judge (max-min bAcc across interfaces) = robustness
    rng = (pivot.max(axis=1) - pivot.min(axis=1)).round(1)
    print("\ninterface RANGE per judge (max-min bAcc, lower=more robust):")
    print(rng.to_string())
    print("\nKendall tau across interface rankings (min tau = most reordering):")
    print(tau.to_string(index=False))
    tau.to_csv(PAPER_TABLES / f"ifc_{tag}_kendall_tau.csv", index=False)
    gaps = pairwise_gaps(df)
    gaps.to_csv(PAPER_TABLES / f"ifc_{tag}_pairwise_gaps.csv", index=False)
    print("\npairwise judge bAcc gaps per interface (cluster-bootstrap CI):")
    print(gaps.to_string(index=False))


def main():
    df = load_records()
    if df.empty:
        print("no records yet in", PILOT_DIR)
        return
    datasets = sorted(df["dataset"].unique())
    print(f"loaded {len(df)} records; judges={sorted(df['judge'].unique())}; "
          f"interfaces={sorted(df['interface'].unique())}; datasets={datasets}")

    for ds in datasets:
        analyze_subset(df[df["dataset"] == ds], f"DATASET = {ds}", ds)

    if len(datasets) > 1:
        analyze_subset(df, "POOLED (all datasets)", "pooled")
        print(f"\n{'='*70}\n### Pooled GEE: judge x interface + dataset (cluster=problem_id in dataset)\n{'='*70}")
        summary, sig = gee_interaction_pooled(df)
        if sig is not None:
            print(f"significant judge x interface interaction terms (p<0.05): {len(sig)}")
            for name, coef, p in sig:
                print(f"  {name}: coef={coef}, p={p}")
        print(summary[:1200])

        print(f"\n{'='*70}\n### 3-way GEE: judge x interface x dataset — joint Wald on the 3-way block\n{'='*70}")
        _, jw = gee_3way(df)
        if jw:
            print(f"3-way interaction block: {jw['n_threeway_terms']} terms, "
                  f"joint Wald chi2={jw['wald_stat']:.1f}, p={jw['p_value']:.4g}")
            print("-> if p<0.05: judge x interface effect is significantly "
                  "HETEROGENEOUS across datasets (the confound reverses by task family).")
        else:
            print("(3-way joint Wald unavailable)")

    print("\ndone")


if __name__ == "__main__":
    main()
