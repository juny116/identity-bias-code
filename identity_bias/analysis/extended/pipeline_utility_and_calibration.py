"""Pipeline utility and calibration."""

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
PAPER_TABLES = REPO / "paper" / "tables"

from identity_bias.analysis.extended.build_critique_tables import load_critique_records

# =====================================================================
# Q10: end-to-end pipeline utility
# =====================================================================


def end_to_end_utility(df_gated_table):
    """For each (critic, condition) compute P(final correct) under one-shot
    and gated pipelines, parameterized over the base correctness rate p of
    incoming solutions.

    P(final correct | wrong input)   = wrong_repaired_correct_rate
    P(final correct | correct input) = 1 - harmful_correction_rate
    P(final correct, overall p)      = p*(1 - hcr) + (1-p)*wrcr
    Net improvement                  = P(final correct) - p
    """
    df = df_gated_table.copy()
    # aggregate datasets within (critic, condition, pipeline) via item-weighted mean
    agg = (
        df.groupby(["critic", "condition", "pipeline"])
        .apply(
            lambda g: pd.Series(
                {
                    "wrcr": np.average(
                        g["wrong_repaired_correct_rate"].fillna(0.0),
                        weights=g["n_wrong"].clip(lower=1),
                    ),
                    "hcr": np.average(
                        g["harmful_correction_rate"].fillna(0.0),
                        weights=g["n_correct"].clip(lower=1),
                    ),
                    "far": np.average(
                        g["wrong_accepted_rate"].fillna(0.0),
                        weights=g["n_wrong"].clip(lower=1),
                    ),
                    "n_wrong": g["n_wrong"].sum(),
                    "n_correct": g["n_correct"].sum(),
                }
            )
        )
        .reset_index()
    )
    # evaluate at p ∈ {0.3, 0.5, 0.7}
    rows = []
    for _, r in agg.iterrows():
        for p in (0.3, 0.5, 0.7):
            p_final = p * (1 - r["hcr"]) + (1 - p) * r["wrcr"]
            rows.append(
                {
                    "critic": r["critic"],
                    "condition": r["condition"],
                    "pipeline": r["pipeline"],
                    "wrcr": r["wrcr"],
                    "hcr": r["hcr"],
                    "far": r["far"],
                    "p_base": p,
                    "p_final_correct": p_final,
                    "net_improvement_pp": (p_final - p) * 100,
                    "n_wrong": int(r["n_wrong"]),
                    "n_correct": int(r["n_correct"]),
                }
            )
    return pd.DataFrame(rows)


# =====================================================================
# Q4: Holm-Bonferroni adjustment of key contrasts
# =====================================================================


def holm_bonferroni(pvals_dict):
    """pvals_dict: {label: raw_p}. Returns dict with raw + adjusted + reject."""
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = []
    running_max = 0.0
    for k, (label, p) in enumerate(items, start=1):
        a = (m - k + 1) * p
        a = max(running_max, min(a, 1.0))
        running_max = a
        adj.append((label, p, a, a < 0.05))
    return pd.DataFrame(adj, columns=["contrast", "p_raw", "p_holm", "reject_at_0.05"])


def caveat_a_pvalues(df_caveat):
    """Bootstrap a p-value for each per-critic Caveat A contrast using the
    same paired bootstrap pivotal: 2 * min(P(diff > 0), P(diff < 0))."""

    out = {}
    contrasts = [
        ("verdict_plus_repair", "verdict_only_neutral", "repair_vs_neutral"),
        ("verdict_only_instructed", "verdict_only_neutral", "instr_vs_neutral"),
        ("verdict_schema_matched", "verdict_only_neutral", "schema_vs_neutral"),
    ]
    for critic in sorted(df_caveat["critic_model"].unique()):
        sub = df_caveat[
            (df_caveat["critic_model"] == critic)
            & (df_caveat["actually_correct"] == False)
            & (df_caveat["condition"] == "anonymous")
        ]
        for a, b, name in contrasts:
            slc = sub[sub["arm"].isin([a, b])]
            if slc["arm"].nunique() < 2:
                continue
            # pivot to wide
            piv = slc.pivot_table(
                index="problem_id", columns="arm", values="is_correct", aggfunc="first"
            ).dropna()
            if len(piv) < 5:
                continue
            diffs = (piv[a].astype(int) - piv[b].astype(int)).to_numpy()
            # paired bootstrap of mean diff
            rng = np.random.default_rng(seed=hash((critic, name)) % (2**32))
            B = 5000
            n = len(diffs)
            boots = []
            for _ in range(B):
                idx = rng.integers(0, n, n)
                boots.append(diffs[idx].mean())
            boots = np.asarray(boots)
            # two-sided p-value via bootstrap pivotal
            p = 2.0 * min((boots <= 0).mean(), (boots >= 0).mean())
            p = max(p, 1.0 / B)
            out[f"{critic}|{name}"] = float(p)
    return out


# =====================================================================
# Q1: per-arm parse-failure rates
# =====================================================================


def per_arm_parse_failures():
    """Raw loader that counts parse failures (records where is_correct is None
    in the JSONL)."""
    from identity_bias.analysis.extended.build_critique_tables import (
        LOGS,
        parse_critic_solver_dataset,
    )

    roots = (
        "caveat_a_run_1",
        "gemma_strong_run_1",
        "verdict_repair_gpt_oss_no_think_run_1",
        "verdict_repair_qwen3_no_think_run_1",
    )
    counts = defaultdict(lambda: [0, 0])  # (critic, arm) -> [n_total, n_fail]
    for root in roots:
        d = LOGS / root
        if not d.exists():
            continue
        for path in d.glob("identity_swap_*.jsonl"):
            parsed = parse_critic_solver_dataset(path.stem)
            if parsed is None:
                continue
            critic, _solver, _ds, arm = parsed
            arm = arm or "legacy_full"
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("type") != "critique":
                        continue
                    c = r["critique"]
                    counts[(critic, arm)][0] += 1
                    if c.get("is_correct") is None:
                        counts[(critic, arm)][1] += 1
    # also walk legacy identity-swap files
    seen = set()
    for path in LOGS.rglob("identity_swap_*.jsonl"):
        if any(str(path).startswith(str(LOGS / r)) for r in roots):
            continue
        parsed = parse_critic_solver_dataset(path.stem)
        if parsed is None:
            continue
        critic, _solver, ds, _ = parsed
        if ds == "bbh":
            continue
        arm = "legacy_full"
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") != "critique":
                    continue
                c = r["critique"]
                counts[(critic, arm)][0] += 1
                if c.get("is_correct") is None:
                    counts[(critic, arm)][1] += 1
    rows = []
    for (critic, arm), (n, nf) in sorted(counts.items()):
        rows.append(
            {
                "critic": critic,
                "arm": arm,
                "n_calls": n,
                "n_parse_fail": nf,
                "parse_fail_rate_pct": (nf / n * 100) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# =====================================================================
# Q3: cost / latency rough estimate
# =====================================================================

COST_PER_ITEM_CALLS = {
    "one_shot_verdict_plus_repair": 1,
    "gated_verdict_then_repair": 1.0,  # plus ~rejection_rate extra calls
    "n_of_k_majority_K5": 5,
}


def cost_table(df_gated_table):
    """Estimate calls/item and relative cost (call-equivalents) for each
    pipeline using observed rejection rate as the gating multiplier."""
    df = df_gated_table.copy()
    # mean rejection rate (1 - wrong_accepted_rate weighted by n_wrong)
    rej_rate = 1 - np.average(
        df["wrong_accepted_rate"].fillna(0.0),
        weights=df["n_wrong"].clip(lower=1),
    )
    # for correct items, also rejected at some rate (= false_rejected_rate)
    # ignore here; approximation: 50/50 wrong:correct, gating triggers on wrong rej
    extra_repair_share = rej_rate / 2  # approx half the inputs are wrong
    gated_calls = 1.0 + extra_repair_share
    rows = [
        {
            "pipeline": "One-shot verdict+repair",
            "calls_per_item": 1.0,
            "calls_relative_to_one_shot": 1.0,
        },
        {
            "pipeline": "Gated verdict-then-repair",
            "calls_per_item": round(gated_calls, 3),
            "calls_relative_to_one_shot": round(gated_calls, 3),
        },
        {
            "pipeline": "N-of-K=5 majority voting",
            "calls_per_item": 5.0,
            "calls_relative_to_one_shot": 5.0,
        },
    ]
    return pd.DataFrame(rows), float(rej_rate)


# =====================================================================
# Q7: per-interface temperature scaling
# =====================================================================


def neg_loglik(T, conf, correct):
    """NLL under temperature-scaled probabilities for a binary outcome."""
    # treat as binary verdict-correct vs incorrect; conf is the model's stated
    # confidence in its verdict, so binary p = sigmoid(logit(conf)/T).
    eps = 1e-6
    conf = np.clip(conf, eps, 1 - eps)
    logit = np.log(conf / (1 - conf))
    p = 1.0 / (1.0 + np.exp(-logit / T))
    p = np.clip(p, eps, 1 - eps)
    return -(correct * np.log(p) + (1 - correct) * np.log(1 - p)).mean()


def ece_10bin(conf, correct):
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    n = len(conf)
    for i in range(10):
        if i < 9:
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
        else:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        if not mask.any():
            continue
        p_emp = correct[mask].mean()
        p_pred = conf[mask].mean()
        w = mask.sum() / n
        ece += w * abs(p_emp - p_pred)
    return float(ece)


def temperature_scaling(df):
    """Fit T per (critic, arm) on confidence vs verdict-correctness."""
    sub = df[df["confidence"].notna()].copy()
    sub["verdict_correct"] = (sub["is_correct"] == sub["actually_correct"]).astype(int)
    sub["confidence"] = sub["confidence"].astype(float).clip(0, 1)
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        for arm in sorted(sub[sub["critic_model"] == critic]["arm"].unique()):
            slc = sub[(sub["critic_model"] == critic) & (sub["arm"] == arm)]
            if len(slc) < 100:
                continue
            conf = slc["confidence"].to_numpy()
            corr = slc["verdict_correct"].to_numpy().astype(float)
            # fit T ∈ [0.1, 10]
            res = minimize_scalar(
                lambda T: neg_loglik(T, conf, corr),
                bounds=(0.1, 10.0),
                method="bounded",
            )
            T_hat = float(res.x)
            # apply
            eps = 1e-6
            c_clipped = np.clip(conf, eps, 1 - eps)
            logit = np.log(c_clipped / (1 - c_clipped))
            conf_scaled = 1.0 / (1.0 + np.exp(-logit / T_hat))
            ece_before = ece_10bin(conf, corr)
            ece_after = ece_10bin(conf_scaled, corr)
            rows.append(
                {
                    "critic": critic,
                    "arm": arm,
                    "n": len(slc),
                    "T_hat": T_hat,
                    "ECE_before": ece_before,
                    "ECE_after": ece_after,
                    "ECE_delta_pp": (ece_after - ece_before) * 100,
                }
            )
    return pd.DataFrame(rows)


# =====================================================================
# main
# =====================================================================


def main():
    print("loading data...")
    df = load_critique_records()
    df_gated = pd.read_csv(PAPER_TABLES / "table5_gated_outcomes.csv")

    print("\n[Q10] end-to-end pipeline utility ...")
    util = end_to_end_utility(df_gated)
    util.to_csv(PAPER_TABLES / "revision_v3_end_to_end_utility.csv", index=False)
    # pretty wide print at p=0.5 only
    print(util[util["p_base"] == 0.5].to_string(index=False))

    print("\n[Q4] Holm-Bonferroni adjusted Caveat A p-values ...")
    pvals = caveat_a_pvalues(df)
    holm = holm_bonferroni(pvals)
    holm.to_csv(PAPER_TABLES / "revision_v3_holm_bonferroni.csv", index=False)
    print(holm.to_string(index=False))

    print("\n[Q1] per-arm parse-failure rates ...")
    paf = per_arm_parse_failures()
    paf.to_csv(PAPER_TABLES / "revision_v3_parse_fail_per_arm.csv", index=False)
    print(paf.to_string(index=False))

    print("\n[Q3] cost / latency rough table ...")
    cost, rej = cost_table(df_gated)
    cost.to_csv(PAPER_TABLES / "revision_v3_cost_per_item.csv", index=False)
    print(f"  observed mean rejection rate (wrong items): {rej:.3f}")
    print(cost.to_string(index=False))

    print("\n[Q7] per-interface temperature scaling ...")
    ts = temperature_scaling(df)
    ts.to_csv(PAPER_TABLES / "revision_v3_temperature_scaling.csv", index=False)
    print(ts.to_string(index=False))

    print("\ndone")


if __name__ == "__main__":
    main()
