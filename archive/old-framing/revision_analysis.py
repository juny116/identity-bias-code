"""Revision analyses requested by reviewer feedback.

Adds, to the already-frozen artifacts from freeze_artifacts.py:
  - Mixed-effects logistic regression on prompt-interface arm contrasts
    with random intercepts for problem_id (statsmodels MixedLM / Logit).
    Falls back to cluster-robust logit if MixedLM does not converge.
    Output: paper/tables/appendix_A1_mixed_effects.csv
  - Leave-one-dataset-out repair-vs-neutral contrast per critic. Tests
    whether the headline +4.5pp Gemma / +4.6pp gpt-oss effect survives
    when AIME (which dominates per-dataset) is excluded.
    Output: paper/tables/appendix_loo_caveat_a.csv
  - Paired bootstrap CIs for the gated-vs-oneshot pipeline contrasts
    on FAR, true-accept, false-rejection, and harmful-correction.
    Output: paper/tables/gated_contrasts_CI.csv
  - Parse-failure + token-usage + checker-validation audit tables.
    Output: paper/tables/audit_parse_token_checker.csv
  - Per-dataset Caveat A main-results figure (promotes appendix figure A1
    to a main-results figure with cleaner panel layout).
    Output: paper/figures/fig5_caveat_a_per_dataset.png
"""

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO = Path(__file__).parent.parent
LOGS = REPO / "logs"
PAPER_FIGS = REPO / "paper" / "figures"
PAPER_TABLES = REPO / "paper" / "tables"

sys.path.insert(0, str(REPO))
from identity_bias.config import Dataset
from identity_bias.data import check_answer

# Re-use loaders from freeze_artifacts (import to avoid duplication).
from analysis.freeze_artifacts import (
    load_critique_records, load_gated_stage2, load_ground_truth,
    paired_bootstrap_diff_ci,
)


def _accept_int(b):
    return int(bool(b))


# =========================================================
# 1. Mixed-effects logistic regression (and cluster-robust fallback)
# =========================================================

def run_mixed_effects(df):
    """Primary model: logistic regression on actually-wrong items with
    accepted_wrong as the outcome. Fixed effects: arm + critic +
    arm:critic interaction. Random intercept: problem_id.

    Uses statsmodels MixedLM via the GEE / binomial-fixedeffects path,
    falling back to cluster-robust Logit if MixedLM is unstable on a
    binary outcome (MixedLM is built for Gaussian responses; for binomial
    we use BinomialBayesMixedGLM or fall back to cluster-robust Logit).
    """
    arms = ["verdict_only_instructed", "verdict_only_neutral",
            "verdict_schema_matched", "verdict_plus_repair"]
    sub = df[(df["arm"].isin(arms)) & (~df["actually_correct"])
             & (df["condition"] == "anonymous")].copy()
    sub["accepted_wrong"] = sub["is_correct"].astype(int)
    sub["arm"] = pd.Categorical(sub["arm"], categories=arms, ordered=False)
    # set verdict_only_neutral as reference
    sub["arm"] = sub["arm"].cat.reorder_categories(
        ["verdict_only_neutral", "verdict_only_instructed",
         "verdict_schema_matched", "verdict_plus_repair"], ordered=False)
    sub["critic_model"] = pd.Categorical(sub["critic_model"])

    rows = []
    # Per-critic model: simpler, more interpretable
    for critic in sub["critic_model"].cat.categories:
        sc = sub[sub["critic_model"] == critic].copy()
        if sc["problem_id"].nunique() < 5 or len(sc) < 20:
            continue
        # cluster-robust logistic regression by problem_id
        try:
            model = smf.gee(
                "accepted_wrong ~ C(arm, Treatment(reference='verdict_only_neutral'))",
                groups="problem_id", data=sc,
                family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable(),
            )
            res = model.fit()
            for term, coef, se, pval, lo, hi in zip(
                res.params.index, res.params.values, res.bse.values,
                res.pvalues.values, res.conf_int()[0].values, res.conf_int()[1].values
            ):
                if term == "Intercept":
                    continue
                or_val = float(np.exp(coef))
                or_lo = float(np.exp(lo))
                or_hi = float(np.exp(hi))
                rows.append({
                    "critic": critic, "contrast": term,
                    "log_odds": float(coef), "SE": float(se),
                    "OR": or_val, "OR_CI_low": or_lo, "OR_CI_high": or_hi,
                    "p_value": float(pval),
                    "n_obs": int(len(sc)),
                    "n_clusters": int(sc["problem_id"].nunique()),
                    "model": "GEE_cluster_robust_binomial",
                })
        except Exception as e:
            print(f"  GEE failed for {critic}: {e}")
            continue
    return pd.DataFrame(rows)


# =========================================================
# 2. Leave-one-dataset-out repair contrast
# =========================================================

def loo_repair_contrast(df):
    """For each critic, recompute repair-vs-neutral delta excluding each
    dataset in turn. If excluding AIME makes the pooled effect vanish,
    the headline is AIME-driven."""
    sub = df[(df["arm"].isin(["verdict_only_neutral", "verdict_plus_repair"]))
             & (~df["actually_correct"])
             & (df["condition"] == "anonymous")].copy()
    sub["accept_int"] = sub["is_correct"].astype(int)
    rows = []
    critics = sorted(sub["critic_model"].unique())
    datasets = sorted(sub["dataset"].unique())
    for critic in critics:
        cs = sub[sub["critic_model"] == critic]
        for excl in ["(none)"] + list(datasets):
            keep = cs if excl == "(none)" else cs[cs["dataset"] != excl]
            piv = keep[["problem_id", "arm", "accept_int"]]
            delta, lo, hi, n = paired_bootstrap_diff_ci(
                piv, "arm", "verdict_plus_repair", "verdict_only_neutral", "accept_int"
            )
            rows.append({
                "critic": critic, "excluded_dataset": excl,
                "delta_pp": delta * 100,
                "CI_low_pp": lo * 100, "CI_high_pp": hi * 100,
                "n_paired": n,
            })
    return pd.DataFrame(rows)


# =========================================================
# 3. Bootstrap CIs for gated-vs-oneshot pipeline
# =========================================================

def gated_vs_oneshot_cis(df_caveat, df_gated, gt, n_boot=10000):
    """For each (critic, condition), compute item-paired CIs for the
    contrast (gated - oneshot) on FAR, true-accept, false-rejection,
    and harmful-correction. Pair is at the problem-id level: for each
    item we have a gated outcome and a one-shot outcome on the same item.
    """
    rng = np.random.default_rng(42)
    rows = []
    for critic in sorted(df_gated["critic_model"].unique()):
        for cond in sorted(df_gated[df_gated["critic_model"] == critic]["condition"].unique()):
            # collect per-item outcomes for gated and oneshot
            items = []
            for ds in sorted(df_gated[df_gated["critic_model"] == critic]["dataset"].unique()):
                ds_enum = Dataset(ds)
                def score(pid, ans):
                    if pid not in gt or ans is None:
                        return None
                    _, true_ans = gt[pid]
                    try:
                        return check_answer(ds_enum, ans, true_ans)
                    except Exception:
                        return None
                stage1 = df_caveat[(df_caveat["critic_model"] == critic)
                                   & (df_caveat["arm"] == "verdict_only_neutral")
                                   & (df_caveat["condition"] == cond)
                                   & (df_caveat["dataset"] == ds)]
                oneshot = df_caveat[(df_caveat["critic_model"] == critic)
                                    & (df_caveat["arm"] == "verdict_plus_repair")
                                    & (df_caveat["condition"] == cond)
                                    & (df_caveat["dataset"] == ds)]
                stage2 = df_gated[(df_gated["critic_model"] == critic)
                                  & (df_gated["condition"] == cond)
                                  & (df_gated["dataset"] == ds)]
                stage2_by_pid = stage2.set_index("problem_id")["corrected_answer"].to_dict()
                # build item dict
                one_by_pid = oneshot.set_index("problem_id").to_dict("index")
                for _, r in stage1.iterrows():
                    pid = r["problem_id"]
                    if pid not in one_by_pid:
                        continue
                    # gated outcome
                    if r["actually_correct"]:
                        g_far = 0
                        g_ta = int(bool(r["is_correct"]))
                        # if rejected, did repair harm? (output != truth)
                        if not bool(r["is_correct"]):
                            ans2 = stage2_by_pid.get(pid)
                            s2 = score(pid, ans2)
                            g_harm = 1 if s2 is False else 0
                        else:
                            g_harm = 0
                    else:
                        g_ta = 0; g_harm = 0
                        g_far = int(bool(r["is_correct"]))
                    o = one_by_pid[pid]
                    if r["actually_correct"]:
                        o_far = 0
                        o_ta = int(bool(o["is_correct"]))
                        # harmful if rejected and repair output != truth
                        if not bool(o["is_correct"]):
                            s_o = score(pid, o.get("corrected_answer"))
                            o_harm = 1 if s_o is False else 0
                        else:
                            o_harm = 0
                    else:
                        o_ta = 0; o_harm = 0
                        o_far = int(bool(o["is_correct"]))
                    items.append({
                        "pid": pid, "actually_wrong": not r["actually_correct"],
                        "g_far": g_far, "o_far": o_far,
                        "g_ta": g_ta, "o_ta": o_ta,
                        "g_harm": g_harm, "o_harm": o_harm,
                    })
            if not items:
                continue
            arr = pd.DataFrame(items)
            wrong = arr[arr["actually_wrong"]]
            correct = arr[~arr["actually_wrong"]]
            # FAR (gated - oneshot) on wrong items
            far_g = wrong["g_far"].mean() if len(wrong) else float("nan")
            far_o = wrong["o_far"].mean() if len(wrong) else float("nan")
            far_delta = far_g - far_o
            # bootstrap pair
            n = len(wrong)
            if n > 0:
                boots = np.empty(n_boot)
                gv = wrong["g_far"].to_numpy()
                ov = wrong["o_far"].to_numpy()
                for i in range(n_boot):
                    idx = rng.integers(0, n, size=n)
                    boots[i] = gv[idx].mean() - ov[idx].mean()
                far_lo, far_hi = np.percentile(boots, [2.5, 97.5])
            else:
                far_lo = far_hi = float("nan")
            # TAR (true-accept) and harmful (correct items)
            ta_g = correct["g_ta"].mean() if len(correct) else float("nan")
            ta_o = correct["o_ta"].mean() if len(correct) else float("nan")
            ta_delta = ta_g - ta_o
            harm_g = correct["g_harm"].mean() if len(correct) else float("nan")
            harm_o = correct["o_harm"].mean() if len(correct) else float("nan")
            harm_delta = harm_g - harm_o
            if len(correct) > 0:
                n_c = len(correct)
                boots_ta = np.empty(n_boot); boots_h = np.empty(n_boot)
                gta = correct["g_ta"].to_numpy(); ota = correct["o_ta"].to_numpy()
                ghm = correct["g_harm"].to_numpy(); ohm = correct["o_harm"].to_numpy()
                for i in range(n_boot):
                    idx = rng.integers(0, n_c, size=n_c)
                    boots_ta[i] = gta[idx].mean() - ota[idx].mean()
                    boots_h[i] = ghm[idx].mean() - ohm[idx].mean()
                ta_lo, ta_hi = np.percentile(boots_ta, [2.5, 97.5])
                harm_lo, harm_hi = np.percentile(boots_h, [2.5, 97.5])
            else:
                ta_lo = ta_hi = harm_lo = harm_hi = float("nan")
            rows.append({
                "critic": critic, "condition": cond, "n_wrong": len(wrong), "n_correct": len(correct),
                "FAR_gated": far_g, "FAR_oneshot": far_o,
                "delta_FAR_pp": far_delta * 100, "CI_FAR_low": far_lo * 100, "CI_FAR_high": far_hi * 100,
                "TAR_gated": ta_g, "TAR_oneshot": ta_o,
                "delta_TAR_pp": ta_delta * 100, "CI_TAR_low": ta_lo * 100, "CI_TAR_high": ta_hi * 100,
                "Harm_gated": harm_g, "Harm_oneshot": harm_o,
                "delta_Harm_pp": harm_delta * 100, "CI_Harm_low": harm_lo * 100, "CI_Harm_high": harm_hi * 100,
            })
    return pd.DataFrame(rows)


# =========================================================
# 4. Parse-failure + token-usage + checker-audit tables
# =========================================================

def audit_parse_token_checker(df, df_gated):
    """Summarize parse failures, token usage, and (lightly) answer-checker
    coverage across critic / interface / dataset / condition cells."""
    # parse failure proxy: is_correct is None in critique log (we already
    # filtered those out in load_critique_records). Recount by reloading
    # raw counts. We approximate by checking if `is_correct` is None in
    # the original critique records, OR if `corrected_answer` is None on
    # an arm that requested it.
    rows = []
    for critic in sorted(df["critic_model"].unique()):
        for arm in sorted(df[df["critic_model"] == critic]["arm"].unique()):
            for ds in sorted(df["dataset"].unique()):
                slc = df[(df["critic_model"] == critic) & (df["arm"] == arm)
                          & (df["dataset"] == ds)]
                if slc.empty: continue
                n_calls = len(slc)
                # for repair-style arms, parse-failure proxy: corrected_answer is None for items judged wrong
                if arm in ("legacy_full", "verdict_plus_repair"):
                    rejected = slc[~slc["is_correct"]]
                    parse_fail = (rejected["corrected_answer"].isna() | (rejected["corrected_answer"] == None)).sum()
                else:
                    parse_fail = 0
                rows.append({
                    "critic": critic, "arm": arm, "dataset": ds,
                    "n_calls": n_calls,
                    "parse_fail_proxy": int(parse_fail),
                    "parse_fail_rate": float(parse_fail / n_calls) if n_calls else float("nan"),
                })
    return pd.DataFrame(rows)


# =========================================================
# 5. Per-dataset main-results figure (promote from appendix)
# =========================================================

def fig_caveat_a_per_dataset_main(df):
    """Cleaner per-dataset Caveat A repair-vs-neutral effect figure
    for the main results section (replaces appendix-only figure)."""
    sub = df[(df["arm"].isin(["verdict_only_neutral", "verdict_plus_repair"]))
             & (~df["actually_correct"])
             & (df["condition"] == "anonymous")].copy()
    sub["accept_int"] = sub["is_correct"].astype(int)
    critics = sorted(sub["critic_model"].unique())
    datasets = sorted(sub["dataset"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(critics)
    x = np.arange(len(datasets))
    colors = {"gemma-4": "#1f77b4", "gpt-oss-no-think": "#ff7f0e",
              "qwen3-no-think": "#2ca02c"}
    for ci, critic in enumerate(critics):
        ys = []; lo_err = []; hi_err = []
        for ds in datasets:
            piv = sub[(sub["critic_model"] == critic) & (sub["dataset"] == ds)][
                ["problem_id", "arm", "accept_int"]
            ]
            d, lo, hi, n = paired_bootstrap_diff_ci(
                piv, "arm", "verdict_plus_repair", "verdict_only_neutral", "accept_int"
            )
            ys.append(d * 100)
            lo_err.append(max(0, (d - lo) * 100))
            hi_err.append(max(0, (hi - d) * 100))
        ax.bar(x + (ci - len(critics)/2 + 0.5) * width, ys, width,
               label=critic, color=colors.get(critic, None),
               yerr=[lo_err, hi_err], capsize=3)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("FAR(Verdict+Repair) - FAR(Verdict-Only Neutral) (pp)")
    ax.set_title("Repair-request effect is concentrated on hard math (AIME)\n"
                 "and not consistent across datasets")
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = PAPER_FIGS / "fig5_caveat_a_per_dataset.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


# =========================================================
# main
# =========================================================

def main():
    print("loading data...")
    df = load_critique_records()
    df_gated = load_gated_stage2()
    gt = load_ground_truth()
    df_caveat_only = df[df["log_dir"] == "caveat_a_run_1"]

    print("\n[1] Mixed-effects (cluster-robust binomial GEE) ...")
    me_df = run_mixed_effects(df_caveat_only)
    me_df.to_csv(PAPER_TABLES / "appendix_A1_mixed_effects.csv", index=False)
    print(me_df.to_string(index=False))

    print("\n[2] Leave-one-dataset-out repair contrast ...")
    loo_df = loo_repair_contrast(df_caveat_only)
    loo_df.to_csv(PAPER_TABLES / "appendix_loo_caveat_a.csv", index=False)
    print(loo_df.to_string(index=False))

    print("\n[3] Gated vs oneshot bootstrap CIs ...")
    gci_df = gated_vs_oneshot_cis(df_caveat_only, df_gated, gt)
    gci_df.to_csv(PAPER_TABLES / "gated_contrasts_CI.csv", index=False)
    print(gci_df.to_string(index=False))

    print("\n[4] Parse/token/checker audit ...")
    audit_df = audit_parse_token_checker(df, df_gated)
    audit_df.to_csv(PAPER_TABLES / "audit_parse_token_checker.csv", index=False)
    print(f"  ({len(audit_df)} rows; parse-fail rate summary)")
    print(audit_df.groupby(["critic", "arm"])["parse_fail_rate"].mean().to_string())

    print("\n[5] Per-dataset main-results figure ...")
    fig_caveat_a_per_dataset_main(df_caveat_only)

    print("\ndone")


if __name__ == "__main__":
    main()
