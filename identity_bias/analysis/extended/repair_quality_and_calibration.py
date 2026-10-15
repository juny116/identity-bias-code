"""Repair quality and calibration."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
LOGS = REPO / "logs"
PAPER_FIGS = REPO / "paper" / "figures"
PAPER_TABLES = REPO / "paper" / "tables"

from identity_bias.analysis.extended.build_critique_tables import (
    load_critique_records,
    load_ground_truth,
)
from identity_bias.data import check_answer
from identity_bias.types import Dataset


def score_answer(ds_enum, pid, ans, gt_by_pid):
    if pid not in gt_by_pid or ans is None:
        return None
    _, gt = gt_by_pid[pid]
    try:
        return check_answer(ds_enum, ans, gt)
    except Exception:
        return None


def repair_quality_table(df, gt_by_pid):
    """For each (critic, arm, condition), compute among rejected wrong items
    (critic correctly identified the wrong solution) what fraction of repairs
    matched ground truth (= repair success rate)."""
    arms = ["legacy_full", "verdict_plus_repair", "verdict_plus_repair_reversed"]
    sub = df[df["arm"].isin(arms) & df["corrected_answer"].notna()].copy()
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        for arm in arms:
            for cond in sorted(sub["condition"].unique()):
                slc = sub[
                    (sub["critic_model"] == critic)
                    & (sub["arm"] == arm)
                    & (sub["condition"] == cond)
                ]
                if slc.empty:
                    continue
                wrong = slc[~slc["actually_correct"]]
                correct = slc[slc["actually_correct"]]
                # repair success among rejected wrong
                rej_wrong = wrong[~wrong["is_correct"]]
                rep_success = 0
                rep_fail = 0
                rep_unscored = 0
                for _, r in rej_wrong.iterrows():
                    ds_enum = Dataset(r["dataset"])
                    s = score_answer(
                        ds_enum, r["problem_id"], r["corrected_answer"], gt_by_pid
                    )
                    if s is True:
                        rep_success += 1
                    elif s is False:
                        rep_fail += 1
                    else:
                        rep_unscored += 1
                # harmful correction among rejected correct
                rej_correct = correct[~correct["is_correct"]]
                harm = 0
                for _, r in rej_correct.iterrows():
                    ds_enum = Dataset(r["dataset"])
                    s = score_answer(
                        ds_enum, r["problem_id"], r["corrected_answer"], gt_by_pid
                    )
                    if s is False:
                        harm += 1
                rows.append(
                    {
                        "critic": critic,
                        "arm": arm,
                        "condition": cond,
                        "n_wrong": len(wrong),
                        "n_rejected_wrong": len(rej_wrong),
                        "repair_success_rate": rep_success / len(rej_wrong)
                        if len(rej_wrong)
                        else float("nan"),
                        "repair_fail_rate": rep_fail / len(rej_wrong)
                        if len(rej_wrong)
                        else float("nan"),
                        "n_correct": len(correct),
                        "n_rejected_correct": len(rej_correct),
                        "harmful_correction_rate": harm / len(rej_correct)
                        if len(rej_correct)
                        else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def confidence_calibration(df):
    """For each (critic, arm), compute ECE (expected calibration error) and
    Brier score using critic confidence as the predictor and (is_correct ==
    actually_correct) as the binary outcome (i.e. was the verdict correct?)."""
    sub = df[df["confidence"].notna()].copy()
    sub["verdict_correct"] = (sub["is_correct"] == sub["actually_correct"]).astype(int)
    sub["confidence"] = sub["confidence"].astype(float).clip(0, 1)
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        for arm in sorted(sub[sub["critic_model"] == critic]["arm"].unique()):
            slc = sub[(sub["critic_model"] == critic) & (sub["arm"] == arm)]
            if len(slc) < 30:
                continue
            conf = slc["confidence"].to_numpy()
            correct = slc["verdict_correct"].to_numpy()
            brier = float(np.mean((conf - correct) ** 2))
            # ECE with 10 bins
            bins = np.linspace(0, 1, 11)
            ece = 0.0
            n = len(slc)
            bin_data = []
            for i in range(10):
                mask = (
                    (conf >= bins[i]) & (conf < bins[i + 1])
                    if i < 9
                    else (conf >= bins[i]) & (conf <= bins[i + 1])
                )
                if not mask.any():
                    continue
                p_emp = correct[mask].mean()
                p_pred = conf[mask].mean()
                w = mask.sum() / n
                ece += w * abs(p_emp - p_pred)
                bin_data.append(
                    {
                        "bin_center": (bins[i] + bins[i + 1]) / 2,
                        "p_pred": p_pred,
                        "p_emp": p_emp,
                        "n": int(mask.sum()),
                    }
                )
            rows.append(
                {
                    "critic": critic,
                    "arm": arm,
                    "n": int(n),
                    "Brier_score": brier,
                    "ECE_10bin": float(ece),
                    "mean_confidence": float(conf.mean()),
                    "mean_correctness": float(correct.mean()),
                    "bin_data": bin_data,
                }
            )
    return rows


def equivalence_test_tost(
    df, critic, arm_a, arm_b, condition="anonymous", margin_pp=5.0
):
    """Two One-Sided Tests (TOST) for equivalence: claim that |delta| < margin.
    For qwen3 null repair contrast: a successful TOST at margin=5pp would
    bound the effect at ±5pp."""
    sub = df[
        (df["critic_model"] == critic)
        & (df["arm"].isin([arm_a, arm_b]))
        & (df["condition"] == condition)
        & (~df["actually_correct"])
        & (df["log_dir"] == "caveat_a_run_1")
    ].copy()
    sub["accept_int"] = sub["is_correct"].astype(int)
    piv = sub.pivot_table(
        index="problem_id", columns="arm", values="accept_int", aggfunc="mean"
    ).dropna()
    if piv.empty:
        return None
    a = piv[arm_a].to_numpy()
    b = piv[arm_b].to_numpy()
    delta = (a - b).mean() * 100  # pp
    se = (a - b).std(ddof=1) / np.sqrt(len(a)) * 100
    # 90% CI for equivalence
    ci_low = delta - 1.645 * se
    ci_hi = delta + 1.645 * se
    equivalent = (ci_low > -margin_pp) and (ci_hi < margin_pp)
    return {
        "critic": critic,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "condition": condition,
        "n_paired": len(a),
        "delta_pp": delta,
        "CI90_low_pp": ci_low,
        "CI90_high_pp": ci_hi,
        "margin_pp": margin_pp,
        "equivalent": equivalent,
    }


def fig_reliability(rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    critics = sorted(set(r["critic"] for r in rows))[:3]
    for ax, critic in zip(axes, critics):
        crit_rows = [
            r
            for r in rows
            if r["critic"] == critic
            and r["arm"]
            in ("legacy_full", "verdict_only_neutral", "verdict_plus_repair")
        ]
        for r in crit_rows:
            bd = r["bin_data"]
            xs = [b["p_pred"] for b in bd]
            ys = [b["p_emp"] for b in bd]
            ns = [b["n"] for b in bd]
            ax.plot(xs, ys, marker="o", label=f"{r['arm']} (ECE={r['ECE_10bin']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.6, label="perfect")
        ax.set_title(critic)
        ax.set_xlabel("predicted confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("empirical verdict-correct rate")
    fig.suptitle("Confidence calibration reliability diagrams")
    plt.tight_layout()
    p = PAPER_FIGS / "fig8_reliability_diagram.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def fig_repair_success(tbl):
    sub = tbl[
        (tbl["condition"] == "anonymous")
        & (tbl["arm"].isin(["legacy_full", "verdict_plus_repair"]))
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    critics = sorted(sub["critic"].unique())
    x = np.arange(len(critics))
    width = 0.35
    legacy = [
        sub[(sub["critic"] == c) & (sub["arm"] == "legacy_full")][
            "repair_success_rate"
        ].mean()
        for c in critics
    ]
    repair = [
        sub[(sub["critic"] == c) & (sub["arm"] == "verdict_plus_repair")][
            "repair_success_rate"
        ].mean()
        for c in critics
    ]
    ax.bar(
        x - width / 2,
        [v * 100 if not np.isnan(v) else 0 for v in legacy],
        width,
        label="legacy full sweep",
        color="#888",
    )
    ax.bar(
        x + width / 2,
        [v * 100 if not np.isnan(v) else 0 for v in repair],
        width,
        label="Caveat A verdict+repair arm",
        color="#1f77b4",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(critics, rotation=15, ha="right")
    ax.set_ylabel("Repair-success rate among correctly rejected wrong items (%)")
    ax.set_title("Repair quality: even when verdict rejects correctly, repair is poor")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    p = PAPER_FIGS / "fig9_repair_success_per_arm.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def main():
    print("loading data...")
    df = load_critique_records()
    gt = load_ground_truth()

    print("\n[1] Repair-quality analysis ...")
    tbl_rq = repair_quality_table(df, gt)
    tbl_rq.to_csv(PAPER_TABLES / "table9_repair_quality.csv", index=False)
    # print pooled summary
    pooled = (
        tbl_rq.groupby(["critic", "arm"])
        .agg(
            n_rej_wrong=("n_rejected_wrong", "sum"),
            success=(
                "repair_success_rate",
                lambda s: (
                    (s * tbl_rq.loc[s.index, "n_rejected_wrong"]).sum()
                    / max(tbl_rq.loc[s.index, "n_rejected_wrong"].sum(), 1)
                ),
            ),
            harm=(
                "harmful_correction_rate",
                lambda s: (
                    (s * tbl_rq.loc[s.index, "n_rejected_correct"]).sum()
                    / max(tbl_rq.loc[s.index, "n_rejected_correct"].sum(), 1)
                ),
            ),
        )
        .reset_index()
    )
    print(pooled.to_string(index=False))
    fig_repair_success(tbl_rq)

    print("\n[2] Confidence calibration ...")
    calib_rows = confidence_calibration(df)
    # save only top-level summary (bin_data dropped for CSV)
    summary_rows = [{k: v for k, v in r.items() if k != "bin_data"} for r in calib_rows]
    pd.DataFrame(summary_rows).to_csv(
        PAPER_TABLES / "table10_confidence_calibration.csv", index=False
    )
    for r in calib_rows:
        if r["arm"] in ("legacy_full", "verdict_only_neutral", "verdict_plus_repair"):
            print(
                f"  {r['critic']:<20} arm={r['arm']:<24} ECE={r['ECE_10bin']:.3f} Brier={r['Brier_score']:.3f} "
                f"mean_conf={r['mean_confidence']:.3f} mean_correct={r['mean_correctness']:.3f} n={r['n']}"
            )
    fig_reliability(calib_rows)

    print("\n[3] Equivalence (TOST) for qwen3 null repair contrast ...")
    for critic in ["qwen3-no-think", "gpt-oss-no-think", "gemma-4"]:
        for margin in (5.0, 3.0):
            tost = equivalence_test_tost(
                df,
                critic,
                "verdict_plus_repair",
                "verdict_only_neutral",
                condition="anonymous",
                margin_pp=margin,
            )
            if tost:
                print(
                    f"  critic={critic} margin={margin}pp: delta={tost['delta_pp']:+.2f}pp "
                    f"CI90=[{tost['CI90_low_pp']:+.2f}, {tost['CI90_high_pp']:+.2f}]  "
                    f"equivalent={tost['equivalent']}"
                )

    print("done")


if __name__ == "__main__":
    main()
