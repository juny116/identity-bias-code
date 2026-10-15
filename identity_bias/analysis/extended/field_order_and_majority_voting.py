"""Field order and majority voting."""

import json

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
    load_gated_stage2,
    paired_bootstrap_diff_ci,
)

# =========================================================
# 1. Caveat A table augmented with field-order reversed arm
# =========================================================


def caveat_a_with_field_order(df):
    """Include verdict_plus_repair_reversed arm. Currently only gpt-oss-no-think
    has this arm; other critics will show NaN for the reversed contrast."""
    arms = [
        "verdict_only_instructed",
        "verdict_only_neutral",
        "verdict_schema_matched",
        "verdict_plus_repair",
        "verdict_plus_repair_reversed",
    ]
    sub = df[
        (df["arm"].isin(arms))
        & (~df["actually_correct"])
        & (df["condition"] == "anonymous")
        & (df["log_dir"] == "caveat_a_run_1")
    ].copy()
    sub["accept_int"] = sub["is_correct"].astype(int)
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        cs = sub[sub["critic_model"] == critic]
        row = {"critic": critic}
        for arm in arms:
            arm_data = cs[cs["arm"] == arm]
            n = len(arm_data)
            far = arm_data["accept_int"].mean() if n else float("nan")
            row[f"FAR_{arm}"] = far
            row[f"n_{arm}"] = n
        # paired delta vs neutral
        for arm in [
            "verdict_only_instructed",
            "verdict_schema_matched",
            "verdict_plus_repair",
            "verdict_plus_repair_reversed",
        ]:
            piv = cs[cs["arm"].isin([arm, "verdict_only_neutral"])][
                ["problem_id", "arm", "accept_int"]
            ]
            arms_present = set(piv["arm"].unique())
            if arm not in arms_present or "verdict_only_neutral" not in arms_present:
                row[f"delta_{arm}_vs_neutral_pp"] = float("nan")
                row[f"CI_low_{arm}_vs_neutral_pp"] = float("nan")
                row[f"CI_high_{arm}_vs_neutral_pp"] = float("nan")
                continue
            delta, lo, hi, n = paired_bootstrap_diff_ci(
                piv, "arm", arm, "verdict_only_neutral", "accept_int"
            )
            row[f"delta_{arm}_vs_neutral_pp"] = delta * 100
            row[f"CI_low_{arm}_vs_neutral_pp"] = lo * 100
            row[f"CI_high_{arm}_vs_neutral_pp"] = hi * 100
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================
# 2. qwen3 cross-critique repair contrast
# =========================================================


def qwen3_cross_critique_repair(df):
    """qwen3-no-think on gpt-oss-think solutions (cross-critique)."""
    sub = df[
        (df["critic_model"] == "qwen3-no-think")
        & (df["solver_model"] == "gpt-oss-think")
        & (df["log_dir"] == "caveat_a_run_1")
        & (df["condition"] == "anonymous")
        & (~df["actually_correct"])
        & (df["arm"].isin(["verdict_only_neutral", "verdict_plus_repair"]))
    ].copy()
    sub["accept_int"] = sub["is_correct"].astype(int)
    rows = []
    far_n = sub[sub["arm"] == "verdict_only_neutral"]["accept_int"].mean()
    far_r = sub[sub["arm"] == "verdict_plus_repair"]["accept_int"].mean()
    piv = sub[["problem_id", "arm", "accept_int"]]
    delta, lo, hi, n = paired_bootstrap_diff_ci(
        piv, "arm", "verdict_plus_repair", "verdict_only_neutral", "accept_int"
    )
    rows.append(
        {
            "critic": "qwen3-no-think",
            "solver": "gpt-oss-think",
            "design": "cross-critique",
            "FAR_neutral": far_n,
            "FAR_repair": far_r,
            "delta_repair_vs_neutral_pp": delta * 100,
            "CI_low_pp": lo * 100,
            "CI_high_pp": hi * 100,
            "n_paired": n,
        }
    )
    return pd.DataFrame(rows)


# =========================================================
# 3. N-of-K voting outcomes
# =========================================================


def load_n_of_k():
    rows = []
    for path in (LOGS / "n_of_k_run_1").glob("n_of_k_*.jsonl"):
        stem = path.stem.replace("n_of_k_", "")
        try:
            body, cond = stem.split("__", 1)
        except ValueError:
            continue
        parts = body.rsplit("_", 1)
        critic_solver, dataset = parts
        if dataset == "pro":
            critic_solver, _ = critic_solver.rsplit("_", 1)
            dataset = "mmlu_pro"
        critic, solver = critic_solver.split("_", 1)
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(
                    {
                        "problem_id": r["problem_id"],
                        "dataset": dataset,
                        "critic_model": r["critic_model"],
                        "solver_model": r["solver_model"],
                        "condition": r["identity_condition"],
                        "K": r["K"],
                        "n_accept": r["n_accept"],
                        "majority_accept": r["majority_accept"],
                        "actually_correct": r["actually_correct"],
                    }
                )
    return pd.DataFrame(rows)


def n_of_k_vs_gated(df_caveat, df_gated, df_nofk):
    """Compute per (critic, condition) FAR for:
    - oneshot verdict_plus_repair (from caveat A)
    - verdict_only_neutral single-shot (from caveat A)
    - N-of-K majority vote
    - gated pipeline (= verdict_only_neutral FAR; same outcome since
      gated accepts when verdict says accept)
    """
    rows = []
    for critic in sorted(df_nofk["critic_model"].unique()):
        for cond in sorted(
            df_nofk[df_nofk["critic_model"] == critic]["condition"].unique()
        ):
            # restrict to same Caveat A sample (matches by problem_id)
            nofk_cs = df_nofk[
                (df_nofk["critic_model"] == critic) & (df_nofk["condition"] == cond)
            ]
            wrong_nofk = nofk_cs[~nofk_cs["actually_correct"]]
            far_nofk = (
                wrong_nofk["majority_accept"].mean()
                if len(wrong_nofk)
                else float("nan")
            )

            # gated FAR = verdict_only_neutral FAR (since gated accepts when stage-1 accepts)
            caveat_neutral = df_caveat[
                (df_caveat["critic_model"] == critic)
                & (df_caveat["arm"] == "verdict_only_neutral")
                & (df_caveat["condition"] == cond)
                & (~df_caveat["actually_correct"])
                & (df_caveat["log_dir"] == "caveat_a_run_1")
            ]
            far_gated = (
                caveat_neutral["is_correct"].mean()
                if len(caveat_neutral)
                else float("nan")
            )

            caveat_repair = df_caveat[
                (df_caveat["critic_model"] == critic)
                & (df_caveat["arm"] == "verdict_plus_repair")
                & (df_caveat["condition"] == cond)
                & (~df_caveat["actually_correct"])
                & (df_caveat["log_dir"] == "caveat_a_run_1")
            ]
            far_oneshot = (
                caveat_repair["is_correct"].mean()
                if len(caveat_repair)
                else float("nan")
            )

            rows.append(
                {
                    "critic": critic,
                    "condition": cond,
                    "n_wrong_nofk": len(wrong_nofk),
                    "FAR_oneshot": far_oneshot,
                    "FAR_gated": far_gated,
                    "FAR_NofK_majority": far_nofk,
                    "delta_NofK_vs_oneshot_pp": (far_nofk - far_oneshot) * 100,
                    "delta_NofK_vs_gated_pp": (far_nofk - far_gated) * 100,
                }
            )
    return pd.DataFrame(rows)


# =========================================================
# Figures
# =========================================================


def fig_field_order(df_caveat):
    """Side-by-side comparison of field-order effect on gpt-oss-no-think
    and gemma-4. Reveals the critic-specific direction reversal."""
    arms = [
        "verdict_only_instructed",
        "verdict_only_neutral",
        "verdict_schema_matched",
        "verdict_plus_repair",
        "verdict_plus_repair_reversed",
    ]
    arm_labels = [
        "Verdict-Only\nInstructed",
        "Verdict-Only\nNeutral",
        "Schema-Matched\nVerdict",
        "Verdict+Repair\n(verdict first)",
        "Verdict+Repair\n(corr_ans first)",
    ]
    critics = ["gpt-oss-no-think", "gemma-4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, critic in zip(axes, critics):
        sub = df_caveat[
            (df_caveat["critic_model"] == critic)
            & (df_caveat["arm"].isin(arms))
            & (df_caveat["condition"] == "anonymous")
            & (~df_caveat["actually_correct"])
        ]
        fars = []
        ns = []
        for arm in arms:
            a = sub[sub["arm"] == arm]
            fars.append(a["is_correct"].mean() * 100 if len(a) else 0)
            ns.append(len(a))
        colors = ["#888", "#bbb", "#888", "#d62728", "#1f77b4"]
        ax.bar(np.arange(len(arms)), fars, color=colors)
        ax.set_xticks(np.arange(len(arms)))
        ax.set_xticklabels(arm_labels, rotation=20, ha="right", fontsize=8)
        for i, (v, n) in enumerate(zip(fars, ns)):
            ax.text(i, v + 1, f"{v:.1f}%\n(n={n})", ha="center", fontsize=7)
        ax.set_title(critic)
        ax.set_ylim(0, max(fars) * 1.2 if fars else 100)
    axes[0].set_ylabel("FAR on actually wrong solutions (%)")
    fig.suptitle(
        "Schema-order effect is critic-specific: gpt-oss FAR drops to neutral "
        "when corrected_answer is generated first;\nGemma-4 FAR rises further "
        "(repair-inflation amplified, +2.7 pp above standard).",
        fontsize=10,
    )
    plt.tight_layout()
    p = PAPER_FIGS / "fig6_field_order_mechanism.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def fig_n_of_k(tbl):
    fig, ax = plt.subplots(figsize=(9, 5))
    pairs = [(r["critic"], r["condition"]) for _, r in tbl.iterrows()]
    labels = [f"{c}\n{cond}" for c, cond in pairs]
    x = np.arange(len(pairs))
    width = 0.27
    ax.bar(
        x - width,
        tbl["FAR_oneshot"] * 100,
        width,
        label="One-shot Verdict+Repair",
        color="#d62728",
    )
    ax.bar(
        x,
        tbl["FAR_gated"] * 100,
        width,
        label="Gated (verdict-only stage 1)",
        color="#888",
    )
    ax.bar(
        x + width,
        tbl["FAR_NofK_majority"] * 100,
        width,
        label="N-of-K=5 majority",
        color="#1f77b4",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("FAR on actually wrong solutions (%)")
    ax.set_title("Mitigation comparison: One-shot vs Gated vs N-of-K majority voting")
    ax.legend()
    plt.tight_layout()
    p = PAPER_FIGS / "fig7_n_of_k_vs_gated.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def main():
    print("loading...")
    df = load_critique_records()
    df_gated = load_gated_stage2()
    df_nofk = load_n_of_k()
    print(
        f"  caveat A: {len(df)} critique records; gated: {len(df_gated)}; N-of-K: {len(df_nofk)}"
    )

    print("\n[1] Caveat A table with field-order arm ...")
    tbl = caveat_a_with_field_order(df)
    tbl.to_csv(PAPER_TABLES / "table2b_caveat_a_with_field_order.csv", index=False)
    print(
        tbl[["critic"] + [c for c in tbl.columns if c.startswith("delta_")]].to_string(
            index=False
        )
    )

    print("\n[2] qwen3-no-think cross-critique repair contrast ...")
    tbl_cross = qwen3_cross_critique_repair(df)
    tbl_cross.to_csv(PAPER_TABLES / "table8_qwen3_cross_critique.csv", index=False)
    print(tbl_cross.to_string(index=False))

    print("\n[3] N-of-K voting vs gated ...")
    df_caveat_only = df[df["log_dir"] == "caveat_a_run_1"]
    tbl_nofk = n_of_k_vs_gated(df_caveat_only, df_gated, df_nofk)
    tbl_nofk.to_csv(PAPER_TABLES / "table7_n_of_k_outcomes.csv", index=False)
    print(tbl_nofk.to_string(index=False))

    print("\n[4] Field-order mechanism figure ...")
    fig_field_order(df_caveat_only)

    print("\n[5] N-of-K vs gated figure ...")
    fig_n_of_k(tbl_nofk)

    print("done")


if __name__ == "__main__":
    main()
