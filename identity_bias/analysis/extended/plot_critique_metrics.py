"""Plot critique metrics."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
PAPER_FIGS = REPO / "paper" / "figures"
PAPER_TABLES = REPO / "paper" / "tables"


def fig1_paired_repair_inflation():
    """Figure 1: paired-repair-inflation-two-critic.
    Phase E: FAR_initial vs FAR_final per critic, paired CIs."""
    df = pd.read_csv(PAPER_TABLES / "table4_phaseE_paired_verdict_vs_full.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    critics = df["critic_model"].tolist()
    init = df["FAR_initial"].tolist()
    final = df["FAR_final"].tolist()
    deltas = df["delta_full_minus_initial_pp"].tolist()
    ci_low = df["CI_low_pp"].tolist()
    ci_high = df["CI_high_pp"].tolist()

    x = np.arange(len(critics))
    width = 0.35
    ax.bar(x - width / 2, init, width, label="Verdict-only (initial)", color="#888")
    ax.bar(
        x + width / 2, final, width, label="Verdict + repair (full)", color="#1f77b4"
    )
    for i, (a, b, d, lo, hi) in enumerate(zip(init, final, deltas, ci_low, ci_high)):
        ax.annotate(
            f"+{d:.1f} pp\n[{lo:.1f}, {hi:.1f}]",
            xy=(i + width / 2, b),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(critics)
    ax.set_ylabel("FAR on actually wrong solutions")
    ax.set_title(
        "Phase E: full critique inflates FAR vs verdict-only call\n(paired across 6 source conditions × 4 datasets)"
    )
    ax.legend(loc="upper left")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    p = PAPER_FIGS / "fig1_paired_repair_inflation.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def fig2_caveat_a_forest():
    """Figure 2: caveat-A-prompt-arm-forest-plot.
    Per critic, contrast of {instructed, schema_matched, plus_repair} vs neutral."""
    df = pd.read_csv(PAPER_TABLES / "table2_caveat_a_arm_per_critic.csv")
    critics = df["critic_model"].tolist()
    contrasts = [
        ("verdict_only_instructed", "Verdict-Only Instructed - Neutral", "#888"),
        ("verdict_schema_matched", "Schema-Matched - Neutral", "#666"),
        ("verdict_plus_repair", "Verdict+Repair - Neutral", "#d62728"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4 + 0.5 * len(critics)))
    y_offset = 0.25
    y_positions = []
    y_labels = []
    for ci, critic in enumerate(critics):
        for ki, (arm, label, color) in enumerate(contrasts):
            row = df[df["critic_model"] == critic].iloc[0]
            d = row[f"delta_{arm}_vs_neutral_pp"]
            lo = row[f"CI_low_{arm}_vs_neutral_pp"]
            hi = row[f"CI_high_{arm}_vs_neutral_pp"]
            y = -(ci * (len(contrasts) + 1) + ki)
            y_positions.append(y)
            y_labels.append(f"{critic}\n{label}" if ki == 1 else f"  {label}")
            ax.errorbar(
                d,
                y,
                xerr=[[d - lo], [hi - d]],
                fmt="o",
                color=color,
                capsize=4,
                markersize=7,
            )
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("FAR delta vs Verdict-Only Neutral (percentage points)")
    ax.set_title(
        "Caveat A ablation: prompt-interface contrasts\n(anonymous wrong solutions, paired bootstrap 95% CI)"
    )
    plt.tight_layout()
    p = PAPER_FIGS / "fig2_caveat_a_forest.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def fig3_weak_per_dataset_forest():
    """Figure 3: forest-plot-WEAK-effect-per-dataset.
    For each (critic, dataset), WEAK_MODEL minus anonymous FAR with CIs."""
    # use the legacy_full / Phase A data via Table 3 derivation
    # Build from raw legacy data by re-reading the underlying table.
    # We approximated this in compute_weak_strong_asymmetry; rebuild per-dataset here.
    # For simplicity rebuild from raw freeze_artifacts run logs
    df_raw = pd.read_csv(PAPER_TABLES / "table3_identity_5critics.csv")
    weak_rows = df_raw[df_raw["condition"] == "weak_model"]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(weak_rows))))
    y_positions = []
    y_labels = []
    for i, (_, r) in enumerate(weak_rows.iterrows()):
        y = -i
        d = r.get("delta_vs_anon_pp")
        lo = r.get("CI_low_pp")
        hi = r.get("CI_high_pp")
        if any(pd.isna([d, lo, hi])):
            continue
        ax.errorbar(
            d,
            y,
            xerr=[[d - lo], [hi - d]],
            fmt="o",
            color="#1f77b4",
            capsize=4,
            markersize=6,
        )
        y_positions.append(y)
        y_labels.append(f"{r['critic_model']}")
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("FAR(WEAK_MODEL) - FAR(anonymous) (pp)")
    ax.set_title(
        "WEAK_MODEL framing effect per critic\n(pooled across datasets, paired bootstrap 95% CI)"
    )
    plt.tight_layout()
    p = PAPER_FIGS / "fig3_weak_per_critic_forest.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def fig4_weak_strong_asymmetry():
    """Figure 4: WEAK and STRONG vs anon, per critic."""
    df = pd.read_csv(PAPER_TABLES / "figure4_weak_strong_asymmetry.csv")
    # filter out low-n entries
    df = df[df["n_paired"] >= 100]
    critics = sorted(df["critic_model"].unique())
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(critics))
    for i, critic in enumerate(critics):
        cs = df[df["critic_model"] == critic]
        weak = cs[cs["contrast"] == "weak_model_minus_anon"]
        strong = cs[cs["contrast"] == "strong_model_minus_anon"]
        if not weak.empty:
            w = weak.iloc[0]
            ax.bar(
                i - width / 2,
                w["delta_pp"],
                width,
                color="#1f77b4",
                label="WEAK_MODEL - anon" if i == 0 else None,
                yerr=[
                    [w["delta_pp"] - w["CI_low_pp"]],
                    [w["CI_high_pp"] - w["delta_pp"]],
                ],
                capsize=4,
            )
        if not strong.empty:
            s = strong.iloc[0]
            ax.bar(
                i + width / 2,
                s["delta_pp"],
                width,
                color="#d62728",
                label="STRONG_MODEL - anon" if i == 0 else None,
                yerr=[
                    [s["delta_pp"] - s["CI_low_pp"]],
                    [s["CI_high_pp"] - s["delta_pp"]],
                ],
                capsize=4,
            )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(critics, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("FAR delta vs anonymous (pp)")
    ax.set_title("Asymmetric competence framing: STRONG does not mirror WEAK")
    ax.legend()
    plt.tight_layout()
    p = PAPER_FIGS / "fig4_weak_strong_asymmetry.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def figA1_caveat_a_per_dataset():
    """Appendix A1: per-dataset repair-interface effect, per critic."""
    df = pd.read_csv(PAPER_TABLES / "appendix_A1_caveat_a_per_dataset.csv")
    critics = sorted(df["critic_model"].unique())
    fig, ax = plt.subplots(figsize=(9, 4 + 0.4 * len(df)))
    y_positions = []
    y_labels = []
    colors = {
        "gemma-4": "#1f77b4",
        "gpt-oss-no-think": "#ff7f0e",
        "qwen3-no-think": "#2ca02c",
    }
    idx = 0
    for critic in critics:
        cs = df[df["critic_model"] == critic]
        for _, r in cs.iterrows():
            y = -idx
            ax.errorbar(
                r["delta_pp"],
                y,
                xerr=[
                    [r["delta_pp"] - r["CI_low_pp"]],
                    [r["CI_high_pp"] - r["delta_pp"]],
                ],
                fmt="o",
                color=colors.get(critic, "#666"),
                capsize=4,
                markersize=6,
            )
            y_positions.append(y)
            y_labels.append(f"{critic} / {r['dataset']} (n={r['n_paired']})")
            idx += 1
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("FAR(Verdict+Repair) - FAR(Verdict-Only Neutral) (pp)")
    ax.set_title("Per-dataset repair-interface effect")
    plt.tight_layout()
    p = PAPER_FIGS / "figA1_caveat_a_per_dataset.png"
    fig.savefig(p, dpi=150)
    print(f"  saved {p}")


def main():
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)

    print("generating figures from frozen tables...")
    fig1_paired_repair_inflation()
    fig2_caveat_a_forest()
    fig3_weak_per_dataset_forest()
    fig4_weak_strong_asymmetry()
    figA1_caveat_a_per_dataset()
    print("done")


if __name__ == "__main__":
    main()
