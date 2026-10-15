"""Verdict repair agreement."""

import json

import matplotlib

from identity_bias.runtime.paths import workspace_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = workspace_root() / "logs"
OUT_DIR = workspace_root() / "preliminary_out"

# Only load records from the verdict-repair runs (have initial_* fields)
VR_RUN_DIRS = [
    "verdict_repair_gpt_oss_no_think_run_1",
    "verdict_repair_qwen3_no_think_run_1",
]


def load_vr_records():
    records = []
    for run in VR_RUN_DIRS:
        run_dir = LOGS_DIR / run
        if not run_dir.exists():
            continue
        for path in run_dir.glob("identity_swap_*.jsonl"):
            stem = path.stem.replace("identity_swap_", "")
            try:
                critic_solver, dataset = stem.rsplit("_", 1)
                if dataset == "pro":
                    critic_solver, _ = critic_solver.rsplit("_", 1)
                    dataset = "mmlu_pro"
            except ValueError:
                continue
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("type") != "critique":
                        continue
                    crit = rec["critique"]
                    records.append(
                        {
                            "problem_id": crit["problem_id"],
                            "dataset": dataset,
                            "run": run,
                            "critic_model": crit["critic_model"],
                            "solver_model": crit["solver_model"],
                            "condition": crit["identity_condition"],
                            "is_correct": bool(crit["is_correct"]),  # final
                            "initial_is_correct": crit.get(
                                "initial_is_correct"
                            ),  # first-stage
                            "corrected_answer": crit.get("corrected_answer"),
                            "actually_correct": bool(crit["actually_correct"]),
                        }
                    )
    return records


def main():
    OUT_DIR.mkdir(exist_ok=True)

    records = load_vr_records()
    print(f"loaded {len(records)} verdict-repair records")
    if not records:
        print("no records found")
        return

    critics = sorted({r["critic_model"] for r in records})
    print(f"critics with VR data: {critics}")
    print(f"datasets: {sorted({r['dataset'] for r in records})}")
    print(f"conditions: {sorted({r['condition'] for r in records})}")

    # Wrong solutions only (FAR domain)
    wrong = [r for r in records if not r["actually_correct"]]
    # Drop any without initial verdict (defensive)
    wrong = [r for r in wrong if r["initial_is_correct"] is not None]
    print(f"wrong + has initial verdict: {len(wrong)}")

    conds_order = [
        "anonymous",
        "weak_model",
        "strong_model",
        "self",
        "other_model",
        "human",
    ]
    available_conds = [
        c for c in conds_order if any(r["condition"] == c for r in wrong)
    ]

    for critic in critics:
        print(f"\n===== {critic} =====\n")
        cw = [r for r in wrong if r["critic_model"] == critic]
        print(f"  wrong-solution records: {len(cw)}")

        # Per-condition: initial accept rate (FAR_initial) and final accept rate (FAR_final)
        print(
            f"\n  {'condition':<14} {'FAR_init':<11} {'FAR_final':<11} {'flip(init→final)':<18} {'repair_success_among_init_reject':<35}  n"
        )
        for cond in available_conds:
            subset = [r for r in cw if r["condition"] == cond]
            if not subset:
                continue
            n = len(subset)
            far_init = sum(r["initial_is_correct"] for r in subset) / n
            far_final = sum(r["is_correct"] for r in subset) / n
            # flips: initial verdict changed in final call
            flips = sum(1 for r in subset if r["initial_is_correct"] != r["is_correct"])
            flip_rate = flips / n
            # among items the critic INITIALLY rejected (initial_is_correct=False),
            # what fraction did the final call correct? We approximate "correction success" as:
            # the corrected_answer matched ground truth. But we don't have ground truth string
            # in the critique record cleanly. As a proxy, count repair "successful" if the
            # final is_correct flipped from initial False to something useful (corrected_answer != None
            # and final is_correct == False — meaning model maintained the rejection AND offered a fix).
            init_rejected = [r for r in subset if not r["initial_is_correct"]]
            if init_rejected:
                offered_fix = sum(
                    1 for r in init_rejected if r["corrected_answer"] is not None
                )
                repair_rate = offered_fix / len(init_rejected)
                print(
                    f"  {cond:<14} {far_init:.3f}      {far_final:.3f}      {flip_rate:.3f} ({flips:>4})      {repair_rate:.3f} ({offered_fix}/{len(init_rejected)})              n={n}"
                )
            else:
                print(
                    f"  {cond:<14} {far_init:.3f}      {far_final:.3f}      {flip_rate:.3f}       -                                  n={n}"
                )

        # KEY summary table: WEAK-anon and STRONG-anon deltas for FAR_initial vs FAR_final
        print(f"\n  --- mechanism summary: where does the WEAK effect happen? ---")
        anon = [r for r in cw if r["condition"] == "anonymous"]
        weak = [r for r in cw if r["condition"] == "weak_model"]
        strong = [r for r in cw if r["condition"] == "strong_model"]
        if anon and weak and strong:
            init_anon = sum(r["initial_is_correct"] for r in anon) / len(anon)
            init_weak = sum(r["initial_is_correct"] for r in weak) / len(weak)
            init_strong = sum(r["initial_is_correct"] for r in strong) / len(strong)
            final_anon = sum(r["is_correct"] for r in anon) / len(anon)
            final_weak = sum(r["is_correct"] for r in weak) / len(weak)
            final_strong = sum(r["is_correct"] for r in strong) / len(strong)
            print(
                f"  WEAK - anon FAR_init  = {(init_weak - init_anon):+.3f}    FAR_final = {(final_weak - final_anon):+.3f}"
            )
            print(
                f"  STRONG - anon FAR_init = {(init_strong - init_anon):+.3f}    FAR_final = {(final_strong - final_anon):+.3f}"
            )
            print(
                f"  → if WEAK FAR_init delta ~= WEAK FAR_final delta, effect is at the *verdict* stage (epistemic deference)."
            )
            print(
                f"  → if WEAK FAR_init delta much smaller than FAR_final delta, effect is at the *repair* stage."
            )

    # ---------- plot: per-critic FAR_init vs FAR_final by condition ----------
    fig, axes = plt.subplots(
        1, len(critics), figsize=(6 * len(critics), 5), squeeze=False
    )
    axes = axes[0]
    for ax, critic in zip(axes, critics):
        cw = [r for r in wrong if r["critic_model"] == critic]
        x_pos = np.arange(len(available_conds))
        far_init_vals, far_final_vals = [], []
        for cond in available_conds:
            subset = [r for r in cw if r["condition"] == cond]
            if not subset:
                far_init_vals.append(np.nan)
                far_final_vals.append(np.nan)
                continue
            far_init_vals.append(
                sum(r["initial_is_correct"] for r in subset) / len(subset)
            )
            far_final_vals.append(sum(r["is_correct"] for r in subset) / len(subset))
        width = 0.35
        ax.bar(
            x_pos - width / 2,
            far_init_vals,
            width,
            label="FAR (initial verdict)",
            color="#888",
        )
        ax.bar(
            x_pos + width / 2,
            far_final_vals,
            width,
            label="FAR (full verdict+repair)",
            color="#1f77b4",
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(available_conds, rotation=15, ha="right")
        ax.set_ylabel("False-Accept Rate on wrong solutions")
        ax.set_title(f"{critic}\ndoes framing move the verdict or the repair?")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    plt.tight_layout()
    p = OUT_DIR / "verdict_vs_repair.png"
    fig.savefig(p, dpi=130)
    print(f"\nplot saved: {p}")


if __name__ == "__main__":
    main()
