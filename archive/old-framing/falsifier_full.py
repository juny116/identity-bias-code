"""
Final falsifier analysis: STRONG vs WEAK vs anonymous across critic families.

Loads all identity_swap data including new STRONG_MODEL runs:
  - gemma-4 (4 datasets, all 6 conditions, full N)
  - qwen3-no-think (4 datasets, STRONG added to existing 5 conditions)
  - gpt-oss-no-think (4 datasets, STRONG added to existing 5 conditions)
  - gpt-oss-think (AIME only, STRONG added)
  - qwen3 (think) — STRONG not added (aborted Phase B-2 due to speed)

Centerpiece plot: per-critic, WEAK-anon and STRONG-anon deltas (pooled across all
datasets). If STRONG is consistently null across all critics, Plan B narrative
("low-competence framing is asymmetric skepticism intervention") is supported.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = Path(__file__).parent.parent / "logs"
OUT_DIR = Path(__file__).parent / "preliminary_out"
OUT_DIR.mkdir(exist_ok=True)
EXCLUDE_DATASETS = {"bbh"}


def load_all_critiques():
    records = []
    for path in LOGS_DIR.rglob("identity_swap_*.jsonl"):
        stem = path.stem.replace("identity_swap_", "")
        try:
            critic_solver, dataset = stem.rsplit("_", 1)
            if dataset == "pro":
                critic_solver, _ = critic_solver.rsplit("_", 1)
                dataset = "mmlu_pro"
        except ValueError:
            continue
        if dataset in EXCLUDE_DATASETS:
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
                records.append({
                    "problem_id": crit["problem_id"],
                    "dataset": dataset,
                    "critic_model": crit["critic_model"],
                    "solver_model": crit["solver_model"],
                    "condition": crit["identity_condition"],
                    "is_correct": bool(crit["is_correct"]),
                    "actually_correct": bool(crit["actually_correct"]),
                })
    return records


def main():
    records = load_all_critiques()
    print(f"loaded {len(records)} total records")

    # Inventory: which critics have STRONG data, and on how many datasets
    print("\n=== STRONG_MODEL availability by critic ===")
    by_critic_ds = defaultdict(set)
    for r in records:
        if r["condition"] == "strong_model":
            by_critic_ds[r["critic_model"]].add(r["dataset"])
    for cm in sorted(by_critic_ds.keys()):
        ds = sorted(by_critic_ds[cm])
        print(f"  {cm:<24} datasets: {ds}")

    # ---------- 1. Per-critic FAR (anon vs WEAK vs STRONG) on wrong solutions ----------
    print("\n=== [1] Per-critic FAR on wrong solutions (matched datasets per critic) ===")
    print(f"{'critic':<24} {'anon':<14} {'weak':<14} {'strong':<14}  ds with STRONG")
    summary = []
    for cm in sorted(by_critic_ds.keys()):
        valid_ds = by_critic_ds[cm]
        wrong = [r for r in records
                 if r["critic_model"] == cm
                 and not r["actually_correct"]
                 and r["dataset"] in valid_ds]
        far_by_cond = {}
        for cond in ["anonymous", "weak_model", "strong_model"]:
            subset = [r for r in wrong if r["condition"] == cond]
            if not subset:
                far_by_cond[cond] = (None, 0)
            else:
                far = sum(r["is_correct"] for r in subset) / len(subset)
                far_by_cond[cond] = (far, len(subset))
        a = far_by_cond["anonymous"]
        w = far_by_cond["weak_model"]
        s = far_by_cond["strong_model"]
        row = f"{cm:<24}"
        for cond_val in (a, w, s):
            row += f" {cond_val[0]:.3f}(n={cond_val[1]:>4})" if cond_val[0] is not None else f" {'-':<14}"
        row += f"  {sorted(valid_ds)}"
        print(row)
        summary.append((cm, a, w, s))

    # ---------- 2. WEAK-anon and STRONG-anon deltas ----------
    print("\n=== [2] WEAK-anon and STRONG-anon FAR deltas (per critic) ===")
    print(f"  Plan B prediction:")
    print(f"    WEAK-anon < 0 (skepticism intervention) - across critics")
    print(f"    STRONG-anon ~ 0 (no symmetric competence effect)")
    print(f"{'critic':<24} {'WEAK-anon':<14} {'STRONG-anon':<14}")
    deltas = {}
    for cm, a, w, s in summary:
        if a[0] is None or w[0] is None or s[0] is None:
            print(f"  {cm:<24} - (missing condition)")
            continue
        wd = w[0] - a[0]
        sd = s[0] - a[0]
        deltas[cm] = (wd, sd)
        print(f"  {cm:<24} {wd:+.3f}        {sd:+.3f}")

    # ---------- 3. Plot: per-critic WEAK and STRONG vs anon deltas ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cms = list(deltas.keys())
    if cms:
        weak_vals = [deltas[c][0] for c in cms]
        strong_vals = [deltas[c][1] for c in cms]
        x = np.arange(len(cms))
        width = 0.35
        ax = axes[0]
        ax.bar(x - width/2, weak_vals, width, label="WEAK_MODEL", color="#1f77b4")
        ax.bar(x + width/2, strong_vals, width, label="STRONG_MODEL", color="#d62728")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(cms, rotation=20, ha="right")
        ax.set_ylabel("FAR delta vs anonymous (pp)")
        ax.set_title("Per-critic: WEAK reduces FAR, STRONG essentially null")
        ax.legend()

    # right: per-critic per-dataset FAR(strong) - FAR(anon)
    print("\n=== [3] Per-critic per-dataset STRONG-anon FAR delta ===")
    dataset_critic_strong = defaultdict(dict)
    for cm in cms:
        for ds in by_critic_ds[cm]:
            anon_recs = [r for r in records if r["critic_model"] == cm
                         and r["condition"] == "anonymous"
                         and r["dataset"] == ds
                         and not r["actually_correct"]]
            strong_recs = [r for r in records if r["critic_model"] == cm
                           and r["condition"] == "strong_model"
                           and r["dataset"] == ds
                           and not r["actually_correct"]]
            if not anon_recs or not strong_recs:
                continue
            a_far = sum(r["is_correct"] for r in anon_recs) / len(anon_recs)
            s_far = sum(r["is_correct"] for r in strong_recs) / len(strong_recs)
            dataset_critic_strong[ds][cm] = (s_far - a_far, len(strong_recs))
    datasets_ordered = ["aime", "gpqa", "mmlu_pro", "olympiad"]
    print(f"{'critic':<24}" + "".join(f"  {ds:<14}" for ds in datasets_ordered))
    for cm in cms:
        row = f"{cm:<24}"
        for ds in datasets_ordered:
            cell = dataset_critic_strong[ds].get(cm)
            if cell is None:
                row += f"  {'-':<14}"
            else:
                row += f"  {cell[0]:+.3f}(n={cell[1]:>3})"
        print(row)

    ax = axes[1]
    width = 0.8 / max(len(cms), 1)
    x = np.arange(len(datasets_ordered))
    for i, cm in enumerate(cms):
        ys = []
        for ds in datasets_ordered:
            cell = dataset_critic_strong[ds].get(cm)
            ys.append(cell[0] if cell else np.nan)
        ax.bar(x + i * width, ys, width, label=cm)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x + (len(cms) - 1) * width / 2)
    ax.set_xticklabels(datasets_ordered)
    ax.set_ylabel("FAR(STRONG) - FAR(anon) (pp)")
    ax.set_title("STRONG effect per dataset & critic — generalization check")
    ax.legend(fontsize=8)

    plt.tight_layout()
    p = OUT_DIR / "falsifier_full.png"
    fig.savefig(p, dpi=130)
    print(f"\nplot saved: {p}")

    # ---------- 4. Sample size info ----------
    print("\n=== [4] sample sizes (wrong solutions) per critic per condition ===")
    for cm in cms:
        print(f"  {cm}:")
        for cond in ["anonymous", "weak_model", "strong_model", "self", "human", "other_model"]:
            n = sum(1 for r in records if r["critic_model"] == cm
                    and r["condition"] == cond
                    and not r["actually_correct"]
                    and r["dataset"] in by_critic_ds[cm])
            print(f"    {cond:<14} n={n}")


if __name__ == "__main__":
    main()
