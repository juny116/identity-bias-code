"""
Day-0 sanity checks for the preliminary deference analysis.

(A) Leave-one-out ambiguity prior: compute ambiguity using ONLY the
    `anonymous` condition records (no `weak_model`, no `self`) to remove
    circular bias. If the deference pattern survives, the thesis is robust.

(B) Per-dataset breakdown: same WEAK-SELF FAR delta but split by
    aime / gpqa / mmlu_pro / olympiad to test generalization.

Outputs two PNGs and prints summary tables.
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


def bin_amb(a):
    if a is None:
        return None
    if a < 0.15:
        return "low"
    if a < 0.35:
        return "med"
    return "high"


BINS = ["low", "med", "high"]


# ---------- (A) leave-one-out: anonymous-only prior ----------

def analysis_A(records):
    """Ambiguity computed from anonymous condition only (decoupled from weak/self)."""
    anon = [r for r in records if r["condition"] == "anonymous"]
    per_item = defaultdict(list)
    for r in anon:
        key = (r["problem_id"], r["solver_model"], r["actually_correct"])
        per_item[key].append(r["is_correct"])
    amb_lookup = {
        key: min(sum(v) / len(v), 1 - sum(v) / len(v))
        for key, v in per_item.items()
    }

    for r in records:
        key = (r["problem_id"], r["solver_model"], r["actually_correct"])
        r["amb_bin"] = bin_amb(amb_lookup.get(key))

    wrong = [r for r in records if not r["actually_correct"] and r["amb_bin"] is not None]
    print(f"[A] anonymous-only prior: wrong-records covered = {len(wrong)}")

    # pooled FAR by condition x bin
    cells = defaultdict(lambda: {"acc": 0, "n": 0})
    for r in wrong:
        cells[(r["condition"], r["amb_bin"])]["acc"] += int(r["is_correct"])
        cells[(r["condition"], r["amb_bin"])]["n"] += 1

    conds = sorted({r["condition"] for r in wrong})
    print("\n[A] FAR (pooled), prior = anonymous-only:")
    print(f"{'condition':<14}" + "".join(f"  {b:<11}" for b in BINS))
    for c in conds:
        row = f"{c:<14}"
        for b in BINS:
            cell = cells[(c, b)]
            row += f"  {cell['acc']/cell['n']:.3f}({cell['n']:>4})" if cell["n"] else f"  {'-':<11}"
        print(row)

    # WEAK - SELF delta by critic
    print("\n[A] WEAK - SELF delta by critic x bin:")
    crit_models = sorted({r["critic_model"] for r in wrong})
    print(f"{'critic':<22}" + "".join(f"  {b:<14}" for b in BINS))
    plot_data = defaultdict(dict)
    for cm in crit_models:
        row = f"{cm:<22}"
        for b in BINS:
            S = [r for r in wrong if r["critic_model"] == cm and r["amb_bin"] == b and r["condition"] == "self"]
            W = [r for r in wrong if r["critic_model"] == cm and r["amb_bin"] == b and r["condition"] == "weak_model"]
            if not S or not W:
                row += f"  {'-':<14}"
                continue
            s_far = sum(r["is_correct"] for r in S) / len(S)
            w_far = sum(r["is_correct"] for r in W) / len(W)
            d = w_far - s_far
            plot_data[cm][b] = d
            row += f"  {d:+.3f}({len(S)},{len(W)})"
        print(row)

    # plot
    fig, ax = plt.subplots(figsize=(8, 5))
    cm_list = sorted(plot_data.keys())
    width = 0.8 / max(len(cm_list), 1)
    x = np.arange(len(BINS))
    for i, cm in enumerate(cm_list):
        ys = [plot_data[cm].get(b, np.nan) for b in BINS]
        ax.bar(x + i * width, ys, width, label=cm)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x + (len(cm_list) - 1) * width / 2)
    ax.set_xticklabels([f"{b}\n(amb)" for b in BINS])
    ax.set_ylabel("FAR(WEAK) - FAR(SELF)")
    ax.set_title("Leave-one-out sanity: ambiguity prior = anonymous-only")
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = OUT_DIR / "sanity_A_leave_one_out.png"
    fig.savefig(p, dpi=130)
    print(f"\n[A] saved {p}")


# ---------- (B) per-dataset ----------

def analysis_B(records):
    """Re-use anonymous-only prior, split deltas by dataset."""
    anon = [r for r in records if r["condition"] == "anonymous"]
    per_item = defaultdict(list)
    for r in anon:
        key = (r["problem_id"], r["solver_model"], r["actually_correct"])
        per_item[key].append(r["is_correct"])
    amb_lookup = {
        key: min(sum(v) / len(v), 1 - sum(v) / len(v))
        for key, v in per_item.items()
    }
    for r in records:
        key = (r["problem_id"], r["solver_model"], r["actually_correct"])
        r["amb_bin"] = bin_amb(amb_lookup.get(key))

    wrong = [r for r in records if not r["actually_correct"] and r["amb_bin"] is not None]
    datasets = sorted({r["dataset"] for r in wrong})
    crit_models = sorted({r["critic_model"] for r in wrong})

    print("\n[B] Per-dataset WEAK - SELF delta (pooled over critics) by ambiguity bin:")
    print(f"{'dataset':<12}" + "".join(f"  {b:<14}" for b in BINS))
    plot_rows = defaultdict(dict)
    for ds in datasets:
        row = f"{ds:<12}"
        for b in BINS:
            S = [r for r in wrong if r["dataset"] == ds and r["amb_bin"] == b and r["condition"] == "self"]
            W = [r for r in wrong if r["dataset"] == ds and r["amb_bin"] == b and r["condition"] == "weak_model"]
            if not S or not W:
                row += f"  {'-':<14}"
                continue
            s_far = sum(r["is_correct"] for r in S) / len(S)
            w_far = sum(r["is_correct"] for r in W) / len(W)
            d = w_far - s_far
            plot_rows[ds][b] = d
            row += f"  {d:+.3f}({len(S)},{len(W)})"
        print(row)

    # plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ds_list = sorted(plot_rows.keys())
    width = 0.8 / max(len(ds_list), 1)
    x = np.arange(len(BINS))
    for i, ds in enumerate(ds_list):
        ys = [plot_rows[ds].get(b, np.nan) for b in BINS]
        ax.bar(x + i * width, ys, width, label=ds)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x + (len(ds_list) - 1) * width / 2)
    ax.set_xticklabels([f"{b}\n(amb)" for b in BINS])
    ax.set_ylabel("FAR(WEAK) - FAR(SELF)")
    ax.set_title("Per-dataset WEAK-vs-SELF (anonymous-only prior)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = OUT_DIR / "sanity_B_per_dataset.png"
    fig.savefig(p, dpi=130)
    print(f"\n[B] saved {p}")


if __name__ == "__main__":
    records = load_all_critiques()
    print(f"loaded {len(records)} records")
    analysis_A(records)
    analysis_B(records)
