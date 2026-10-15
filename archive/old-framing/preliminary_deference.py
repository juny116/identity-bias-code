"""
Preliminary deference-under-uncertainty analysis.

Tests the prediction (from research_direction.md round 2):
  - Identity labels barely shift critique behavior.
  - WEAK_MODEL framing reduces False-Accept-Rate (FAR) on wrong solutions.
  - The reduction should be largest on AMBIGUOUS items (high critic disagreement)
    and smallest on items the critic is confident about.

Pipeline:
  1. Load every identity_swap_*.jsonl under logs/ (all subdirs, all critic configs).
  2. For each problem, compute item-level ambiguity from critic accept rate.
  3. Restrict to actually-wrong solutions (where FAR is defined).
  4. Stratify FAR by (condition x ambiguity bin x critic_model).
  5. Print tables and write a PNG plot.

This is Day-0 evidence: zero new experiments, just re-cuts existing data.
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

# Drop BBH (too easy per project memory).
EXCLUDE_DATASETS = {"bbh"}


def load_all_critiques():
    """Walk logs/ for every identity_swap_*.jsonl, return flat list of critique dicts."""
    records = []
    for path in LOGS_DIR.rglob("identity_swap_*.jsonl"):
        # filename pattern: identity_swap_{critic}_on_{solver}_{dataset}.jsonl
        stem = path.stem.replace("identity_swap_", "")
        try:
            critic_solver, dataset = stem.rsplit("_", 1)
            # mmlu_pro is two tokens
            if dataset == "pro":
                critic_solver, _ = critic_solver.rsplit("_", 1)
                dataset = "mmlu_pro"
        except ValueError:
            continue
        if dataset in EXCLUDE_DATASETS:
            continue
        run_tag = path.parent.name
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
                    "is_correct": bool(crit["is_correct"]),       # critic verdict
                    "actually_correct": bool(crit["actually_correct"]),
                    "confidence": crit.get("confidence"),
                    "run_tag": run_tag,
                })
    return records


def compute_ambiguity(records):
    """
    Ambiguity = how often the pool of critic+condition+run combinations agreed
    on accept vs reject for a given problem (keyed by (problem_id, solver_model)).
    Closer to 0.5 accept-rate = more ambiguous.

    We compute it pooled across ALL conditions and critic configs. This is
    biased toward conservatism: condition-specific signal must overcome a
    regression-to-the-mean drag. If we still see an interaction, that's robust.
    """
    per_item = defaultdict(list)
    for r in records:
        key = (r["problem_id"], r["solver_model"], r["actually_correct"])
        per_item[key].append(r["is_correct"])

    ambiguity = {}
    for key, votes in per_item.items():
        accept_rate = sum(votes) / len(votes)
        ambiguity[key] = {
            "accept_rate": accept_rate,
            "n_votes": len(votes),
            # distance from edges: 0 (unanimous) .. 0.5 (perfect split)
            "ambiguity": min(accept_rate, 1 - accept_rate),
        }
    return ambiguity


def bin_ambiguity(a):
    """3-bin discretization on the [0, 0.5] ambiguity scale."""
    if a < 0.15:
        return "low"      # near-unanimous critic vote
    if a < 0.35:
        return "med"
    return "high"         # critics genuinely split


def main():
    records = load_all_critiques()
    print(f"loaded {len(records)} critique records")
    if not records:
        return

    ambiguity = compute_ambiguity(records)

    # attach ambiguity bin to each record
    for r in records:
        key = (r["problem_id"], r["solver_model"], r["actually_correct"])
        info = ambiguity.get(key)
        r["accept_rate_overall"] = info["accept_rate"] if info else None
        r["ambiguity"] = info["ambiguity"] if info else None
        r["amb_bin"] = bin_ambiguity(info["ambiguity"]) if info else None

    # filter to wrong solutions only (FAR is defined on those)
    wrong = [r for r in records if not r["actually_correct"] and r["amb_bin"] is not None]
    print(f"wrong-solution records: {len(wrong)}")

    # ---------- 1. condition x ambiguity_bin FAR (pooled across critic models) ----------
    cells = defaultdict(lambda: {"accept": 0, "n": 0})
    for r in wrong:
        cells[(r["condition"], r["amb_bin"])]["accept"] += int(r["is_correct"])
        cells[(r["condition"], r["amb_bin"])]["n"] += 1

    conditions = sorted({r["condition"] for r in wrong})
    bins = ["low", "med", "high"]

    print("\n=== FAR on wrong solutions: condition x ambiguity (pooled critics) ===")
    header = f"{'condition':<14}" + "".join(f"  {b:<10}" for b in bins)
    print(header)
    table_rows = []
    for cond in conditions:
        row = f"{cond:<14}"
        for b in bins:
            c = cells[(cond, b)]
            if c["n"] == 0:
                row += f"  {'-':<10}"
            else:
                far = c["accept"] / c["n"]
                row += f"  {far:.3f}({c['n']:>4})"
        print(row)
        table_rows.append((cond, [(cells[(cond, b)]['accept'], cells[(cond, b)]['n']) for b in bins]))

    # ---------- 2. WEAK vs SELF delta by bin, per critic model ----------
    print("\n=== WEAK_MODEL minus SELF FAR delta, by critic_model x ambiguity ===")
    print(f"  (negative = WEAK reduces over-acceptance vs SELF; this is the predicted direction)")
    critic_models = sorted({r["critic_model"] for r in wrong})
    deltas_for_plot = defaultdict(dict)
    header = f"{'critic':<22}" + "".join(f"  {b:<14}" for b in bins)
    print(header)
    for cm in critic_models:
        row = f"{cm:<22}"
        for b in bins:
            self_records = [r for r in wrong if r["critic_model"] == cm and r["amb_bin"] == b and r["condition"] == "self"]
            weak_records = [r for r in wrong if r["critic_model"] == cm and r["amb_bin"] == b and r["condition"] == "weak_model"]
            if not self_records or not weak_records:
                row += f"  {'-':<14}"
                continue
            self_far = sum(r["is_correct"] for r in self_records) / len(self_records)
            weak_far = sum(r["is_correct"] for r in weak_records) / len(weak_records)
            delta = weak_far - self_far
            deltas_for_plot[cm][b] = (delta, len(self_records), len(weak_records))
            row += f"  {delta:+.3f}({len(self_records)},{len(weak_records)})"
        print(row)

    # ---------- 3. plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: condition x ambiguity FAR
    ax = axes[0]
    width = 0.8 / len(conditions)
    x = np.arange(len(bins))
    for i, cond in enumerate(conditions):
        ys = []
        for b in bins:
            c = cells[(cond, b)]
            ys.append(c["accept"] / c["n"] if c["n"] else np.nan)
        ax.bar(x + i * width, ys, width, label=cond)
    ax.set_xticks(x + (len(conditions) - 1) * width / 2)
    ax.set_xticklabels([f"{b}\n(amb)" for b in bins])
    ax.set_ylabel("False-Accept Rate (wrong solutions)")
    ax.set_title("FAR by condition x item ambiguity (all critic models pooled)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 1)

    # right: WEAK-SELF delta by critic
    ax = axes[1]
    cm_list = sorted(deltas_for_plot.keys())
    width = 0.8 / max(len(cm_list), 1)
    for i, cm in enumerate(cm_list):
        ys = []
        for b in bins:
            v = deltas_for_plot[cm].get(b)
            ys.append(v[0] if v else np.nan)
        ax.bar(x + i * width, ys, width, label=cm)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x + (len(cm_list) - 1) * width / 2)
    ax.set_xticklabels([f"{b}\n(amb)" for b in bins])
    ax.set_ylabel("FAR(WEAK_MODEL) - FAR(SELF)")
    ax.set_title("Predicted: most negative in HIGH-ambiguity bin")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / "preliminary_deference.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nplot saved: {out_path}")

    # ---------- 4. sanity numbers ----------
    print("\n=== sanity ===")
    print(f"unique problems: {len({r['problem_id'] for r in records})}")
    print(f"unique critic_models: {sorted({r['critic_model'] for r in records})}")
    print(f"unique conditions: {sorted({r['condition'] for r in records})}")
    print(f"unique datasets: {sorted({r['dataset'] for r in records})}")
    print(f"ambiguity bin counts (wrong-only): " +
          ", ".join(f"{b}={sum(1 for r in wrong if r['amb_bin'] == b)}" for b in bins))


if __name__ == "__main__":
    main()
