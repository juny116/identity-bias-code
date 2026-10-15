"""Strong author label effects."""

import json
from collections import defaultdict

import matplotlib

from identity_bias.runtime.paths import workspace_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS_DIR = workspace_root() / "logs"
OUT_DIR = workspace_root() / "preliminary_out"
EXCLUDE_DATASETS = {"bbh"}
NEW_RUN = "gemma_strong_run_1"


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
                records.append(
                    {
                        "problem_id": crit["problem_id"],
                        "dataset": dataset,
                        "critic_model": crit["critic_model"],
                        "solver_model": crit["solver_model"],
                        "condition": crit["identity_condition"],
                        "is_correct": bool(crit["is_correct"]),
                        "actually_correct": bool(crit["actually_correct"]),
                        "run_tag": run_tag,
                    }
                )
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


def main():
    OUT_DIR.mkdir(exist_ok=True)

    records = load_all_critiques()
    print(f"loaded {len(records)} records total")
    gemma = [r for r in records if r["critic_model"] == "gemma-4"]
    print(f"gemma-4 records: {len(gemma)}")
    print(f"  conditions present: {sorted({r['condition'] for r in gemma})}")
    print(f"  datasets present:   {sorted({r['dataset'] for r in gemma})}")

    # ---------- 1. plain per-condition FAR (gemma only, wrong solutions) ----------
    print("\n=== [1] gemma-4 FAR by condition (wrong solutions only) ===")
    wrong_gemma = [r for r in gemma if not r["actually_correct"]]
    print(f"wrong-solution records: {len(wrong_gemma)}")
    by_cond = defaultdict(list)
    for r in wrong_gemma:
        by_cond[r["condition"]].append(r["is_correct"])
    for cond in [
        "anonymous",
        "human",
        "other_model",
        "self",
        "weak_model",
        "strong_model",
    ]:
        votes = by_cond.get(cond, [])
        if not votes:
            print(f"  {cond:<14} n=0")
            continue
        far = sum(votes) / len(votes)
        print(f"  {cond:<14} FAR={far:.3f}  n={len(votes)}")

    # ---------- 2. anonymous-only prior ambiguity, stratified FAR ----------
    # Use pooled records from ALL critics for the ambiguity prior so the
    # binning is comparable to prior preliminary analysis.
    anon = [r for r in records if r["condition"] == "anonymous"]
    per_item = defaultdict(list)
    for r in anon:
        key = (r["problem_id"], r["solver_model"], r["actually_correct"])
        per_item[key].append(r["is_correct"])
    amb_lookup = {
        key: min(sum(v) / len(v), 1 - sum(v) / len(v)) for key, v in per_item.items()
    }

    for r in records:
        key = (r["problem_id"], r["solver_model"], r["actually_correct"])
        r["amb_bin"] = bin_amb(amb_lookup.get(key))

    wrong = [
        r for r in records if not r["actually_correct"] and r["amb_bin"] is not None
    ]
    wrong_gemma = [r for r in wrong if r["critic_model"] == "gemma-4"]

    print("\n=== [2] gemma-4 FAR by condition x ambiguity bin ===")
    print(f"{'condition':<14}" + "".join(f"  {b:<11}" for b in BINS))
    cells = defaultdict(lambda: {"a": 0, "n": 0})
    for r in wrong_gemma:
        cells[(r["condition"], r["amb_bin"])]["a"] += int(r["is_correct"])
        cells[(r["condition"], r["amb_bin"])]["n"] += 1
    conds_order = [
        "anonymous",
        "human",
        "other_model",
        "self",
        "weak_model",
        "strong_model",
    ]
    for cond in conds_order:
        row = f"{cond:<14}"
        for b in BINS:
            c = cells[(cond, b)]
            row += f"  {c['a'] / c['n']:.3f}({c['n']:>3})" if c["n"] else f"  {'-':<11}"
        print(row)

    # ---------- 3. KEY: STRONG vs anonymous and WEAK vs anonymous deltas, gemma ----------
    print(
        "\n=== [3] FALSIFIER: gemma STRONG-anon and WEAK-anon FAR delta by ambiguity ==="
    )
    print("  Predicted: STRONG-anon > 0 (more over-acceptance) in HIGH bin")
    print("             WEAK-anon < 0 (less over-acceptance) in HIGH bin")
    print(f"{'comparison':<22}" + "".join(f"  {b:<14}" for b in BINS))
    deltas = {}
    for cond in ["weak_model", "strong_model"]:
        row = f"FAR({cond})-anon ".ljust(22)
        deltas[cond] = {}
        for b in BINS:
            anon_cell = cells[("anonymous", b)]
            cond_cell = cells[(cond, b)]
            if anon_cell["n"] == 0 or cond_cell["n"] == 0:
                row += f"  {'-':<14}"
                continue
            anon_far = anon_cell["a"] / anon_cell["n"]
            cond_far = cond_cell["a"] / cond_cell["n"]
            d = cond_far - anon_far
            deltas[cond][b] = d
            row += f"  {d:+.3f}({anon_cell['n']},{cond_cell['n']:>3})"
        print(row)

    # also pooled critics for context
    print("\n=== [4] pooled-critics WEAK/STRONG vs anon deltas (uses everything) ===")
    print(f"  STRONG only available on gemma; WEAK on all critics. This is for sanity.")
    pooled_cells = defaultdict(lambda: {"a": 0, "n": 0})
    for r in wrong:
        pooled_cells[(r["condition"], r["amb_bin"])]["a"] += int(r["is_correct"])
        pooled_cells[(r["condition"], r["amb_bin"])]["n"] += 1
    print(f"{'comparison':<22}" + "".join(f"  {b:<14}" for b in BINS))
    for cond in ["weak_model", "strong_model"]:
        row = f"FAR({cond})-anon ".ljust(22)
        for b in BINS:
            ac = pooled_cells[("anonymous", b)]
            cc = pooled_cells[(cond, b)]
            if ac["n"] == 0 or cc["n"] == 0:
                row += f"  {'-':<14}"
                continue
            d = (cc["a"] / cc["n"]) - (ac["a"] / ac["n"])
            row += f"  {d:+.3f}({ac['n']:>5},{cc['n']:>3})"
        print(row)

    # ---------- 5. plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: gemma FAR by condition x bin
    ax = axes[0]
    width = 0.8 / len(conds_order)
    x = np.arange(len(BINS))
    palette = {
        "anonymous": "#888",
        "human": "#999",
        "other_model": "#aaa",
        "self": "#666",
        "weak_model": "#1f77b4",
        "strong_model": "#d62728",
    }
    for i, cond in enumerate(conds_order):
        ys = []
        for b in BINS:
            c = cells[(cond, b)]
            ys.append(c["a"] / c["n"] if c["n"] else np.nan)
        ax.bar(x + i * width, ys, width, label=cond, color=palette.get(cond, None))
    ax.set_xticks(x + (len(conds_order) - 1) * width / 2)
    ax.set_xticklabels([f"{b}\n(amb)" for b in BINS])
    ax.set_ylabel("FAR on wrong solutions (gemma-4 only)")
    ax.set_title("Gemma-4: condition x ambiguity (AIME)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper right")

    # right: WEAK and STRONG vs anon deltas
    ax = axes[1]
    width = 0.35
    weak_ys = [deltas.get("weak_model", {}).get(b, np.nan) for b in BINS]
    strong_ys = [deltas.get("strong_model", {}).get(b, np.nan) for b in BINS]
    ax.bar(x - width / 2, weak_ys, width, label="weak - anon", color="#1f77b4")
    ax.bar(x + width / 2, strong_ys, width, label="strong - anon", color="#d62728")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(amb)" for b in BINS])
    ax.set_ylabel("FAR delta vs anonymous (gemma-4)")
    ax.set_title("Falsifier: STRONG should go OPPOSITE of WEAK (esp. high-amb)")
    ax.legend()
    plt.tight_layout()
    p = OUT_DIR / "falsifier_strong_gemma.png"
    fig.savefig(p, dpi=130)
    print(f"\nplot saved: {p}")


if __name__ == "__main__":
    main()
