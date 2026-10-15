"""Solution length sensitivity."""

import json

import numpy as np
import pandas as pd

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
LOGS = REPO / "logs"
PAPER_TABLES = REPO / "paper" / "tables"

from identity_bias.analysis.extended.build_critique_tables import load_critique_records


def load_solution_lengths():
    """Return dict (solver, dataset, problem_id) -> char_length of CoT+answer."""
    out = {}
    for path in LOGS.rglob("solutions_*.jsonl"):
        stem = path.stem  # solutions_<solver>_<dataset>
        body = stem.replace("solutions_", "")
        if "_" not in body:
            continue
        solver, _, dataset = body.rpartition("_")
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") != "solution":
                    continue
                sol = r["solution"]
                pid = sol["problem_id"]
                cot = sol.get("chain_of_thought") or ""
                ans = sol.get("final_answer") or ""
                length = len(cot) + len(ans)
                out[(solver, dataset, pid)] = length
    return out


def main():
    print("loading critique records...")
    df = load_critique_records()
    sub = df[
        df["arm"].isin(["verdict_only_neutral", "verdict_plus_repair"])
        & df["condition"].eq("anonymous")
        & (df["actually_correct"] == False)
    ].copy()
    print(f"  {len(sub)} wrong/anon records in two key arms")

    print("loading solution lengths...")
    sl = load_solution_lengths()
    print(f"  {len(sl)} solutions catalogued")

    sub["sol_len_chars"] = sub.apply(
        lambda r: sl.get((r["solver_model"], r["dataset"], r["problem_id"])),
        axis=1,
    )
    n_missing = sub["sol_len_chars"].isna().sum()
    print(f"  {n_missing} records have no length match (dropped)")
    sub = sub.dropna(subset=["sol_len_chars"]).copy()
    sub["sol_len_chars"] = sub["sol_len_chars"].astype(int)

    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        cs = sub[sub["critic_model"] == critic]
        # quartiles per critic
        q = cs["sol_len_chars"].quantile([0.25, 0.5, 0.75]).to_list()

        def assign_q(x):
            return (
                "Q1"
                if x <= q[0]
                else "Q2"
                if x <= q[1]
                else "Q3"
                if x <= q[2]
                else "Q4"
            )

        cs = cs.assign(quartile=cs["sol_len_chars"].map(assign_q))
        for qname in ["Q1", "Q2", "Q3", "Q4"]:
            qsub = cs[cs["quartile"] == qname]
            piv = qsub.pivot_table(
                index="problem_id",
                columns="arm",
                values="is_correct",
                aggfunc="first",
            ).dropna()
            if len(piv) < 5:
                continue
            far_neutral = piv["verdict_only_neutral"].mean() * 100
            far_repair = piv["verdict_plus_repair"].mean() * 100
            diff = far_repair - far_neutral
            # paired bootstrap CI
            diffs = (
                piv["verdict_plus_repair"].astype(int)
                - piv["verdict_only_neutral"].astype(int)
            ).to_numpy()
            rng = np.random.default_rng(seed=42)
            B = 5000
            n = len(diffs)
            boots = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(B)])
            ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975]) * 100
            len_lo = qsub["sol_len_chars"].min()
            len_hi = qsub["sol_len_chars"].max()
            rows.append(
                {
                    "critic": critic,
                    "quartile": qname,
                    "n_items": len(piv),
                    "len_lo": int(len_lo),
                    "len_hi": int(len_hi),
                    "far_neutral_pct": round(far_neutral, 1),
                    "far_repair_pct": round(far_repair, 1),
                    "diff_pp": round(diff, 2),
                    "ci_lo_pp": round(ci_lo, 2),
                    "ci_hi_pp": round(ci_hi, 2),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(PAPER_TABLES / "revision_v3_length_quartiles.csv", index=False)
    print("\nrepair-vs-neutral FAR contrast by solution length quartile:")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
