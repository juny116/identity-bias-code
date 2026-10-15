"""Frozen artifact generator.

Reads all critique logs (Caveat A ablation + gated mitigation + existing
identity-swap sweeps) and produces the tables and figures specified in
paper/figure_spec.md. Outputs go to paper/tables/ as CSV and paper/figures/
as PNG. A consolidated JSON of headline numbers is written to
paper/results_numbers.json so the paper LaTeX TODOs can be filled in.

Key statistical machinery:
  - paired_bootstrap_diff_ci: paired bootstrap over problem_id, 10k resamples
  - per-critic FAR / FRR aggregations on actually-wrong / actually-correct items
  - WEAK-vs-anonymous and STRONG-vs-anonymous deltas with paired CIs
  - Mixed-effects logistic regression deferred to a separate appendix step
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- repo paths ---
REPO = Path(__file__).parent.parent
LOGS = REPO / "logs"
PAPER_FIGS = REPO / "paper" / "figures"
PAPER_TABLES = REPO / "paper" / "tables"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)
PAPER_TABLES.mkdir(parents=True, exist_ok=True)

# Make identity_bias importable for check_answer
sys.path.insert(0, str(REPO))
from identity_bias.config import Dataset
from identity_bias.data import check_answer


# --- data loaders ---

def parse_critic_solver_dataset(stem: str):
    """identity_swap_<critic>_on_<solver>_<dataset>[__<arm>]"""
    body = stem.replace("identity_swap_", "")
    arm = None
    if "__" in body:
        body, arm = body.split("__", 1)
    parts = body.rsplit("_", 1)
    if len(parts) != 2:
        return None
    critic_solver, dataset = parts
    if dataset == "pro":
        critic_solver, _ = critic_solver.rsplit("_", 1)
        dataset = "mmlu_pro"
    if "_on_" in critic_solver:
        critic, solver = critic_solver.split("_on_", 1)
    else:
        critic = solver = critic_solver
    return critic, solver, dataset, arm


def load_critique_records(roots=("caveat_a_run_1", "gemma_strong_run_1",
                                  "verdict_repair_gpt_oss_no_think_run_1",
                                  "verdict_repair_qwen3_no_think_run_1"),
                          include_legacy_identity=True):
    """Load critique records as a long-format pandas DataFrame.

    `arm` is the Caveat A prompt-interface arm when applicable;
    otherwise 'legacy_full' for the original full-call critic runs
    (identity-swap sweeps) and 'legacy_verdict_only' / 'legacy_full_two_stage'
    for Phase E two-stage records (which logged both stages).
    """
    rows = []
    # Caveat A and other __arm-tagged log dirs
    for root in roots:
        d = LOGS / root
        if not d.exists():
            continue
        for path in d.glob("identity_swap_*.jsonl"):
            parsed = parse_critic_solver_dataset(path.stem)
            if parsed is None:
                continue
            critic, solver, dataset, arm = parsed
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("type") != "critique":
                        continue
                    c = r["critique"]
                    rows.append({
                        "log_dir": root,
                        "problem_id": c["problem_id"],
                        "dataset": dataset,
                        "critic_model": c["critic_model"],
                        "solver_model": c["solver_model"],
                        "condition": c["identity_condition"],
                        "arm": arm if arm is not None else "legacy_full",
                        "is_correct": c.get("is_correct"),
                        "corrected_answer": c.get("corrected_answer"),
                        "confidence": c.get("confidence"),
                        "actually_correct": c["actually_correct"],
                        "initial_is_correct": c.get("initial_is_correct"),
                        "initial_error_description": c.get("initial_error_description"),
                        "initial_confidence": c.get("initial_confidence"),
                    })

    if include_legacy_identity:
        # Phase A/B identity-swap sweeps (legacy_full, no two-stage)
        for path in LOGS.rglob("identity_swap_*.jsonl"):
            # skip files we already loaded above
            if any(str(path).startswith(str(LOGS / r)) for r in roots):
                continue
            parsed = parse_critic_solver_dataset(path.stem)
            if parsed is None:
                continue
            critic, solver, dataset, _arm = parsed
            if dataset == "bbh":  # dropped from project
                continue
            with open(path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("type") != "critique":
                        continue
                    c = r["critique"]
                    rows.append({
                        "log_dir": path.parent.name,
                        "problem_id": c["problem_id"],
                        "dataset": dataset,
                        "critic_model": c["critic_model"],
                        "solver_model": c["solver_model"],
                        "condition": c["identity_condition"],
                        "arm": "legacy_full",
                        "is_correct": c.get("is_correct"),
                        "corrected_answer": c.get("corrected_answer"),
                        "confidence": c.get("confidence"),
                        "actually_correct": c["actually_correct"],
                        "initial_is_correct": c.get("initial_is_correct"),
                        "initial_error_description": c.get("initial_error_description"),
                        "initial_confidence": c.get("initial_confidence"),
                    })

    df = pd.DataFrame(rows)
    # drop records with None is_correct (parse failures)
    df = df[df["is_correct"].notna()].copy()
    df["is_correct"] = df["is_correct"].astype(bool)
    df["actually_correct"] = df["actually_correct"].astype(bool)
    return df


def load_gated_stage2(roots=("gated_run_1",)):
    """Stage-2 repair-only outputs for the gated pipeline."""
    rows = []
    for root in roots:
        d = LOGS / root
        if not d.exists():
            continue
        for path in d.glob("gated_stage2_*.jsonl"):
            # filename: gated_stage2_<critic>_<solver>_<dataset>__<condition>.jsonl
            stem = path.stem.replace("gated_stage2_", "")
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
                    rows.append({
                        "problem_id": r["problem_id"],
                        "dataset": dataset,
                        "critic_model": r["critic_model"],
                        "solver_model": r["solver_model"],
                        "condition": r["identity_condition"],
                        "stage1_is_correct": r.get("stage1_is_correct"),
                        "actually_correct": r["actually_correct"],
                        "corrected_answer": r.get("corrected_answer"),
                    })
    df = pd.DataFrame(rows)
    return df


def load_ground_truth():
    """Pull ground_truth from one of the solutions files per dataset, keyed by
    problem_id. Used to score corrected_answer outputs."""
    gt = {}
    for path in LOGS.rglob("solutions_*.jsonl"):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("type") != "solution":
                    continue
                p = r["problem"]
                if p["id"] not in gt:
                    gt[p["id"]] = (p.get("dataset"), p.get("ground_truth"))
    return gt


# --- stats utilities ---

def paired_bootstrap_diff_ci(df, group_col, a_label, b_label, value_col,
                             n_resamples=10000, conf=0.95, rng=None):
    """Paired bootstrap of (mean(value_col | group=a) - mean(value_col | group=b))
    where rows are aligned by problem_id (caller must pivot so each problem_id
    has both labels present). Returns (delta, lo, hi).

    Input df should have columns: problem_id, group_col, value_col.
    """
    pivoted = (df.pivot_table(index="problem_id",
                              columns=group_col, values=value_col,
                              aggfunc="mean")  # mean across replicates
                .dropna(subset=[a_label, b_label]))
    if pivoted.empty:
        return float("nan"), float("nan"), float("nan"), 0
    a = pivoted[a_label].to_numpy(dtype=float)
    b = pivoted[b_label].to_numpy(dtype=float)
    delta = float((a - b).mean())
    n = len(pivoted)
    rng = rng or np.random.default_rng(42)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = (a[idx] - b[idx]).mean()
    lo = float(np.percentile(boots, (1 - conf) / 2 * 100))
    hi = float(np.percentile(boots, (1 + conf) / 2 * 100))
    return delta, lo, hi, n


# --- analysis functions ---

def compute_caveat_a_table(df):
    """Table 2: Caveat A FAR by arm × critic (pooled), with paired CIs vs neutral."""
    arms = ["verdict_only_instructed", "verdict_only_neutral",
            "verdict_schema_matched", "verdict_plus_repair"]
    sub = df[(df["arm"].isin(arms)) & (~df["actually_correct"])
             & (df["condition"] == "anonymous")].copy()
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        cs = sub[sub["critic_model"] == critic]
        row = {"critic_model": critic}
        for arm in arms:
            arm_data = cs[cs["arm"] == arm]
            n = len(arm_data)
            far = arm_data["is_correct"].mean() if n else float("nan")
            row[f"FAR_{arm}"] = far
            row[f"n_{arm}"] = n
        # paired delta vs neutral
        for arm in ["verdict_only_instructed", "verdict_schema_matched", "verdict_plus_repair"]:
            piv = cs[cs["arm"].isin([arm, "verdict_only_neutral"])][
                ["problem_id", "arm", "is_correct"]
            ].copy()
            piv["accept_int"] = piv["is_correct"].astype(int)
            delta, lo, hi, n = paired_bootstrap_diff_ci(
                piv, "arm", arm, "verdict_only_neutral", "accept_int"
            )
            row[f"delta_{arm}_vs_neutral_pp"] = delta * 100
            row[f"CI_low_{arm}_vs_neutral_pp"] = lo * 100
            row[f"CI_high_{arm}_vs_neutral_pp"] = hi * 100
            row[f"npaired_{arm}_vs_neutral"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def compute_caveat_a_per_dataset(df):
    """Per-dataset Caveat A delta_repair_vs_neutral_pp (for appendix figure A1)."""
    arms = ["verdict_only_neutral", "verdict_plus_repair"]
    sub = df[(df["arm"].isin(arms)) & (~df["actually_correct"])
             & (df["condition"] == "anonymous")].copy()
    sub["accept_int"] = sub["is_correct"].astype(int)
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        for ds in sorted(sub["dataset"].unique()):
            cs = sub[(sub["critic_model"] == critic) & (sub["dataset"] == ds)]
            if cs.empty:
                continue
            delta, lo, hi, n = paired_bootstrap_diff_ci(
                cs[["problem_id", "arm", "accept_int"]],
                "arm", "verdict_plus_repair", "verdict_only_neutral", "accept_int"
            )
            rows.append({
                "critic_model": critic, "dataset": ds,
                "delta_pp": delta * 100, "CI_low_pp": lo * 100, "CI_high_pp": hi * 100,
                "n_paired": n,
            })
    return pd.DataFrame(rows)


def compute_baseline_table(df):
    """Table 1: baseline FAR (wrong) and FRR (correct) by critic × dataset × arm."""
    arms = ["legacy_full", "verdict_only_instructed", "verdict_only_neutral",
            "verdict_plus_repair"]
    rows = []
    for critic in sorted(df["critic_model"].unique()):
        for ds in sorted(df["dataset"].unique()):
            for arm in arms:
                slice_anon = df[(df["critic_model"] == critic) & (df["dataset"] == ds)
                                & (df["arm"] == arm) & (df["condition"] == "anonymous")]
                if slice_anon.empty:
                    continue
                wrong = slice_anon[~slice_anon["actually_correct"]]
                correct = slice_anon[slice_anon["actually_correct"]]
                if len(wrong) < 5 and len(correct) < 5:
                    continue
                rows.append({
                    "critic": critic, "dataset": ds, "interface": arm,
                    "n_wrong": len(wrong),
                    "FAR": wrong["is_correct"].mean() if len(wrong) else float("nan"),
                    "n_correct": len(correct),
                    "FRR": (~correct["is_correct"]).mean() if len(correct) else float("nan"),
                })
    return pd.DataFrame(rows)


def compute_phase_e_paired(df):
    """Figure 1 + Table 4 data. Phase E two-stage records have initial_is_correct
    AND is_correct on the same record (initial = verdict-only, final = full).
    Compute paired FAR_initial vs FAR_final per critic, pooled across datasets+conditions."""
    sub = df[(df["initial_is_correct"].notna()) & (~df["actually_correct"])].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["initial_accept_int"] = sub["initial_is_correct"].astype(bool).astype(int)
    sub["final_accept_int"] = sub["is_correct"].astype(int)
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        cs = sub[sub["critic_model"] == critic]
        n = len(cs)
        far_init = cs["initial_accept_int"].mean()
        far_final = cs["final_accept_int"].mean()
        # paired per problem_id+condition: use (init - final) at item level
        # paired bootstrap over (problem_id, condition) clusters
        keys = list(zip(cs["problem_id"], cs["condition"]))
        deltas = (cs["final_accept_int"] - cs["initial_accept_int"]).to_numpy(dtype=float)
        n_b = len(deltas)
        rng = np.random.default_rng(42)
        boots = np.empty(10000)
        for i in range(10000):
            idx = rng.integers(0, n_b, size=n_b)
            boots[i] = deltas[idx].mean()
        delta_mean = float(deltas.mean())
        lo = float(np.percentile(boots, 2.5))
        hi = float(np.percentile(boots, 97.5))
        rows.append({"critic_model": critic, "n": n,
                     "FAR_initial": far_init, "FAR_final": far_final,
                     "delta_full_minus_initial_pp": delta_mean * 100,
                     "CI_low_pp": lo * 100, "CI_high_pp": hi * 100})
    return pd.DataFrame(rows)


def compute_identity_5critic_table(df):
    """Table 3: identity-condition FAR across 5 critics (legacy_full only)."""
    sub = df[(df["arm"] == "legacy_full") & (~df["actually_correct"])].copy()
    sub["accept_int"] = sub["is_correct"].astype(int)
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        cs = sub[sub["critic_model"] == critic]
        if cs.empty: continue
        # matched datasets across conditions
        for cond in sorted(cs["condition"].unique()):
            cond_data = cs[cs["condition"] == cond]
            n = len(cond_data)
            if n < 5: continue
            far = cond_data["accept_int"].mean()
            row = {"critic_model": critic, "condition": cond, "n_wrong": n, "FAR": far}
            # delta vs anonymous, paired
            if cond != "anonymous":
                piv = cs[cs["condition"].isin([cond, "anonymous"])][
                    ["problem_id", "condition", "accept_int"]
                ]
                delta, lo, hi, n_pair = paired_bootstrap_diff_ci(
                    piv, "condition", cond, "anonymous", "accept_int"
                )
                row.update({"delta_vs_anon_pp": delta * 100,
                            "CI_low_pp": lo * 100, "CI_high_pp": hi * 100,
                            "n_paired": n_pair})
            rows.append(row)
    return pd.DataFrame(rows)


def compute_weak_strong_asymmetry(df):
    """Figure 4 / Table for WEAK and STRONG vs anonymous deltas, per critic.
    Uses legacy_full data for the 5-critic comparison."""
    sub = df[(df["arm"] == "legacy_full") & (~df["actually_correct"])].copy()
    sub["accept_int"] = sub["is_correct"].astype(int)
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        cs = sub[sub["critic_model"] == critic]
        if cs.empty: continue
        for tgt in ("weak_model", "strong_model"):
            piv = cs[cs["condition"].isin([tgt, "anonymous"])][
                ["problem_id", "condition", "accept_int"]
            ]
            delta, lo, hi, n = paired_bootstrap_diff_ci(
                piv, "condition", tgt, "anonymous", "accept_int"
            )
            if n == 0:
                continue
            rows.append({"critic_model": critic, "contrast": f"{tgt}_minus_anon",
                         "delta_pp": delta * 100, "CI_low_pp": lo * 100,
                         "CI_high_pp": hi * 100, "n_paired": n})
    return pd.DataFrame(rows)


def compute_gated_outcomes(df_caveat, df_gated, ground_truth):
    """Table 5: gated-pipeline-outcome-breakdown.

    Combines:
      - Caveat A arm=verdict_only_neutral as stage-1 verdict
      - Gated stage-2 repair output for items where stage-1 rejected
      - Caveat A arm=verdict_plus_repair as the one-shot pipeline comparison
    """
    rows = []
    for critic in sorted(df_gated["critic_model"].unique()):
        for cond in sorted(df_gated[df_gated["critic_model"] == critic]["condition"].unique()):
            for ds in sorted(df_gated[df_gated["critic_model"] == critic]["dataset"].unique()):
                # gated pipeline: stage-1 from caveat A neutral
                stage1 = df_caveat[(df_caveat["critic_model"] == critic)
                                   & (df_caveat["arm"] == "verdict_only_neutral")
                                   & (df_caveat["condition"] == cond)
                                   & (df_caveat["dataset"] == ds)]
                stage2 = df_gated[(df_gated["critic_model"] == critic)
                                  & (df_gated["condition"] == cond)
                                  & (df_gated["dataset"] == ds)]
                oneshot = df_caveat[(df_caveat["critic_model"] == critic)
                                    & (df_caveat["arm"] == "verdict_plus_repair")
                                    & (df_caveat["condition"] == cond)
                                    & (df_caveat["dataset"] == ds)]
                if stage1.empty:
                    continue

                # build per-item outcomes
                ds_enum = Dataset(ds)
                def score(pid, ans):
                    if pid not in ground_truth or ans is None:
                        return None
                    _, gt = ground_truth[pid]
                    try:
                        return check_answer(ds_enum, ans, gt)
                    except Exception:
                        return None

                # gated outcomes
                wrong = stage1[~stage1["actually_correct"]]
                correct = stage1[stage1["actually_correct"]]
                stage2_by_pid = stage2.set_index("problem_id")["corrected_answer"].to_dict()

                wrong_accept = ((wrong["is_correct"] == True)).sum()  # stage1 accepted
                wrong_rej = wrong[wrong["is_correct"] == False]
                wrong_rep_ok = 0; wrong_rep_bad = 0
                for _, r in wrong_rej.iterrows():
                    ans = stage2_by_pid.get(r["problem_id"])
                    s = score(r["problem_id"], ans)
                    if s is True: wrong_rep_ok += 1
                    elif s is False: wrong_rep_bad += 1
                    # else parse failure → counted as bad
                    else: wrong_rep_bad += 1
                n_wrong = len(wrong)

                correct_accept = (correct["is_correct"] == True).sum()
                correct_rej = correct[correct["is_correct"] == False]
                correct_false_rej = 0; correct_harm = 0
                for _, r in correct_rej.iterrows():
                    ans = stage2_by_pid.get(r["problem_id"])
                    s = score(r["problem_id"], ans)
                    if s is False:
                        correct_harm += 1
                    correct_false_rej += 1
                n_correct = len(correct)

                # one-shot pipeline: from caveat A verdict_plus_repair
                wrong1 = oneshot[~oneshot["actually_correct"]]
                correct1 = oneshot[oneshot["actually_correct"]]
                onesh_wrong_accept = (wrong1["is_correct"] == True).sum()
                onesh_wrong_rep_ok = 0; onesh_wrong_rep_bad = 0
                for _, r in wrong1[wrong1["is_correct"] == False].iterrows():
                    s = score(r["problem_id"], r["corrected_answer"])
                    if s is True: onesh_wrong_rep_ok += 1
                    elif s is False: onesh_wrong_rep_bad += 1
                    else: onesh_wrong_rep_bad += 1
                onesh_correct_accept = (correct1["is_correct"] == True).sum()
                onesh_correct_harm = 0
                for _, r in correct1[correct1["is_correct"] == False].iterrows():
                    s = score(r["problem_id"], r["corrected_answer"])
                    if s is False:
                        onesh_correct_harm += 1
                onesh_n_wrong = len(wrong1)
                onesh_n_correct = len(correct1)

                rows.append({
                    "critic": critic, "condition": cond, "dataset": ds,
                    "pipeline": "gated",
                    "n_wrong": n_wrong,
                    "wrong_accepted_rate": wrong_accept / n_wrong if n_wrong else float("nan"),
                    "wrong_repaired_correct_rate": wrong_rep_ok / n_wrong if n_wrong else float("nan"),
                    "wrong_repaired_incorrect_rate": wrong_rep_bad / n_wrong if n_wrong else float("nan"),
                    "n_correct": n_correct,
                    "correct_accepted_rate": correct_accept / n_correct if n_correct else float("nan"),
                    "false_rejected_rate": correct_false_rej / n_correct if n_correct else float("nan"),
                    "harmful_correction_rate": correct_harm / n_correct if n_correct else float("nan"),
                })
                rows.append({
                    "critic": critic, "condition": cond, "dataset": ds,
                    "pipeline": "oneshot",
                    "n_wrong": onesh_n_wrong,
                    "wrong_accepted_rate": onesh_wrong_accept / onesh_n_wrong if onesh_n_wrong else float("nan"),
                    "wrong_repaired_correct_rate": onesh_wrong_rep_ok / onesh_n_wrong if onesh_n_wrong else float("nan"),
                    "wrong_repaired_incorrect_rate": onesh_wrong_rep_bad / onesh_n_wrong if onesh_n_wrong else float("nan"),
                    "n_correct": onesh_n_correct,
                    "correct_accepted_rate": onesh_correct_accept / onesh_n_correct if onesh_n_correct else float("nan"),
                    "false_rejected_rate": float("nan"),  # uses one-shot reject == verdict reject
                    "harmful_correction_rate": onesh_correct_harm / onesh_n_correct if onesh_n_correct else float("nan"),
                })
    return pd.DataFrame(rows)


# --- main ---

def main():
    print("loading critique records...")
    df = load_critique_records()
    print(f"  {len(df)} critique records loaded")
    print(f"  critics: {sorted(df['critic_model'].unique())}")
    print(f"  arms: {sorted(df['arm'].unique())}")

    print("\nloading gated stage-2 records...")
    df_gated = load_gated_stage2()
    print(f"  {len(df_gated)} gated stage-2 records loaded")

    print("\nloading ground truth...")
    gt = load_ground_truth()
    print(f"  {len(gt)} problems with ground truth")

    # --- frozen tables ---
    out = {}

    print("\n[Table 2] caveat-A-FAR-by-arm-per-critic ...")
    tbl_caveat = compute_caveat_a_table(df)
    tbl_caveat.to_csv(PAPER_TABLES / "table2_caveat_a_arm_per_critic.csv", index=False)
    print(tbl_caveat.to_string(index=False))
    out["table2_caveat_a"] = tbl_caveat.to_dict(orient="records")

    print("\n[Appendix A1] per-dataset Caveat A repair-vs-neutral delta ...")
    tbl_caveat_ds = compute_caveat_a_per_dataset(df)
    tbl_caveat_ds.to_csv(PAPER_TABLES / "appendix_A1_caveat_a_per_dataset.csv", index=False)
    print(tbl_caveat_ds.to_string(index=False))
    out["appendix_A1_caveat_a_per_dataset"] = tbl_caveat_ds.to_dict(orient="records")

    print("\n[Table 1] baseline FAR/FRR by critic × dataset × interface ...")
    tbl_baseline = compute_baseline_table(df)
    tbl_baseline.to_csv(PAPER_TABLES / "table1_baseline.csv", index=False)
    print(f"  ({len(tbl_baseline)} rows)")
    out["table1_baseline"] = tbl_baseline.to_dict(orient="records")

    print("\n[Figure 1 / Table 4] Phase E paired verdict-vs-full FAR ...")
    tbl_phaseE = compute_phase_e_paired(df)
    tbl_phaseE.to_csv(PAPER_TABLES / "table4_phaseE_paired_verdict_vs_full.csv", index=False)
    print(tbl_phaseE.to_string(index=False))
    out["table4_phaseE"] = tbl_phaseE.to_dict(orient="records")

    print("\n[Table 3] identity-condition FAR across 5 critics ...")
    tbl_identity = compute_identity_5critic_table(df)
    tbl_identity.to_csv(PAPER_TABLES / "table3_identity_5critics.csv", index=False)
    print(f"  ({len(tbl_identity)} rows)")
    out["table3_identity"] = tbl_identity.to_dict(orient="records")

    print("\n[Figure 4 data] WEAK / STRONG asymmetry per critic ...")
    tbl_asym = compute_weak_strong_asymmetry(df)
    tbl_asym.to_csv(PAPER_TABLES / "figure4_weak_strong_asymmetry.csv", index=False)
    print(tbl_asym.to_string(index=False))
    out["figure4_asymmetry"] = tbl_asym.to_dict(orient="records")

    print("\n[Table 5] gated pipeline outcome breakdown ...")
    df_caveat_a_only = df[df["log_dir"] == "caveat_a_run_1"]
    tbl_gated = compute_gated_outcomes(df_caveat_a_only, df_gated, gt)
    tbl_gated.to_csv(PAPER_TABLES / "table5_gated_outcomes.csv", index=False)
    print(f"  ({len(tbl_gated)} rows)")
    out["table5_gated"] = tbl_gated.to_dict(orient="records")

    # --- headline numbers JSON for paper TODOs ---
    headline = {}
    # primary: verdict_plus_repair vs verdict_only_neutral per critic
    for r in out["table2_caveat_a"]:
        c = r["critic_model"]
        headline[f"{c}_repair_vs_neutral_pp"] = r["delta_verdict_plus_repair_vs_neutral_pp"]
        headline[f"{c}_repair_vs_neutral_CI_low"] = r["CI_low_verdict_plus_repair_vs_neutral_pp"]
        headline[f"{c}_repair_vs_neutral_CI_high"] = r["CI_high_verdict_plus_repair_vs_neutral_pp"]
        headline[f"{c}_instructed_vs_neutral_pp"] = r["delta_verdict_only_instructed_vs_neutral_pp"]
        headline[f"{c}_schema_vs_neutral_pp"] = r["delta_verdict_schema_matched_vs_neutral_pp"]

    # pooled headline across the two no-think self-critique critics + gemma
    pooled = {"critics": list(set(r["critic_model"] for r in out["table2_caveat_a"]))}
    pooled["mean_repair_vs_neutral_pp"] = float(
        np.mean([r["delta_verdict_plus_repair_vs_neutral_pp"] for r in out["table2_caveat_a"]])
    )
    pooled["mean_instructed_vs_neutral_pp"] = float(
        np.mean([r["delta_verdict_only_instructed_vs_neutral_pp"] for r in out["table2_caveat_a"]])
    )
    headline["pooled"] = pooled

    with open(REPO / "paper" / "results_numbers.json", "w") as f:
        json.dump({"headline": headline, "full_tables": out}, f, indent=2, default=str)
    print(f"\nresults_numbers.json written ({(REPO / 'paper' / 'results_numbers.json').stat().st_size} bytes)")


if __name__ == "__main__":
    main()
