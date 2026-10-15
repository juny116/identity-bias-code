"""Two additional appendix tables from Codex's brainstorm:
  1. Per-dataset repair-quality breakdown (which datasets have bad repair?)
  2. Macro (equal-dataset-weighted) summary for repair-quality and calibration
     metrics, to address reviewer concern that pooled metrics may be
     dominated by large legacy datasets.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
PAPER_TABLES = REPO / "paper" / "tables"

sys.path.insert(0, str(REPO))
from identity_bias.config import Dataset
from identity_bias.data import check_answer
from analysis.freeze_artifacts import load_critique_records, load_ground_truth


def score(ds_enum, pid, ans, gt_by_pid):
    if pid not in gt_by_pid or ans is None:
        return None
    _, gt = gt_by_pid[pid]
    try:
        return check_answer(ds_enum, ans, gt)
    except Exception:
        return None


def per_dataset_repair_quality(df, gt):
    """Repair-success and harmful-correction rates per critic × arm × dataset.
    For rejected wrong items, success = corrected_answer matches GT.
    For rejected correct items, harmful = corrected_answer differs from GT."""
    arms = ["legacy_full", "verdict_plus_repair", "verdict_plus_repair_reversed"]
    sub = df[df["arm"].isin(arms) & df["corrected_answer"].notna()].copy()
    rows = []
    for critic in sorted(sub["critic_model"].unique()):
        for arm in arms:
            for ds in sorted(sub["dataset"].unique()):
                slc = sub[(sub["critic_model"] == critic) & (sub["arm"] == arm)
                          & (sub["dataset"] == ds)]
                if slc.empty:
                    continue
                wrong = slc[~slc["actually_correct"]]
                correct = slc[slc["actually_correct"]]
                rej_wrong = wrong[~wrong["is_correct"]]
                rep_success = 0; rep_fail = 0
                for _, r in rej_wrong.iterrows():
                    ds_enum = Dataset(r["dataset"])
                    s = score(ds_enum, r["problem_id"], r["corrected_answer"], gt)
                    if s is True: rep_success += 1
                    elif s is False: rep_fail += 1
                    else: rep_fail += 1  # parse fail counts as fail
                rej_correct = correct[~correct["is_correct"]]
                harm = 0
                for _, r in rej_correct.iterrows():
                    ds_enum = Dataset(r["dataset"])
                    s = score(ds_enum, r["problem_id"], r["corrected_answer"], gt)
                    if s is False: harm += 1
                if len(rej_wrong) == 0 and len(rej_correct) == 0:
                    continue
                rows.append({
                    "critic": critic, "arm": arm, "dataset": ds,
                    "n_rejected_wrong": len(rej_wrong),
                    "repair_success_rate": rep_success / len(rej_wrong) if len(rej_wrong) else float("nan"),
                    "n_rejected_correct": len(rej_correct),
                    "harmful_correction_rate": harm / len(rej_correct) if len(rej_correct) else float("nan"),
                })
    return pd.DataFrame(rows)


def macro_summary(df_pdq, df_calib):
    """Equal-dataset-weighted (macro) averages of repair-quality and calibration
    metrics. Addresses reviewer concern that pooled values are dominated by
    large legacy datasets."""
    rows = []
    # repair quality
    for critic in sorted(df_pdq["critic"].unique()):
        for arm in sorted(df_pdq[df_pdq["critic"] == critic]["arm"].unique()):
            sub = df_pdq[(df_pdq["critic"] == critic) & (df_pdq["arm"] == arm)]
            if sub.empty:
                continue
            macro_success = sub["repair_success_rate"].mean()
            macro_harm = sub["harmful_correction_rate"].mean(skipna=True)
            rows.append({
                "critic": critic, "arm": arm, "metric": "macro_repair_success",
                "value": macro_success, "n_datasets": len(sub),
            })
            rows.append({
                "critic": critic, "arm": arm, "metric": "macro_harmful_correction",
                "value": macro_harm, "n_datasets": len(sub),
            })

    return pd.DataFrame(rows)


def calibration_per_dataset(df):
    """ECE per (critic, arm, dataset)."""
    sub = df[df["confidence"].notna()].copy()
    sub["verdict_correct"] = (sub["is_correct"] == sub["actually_correct"]).astype(int)
    sub["confidence"] = sub["confidence"].astype(float).clip(0, 1)
    rows = []
    arms = ["legacy_full", "verdict_only_neutral", "verdict_plus_repair"]
    for critic in sorted(sub["critic_model"].unique()):
        for arm in arms:
            for ds in sorted(sub["dataset"].unique()):
                slc = sub[(sub["critic_model"] == critic) & (sub["arm"] == arm)
                          & (sub["dataset"] == ds)]
                if len(slc) < 30: continue
                conf = slc["confidence"].to_numpy()
                correct = slc["verdict_correct"].to_numpy()
                bins = np.linspace(0, 1, 11)
                ece = 0.0
                for i in range(10):
                    mask = (conf >= bins[i]) & (conf < bins[i + 1]) if i < 9 else (conf >= bins[i]) & (conf <= bins[i + 1])
                    if not mask.any(): continue
                    p_emp = correct[mask].mean()
                    p_pred = conf[mask].mean()
                    w = mask.sum() / len(slc)
                    ece += w * abs(p_emp - p_pred)
                rows.append({
                    "critic": critic, "arm": arm, "dataset": ds,
                    "n": len(slc),
                    "ECE_10bin": float(ece),
                    "mean_conf": float(conf.mean()),
                    "mean_correct": float(correct.mean()),
                })
    return pd.DataFrame(rows)


def main():
    print("loading data...")
    df = load_critique_records()
    gt = load_ground_truth()

    print("\n[1] Per-dataset repair-quality ...")
    pdq = per_dataset_repair_quality(df, gt)
    pdq.to_csv(PAPER_TABLES / "appendix_per_dataset_repair_quality.csv", index=False)
    print(pdq.to_string(index=False))

    print("\n[2] Macro (equal-dataset-weighted) summary ...")
    macro = macro_summary(pdq, None)
    macro.to_csv(PAPER_TABLES / "appendix_macro_repair_quality.csv", index=False)
    print(macro.to_string(index=False))

    print("\n[3] Per-dataset calibration (ECE) ...")
    pdc = calibration_per_dataset(df)
    pdc.to_csv(PAPER_TABLES / "appendix_per_dataset_calibration.csv", index=False)
    print(pdc.to_string(index=False))

    # macro ECE
    print("\n[4] Macro (equal-dataset-weighted) ECE ...")
    macro_ece = pdc.groupby(["critic", "arm"])["ECE_10bin"].mean().reset_index()
    macro_ece.columns = ["critic", "arm", "macro_ECE_10bin"]
    macro_ece.to_csv(PAPER_TABLES / "appendix_macro_calibration.csv", index=False)
    print(macro_ece.to_string(index=False))

    print("done")


if __name__ == "__main__":
    main()
