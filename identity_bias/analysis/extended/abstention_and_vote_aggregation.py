"""Abstention and vote aggregation."""

import json

import numpy as np
import pandas as pd

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
LOGS = REPO / "logs"
PAPER_TABLES = REPO / "paper" / "tables"

from identity_bias.analysis.extended.build_critique_tables import load_critique_records

# ---------------------------------------------------------------------------
# 1. Think vs no-think comparison
# ---------------------------------------------------------------------------


def think_vs_no_think(df):
    """Per-critic baseline (legacy_full) FAR and repair-success comparison."""
    sub = df[df["arm"] == "legacy_full"].copy()
    sub_wrong = sub[~sub["actually_correct"]]

    rows = []
    pairs = [
        ("qwen3", "qwen3-no-think"),
        ("gpt-oss-think", "gpt-oss-no-think"),
    ]
    for think, no_think in pairs:
        for critic in (think, no_think):
            cs = sub_wrong[sub_wrong["critic_model"] == critic]
            n = len(cs)
            if n == 0:
                continue
            far = cs["is_correct"].mean() * 100
            # repair success: among rejected wrong items, how often is corrected_answer right?
            # We don't have direct gt comparison here; approximate via correct→wrong flip
            # Use existing repair quality from table9 instead — load.
            rows.append(
                {
                    "critic": critic,
                    "n_wrong": n,
                    "far_pct": round(far, 2),
                    "thinking": "ON" if critic == think else "OFF",
                }
            )

    # also load repair quality from existing table9
    try:
        rq = pd.read_csv(PAPER_TABLES / "table9_repair_quality.csv")
        # adapt as needed
    except FileNotFoundError:
        rq = None
    return pd.DataFrame(rows), rq


# ---------------------------------------------------------------------------
# 2. Abstain-threshold curve from confidence
# ---------------------------------------------------------------------------


def abstain_threshold_curve(df):
    """For each (critic, arm), sweep confidence threshold tau in [0.5, 1.0].
    Item is "abstained" if confidence < tau. Among non-abstained items, compute
    FAR (on wrong items) and coverage (fraction kept)."""
    arms = ["legacy_full", "verdict_only_neutral", "verdict_plus_repair"]
    sub = df[df["arm"].isin(arms) & df["confidence"].notna()].copy()
    sub["confidence"] = sub["confidence"].astype(float).clip(0, 1)
    sub_wrong = sub[~sub["actually_correct"]]

    taus = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    rows = []
    for critic in sorted(sub_wrong["critic_model"].unique()):
        for arm in arms:
            cs = sub_wrong[
                (sub_wrong["critic_model"] == critic) & (sub_wrong["arm"] == arm)
            ]
            if len(cs) < 50:
                continue
            for tau in taus:
                kept = cs[cs["confidence"] >= tau]
                if len(kept) < 5:
                    continue
                far = kept["is_correct"].mean() * 100
                cov = len(kept) / len(cs) * 100
                rows.append(
                    {
                        "critic": critic,
                        "arm": arm,
                        "tau": tau,
                        "n_kept": len(kept),
                        "coverage_pct": round(cov, 1),
                        "far_pct": round(far, 2),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. N-of-K=5 score aggregation
# ---------------------------------------------------------------------------


def n_of_k_aggregation():
    """Re-analyze N-of-K=5 logs to compute majority verdict, mean-confidence-thresholded
    verdict, and compare to single-call FAR."""
    base = LOGS / "n_of_k_run_1"
    if not base.exists():
        return pd.DataFrame()

    # Aggregate per (critic, condition, dataset, problem_id) across the 5 samples.
    records = []
    for path in base.glob("identity_swap_*.jsonl"):
        stem = path.stem.replace("identity_swap_", "")
        # critic_on_solver_dataset__samplek
        body = stem
        sample_k = None
        if "__sample" in body:
            body, sample_k = body.rsplit("__sample", 1)
            try:
                sample_k = int(sample_k)
            except ValueError:
                sample_k = None
        parts = body.split("_on_")
        if len(parts) != 2:
            continue
        critic = parts[0]
        rest = parts[1]
        ds = None
        for cand in ("aime", "gpqa", "mmlu_pro", "olympiad"):
            if rest.endswith("_" + cand):
                ds = cand
                break
        if ds is None:
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
                records.append(
                    {
                        "critic": critic,
                        "dataset": ds,
                        "sample_k": sample_k,
                        "problem_id": c["problem_id"],
                        "condition": c["identity_condition"],
                        "is_correct": c.get("is_correct"),
                        "actually_correct": c["actually_correct"],
                        "confidence": c.get("confidence"),
                    }
                )
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df = df[df["is_correct"].notna()]
    df_wrong = df[~df["actually_correct"]]

    rows = []
    for critic in sorted(df_wrong["critic"].unique()):
        for cond in sorted(df_wrong["condition"].unique()):
            grp = df_wrong[
                (df_wrong["critic"] == critic) & (df_wrong["condition"] == cond)
            ]
            piv_v = grp.pivot_table(
                index=["dataset", "problem_id"],
                columns="sample_k",
                values="is_correct",
                aggfunc="first",
            )
            piv_c = grp.pivot_table(
                index=["dataset", "problem_id"],
                columns="sample_k",
                values="confidence",
                aggfunc="first",
            )
            if piv_v.empty:
                continue
            sample_cols = [k for k in piv_v.columns if k is not None]
            piv_v = piv_v[sample_cols]
            piv_c = piv_c[sample_cols] if sample_cols else piv_c
            piv_v = piv_v.dropna()
            piv_c = piv_c.loc[piv_v.index]
            if len(piv_v) < 20:
                continue
            # single-call FAR (use sample_k=0 if present, else first column)
            first = sample_cols[0]
            far_single = piv_v[first].astype(float).mean() * 100
            # majority verdict
            far_maj = (
                piv_v.astype(int).sum(axis=1) >= (len(sample_cols) / 2 + 0.5)
            ).mean() * 100
            # mean-confidence-weighted: accept if mean conf > 0.5 AND majority accept
            mean_conf = piv_c.astype(float).mean(axis=1)
            far_conf_gate = (
                (piv_v.astype(int).sum(axis=1) >= (len(sample_cols) / 2 + 0.5))
                & (mean_conf > 0.7)
            ).mean() * 100
            rows.append(
                {
                    "critic": critic,
                    "condition": cond,
                    "n_items": len(piv_v),
                    "far_single_pct": round(far_single, 2),
                    "far_majority5_pct": round(far_maj, 2),
                    "far_majority_conf_gate_pct": round(far_conf_gate, 2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Interface vs Identity variation (NEW centerpiece table)
# ---------------------------------------------------------------------------


def interface_vs_identity(df):
    """For each critic, compute:
    - identity 4-label FAR range (anon/human/other/self)
    - Caveat A interface FAR range across 5 arms (instructed, neutral,
      schema_matched, V+R, V+R reversed)
    """
    rows = []
    identity_labels = ["anonymous", "human", "other_model", "self"]
    interface_arms = [
        "verdict_only_instructed",
        "verdict_only_neutral",
        "verdict_schema_matched",
        "verdict_plus_repair",
        "verdict_plus_repair_reversed",
    ]

    for critic in sorted(df["critic_model"].unique()):
        # identity range (on legacy_full or any available paired-condition data)
        legacy = df[
            (df["critic_model"] == critic)
            & (df["arm"] == "legacy_full")
            & (~df["actually_correct"])
        ]
        if legacy.empty:
            continue
        id_fars = []
        for c in identity_labels:
            cs = legacy[legacy["condition"] == c]
            if len(cs) >= 50:
                id_fars.append(cs["is_correct"].mean() * 100)
        id_range = max(id_fars) - min(id_fars) if len(id_fars) >= 2 else float("nan")

        # interface range (Caveat A on anonymous wrong items)
        ca = df[
            (df["critic_model"] == critic)
            & (df["arm"].isin(interface_arms))
            & (df["condition"] == "anonymous")
            & (~df["actually_correct"])
        ]
        iface_fars = {}
        for arm in interface_arms:
            cs = ca[ca["arm"] == arm]
            if len(cs) >= 50:
                iface_fars[arm] = cs["is_correct"].mean() * 100
        iface_range = (
            max(iface_fars.values()) - min(iface_fars.values())
            if len(iface_fars) >= 2
            else float("nan")
        )

        rows.append(
            {
                "critic": critic,
                "id_n_labels": len(id_fars),
                "id_range_pp": round(id_range, 2) if not np.isnan(id_range) else None,
                "iface_n_arms": len(iface_fars),
                "iface_range_pp": round(iface_range, 2)
                if not np.isnan(iface_range)
                else None,
                "ratio_iface_over_id": round(iface_range / id_range, 2)
                if (id_range and not np.isnan(id_range) and id_range > 0)
                else None,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    print("loading critique records ...")
    df = load_critique_records()
    print(f"  {len(df)} records")

    print("\n[1] Think vs no-think ...")
    tn, _ = think_vs_no_think(df)
    tn.to_csv(PAPER_TABLES / "revision_v4_think_vs_nothink.csv", index=False)
    print(tn.to_string(index=False))

    print("\n[2] Abstain-threshold curve ...")
    at = abstain_threshold_curve(df)
    at.to_csv(PAPER_TABLES / "revision_v4_abstain_threshold.csv", index=False)
    print(at.to_string(index=False))

    print("\n[3] N-of-K aggregation ...")
    nk = n_of_k_aggregation()
    nk.to_csv(PAPER_TABLES / "revision_v4_n_of_k_aggregation.csv", index=False)
    print(nk.to_string(index=False))

    print("\n[4] Interface vs Identity variation ...")
    iv = interface_vs_identity(df)
    iv.to_csv(PAPER_TABLES / "revision_v4_interface_vs_identity.csv", index=False)
    print(iv.to_string(index=False))

    print("\ndone")


if __name__ == "__main__":
    main()
