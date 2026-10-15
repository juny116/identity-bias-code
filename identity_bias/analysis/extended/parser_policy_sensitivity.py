"""Parser policy sensitivity."""

import numpy as np
import pandas as pd

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
LOGS = REPO / "logs"
PAPER_TABLES = REPO / "paper" / "tables"

from identity_bias.analysis.extended.audit_critique_parsing import (
    load_caveat_a_with_raw,
    replay_parse,
)


def paired_bootstrap_diff(diffs, B=5000, seed=42):
    """Return mean and (2.5%, 97.5%) CI of mean(diffs) via bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(B)])
    return (
        float(diffs.mean()),
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


# ---------------------------------------------------------------------------
# Option 1: all Caveat A arm contrasts, parser-aware
# ---------------------------------------------------------------------------

ARMS_VS_NEUTRAL = [
    ("verdict_only_instructed", "instructed_vs_neutral"),
    ("verdict_schema_matched", "schema_vs_neutral"),
    ("verdict_plus_repair", "repair_vs_neutral"),
    ("verdict_plus_repair_reversed", "repair_reversed_vs_neutral"),
]


def contrasts_all_arms(df):
    """For each (critic, arm_pair) compute the paired contrast vs verdict_only_neutral
    on anonymous wrong items, with parse failures included vs excluded."""
    sub = df[(df["condition"] == "anonymous") & (~df["actually_correct"])].copy()
    rows = []
    for critic in sorted(sub["critic"].unique()):
        for arm, contrast_name in ARMS_VS_NEUTRAL:
            arms_present = sub[sub["critic"] == critic]["arm"].unique()
            if arm not in arms_present or "verdict_only_neutral" not in arms_present:
                continue
            for include_pf in (True, False):
                if include_pf:
                    slc = sub[
                        (sub["critic"] == critic)
                        & sub["arm"].isin([arm, "verdict_only_neutral"])
                    ]
                else:
                    slc = sub[
                        (sub["critic"] == critic)
                        & sub["arm"].isin([arm, "verdict_only_neutral"])
                        & sub["parse_ok"]
                    ]
                piv = slc.pivot_table(
                    index="problem_id",
                    columns="arm",
                    values="is_correct",
                    aggfunc="first",
                ).dropna()
                if len(piv) < 5:
                    continue
                far_neutral = piv["verdict_only_neutral"].astype(int).mean() * 100
                far_other = piv[arm].astype(int).mean() * 100
                diffs = (
                    piv[arm].astype(int) - piv["verdict_only_neutral"].astype(int)
                ).to_numpy()
                d, lo, hi = paired_bootstrap_diff(diffs)
                rows.append(
                    {
                        "critic": critic,
                        "contrast": contrast_name,
                        "parse_fails": "included" if include_pf else "excluded",
                        "n_pairs": len(piv),
                        "far_neutral_pct": round(far_neutral, 2),
                        "far_other_pct": round(far_other, 2),
                        "diff_pp": round(d * 100, 2),
                        "ci_lo_pp": round(lo * 100, 2),
                        "ci_hi_pp": round(hi * 100, 2),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Option 1b: parser-policy sensitivity
# ---------------------------------------------------------------------------

POLICIES = ["accept", "reject", "drop"]


def policy_apply(piv_vals, piv_ok, policy):
    """piv_vals: bool DataFrame of is_correct per arm. piv_ok: bool DataFrame of parse_ok.
    Returns int DataFrame of treated is_correct after applying policy."""
    out = piv_vals.astype(int).copy()
    if policy == "accept":
        # current default: parse fail -> is_correct=True (already in piv_vals from logger fallback)
        return out
    if policy == "reject":
        # parse fail -> is_correct = False
        out = out.where(piv_ok, other=0)
        return out
    if policy == "drop":
        # drop pairs where ANY arm has parse fail
        keep = piv_ok.all(axis=1)
        return out.loc[keep]
    raise ValueError(policy)


def policy_sensitivity(df, arms_pair=("verdict_only_neutral", "verdict_plus_repair")):
    """Compute V+R-vs-Neutral contrast on anonymous wrong items under three
    parser-fallback policies for each critic."""
    sub = df[
        (df["condition"] == "anonymous")
        & (~df["actually_correct"])
        & df["arm"].isin(arms_pair)
    ].copy()
    rows = []
    a, b = arms_pair
    for critic in sorted(sub["critic"].unique()):
        cs = sub[sub["critic"] == critic]
        arms_present = set(cs["arm"].unique())
        if a not in arms_present or b not in arms_present:
            continue
        piv_v = cs.pivot_table(
            index="problem_id", columns="arm", values="is_correct", aggfunc="first"
        ).dropna()
        piv_o = cs.pivot_table(
            index="problem_id", columns="arm", values="parse_ok", aggfunc="first"
        ).reindex(piv_v.index)
        if len(piv_v) < 5:
            continue
        for policy in POLICIES:
            piv_t = policy_apply(piv_v, piv_o, policy)
            if len(piv_t) < 5:
                continue
            far_a = piv_t[a].mean() * 100
            far_b = piv_t[b].mean() * 100
            diffs = (piv_t[b] - piv_t[a]).to_numpy()
            d, lo, hi = paired_bootstrap_diff(diffs)
            rows.append(
                {
                    "critic": critic,
                    "contrast": f"{b}_vs_{a}",
                    "policy": policy,
                    "n_pairs": len(piv_t),
                    "far_a_pct": round(far_a, 2),
                    "far_b_pct": round(far_b, 2),
                    "diff_pp": round(d * 100, 2),
                    "ci_lo_pp": round(lo * 100, 2),
                    "ci_hi_pp": round(hi * 100, 2),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Option 3: per-condition parser-aware contrast
# ---------------------------------------------------------------------------


def per_condition_contrast(df, arms=("verdict_only_neutral", "verdict_plus_repair")):
    """V+R vs Neutral contrast within each (critic, identity_condition) on wrong items,
    parse-fails included vs excluded."""
    sub = df[df["arm"].isin(arms) & (~df["actually_correct"])].copy()
    rows = []
    for critic in sorted(sub["critic"].unique()):
        conds = sorted(sub[sub["critic"] == critic]["condition"].unique())
        for cond in conds:
            for include_pf in (True, False):
                if include_pf:
                    slc = sub[(sub["critic"] == critic) & (sub["condition"] == cond)]
                else:
                    slc = sub[
                        (sub["critic"] == critic)
                        & (sub["condition"] == cond)
                        & sub["parse_ok"]
                    ]
                piv = slc.pivot_table(
                    index="problem_id",
                    columns="arm",
                    values="is_correct",
                    aggfunc="first",
                ).dropna()
                if len(piv) < 20:
                    continue
                a, b = arms
                far_a = piv[a].astype(int).mean() * 100
                far_b = piv[b].astype(int).mean() * 100
                diffs = (piv[b].astype(int) - piv[a].astype(int)).to_numpy()
                d, lo, hi = paired_bootstrap_diff(diffs)
                rows.append(
                    {
                        "critic": critic,
                        "condition": cond,
                        "parse_fails": "included" if include_pf else "excluded",
                        "n_pairs": len(piv),
                        "far_neutral_pct": round(far_a, 2),
                        "far_repair_pct": round(far_b, 2),
                        "diff_pp": round(d * 100, 2),
                        "ci_lo_pp": round(lo * 100, 2),
                        "ci_hi_pp": round(hi * 100, 2),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    print("loading Caveat A records ...")
    df = load_caveat_a_with_raw()
    print(f"  {len(df)} records")

    print("replaying parser ...")
    df["parse_ok"] = df["raw_response"].map(replay_parse)
    df["parse_fail"] = ~df["parse_ok"]

    print("\n[Option 1] paired contrasts for ALL Caveat A arms vs neutral ...")
    out1 = contrasts_all_arms(df)
    out1.to_csv(PAPER_TABLES / "revision_v4_all_arms_parser_aware.csv", index=False)
    print(out1.to_string(index=False))

    print("\n[Option 1b] parser-policy sensitivity for V+R vs neutral ...")
    out1b_vr = policy_sensitivity(
        df, arms_pair=("verdict_only_neutral", "verdict_plus_repair")
    )
    out1b_vr.to_csv(PAPER_TABLES / "revision_v4_policy_sensitivity_vr.csv", index=False)
    print(out1b_vr.to_string(index=False))

    print(
        "\n[Option 1b] parser-policy sensitivity for field-order (V+R reversed vs neutral) ..."
    )
    out1b_fr = policy_sensitivity(
        df, arms_pair=("verdict_only_neutral", "verdict_plus_repair_reversed")
    )
    out1b_fr.to_csv(
        PAPER_TABLES / "revision_v4_policy_sensitivity_field_order.csv", index=False
    )
    print(out1b_fr.to_string(index=False))

    print("\n[Option 3] per-condition parser-aware V+R vs neutral ...")
    out3 = per_condition_contrast(df)
    out3.to_csv(
        PAPER_TABLES / "revision_v4_per_condition_parser_aware.csv", index=False
    )
    print(out3.to_string(index=False))

    print("\ndone")


if __name__ == "__main__":
    main()
