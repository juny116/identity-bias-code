"""Audit critique parsing."""

import json
import re

import numpy as np
import pandas as pd

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
LOGS = REPO / "logs"
PAPER_TABLES = REPO / "paper" / "tables"


def strip_thinking(text):
    """Remove <think>...</think> blocks (replica of identity_critic.strip_thinking)."""
    if text is None:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def replay_parse(text):
    """Return True if any parse path would succeed; False if fallback triggers."""
    if text is None or not text:
        return False
    text = strip_thinking(text)
    # 1. direct JSON
    try:
        d = json.loads(text)
        if isinstance(d, dict) and "is_correct" in d:
            return True
    except json.JSONDecodeError:
        pass
    # 2. fenced JSON block
    m = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            if isinstance(d, dict) and "is_correct" in d:
                return True
        except json.JSONDecodeError:
            pass
    # 3. nested-brace JSON with is_correct
    m = re.search(
        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\"is_correct\"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        text,
        re.DOTALL,
    )
    if not m:
        m = re.search(r"\{.*\"is_correct\".*\}", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict) and "is_correct" in d:
                return True
        except json.JSONDecodeError:
            pass
    return False


def load_caveat_a_with_raw():
    """Load Caveat A records with raw_response intact (for parse-replay)."""
    rows = []
    roots = (
        "caveat_a_run_1",
        "verdict_repair_gpt_oss_no_think_run_1",
        "verdict_repair_qwen3_no_think_run_1",
    )
    for root in roots:
        d = LOGS / root
        if not d.exists():
            continue
        for path in d.glob("identity_swap_*.jsonl"):
            stem = path.stem.replace("identity_swap_", "")
            arm = None
            if "__" in stem:
                stem, arm = stem.split("__", 1)
            parts = stem.split("_on_")
            if len(parts) != 2:
                continue
            critic = parts[0]
            rest = parts[1]
            # dataset is the trailing token
            for ds in ("aime", "gpqa", "mmlu_pro", "olympiad"):
                if rest.endswith("_" + ds):
                    dataset = ds
                    break
            else:
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
                    rows.append(
                        {
                            "critic": critic,
                            "dataset": dataset,
                            "arm": arm or "legacy_full",
                            "condition": c["identity_condition"],
                            "problem_id": c["problem_id"],
                            "is_correct": c.get("is_correct"),
                            "actually_correct": c["actually_correct"],
                            "raw_response": c.get("raw_response"),
                        }
                    )
    df = pd.DataFrame(rows)
    return df


def main():
    print("loading Caveat A records (with raw_response)...")
    df = load_caveat_a_with_raw()
    print(f"  loaded {len(df)} records")
    print(f"  arms: {sorted(df['arm'].unique())}")

    print("\nreplaying parser on raw_response ...")
    df["parse_ok"] = df["raw_response"].map(replay_parse)
    df["parse_fail"] = ~df["parse_ok"]

    # per-cell parse-fail summary
    print("\nper-arm parse-failure rates (reconstructed) ...")
    summary = (
        df.groupby(["critic", "arm"])
        .agg(n_calls=("problem_id", "size"), n_parse_fail=("parse_fail", "sum"))
        .reset_index()
    )
    summary["parse_fail_pct"] = summary["n_parse_fail"] / summary["n_calls"] * 100
    print(summary.to_string(index=False))
    summary.to_csv(PAPER_TABLES / "revision_v4_parse_fail_retro.csv", index=False)

    # Recompute V+R vs Neutral contrast excluding parse failures
    print("\nV+R vs Neutral contrast on actually-wrong anonymous items ...")
    sub = df[
        (df["arm"].isin(["verdict_only_neutral", "verdict_plus_repair"]))
        & (df["condition"] == "anonymous")
        & (df["actually_correct"] == False)
    ].copy()

    rows = []
    for critic in sorted(sub["critic"].unique()):
        for include_pf in (True, False):
            if include_pf:
                slc = sub[sub["critic"] == critic]
            else:
                slc = sub[(sub["critic"] == critic) & sub["parse_ok"]]
            piv = slc.pivot_table(
                index="problem_id",
                columns="arm",
                values="is_correct",
                aggfunc="first",
            ).dropna()
            if len(piv) < 5:
                continue
            far_n = piv["verdict_only_neutral"].astype(int).mean() * 100
            far_r = piv["verdict_plus_repair"].astype(int).mean() * 100
            diff = far_r - far_n
            diffs = (
                piv["verdict_plus_repair"].astype(int)
                - piv["verdict_only_neutral"].astype(int)
            ).to_numpy()
            rng = np.random.default_rng(42)
            B = 5000
            n = len(diffs)
            boots = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(B)])
            ci_lo, ci_hi = np.quantile(boots, [0.025, 0.975]) * 100
            rows.append(
                {
                    "critic": critic,
                    "parse_fails": "included" if include_pf else "excluded",
                    "n_pairs": len(piv),
                    "far_neutral_pct": round(far_n, 2),
                    "far_repair_pct": round(far_r, 2),
                    "diff_pp": round(diff, 2),
                    "ci_lo_pp": round(ci_lo, 2),
                    "ci_hi_pp": round(ci_hi, 2),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(
        PAPER_TABLES / "revision_v4_caveat_a_parsefail_sensitivity.csv", index=False
    )
    print(out.to_string(index=False))

    print("\nper-dataset breakdown ...")
    rows2 = []
    for critic in sorted(sub["critic"].unique()):
        for ds in sorted(sub["dataset"].unique()):
            for include_pf in (True, False):
                if include_pf:
                    slc = sub[(sub["critic"] == critic) & (sub["dataset"] == ds)]
                else:
                    slc = sub[
                        (sub["critic"] == critic)
                        & (sub["dataset"] == ds)
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
                far_n = piv["verdict_only_neutral"].astype(int).mean() * 100
                far_r = piv["verdict_plus_repair"].astype(int).mean() * 100
                rows2.append(
                    {
                        "critic": critic,
                        "dataset": ds,
                        "parse_fails": "included" if include_pf else "excluded",
                        "n_pairs": len(piv),
                        "far_neutral_pct": round(far_n, 2),
                        "far_repair_pct": round(far_r, 2),
                        "diff_pp": round(far_r - far_n, 2),
                    }
                )
    out2 = pd.DataFrame(rows2)
    out2.to_csv(
        PAPER_TABLES / "revision_v4_caveat_a_parsefail_per_dataset.csv", index=False
    )
    print(out2.to_string(index=False))

    print("\ndone")


if __name__ == "__main__":
    main()
