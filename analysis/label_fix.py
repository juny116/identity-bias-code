"""Apply robust MCQ label corrections (from relabel_mcq.py) to a loaded records
DataFrame, non-destructively. Overrides `actually_correct` for GPQA/MMLU-Pro
candidates where a confident robust option-letter relabel exists.
"""
import json
from pathlib import Path

_PATH = Path(__file__).parent.parent / "paper" / "tables" / "corrected_labels.json"


def load_corrections():
    if _PATH.exists():
        return json.load(open(_PATH))
    return {}


def apply(df):
    """Override df['actually_correct'] using corrected_labels.json keyed by
    dataset|problem_id|solver. Returns (df, n_overridden, n_flipped)."""
    corr = load_corrections()
    if not corr or df.empty:
        return df, 0, 0
    n_over = n_flip = 0
    col = df.columns
    if not {"dataset", "problem_id", "solver", "actually_correct"} <= set(col):
        return df, 0, 0
    ac = df["actually_correct"].tolist()
    keys = (df["dataset"].astype(str) + "|" + df["problem_id"].astype(str)
            + "|" + df["solver"].astype(str)).tolist()
    for i, k in enumerate(keys):
        if k in corr:
            new = bool(corr[k])
            n_over += 1
            if bool(ac[i]) != new:
                n_flip += 1
                ac[i] = new
    df = df.copy()
    df["actually_correct"] = ac
    return df, n_over, n_flip
