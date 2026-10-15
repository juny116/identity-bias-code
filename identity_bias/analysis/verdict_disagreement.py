"""Measure per-item verdict disagreement across interfaces."""

from collections import defaultdict

import pandas as pd

from identity_bias.analysis.records import load_judge_records
from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
PILOT = REPO / "logs" / "interface_confound_pilot"


def main():
    df = load_judge_records(PILOT)
    if df.empty:
        print("No judge records found")
        return
    df = df[df["parse_ok"]].copy()

    out = []
    for (judge, ds), g in df.groupby(["judge", "dataset"]):
        # candidate key = (problem_id, solver); collect verdicts across interfaces
        cand = defaultdict(list)
        cand_label = {}
        for _, r in g.iterrows():
            k = (r["problem_id"], r["solver"])
            cand[k].append(bool(r["is_correct"]))
            cand_label[k] = bool(r["actually_correct"])

        def frac_disagree(items):
            multi = [k for k in items if len(cand[k]) >= 2]
            if not multi:
                return float("nan"), 0
            dis = sum(1 for k in multi if len(set(cand[k])) > 1)
            return 100 * dis / len(multi), len(multi)

        allk = list(cand)
        wrongk = [k for k in allk if not cand_label[k]]
        corrk = [k for k in allk if cand_label[k]]
        d_all, n_all = frac_disagree(allk)
        d_w, n_w = frac_disagree(wrongk)
        d_c, n_c = frac_disagree(corrk)
        out.append(
            {
                "judge": judge,
                "dataset": ds,
                "pct_items_disagree": round(d_all, 1),
                "n_items": n_all,
                "pct_wrong_disagree": round(d_w, 1),
                "n_wrong": n_w,
                "pct_correct_disagree": round(d_c, 1),
                "n_correct": n_c,
            }
        )
    res = pd.DataFrame(out).sort_values(["dataset", "judge"])
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    res.to_csv(REPO / "paper" / "tables" / "ifc_item_disagreement.csv", index=False)
    print(
        f"\nmean % candidates that flip verdict across interfaces: "
        f"{res['pct_items_disagree'].mean():.1f}%"
    )


if __name__ == "__main__":
    main()
