"""Estimate answer-label changes after robust option-letter and integer extraction."""

import json
import re

from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
SAMPLES = REPO / "logs" / "interface_confound_samples"
OUT = REPO / "paper" / "tables" / "label_noise_flips.csv"

MCQ = {"gpqa": "ABCD", "mmlu_pro": "ABCDEFGHIJ"}


def robust_letter(s, valid):
    """Extract a single option letter from a messy final-answer string."""
    if s is None:
        return None
    t = str(s).strip()
    # strip common LaTeX wrappers
    t = re.sub(r"\\(?:text|mathrm|mathbf|boxed|textbf)\s*\{([^}]*)\}", r"\1", t)
    t = t.replace("$", "").replace("\\", "")
    up = t.upper()
    # 1) explicit "answer is C" / "option C" / "(C)" / "C." patterns
    m = re.search(r"\b(?:ANSWER|OPTION|CHOICE)\b[^A-Z0-9]{0,8}([A-Z])\b", up)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r"\(\s*([A-Z])\s*\)", up)
    if m and m.group(1) in valid:
        return m.group(1)
    # 2) a standalone valid letter (prefer the LAST one, often the conclusion)
    cands = [c for c in re.findall(r"\b([A-Z])\b", up) if c in valid]
    if cands:
        return cands[-1]
    # 3) bare single char
    cleaned = up.strip("() .")
    if len(cleaned) == 1 and cleaned in valid:
        return cleaned
    return None


def robust_int(s):
    if s is None:
        return None
    t = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", str(s))
    nums = re.findall(r"-?\d{1,4}", t)
    if nums:
        try:
            return int(nums[-1])
        except ValueError:
            return None
    return None


def main():
    import csv

    flips = []
    summary = []
    for ds in ["aime", "gpqa", "olympiad", "mmlu_pro"]:
        f = list(SAMPLES.glob(f"{ds}_w*_seed42.json"))
        if not f:
            continue
        pool = json.load(open(f[0]))["pool"]
        n = len(pool)
        if ds in MCQ:
            valid = set(MCQ[ds])
            n_flip = 0
            n_unresolved = 0
            for c in pool:
                gt = str(c.get("ground_truth", "")).strip().upper().strip("()")
                new = robust_letter(c.get("final_answer"), valid)
                if new is None:
                    n_unresolved += 1
                    continue
                new_label = new == gt
                if new_label != bool(c["is_correct"]):
                    n_flip += 1
                    flips.append(
                        {
                            "dataset": ds,
                            "problem_id": c["problem_id"],
                            "solver": c["solver"],
                            "ground_truth": gt,
                            "extracted_final_answer": str(c.get("final_answer"))[:50],
                            "robust_letter": new,
                            "stored_label": c["is_correct"],
                            "robust_label": new_label,
                        }
                    )
            summary.append((ds, n, n_flip, n_unresolved))
        else:
            # math: flag extraction edge cases only (no re-judge; math_verify is used)
            n_edge = 0
            for c in pool:
                fa = str(c.get("final_answer", "")).strip()
                cot = c.get("chain_of_thought", "") or ""
                if (
                    fa == ""
                    or len(re.findall(r"\\boxed\{", cot)) > 1
                    or (ds == "aime" and robust_int(fa) is None)
                ):
                    n_edge += 1
            summary.append((ds, n, None, n_edge))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if flips:
        with open(OUT, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flips[0].keys()))
            w.writeheader()
            w.writerows(flips)
    print(f"{'dataset':9} {'n':>4} {'label_flips':>12} {'unresolved/edge':>16}")
    for ds, n, nf, extra in summary:
        flip_s = f"{nf} ({100 * nf / n:.1f}%)" if nf is not None else "n/a (math)"
        kind = "unresolved-letter" if ds in MCQ else "extract-edge-flag"
        print(f"{ds:9} {n:>4} {flip_s:>12}   {extra} {kind}")
    print(
        f"\nMCQ label flips (robust re-extraction disagrees with stored label) "
        f"-> {OUT if flips else '(none)'}"
    )
    print(
        "NOTE: math/olympiad use math_verify (LaTeX-aware) so we only flag "
        "extraction edge cases there for the manual sheet, not auto-flip."
    )


if __name__ == "__main__":
    main()
