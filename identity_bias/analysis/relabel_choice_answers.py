"""Build a label-correction map from confidently extracted choice answers."""

import json
import re

from identity_bias.analysis.estimate_label_errors import (  # reuse normalizer
    MCQ,
    robust_letter,
)
from identity_bias.runtime.paths import workspace_root

REPO = workspace_root()
SAMPLES = REPO / "logs" / "interface_confound_samples"
OUT = REPO / "paper" / "tables" / "corrected_labels.json"


def confident(final_answer, letter):
    """Trust a flip only when the answer STARTS with an explicit option marker
    (\\text{(C)}, \\text{C}, (C), C), C., \\boxed{...C}, or a bare letter). This
    catches clearly-correct answers even when followed by explanatory text
    (e.g. '(C)\\;4.12 MeV'), while excluding prose blobs where a stray standalone
    letter was picked up mid-sentence."""
    if letter is None:
        return False
    t = str(final_answer or "").strip().lstrip("$").lstrip()
    L = re.escape(letter)
    prefix = [
        rf"^\\text\s*\{{?\(?\s*{L}",  # \text{(C)} or \text{C}
        rf"^\\boxed\s*\{{[^}}]*{L}",  # \boxed{...C}
        rf"^\(\s*{L}\s*\)",  # (C)
        rf"^{L}\s*[\).:\-]",  # C) C. C: C-
        rf"^{L}\s*$",  # bare C
    ]
    return any(re.match(p, t, re.IGNORECASE) for p in prefix)


def main():
    corrections = {}
    flipped, left, unresolved = [], 0, []
    for ds, opts in MCQ.items():
        f = list(SAMPLES.glob(f"{ds}_w*_seed42.json"))
        if not f:
            continue
        valid = set(opts)
        for c in json.load(open(f[0]))["pool"]:
            gt = str(c.get("ground_truth", "")).strip().upper().strip("()")
            letter = robust_letter(c.get("final_answer"), valid)
            key = f"{ds}|{c['problem_id']}|{c['solver']}"
            if letter is not None and confident(c.get("final_answer"), letter):
                new_label = letter == gt
                corrections[key] = new_label  # canonical robust label
                if new_label != bool(c["is_correct"]):
                    flipped.append(
                        (
                            key,
                            c["is_correct"],
                            new_label,
                            str(c.get("final_answer"))[:40],
                        )
                    )
                else:
                    left += 1
            else:
                unresolved.append(key)  # leave stored label, flag

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(corrections, open(OUT, "w"))
    print(
        f"wrote {OUT}: {len(corrections)} confident MCQ labels "
        f"({len(flipped)} FLIPPED vs stored, {left} confirmed same)"
    )
    print(f"left unresolved/ambiguous (stored label kept, flagged): {len(unresolved)}")
    print("\nflipped candidates (stored -> robust):")
    for k, old, new, fa in flipped:
        print(f"  {k}: {old} -> {new}   final={fa!r}")


if __name__ == "__main__":
    main()
