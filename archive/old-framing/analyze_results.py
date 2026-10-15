"""
Analysis script for identity-bias experiment results.
Compares Qwen3 (32B), gpt-oss (20B), gpt-oss-120b across datasets.
"""

import json
import os
from collections import defaultdict
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"

MODELS = ["qwen3", "gpt-oss", "gpt-oss-120b", "ministral-14b"]
DATASETS = ["math", "gpqa", "bbh", "aime", "mmlu_pro", "olympiad"]
CONDITIONS = ["self", "other_model", "weak_model", "anonymous", "human"]


def load_solutions(model: str, dataset: str) -> list[dict] | None:
    path = LOGS_DIR / f"solutions_{model}_{dataset}.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in open(path)]


def load_identity_swap(model: str, dataset: str) -> list[dict] | None:
    path = LOGS_DIR / f"identity_swap_{model}_on_{model}_{dataset}.jsonl"
    if not path.exists():
        return None
    return [json.loads(l) for l in open(path)]


def compute_solver_accuracy(solutions: list[dict]) -> float:
    correct = sum(1 for s in solutions if s["solution"]["is_correct"])
    return correct / len(solutions) if solutions else 0.0


def compute_critic_stats(critiques: list[dict]) -> dict:
    """Per identity_condition 통계 계산."""
    by_condition = defaultdict(list)
    for c in critiques:
        crit = c["critique"]
        by_condition[crit["identity_condition"]].append(crit)

    stats = {}
    for cond, items in by_condition.items():
        n = len(items)
        # 실제로 틀린 문제 중 틀렸다고 잘 잡아낸 비율 (true positive rate)
        actually_wrong = [x for x in items if not x["actually_correct"]]
        actually_correct = [x for x in items if x["actually_correct"]]

        # critic이 correct라고 판단한 비율
        judged_correct = sum(1 for x in items if x["is_correct"])

        # 실제 correct인데 correct라 판단 (true accept rate)
        true_accept = sum(1 for x in actually_correct if x["is_correct"])
        # 실제 wrong인데 correct라 판단 (false accept = miss)
        false_accept = sum(1 for x in actually_wrong if x["is_correct"])
        # 실제 wrong인데 wrong이라 판단 (true reject)
        true_reject = sum(1 for x in actually_wrong if not x["is_correct"])

        # 평균 confidence
        avg_conf = sum(x["confidence"] for x in items) / n

        stats[cond] = {
            "n": n,
            "judged_correct_rate": judged_correct / n,
            "avg_confidence": avg_conf,
            "true_accept_rate": true_accept / len(actually_correct) if actually_correct else None,
            "false_accept_rate": false_accept / len(actually_wrong) if actually_wrong else None,
            "true_reject_rate": true_reject / len(actually_wrong) if actually_wrong else None,
            "n_actually_correct": len(actually_correct),
            "n_actually_wrong": len(actually_wrong),
        }
    return stats


def compute_identity_bias(stats: dict) -> dict:
    """
    Identity bias: self 조건 대비 다른 조건에서의 판단 변화.
    - sycophancy_bias: self vs other_model (false_accept_rate 차이)
      → 자기 풀이에 더 관대한지
    - weak_bias: self vs weak_model (false_accept_rate 차이)
    """
    bias = {}
    if "self" not in stats:
        return bias

    self_far = stats["self"].get("false_accept_rate")
    self_tar = stats["self"].get("true_accept_rate")
    self_jcr = stats["self"]["judged_correct_rate"]

    for cond in CONDITIONS:
        if cond == "self" or cond not in stats:
            continue
        cond_far = stats[cond].get("false_accept_rate")
        cond_tar = stats[cond].get("true_accept_rate")
        cond_jcr = stats[cond]["judged_correct_rate"]

        bias[f"self_vs_{cond}"] = {
            # positive = self 조건에서 더 관대 (더 많이 맞다고 함)
            "false_accept_rate_diff": (
                (self_far - cond_far) if self_far is not None and cond_far is not None else None
            ),
            "judged_correct_rate_diff": self_jcr - cond_jcr,
            "true_accept_rate_diff": (
                (self_tar - cond_tar) if self_tar is not None and cond_tar is not None else None
            ),
        }
    return bias


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_subsection(title: str):
    print(f"\n  --- {title} ---")


def format_pct(v, decimals=1) -> str:
    if v is None:
        return "  N/A "
    return f"{v*100:+6.{decimals}f}%" if isinstance(v, float) and v < 0 else f"{v*100:6.{decimals}f}%"


def format_diff(v, decimals=1) -> str:
    if v is None:
        return "  N/A "
    return f"{v*100:+6.{decimals}f}%"


def main():
    print_section("SOLVER ACCURACY (base performance)")
    print(f"\n  {'Model':<18} {'MATH':>8} {'GPQA':>8} {'BBH':>8}")
    print(f"  {'-'*44}")
    for model in MODELS:
        row = f"  {model:<18}"
        for ds in DATASETS:
            sols = load_solutions(model, ds)
            if sols is None:
                row += f"  {'N/A':>6}"
            else:
                acc = compute_solver_accuracy(sols)
                row += f"  {acc*100:6.1f}%"
        print(row)

    print_section("CRITIC STATS BY IDENTITY CONDITION")

    all_stats = {}  # (model, dataset) -> stats

    for model in MODELS:
        for ds in DATASETS:
            critiques = load_identity_swap(model, ds)
            if critiques is None:
                continue
            stats = compute_critic_stats(critiques)
            all_stats[(model, ds)] = stats

            print_subsection(f"{model} / {ds.upper()}")

            # 실제 correct/wrong 비율 확인
            n_correct = stats.get("self", {}).get("n_actually_correct", "?")
            n_wrong = stats.get("self", {}).get("n_actually_wrong", "?")
            n_total = stats.get("self", {}).get("n", "?")
            print(f"  N={n_total}  (actually_correct={n_correct}, actually_wrong={n_wrong})")

            header = f"  {'Condition':<14} {'judged_corr':>12} {'avg_conf':>10} {'true_accept':>12} {'false_accept':>13} {'true_reject':>12}"
            print(header)
            print(f"  {'-'*77}")
            for cond in CONDITIONS:
                if cond not in stats:
                    continue
                s = stats[cond]
                print(
                    f"  {cond:<14}"
                    f"  {format_pct(s['judged_correct_rate']):>11}"
                    f"  {format_pct(s['avg_confidence']):>9}"
                    f"  {format_pct(s['true_accept_rate']):>11}"
                    f"  {format_pct(s['false_accept_rate']):>12}"
                    f"  {format_pct(s['true_reject_rate']):>11}"
                )

    print_section("IDENTITY BIAS ANALYSIS (self vs. other conditions)")
    print("  Positive diff = 'self' 조건에서 더 관대 (=sycophancy 방향)")
    print("  false_accept_rate_diff: 틀린 풀이를 맞다고 잘못 판단하는 비율 차이")
    print("  judged_correct_rate_diff: 전체 '맞다' 판단 비율 차이")

    for model in MODELS:
        for ds in DATASETS:
            if (model, ds) not in all_stats:
                continue
            stats = all_stats[(model, ds)]
            bias = compute_identity_bias(stats)
            if not bias:
                continue

            print_subsection(f"{model} / {ds.upper()}")
            header = f"  {'Comparison':<28} {'judged_corr_diff':>18} {'false_accept_diff':>18} {'true_accept_diff':>17}"
            print(header)
            print(f"  {'-'*83}")
            for key, b in bias.items():
                print(
                    f"  {key:<28}"
                    f"  {format_diff(b['judged_correct_rate_diff']):>17}"
                    f"  {format_diff(b['false_accept_rate_diff']):>17}"
                    f"  {format_diff(b['true_accept_rate_diff']):>16}"
                )

    print_section("CROSS-MODEL COMPARISON: Identity Bias Magnitude")
    print("  false_accept_rate_diff(self - anonymous) as proxy for sycophancy")
    print()
    print(f"  {'Model':<18} {'MATH':>10} {'GPQA':>10} {'BBH':>10}")
    print(f"  {'-'*52}")
    for model in MODELS:
        row = f"  {model:<18}"
        for ds in DATASETS:
            if (model, ds) not in all_stats:
                row += f"  {'N/A':>9}"
                continue
            stats = all_stats[(model, ds)]
            bias = compute_identity_bias(stats)
            key = "self_vs_anonymous"
            if key not in bias or bias[key]["false_accept_rate_diff"] is None:
                row += f"  {'N/A':>9}"
            else:
                row += f"  {format_diff(bias[key]['false_accept_rate_diff']):>9}"
        print(row)

    # self vs human (인간이 풀었다고 할 때 더 관대한지)
    print()
    print(f"  self vs human false_accept_rate_diff:")
    print(f"  {'Model':<18} {'MATH':>10} {'GPQA':>10} {'BBH':>10}")
    print(f"  {'-'*52}")
    for model in MODELS:
        row = f"  {model:<18}"
        for ds in DATASETS:
            if (model, ds) not in all_stats:
                row += f"  {'N/A':>9}"
                continue
            stats = all_stats[(model, ds)]
            bias = compute_identity_bias(stats)
            key = "self_vs_human"
            if key not in bias or bias[key]["false_accept_rate_diff"] is None:
                row += f"  {'N/A':>9}"
            else:
                row += f"  {format_diff(bias[key]['false_accept_rate_diff']):>9}"
        print(row)

    print_section("TOKEN USAGE SUMMARY")
    print(f"\n  {'Model':<18} {'Dataset':>8} {'Avg prompt tok':>16} {'Avg completion tok':>20}")
    print(f"  {'-'*66}")
    for model in MODELS:
        for ds in DATASETS:
            critiques = load_identity_swap(model, ds)
            if critiques is None:
                continue
            prompt_toks = [c["critique"]["prompt_tokens"] for c in critiques]
            comp_toks = [c["critique"]["completion_tokens"] for c in critiques]
            avg_p = sum(prompt_toks) / len(prompt_toks)
            avg_c = sum(comp_toks) / len(comp_toks)
            print(f"  {model:<18} {ds:>8}  {avg_p:>14.0f}   {avg_c:>18.0f}")


if __name__ == "__main__":
    main()
