"""Unified entry point for authorship and judge-interface experiments."""

import argparse
import ast
import os
import runpy
import sys
from pathlib import Path

COMMANDS = {
    "run": {
        "generate-solutions": "identity_bias.workflows.generate_solutions",
        "evaluate-authorship": "identity_bias.workflows.evaluate_authorship",
        "evaluate-pairwise-interfaces": "identity_bias.workflows.evaluate_pairwise_interfaces",
        "compare-solver-critic": "identity_bias.workflows.compare_solver_critic",
        "evaluate-verdict-interfaces": "identity_bias.workflows.evaluate_verdict_interfaces",
    },
    "analyze": {
        "sample-label-audit": "identity_bias.analysis.sample_label_audit",
        "pairwise-preference-metrics": "identity_bias.analysis.pairwise_preference_metrics",
        "remove-truncated-responses": "identity_bias.analysis.remove_truncated_responses",
        "estimate-label-errors": "identity_bias.analysis.estimate_label_errors",
        "audit-verdict-parsing": "identity_bias.analysis.audit_verdict_parsing",
        "relabel-choice-answers": "identity_bias.analysis.relabel_choice_answers",
        "verdict-interface-metrics": "identity_bias.analysis.verdict_interface_metrics",
        "compare-reasoning-modes": "identity_bias.analysis.compare_reasoning_modes",
        "compare-reference-answers": "identity_bias.analysis.compare_reference_answers",
        "verdict-disagreement": "identity_bias.analysis.verdict_disagreement",
    },
    "data": {"prepare-rewardbench": "identity_bias.data.prepare_rewardbench"},
    "run-extended": {
        "evaluate-majority-voting": "identity_bias.workflows.extended.evaluate_majority_voting",
        "repair-rejected-solutions": "identity_bias.workflows.extended.repair_rejected_solutions",
        "evaluate-critique-interfaces": "identity_bias.workflows.extended.evaluate_critique_interfaces",
        "compare-session-context": "identity_bias.workflows.extended.compare_session_context",
    },
    "analyze-extended": {
        "dataset-repair-and-calibration": "identity_bias.analysis.extended.dataset_repair_and_calibration",
        "plot-critique-metrics": "identity_bias.analysis.extended.plot_critique_metrics",
        "solver-critic-summary": "identity_bias.analysis.extended.solver_critic_summary",
        "cross-model-authorship-effects": "identity_bias.analysis.extended.cross_model_authorship_effects",
        "ambiguity-sensitivity-checks": "identity_bias.analysis.extended.ambiguity_sensitivity_checks",
        "pipeline-utility-and-calibration": "identity_bias.analysis.extended.pipeline_utility_and_calibration",
        "field-order-and-majority-voting": "identity_bias.analysis.extended.field_order_and_majority_voting",
        "strong-author-label-effects": "identity_bias.analysis.extended.strong_author_label_effects",
        "repair-quality-and-calibration": "identity_bias.analysis.extended.repair_quality_and_calibration",
        "audit-critique-parsing": "identity_bias.analysis.extended.audit_critique_parsing",
        "abstention-and-vote-aggregation": "identity_bias.analysis.extended.abstention_and_vote_aggregation",
        "parser-policy-sensitivity": "identity_bias.analysis.extended.parser_policy_sensitivity",
        "solution-length-sensitivity": "identity_bias.analysis.extended.solution_length_sensitivity",
        "verdict-repair-agreement": "identity_bias.analysis.extended.verdict_repair_agreement",
        "authorship-and-ambiguity": "identity_bias.analysis.extended.authorship_and_ambiguity",
        "mixed-effects-and-robustness": "identity_bias.analysis.extended.mixed_effects_and_robustness",
        "build-critique-tables": "identity_bias.analysis.extended.build_critique_tables",
    },
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", help="Root containing logs and analysis outputs")
    parser.add_argument(
        "--models-file", help="JSON mapping profile names to model fields"
    )
    parser.add_argument("command", choices=["list", *COMMANDS])
    parser.add_argument("tool", nargs="?")
    parser.add_argument("tool_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "list":
        for group, entries in COMMANDS.items():
            print(group + ": " + ", ".join(sorted(entries)))
        return
    tools = COMMANDS[args.command]
    if not args.tool:
        print(", ".join(sorted(tools)))
        return
    name = args.tool.replace("_", "-")
    if name not in tools:
        parser.error(f"unknown {args.command} tool: {args.tool}")
    if args.workspace:
        os.environ["IDENTITY_BIAS_WORKSPACE"] = str(
            Path(args.workspace).expanduser().resolve()
        )
    if args.models_file:
        os.environ["IDENTITY_BIAS_MODELS"] = str(
            Path(args.models_file).expanduser().resolve()
        )
    module = tools[name]
    if args.command.startswith("analyze") and any(
        a in ("-h", "--help") for a in args.tool_args
    ):
        source = (
            Path(__file__).parent.parent.joinpath(*module.split(".")).with_suffix(".py")
        )
        print(ast.get_docstring(ast.parse(source.read_text())) or name)
        print(
            "\nUse --workspace PATH before the command to select the input/output root."
        )
        return
    previous = sys.argv
    try:
        sys.argv = [module, *args.tool_args]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
