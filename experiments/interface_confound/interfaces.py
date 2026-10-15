"""Interface definitions for the judge-interface-confound experiment.

Each interface is a (system prompt + JSON output schema) pair given to a judge
that must decide whether a candidate solution is correct. The judged content
(problem + candidate solution + candidate final answer) is identical across all
interfaces; only the output contract changes.

Field ORDER in the schema matters: vLLM structured decoding emits properties in
the order they appear in `properties`, so e.g. repair_then_verdict forces the
model to produce the corrected answer before committing to a verdict.

All interfaces expose, after parsing:
  - is_correct (bool)        : the verdict on the candidate solution
  - confidence (float|None)  : stated confidence in [0,1] when the schema has it
  - corrected_answer (str|None): only for repair interfaces

Constrained decoding is enforced via OpenAI-style response_format json_schema
with strict=True, so a schema-valid object is guaranteed unless generation is
truncated (which we log as a non-verdict, never as accept).
"""

from dataclasses import dataclass


SHARED_TASK_PREAMBLE = (
    "You are an expert grader checking a candidate solution to a reasoning "
    "problem. The solution may contain subtle errors in logic, arithmetic, or "
    "final answer. Do not assume it is correct. Judge whether the candidate's "
    "FINAL ANSWER is correct for the given problem."
)


def build_user_message(question: str, chain_of_thought: str, final_answer: str) -> str:
    """The judged content. Identical across all interfaces and anonymous
    (no solver identity is revealed)."""
    return (
        f"PROBLEM:\n{question}\n\n"
        f"CANDIDATE SOLUTION:\n{chain_of_thought}\n\n"
        f"CANDIDATE FINAL ANSWER:\n{final_answer}"
    )


@dataclass
class Interface:
    name: str
    system_prompt: str
    schema: dict  # JSON schema for response_format


# ---------------------------------------------------------------------------
# 1. verdict_only — minimal: just the verdict + confidence
# ---------------------------------------------------------------------------
VERDICT_ONLY = Interface(
    name="verdict_only",
    system_prompt=(
        SHARED_TASK_PREAMBLE
        + "\n\nReturn only your verdict: whether the candidate final answer is "
        "correct, and your confidence in that verdict from 0 to 1."
    ),
    schema={
        "type": "object",
        "properties": {
            "is_correct": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["is_correct", "confidence"],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------------------
# 2. rubric_verdict — assess against an explicit rubric first, then verdict
# ---------------------------------------------------------------------------
RUBRIC_VERDICT = Interface(
    name="rubric_verdict",
    system_prompt=(
        SHARED_TASK_PREAMBLE
        + "\n\nFirst assess the solution against these criteria: (1) is the "
        "final answer correct for the problem; (2) is the reasoning logically "
        "valid; (3) are the calculations free of arithmetic errors. Summarize "
        "your rubric assessment, then give your verdict and confidence."
    ),
    schema={
        "type": "object",
        "properties": {
            "rubric_assessment": {"type": "string"},
            "is_correct": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["rubric_assessment", "is_correct", "confidence"],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------------------
# 3. score_then_verdict — numeric quality score FIRST, then boolean verdict
# ---------------------------------------------------------------------------
SCORE_THEN_VERDICT = Interface(
    name="score_then_verdict",
    system_prompt=(
        SHARED_TASK_PREAMBLE
        + "\n\nFirst rate the overall quality of the solution on an integer "
        "scale from 0 (completely wrong) to 10 (fully correct). Then give your "
        "binary verdict on whether the final answer is correct, and your "
        "confidence."
    ),
    schema={
        "type": "object",
        "properties": {
            "quality_score": {"type": "integer", "minimum": 0, "maximum": 10},
            "is_correct": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["quality_score", "is_correct", "confidence"],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------------------
# 4. verdict_plus_repair — verdict FIRST, then a corrected answer
# ---------------------------------------------------------------------------
VERDICT_PLUS_REPAIR = Interface(
    name="verdict_plus_repair",
    system_prompt=(
        SHARED_TASK_PREAMBLE
        + "\n\nGive your verdict on whether the candidate final answer is "
        "correct. Then, if it is wrong, provide the corrected final answer "
        "(if it is correct, repeat the candidate final answer). Also report "
        "your confidence."
    ),
    schema={
        "type": "object",
        "properties": {
            "is_correct": {"type": "boolean"},
            "corrected_answer": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["is_correct", "corrected_answer", "confidence"],
        "additionalProperties": False,
    },
)


# ---------------------------------------------------------------------------
# 5. repair_then_verdict — produce the correct answer FIRST, then verdict
# ---------------------------------------------------------------------------
REPAIR_THEN_VERDICT = Interface(
    name="repair_then_verdict",
    system_prompt=(
        SHARED_TASK_PREAMBLE
        + "\n\nFirst, independently determine what you believe the correct "
        "final answer to the problem is. Then decide whether the candidate's "
        "final answer matches it (i.e. whether the candidate is correct), and "
        "report your confidence."
    ),
    schema={
        "type": "object",
        "properties": {
            "corrected_answer": {"type": "string"},
            "is_correct": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["corrected_answer", "is_correct", "confidence"],
        "additionalProperties": False,
    },
)


def with_reasoning_field(iface: "Interface") -> "Interface":
    """Deliberation-in-schema variant of an interface (codex Option 2 for the
    think ablation): prepend a free-text 'reasoning' field as the FIRST schema
    property so the constrained-JSON judge must externalize step-by-step
    deliberation BEFORE any verdict/score/correction field. This gives native
    reasoning a place to land under constrained decoding. It is reported as
    'deliberation-in-schema robustness', not as native chain-of-thought.
    The judged content and field order of the original fields are preserved."""
    new_props = {"reasoning": {"type": "string"}}
    new_props.update(iface.schema["properties"])  # reasoning first, rest unchanged
    new_schema = {
        "type": "object",
        "properties": new_props,
        "required": ["reasoning"] + list(iface.schema["required"]),
        "additionalProperties": False,
    }
    new_prompt = (
        iface.system_prompt
        + "\n\nThink step by step FIRST: in the 'reasoning' field, work through "
        "the problem and the candidate's solution in detail before filling in "
        "any other field. Your final verdict must follow from that reasoning."
    )
    return Interface(name=iface.name, system_prompt=new_prompt, schema=new_schema)


INTERFACES = {
    iface.name: iface
    for iface in [
        VERDICT_ONLY,
        RUBRIC_VERDICT,
        SCORE_THEN_VERDICT,
        VERDICT_PLUS_REPAIR,
        REPAIR_THEN_VERDICT,
    ]
}

# Default ordered list used by the minimal first run.
FIRST_RUN_INTERFACES = [
    "verdict_only",
    "rubric_verdict",
    "score_then_verdict",
    "verdict_plus_repair",
    "repair_then_verdict",
]
