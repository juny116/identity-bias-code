"""Pairwise (A vs B) preference interfaces for the subjective/no-gold extension.

These are the subjective analogues of the objective output interfaces (codex
mapping). The judged content (prompt + response A + response B) is identical
across interfaces; only the output contract changes. Field ORDER matters under
constrained decoding (e.g. score/feedback BEFORE vs the verdict).

All expose, after parsing: choice in {"A","B"} and confidence in [0,1].
"""
from dataclasses import dataclass

PREAMBLE = (
    "You are comparing two assistant responses (A and B) to the same user "
    "prompt and deciding which response is better overall (more helpful, correct, "
    "and appropriate). There is no single gold answer; use your judgment."
)


def build_pairwise_message(prompt, resp_a, resp_b):
    return (f"USER PROMPT:\n{prompt}\n\n"
            f"RESPONSE A:\n{resp_a}\n\n"
            f"RESPONSE B:\n{resp_b}")


_CHOICE = {"type": "string", "enum": ["A", "B"]}
_CONF = {"type": "number", "minimum": 0, "maximum": 1}


@dataclass
class PairwiseInterface:
    name: str
    system_prompt: str
    schema: dict


CHOICE_ONLY = PairwiseInterface(
    "choice_only",
    PREAMBLE + "\n\nReturn only which response is better (A or B) and your confidence.",
    {"type": "object", "properties": {"choice": _CHOICE, "confidence": _CONF},
     "required": ["choice", "confidence"], "additionalProperties": False},
)

RUBRIC_THEN_CHOICE = PairwiseInterface(
    "rubric_then_choice",
    PREAMBLE + "\n\nFirst assess both responses on: (1) helpfulness, (2) correctness, "
    "(3) appropriateness/safety. Summarize that assessment, then choose A or B and give confidence.",
    {"type": "object", "properties": {"rubric_assessment": {"type": "string"},
                                       "choice": _CHOICE, "confidence": _CONF},
     "required": ["rubric_assessment", "choice", "confidence"], "additionalProperties": False},
)

SCORE_THEN_CHOICE = PairwiseInterface(
    "score_then_choice",
    PREAMBLE + "\n\nFirst rate each response on an integer 0-10 overall-quality scale "
    "(score_A and score_B). Then choose which is better (A or B) and give confidence.",
    {"type": "object", "properties": {"score_A": {"type": "integer", "minimum": 0, "maximum": 10},
                                       "score_B": {"type": "integer", "minimum": 0, "maximum": 10},
                                       "choice": _CHOICE, "confidence": _CONF},
     "required": ["score_A", "score_B", "choice", "confidence"], "additionalProperties": False},
)

CHOICE_THEN_FEEDBACK = PairwiseInterface(
    "choice_then_feedback",
    PREAMBLE + "\n\nFirst choose which response is better (A or B). Then give brief "
    "feedback on how the weaker response could be improved. Also report confidence.",
    {"type": "object", "properties": {"choice": _CHOICE, "feedback": {"type": "string"},
                                       "confidence": _CONF},
     "required": ["choice", "feedback", "confidence"], "additionalProperties": False},
)

FEEDBACK_THEN_CHOICE = PairwiseInterface(
    "feedback_then_choice",
    PREAMBLE + "\n\nFirst give brief critique/improvement feedback for each response "
    "(strengths and weaknesses). Then choose which is better (A or B) and give confidence.",
    {"type": "object", "properties": {"feedback": {"type": "string"},
                                       "choice": _CHOICE, "confidence": _CONF},
     "required": ["feedback", "choice", "confidence"], "additionalProperties": False},
)

PAIRWISE_INTERFACES = {i.name: i for i in [
    CHOICE_ONLY, RUBRIC_THEN_CHOICE, SCORE_THEN_CHOICE,
    CHOICE_THEN_FEEDBACK, FEEDBACK_THEN_CHOICE,
]}
