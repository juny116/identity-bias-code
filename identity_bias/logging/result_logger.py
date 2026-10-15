"""JSONL logger for experiment results."""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

from identity_bias.data.base import Problem, Solution
from identity_bias.critic.identity_critic import CritiqueResult


class ResultLogger:
    """Logs experiment results to JSONL files."""

    def __init__(self, log_dir: str | Path, experiment_name: str,
                 resume: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.jsonl"

        if not resume and self.log_file.exists():
            self.log_file.unlink()

    def get_completed_ids(self, record_type: str = "solution") -> set[str]:
        """Get IDs of already-completed records for resume."""
        completed = set()
        if not self.log_file.exists():
            return completed
        with open(self.log_file) as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record["type"] == record_type:
                    if record_type == "solution":
                        completed.add(record["problem"]["id"])
                    elif record_type == "critique":
                        crit = record["critique"]
                        # key = (problem_id, condition)
                        completed.add(f"{crit['problem_id']}_{crit['identity_condition']}")
        return completed

    def log_solution(self, problem: Problem, solution: Solution) -> None:
        """Log a generated solution."""
        record = {
            "type": "solution",
            "problem": asdict(problem),
            "solution": asdict(solution),
            "timestamp": datetime.now().isoformat(),
        }
        self._write(record)

    def log_critique(self, critique: CritiqueResult) -> None:
        """Log a critique result."""
        record = {
            "type": "critique",
            "critique": asdict(critique),
            "timestamp": datetime.now().isoformat(),
        }
        self._write(record)

    def log_metrics(self, metrics: dict) -> None:
        """Log aggregated metrics."""
        record = {
            "type": "metrics",
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self._write(record)

    def _write(self, record: dict) -> None:
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def load_solutions(log_file: str | Path) -> tuple[list[Problem], list[Solution]]:
        """Load problems and solutions from a log file."""
        problems = []
        solutions = []
        with open(log_file) as f:
            for line in f:
                record = json.loads(line)
                if record["type"] == "solution":
                    problems.append(Problem(**record["problem"]))
                    solutions.append(Solution(**record["solution"]))
        return problems, solutions

    @staticmethod
    def load_critiques(log_file: str | Path) -> list[CritiqueResult]:
        """Load critique results from a log file."""
        critiques = []
        with open(log_file) as f:
            for line in f:
                record = json.loads(line)
                if record["type"] == "critique":
                    critiques.append(CritiqueResult(**record["critique"]))
        return critiques
