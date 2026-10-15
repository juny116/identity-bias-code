import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from identity_bias.analysis.label_corrections import apply
from identity_bias.analysis.records import load_judge_records
from identity_bias.critic.identity_critic import IdentityCritic
from identity_bias.data.base import Problem
from identity_bias.io.jsonl import read_records
from identity_bias.io.result_logger import ResultLogger
from identity_bias.llm.base import LLMResponse
from identity_bias.runtime.models import load_model_profiles
from identity_bias.runtime.token_budget import available_tokens
from identity_bias.solver.cot_solver import CoTSolver
from identity_bias.types import Dataset, IdentityCondition, LLMProvider


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append(messages)
        return LLMResponse(next(self.responses), prompt_tokens=10, completion_tokens=5)


class CoreTests(unittest.TestCase):
    def test_solver_to_two_stage_critic_and_resume(self):
        problem = Problem("p1", "What is 6 times 7?", "42", "aime")
        solver = CoTSolver(FakeLLM([r"The answer is \boxed{42}"]), "solver")
        with patch(
            "identity_bias.solver.cot_solver.check_answer",
            side_effect=lambda ds, p, g: p == g,
        ):
            solution = solver.solve(problem, Dataset.AIME)
        self.assertTrue(solution.is_correct)
        llm = FakeLLM(
            [
                '{"is_correct": true, "confidence": 0.8}',
                '{"is_correct": false, "corrected_answer": "41", "confidence": 0.6}',
            ]
        )
        result = IdentityCritic(llm, "critic", two_stage=True).critique(
            problem, solution, IdentityCondition.SELF
        )
        self.assertTrue(result.initial_is_correct)
        self.assertFalse(result.is_correct)
        self.assertEqual(len(llm.calls), 2)
        self.assertNotEqual(llm.calls[0], llm.calls[1])
        with tempfile.TemporaryDirectory() as directory:
            log = ResultLogger(directory, "run")
            log.log_solution(problem, solution)
            log.log_critique(result)
            self.assertEqual(log.get_completed_ids(), {"p1"})
            self.assertEqual(log.get_completed_ids("critique"), {"p1_self"})
            self.assertEqual(
                ResultLogger.load_solutions(log.log_file), ([problem], [solution])
            )
            self.assertEqual(ResultLogger.load_critiques(log.log_file), [result])
            with log.log_file.open("a") as stream:
                stream.write("\n{incomplete")
            self.assertEqual(log.get_completed_ids(), {"p1"})
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                list(read_records(log.log_file))

    def test_profiles_are_external_and_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "profiles.json"
            file.write_text(
                json.dumps(
                    {
                        "local": {
                            "provider": "vllm",
                            "model_name": "dummy",
                            "max_tokens": 123,
                            "base_url": "http://localhost:1234/v1",
                            "api_key_env": "TEST_MODEL_KEY",
                        }
                    }
                )
            )
            with patch.dict(os.environ, {"TEST_MODEL_KEY": "test-key"}):
                profiles = load_model_profiles(file)
            one = profiles["local"]()
            self.assertEqual(one.provider, LLMProvider.VLLM)
            self.assertEqual(one.api_key, "test-key")
            one.temperature = 2
            self.assertEqual(profiles["local"]().temperature, 0)
            self.assertEqual(
                profiles["local"](base_url="override").base_url, "override"
            )
            self.assertEqual(profiles["local"]().base_url, "http://localhost:1234/v1")

    def test_corrections_do_not_mutate_records(self):
        data = pd.DataFrame(
            [dict(dataset="gpqa", problem_id="1", solver="s", actually_correct=False)]
        )
        out, count, flipped = apply(data, corrections={"gpqa|1|s": True})
        self.assertEqual((count, flipped), (1, 1))
        self.assertFalse(data.iloc[0].actually_correct)
        self.assertTrue(out.iloc[0].actually_correct)

    def test_directory_reader_and_empty_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(load_judge_records(directory, corrections={}).empty)
            Path(directory, "a.jsonl").write_text(
                '{"dataset":"gpqa","problem_id":"1","solver":"s","actually_correct":false}\n{broken\n'
            )
            out = load_judge_records(directory, corrections={"gpqa|1|s": True})
            self.assertEqual(len(out), 1)
            self.assertTrue(out.iloc[0].actually_correct)

    def test_budget_preserves_estimation_rule(self):
        with patch(
            "identity_bias.runtime.token_budget.encoder",
            return_value=SimpleNamespace(encode=lambda text: list(text)),
        ):
            self.assertEqual(
                available_tokens([{"content": "1234"}], 1000),
                1000 - int((4 + 4) * 1.25) - 128,
            )

    def test_pairwise_order_and_truncation(self):
        from identity_bias.workflows.evaluate_pairwise_interfaces import (
            PAIRWISE_INTERFACES,
            run_one,
        )

        pair = dict(
            prompt="question", response_A="one", response_B="two", chosen_side="B"
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"choice":"A"}'),
                    finish_reason="stop",
                )
            ]
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: response)
            )
        )
        with patch(
            "identity_bias.workflows.evaluate_pairwise_interfaces.available_tokens",
            return_value=1000,
        ):
            result = run_one(
                client, "dummy", PAIRWISE_INTERFACES["choice_only"], pair, {}, 100, True
            )
            self.assertEqual(result["picked_orig"], "B")
            self.assertTrue(result["picked_chosen"])
            response.choices[0].finish_reason = "length"
            result = run_one(
                client, "dummy", PAIRWISE_INTERFACES["choice_only"], pair, {}, 100, True
            )
            self.assertFalse(result["parse_ok"])
            self.assertIsNone(result["picked_orig"])

    def test_all_failed_generation_does_not_crash(self):
        from identity_bias.workflows import generate_solutions as workflow

        problem = Problem("p1", "q", "1", "aime")
        with tempfile.TemporaryDirectory() as directory:
            args = [
                "generate",
                "--model",
                "fake",
                "--dataset",
                "aime",
                "--log-dir",
                directory,
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(
                    workflow,
                    "MODEL_PRESETS",
                    {"fake": lambda **kwargs: SimpleNamespace(temperature=0)},
                ),
                patch.object(workflow, "load_dataset_problems", return_value=[problem]),
                patch.object(workflow, "create_llm"),
                patch.object(
                    workflow.CoTSolver,
                    "solve",
                    side_effect=RuntimeError("test failure"),
                ),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                workflow.main()
            self.assertIn("No solutions were generated", output.getvalue())

    def test_cli_help_does_not_create_workspaces(self):
        from identity_bias.cli import COMMANDS

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            env = {
                **os.environ,
                "PYTHONPATH": str(root),
                "IDENTITY_BIAS_WORKSPACE": directory,
            }
            commands = [["list"]] + [
                [group, name, "--help"]
                for group, tools in COMMANDS.items()
                for name in tools
            ]
            for command in commands:
                with self.subTest(command=command):
                    proc = subprocess.run(
                        [sys.executable, "-m", "identity_bias", *command],
                        cwd=directory,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
