import sys
from pathlib import Path

from mlbisect.config import Configuration
from mlbisect.executors.fake import FakeExecutor
from mlbisect.executors.local import LocalExecutor
from mlbisect.manifest import Manifest


def _manifest(command: str, timeout_s: float = 5) -> Manifest:
    return Manifest.model_validate(
        {
            "version": 1,
            "name": "executor-test",
            "metric": {
                "name": "accuracy",
                "direction": "higher_is_better",
                "min_effect": 0.02,
            },
            "run": {"command": command, "timeout_s": timeout_s},
            "axes": [
                {"name": "data", "old": "v1", "new": "v2"},
                {"name": "deps", "old": "old", "new": "new"},
            ],
        }
    )


def _python_command(script: Path) -> str:
    return f'"{sys.executable}" "{script}"'


def test_local_executor_sets_environment_and_reads_output_file(tmp_path: Path) -> None:
    script = tmp_path / "eval.py"
    script.write_text(
        "import json, os\n"
        "assert os.environ['MLBISECT_AXIS_DATA'] == 'v2'\n"
        "assert os.environ['MLBISECT_AXIS_DEPS'] == 'old'\n"
        "assert os.environ['MLBISECT_CONFIG'] == 'data'\n"
        "with open(os.environ['MLBISECT_OUTPUT'], 'w') as handle:\n"
        "    json.dump({'metric': int(os.environ['MLBISECT_SEED'])}, handle)\n",
        encoding="utf-8",
    )
    executor = LocalExecutor(_manifest(_python_command(script)), tmp_path)

    result = executor.run(Configuration.from_axes(["data"]), seed=7)

    assert result.ok
    assert result.metric == 7
    assert result.error is None


def test_local_executor_reads_final_stdout_line(tmp_path: Path) -> None:
    script = tmp_path / "eval.py"
    script.write_text(
        "print('progress')\nprint('{\"metric\": 0.8125}')\n", encoding="utf-8"
    )
    executor = LocalExecutor(_manifest(_python_command(script)), tmp_path)

    result = executor.run(Configuration(), seed=0)

    assert result.ok
    assert result.metric == 0.8125


def test_local_executor_reports_nonzero_exit(tmp_path: Path) -> None:
    script = tmp_path / "eval.py"
    script.write_text("raise SystemExit('bad eval')\n", encoding="utf-8")
    executor = LocalExecutor(_manifest(_python_command(script)), tmp_path)

    result = executor.run(Configuration(), seed=0)

    assert not result.ok
    assert result.metric is None
    assert "status 1" in (result.error or "")


def test_fake_executor_is_deterministic_and_applies_interaction() -> None:
    executor = FakeExecutor(
        baseline=1.0,
        effects={"data": -0.1},
        interactions={frozenset({"data", "deps"}): -0.4},
        noise_sd=0.2,
    )
    config = Configuration.from_axes(["data", "deps"])

    first = executor.run(config, seed=13)
    second = executor.run(config, seed=13)

    assert first == second
    assert first.metric is not None
    assert first.metric < 0.5
