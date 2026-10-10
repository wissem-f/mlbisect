from pathlib import Path

from typer.testing import CliRunner

from mlbisect import __version__
from mlbisect.cli import app


def _write_run_manifest(path: Path) -> None:
    path.write_text(
        "version: 1\n"
        "name: run-test\n"
        "metric:\n"
        "  name: accuracy\n"
        "  direction: higher_is_better\n"
        "  min_effect: 0.1\n"
        "run:\n"
        "  command: python eval.py\n"
        "axes:\n"
        "  - name: data\n"
        "    old: v1\n"
        "    new: v2\n"
        "  - name: deps\n"
        "    old: v1\n"
        "    new: v2\n"
        "budget:\n"
        "  max_runs: 50\n"
        "  baseline_runs: 5\n"
        "  min_runs_per_config: 3\n"
        "  max_runs_per_config: 3\n"
        "  batch_size: 1\n",
        encoding="utf-8",
    )


def test_version() -> None:
    result = CliRunner().invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_init_writes_template(tmp_path: Path) -> None:
    path = tmp_path / "mlbisect.yaml"

    result = CliRunner().invoke(app, ["init", str(path)])

    assert result.exit_code == 0
    contents = path.read_text(encoding="utf-8")
    assert "MLBISECT_AXIS_<NAME>" in contents
    assert "name: data" in contents


def test_validate_prints_run_plan(tmp_path: Path) -> None:
    path = tmp_path / "mlbisect.yaml"
    path.write_text(
        "version: 1\n"
        "name: cli-test\n"
        "metric:\n"
        "  name: accuracy\n"
        "  direction: higher_is_better\n"
        "  min_effect: 0.02\n"
        "run:\n"
        "  command: python eval.py\n"
        "axes:\n"
        "  - name: data\n"
        "    old: v1\n"
        "    new: v2\n"
        "  - name: deps\n"
        "    old: old\n"
        "    new: new\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["validate", str(path)])

    assert result.exit_code == 0
    assert "Manifest valid: cli-test" in result.stdout
    assert "Worst-case runs: 40" in result.stdout


def test_noise_runs_an_aa_check(tmp_path: Path) -> None:
    (tmp_path / "eval.py").write_text(
        "print('{\"metric\": 0.9}')\n", encoding="utf-8"
    )
    path = tmp_path / "mlbisect.yaml"
    path.write_text(
        "version: 1\n"
        "name: noise-test\n"
        "metric:\n"
        "  name: accuracy\n"
        "  direction: higher_is_better\n"
        "  min_effect: 0.02\n"
        "run:\n"
        "  command: python eval.py\n"
        "axes:\n"
        "  - name: data\n"
        "    old: v1\n"
        "    new: v2\n"
        "  - name: deps\n"
        "    old: v1\n"
        "    new: v2\n"
        "budget:\n"
        "  max_runs: 10\n"
        "  baseline_runs: 3\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(app, ["noise", str(path)])

    assert result.exit_code == 0
    assert "Noise sd: 0" in result.stdout


def test_run_writes_reports_and_resume_reuses_completed_runs(
    tmp_path: Path, monkeypatch: object
) -> None:
    counter = tmp_path / "counter.txt"
    monkeypatch.setenv("MLBISECT_COUNTER", str(counter))
    (tmp_path / "eval.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "counter = Path(os.environ['MLBISECT_COUNTER'])\n"
        "count = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(count + 1))\n"
        "metric = 0.7 if os.environ['MLBISECT_CONFIG'] == 'data,deps' else 1.0\n"
        "print(json.dumps({'metric': metric}))\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "mlbisect.yaml"
    _write_run_manifest(manifest_path)
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"

    first = CliRunner().invoke(
        app,
        [
            "run",
            str(manifest_path),
            "--out",
            str(markdown_path),
            "--json",
            str(json_path),
        ],
    )
    first_count = int(counter.read_text())
    second = CliRunner().invoke(
        app,
        [
            "run",
            str(manifest_path),
            "--resume",
            "--out",
            str(markdown_path),
            "--json",
            str(json_path),
        ],
    )

    assert first.exit_code == 0, first.stdout
    assert second.exit_code == 0, second.stdout
    assert int(counter.read_text()) == first_count
    assert "Result: culprit found" in markdown_path.read_text(encoding="utf-8")
    assert '"status": "culprit_found"' in json_path.read_text(encoding="utf-8")
    assert (tmp_path / ".mlbisect" / "runs.jsonl").exists()


def test_run_dry_run_does_not_execute_or_write_reports(tmp_path: Path) -> None:
    sentinel = tmp_path / "executed.txt"
    (tmp_path / "eval.py").write_text(
        f"from pathlib import Path\nPath(r'{sentinel}').write_text('yes')\n"
        "print('{\"metric\": 1.0}')\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "mlbisect.yaml"
    _write_run_manifest(manifest_path)

    result = CliRunner().invoke(app, ["run", str(manifest_path), "--dry-run"])

    assert result.exit_code == 0
    assert "Dry run: no evaluations will be executed." in result.stdout
    assert "{data,deps}" in result.stdout
    assert not sentinel.exists()
    assert not (tmp_path / "report.md").exists()
