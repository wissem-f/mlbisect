from pathlib import Path

from mlbisect.config import Configuration
from mlbisect.executors.base import RunResult
from mlbisect.store import RunStore, StoreError


def test_run_store_appends_and_reloads_cache(tmp_path: Path) -> None:
    path = tmp_path / ".mlbisect" / "runs.jsonl"
    store = RunStore(path)
    result = RunResult(
        config=frozenset({"deps", "data"}),
        seed=4,
        metric=0.81,
        ok=True,
        duration_s=1.25,
    )

    store.append(result, timestamp="2026-01-01T00:00:00+00:00")
    assert store.get(Configuration.from_axes(["data", "deps"]), 4) == result
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1

    reloaded = RunStore(path)
    assert reloaded.lookup(["data", "deps"], 4) == result
    assert reloaded.results == (result,)


def test_run_store_cache_uses_latest_duplicate(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.jsonl")
    first = RunResult(frozenset(), 0, 0.8, True, 0.1)
    second = RunResult(frozenset(), 0, 0.7, True, 0.2)

    store.append(first)
    store.append(second)

    assert store.get(Configuration(), 0) == second
    assert len(store.results) == 1


def test_run_store_rejects_mismatched_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    store = RunStore(path, fingerprint="manifest-a")
    store.append(RunResult(frozenset(), 0, 1.0, True, 0.0))

    try:
        RunStore(path, fingerprint="manifest-b")
    except StoreError as exc:
        assert "fingerprint" in str(exc)
    else:
        raise AssertionError("mismatched fingerprint was accepted")
