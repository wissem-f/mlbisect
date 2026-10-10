"""Append-only JSONL persistence and run-result cache."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from mlbisect.config import Configuration
from mlbisect.executors.base import RunResult


class StoreError(ValueError):
    """Raised when a run log cannot be read."""


class RunStore:
    """Persist runs in JSONL and cache them by configuration and seed."""

    def __init__(
        self,
        path: str | Path,
        load_existing: bool = True,
        fingerprint: str | None = None,
    ) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self._cache: dict[tuple[frozenset[str], int], RunResult] = {}
        self._history: dict[tuple[frozenset[str], int], list[RunResult]] = {}
        self._run_count = 0
        if load_existing:
            self._load()

    @property
    def run_count(self) -> int:
        """Return the number of executor attempts represented by the log."""
        return self._run_count

    def append(self, result: RunResult, timestamp: str | None = None) -> None:
        """Append a result and make it available through the cache."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, object] = {
            "config": sorted(result.config),
            "seed": result.seed,
            "metric": result.metric,
            "ok": result.ok,
            "duration_s": result.duration_s,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        }
        if self.fingerprint is not None:
            record["fingerprint"] = self.fingerprint
        if result.error is not None:
            record["error"] = result.error
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
        self._cache[self._key(result.config, result.seed)] = result
        key = self._key(result.config, result.seed)
        self._history.setdefault(key, []).append(result)
        self._run_count += 1

    def get(
        self, config: Configuration | Iterable[str], seed: int
    ) -> RunResult | None:
        """Return the cached result for a configuration and seed, if present."""
        axes = self._axes(config)
        return self._cache.get(self._key(axes, seed))

    def lookup(
        self, config: Configuration | Iterable[str], seed: int
    ) -> RunResult | None:
        """Alias for :meth:`get` used by callers describing cache access."""
        return self.get(config, seed)

    def history(
        self, config: Configuration | Iterable[str], seed: int
    ) -> tuple[RunResult, ...]:
        """Return all attempts for a configuration and seed in log order."""
        axes = self._axes(config)
        return tuple(self._history.get(self._key(axes, seed), ()))

    def attempts_for_config(
        self, config: Configuration | Iterable[str]
    ) -> int:
        """Return all logged attempts for a configuration."""
        axes = self._axes(config)
        return sum(
            len(history)
            for (cached_axes, _), history in self._history.items()
            if cached_axes == axes
        )

    @property
    def results(self) -> tuple[RunResult, ...]:
        """Return one cached result per configuration and seed."""
        return tuple(self._cache.values())

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if self.fingerprint is not None and isinstance(record, dict):
                            record_fingerprint = record.get("fingerprint")
                            if record_fingerprint != self.fingerprint:
                                raise StoreError(
                                    "run log fingerprint does not match the "
                                    "current manifest"
                                )
                        result = self._result_from_record(record)
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as exc:
                        raise StoreError(
                            f"invalid run log record at {self.path}:{line_number}: "
                            f"{exc}"
                        ) from exc
                    self._cache[self._key(result.config, result.seed)] = result
                    key = self._key(result.config, result.seed)
                    self._history.setdefault(key, []).append(result)
                    self._run_count += 1
        except OSError as exc:
            raise StoreError(f"cannot read run log '{self.path}': {exc}") from exc

    @staticmethod
    def _result_from_record(record: object) -> RunResult:
        if not isinstance(record, dict):
            raise TypeError("record must be a JSON object")
        config = record["config"]
        if not isinstance(config, list) or not all(
            isinstance(axis, str) for axis in config
        ):
            raise TypeError("config must be a list of axis names")
        return RunResult(
            config=frozenset(config),
            seed=int(record["seed"]),
            metric=None if record["metric"] is None else float(record["metric"]),
            ok=bool(record["ok"]),
            duration_s=float(record["duration_s"]),
            error=None if record.get("error") is None else str(record["error"]),
        )

    @staticmethod
    def _axes(config: Configuration | Iterable[str]) -> frozenset[str]:
        return config.axes if isinstance(config, Configuration) else frozenset(config)

    @staticmethod
    def _key(
        config: frozenset[str] | Iterable[str], seed: int
    ) -> tuple[frozenset[str], int]:
        return frozenset(config), seed
