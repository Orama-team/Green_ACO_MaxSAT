"""
runner.py
=========
Stage orchestration: parallelism, checkpointing and provenance recording.

Two behaviours matter for reproducibility at this scale.

**Resumability.** A full sweep is long enough that interruptions are expected,
so every task writes an atomic JSON checkpoint on completion and a ``.started``
marker while in flight. On restart, tasks with a checkpoint are skipped and
tasks left with a marker but no checkpoint (i.e. the worker died) are re-run.

**Recorded provenance.** Absolute energy values depend on the hardware and on
how many workers run at once, so every sweep records the machine, library
versions, arguments, seed and worker count next to its results.
"""

from __future__ import annotations

import json
import os
import platform
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from . import paths


def safe_name(text: str) -> str:
    """Filesystem-safe token for an instance/run identifier."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", text)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so an interrupted write cannot corrupt state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


@dataclass
class TaskState:
    """Filesystem state of a single task."""

    checkpoint: Path
    started: Path

    def is_done(self) -> bool:
        return self.checkpoint.exists()

    def was_started(self) -> bool:
        return self.started.exists()


class CheckpointStore:
    """Per-stage checkpoint directory."""

    def __init__(self, stage: str, root: Path | None = None):
        self.dir = (root or paths.CHECKPOINTS) / stage
        self.dir.mkdir(parents=True, exist_ok=True)

    def state(self, *parts) -> TaskState:
        token = "__".join(safe_name(str(p)) for p in parts)
        return TaskState(self.dir / f"{token}.json",
                         self.dir / f"{token}.started")

    def load(self, state: TaskState) -> Optional[Dict]:
        if not state.is_done():
            return None
        try:
            with open(state.checkpoint, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            # A truncated checkpoint counts as absent so the task re-runs.
            return None

    def begin(self, state: TaskState) -> None:
        state.started.parent.mkdir(parents=True, exist_ok=True)
        state.started.write_text(str(os.getpid()), encoding="utf-8")

    def finish(self, state: TaskState, payload: Any) -> None:
        atomic_write_json(state.checkpoint, payload)
        if state.started.exists():
            state.started.unlink()

    def completed_count(self) -> int:
        return len(list(self.dir.glob("*.json")))


def default_workers(requested: int | None = None) -> int:
    """Worker count, defaulting to ``min(4, cpu_count - 1)``.

    The default is only a convenience: the value actually used is recorded in
    ``results/MANIFEST.json``. This matters because energy is measured, and CPU
    contention inflates the joules each iteration costs, so the number of
    workers changes how much search fits in a fixed budget. ``--n-jobs`` should
    therefore be set explicitly and kept fixed across runs that are meant to be
    comparable. See ``tests/probe_cpu_contention.py`` for the measured effect.
    """
    if requested:
        return max(1, int(requested))
    cpu = os.cpu_count() or 4
    return max(1, min(4, cpu - 1))

def run_tasks(
    tasks: List,
    fn: Callable,
    store: CheckpointStore,
    n_jobs: int | None = None,
    size_threshold: int | None = None,
    timeout_s: Optional[float] = None,
    task_key: Callable = lambda t: (str(t),),
    verbose: bool = True,
) -> List[Dict]:
    """Execute ``fn(task)`` for every task, resuming from checkpoints.

    ``fn`` must be picklable, so any non-trivial state it needs (an energy
    meter, for instance) is created inside the worker rather than captured
    from the parent process: a live CodeCarbon tracker holds locks and cannot
    be serialised to a worker.

    Tasks whose instance is larger than ``size_threshold`` clauses run serially
    on the main process: they are both the slowest and the most memory-hungry,
    so parallelising them risks exhausting RAM.
    """
    pending = []
    results: List[Dict] = []

    for task in tasks:
        state = store.state(*task_key(task))
        cached = store.load(state)
        if cached is not None:
            results.append(cached)
        else:
            pending.append((task, state))

    if verbose and results:
        print(f"  [resume] {len(results)} task(s) restored from checkpoints")

    if not pending:
        return results

    workers = default_workers(n_jobs)
    if size_threshold:
        serial = [p for p in pending
                  if getattr(p[0], "n_clauses", 0) > size_threshold]
        serial_ids = {id(p[0]) for p in serial}
        parallel = [p for p in pending if id(p[0]) not in serial_ids]
        # Largest first: a slow task discovered early surfaces problems sooner.
        serial.sort(key=lambda p: -getattr(p[0], "n_clauses", 0))
    else:
        serial, parallel = [], pending

    if verbose and serial:
        print(f"  [split] {len(serial)} large task(s) run serially "
              f"(>{size_threshold} clauses)")

    for task, state in serial:
        results.append(_run_one(fn, task, state, store, timeout_s, verbose))

    if parallel:
        results.extend(_run_parallel(fn, parallel, store, workers, timeout_s,
                                     verbose))

    return results


def _run_one(fn, task, state, store, timeout_s, verbose) -> Dict:
    store.begin(state)
    t0 = time.time()
    try:
        payload = fn(task)
    except Exception as exc:  # pragma: no cover - surfaced in the output
        payload = {"__error__": f"{type(exc).__name__}: {exc}"}
    store.finish(state, payload)
    if verbose:
        print(f"  [done] {_describe(task)} in {time.time() - t0:.1f}s")
    return payload


def _run_parallel(fn, pending, store, workers, timeout_s, verbose) -> List[Dict]:
    try:
        from joblib import Parallel, delayed
    except ImportError:  # pragma: no cover - optional dependency
        if verbose:
            print("  [parallel] joblib unavailable, running serially")
        return [_run_one(fn, t, s, store, timeout_s, verbose)
                for t, s in pending]

    payloads = Parallel(n_jobs=workers, backend="loky", verbose=0)(
        delayed(_worker)(fn, task, state, str(store.dir))
        for task, state in pending
    )
    return [p for p in payloads if p is not None]


def _worker(fn, task, state, store_dir):
    """Child-process entry point: run one task and persist its checkpoint."""
    store = CheckpointStore.__new__(CheckpointStore)
    store.dir = Path(store_dir)
    return _run_one(fn, task, state, store, None, False)


def _describe(task) -> str:
    for attr in ("benchmark", "name", "instance"):
        if hasattr(task, attr):
            return str(getattr(task, attr))[:48]
    return str(task)[:48]

def machine_info() -> Dict[str, Any]:
    """Hardware and interpreter details relevant to measured energy."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import codecarbon

        info["codecarbon"] = getattr(codecarbon, "__version__", "unknown")
    except Exception:
        info["codecarbon"] = None
    return info


def write_manifest(results_dir: Path, payload: Dict[str, Any]) -> Path:
    """Persist a run manifest next to the results it describes."""
    paths.ensure_dirs()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / "MANIFEST.json"
    atomic_write_json(target, payload)
    return target
