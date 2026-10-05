"""Checkpointing and orchestration behaviour in ``runner.py``.

A full sweep is long enough that interruptions are expected, so these are the
properties worth proving: an interrupted run restarts work instead of
resuming garbage, a truncated checkpoint counts as absent, and a completed
run resumes nothing.
"""

from __future__ import annotations

from greenaco.runner import (
    CheckpointStore,
    atomic_write_json,
    default_workers,
    run_tasks,
    safe_name,
)


N_JOBS = 1  # Serial, in-process: parallel workers cannot pickle local test fns.


def test_safe_name_replaces_path_fragments():
    assert safe_name("min-fill/MinFill_R0_myciel5") == "min-fill_MinFill_R0_myciel5"
    assert safe_name("a b.c") == "a_b_c"


def test_atomic_write_round_trips(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"a": [1, 2, 3]})
    assert not (tmp_path / "state.json.tmp").exists()
    import json

    assert json.loads(target.read_text(encoding="utf-8")) == {"a": [1, 2, 3]}


def test_truncated_checkpoint_counts_as_absent(tmp_path):
    store = CheckpointStore("stage", root=tmp_path)
    state = store.state("task")
    state.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state.checkpoint.write_text("{not json", encoding="utf-8")
    assert state.is_done() is True  # the file exists
    assert store.load(state) is None  # ...but cannot be trusted


def test_run_tasks_executes_and_checkpoints(tmp_path):
    store = CheckpointStore("stage", root=tmp_path)
    out = run_tasks(["t1", "t2"], lambda task: {"echo": task},
                    store=store, n_jobs=N_JOBS, verbose=False)
    assert [row["echo"] for row in out] == ["t1", "t2"]
    assert store.completed_count() == 2


def test_run_tasks_resumes_completed_work(tmp_path):
    """A second call must not re-run tasks that already checkpointed."""
    store = CheckpointStore("stage", root=tmp_path)
    calls = []

    def fn(task):
        calls.append(task)
        return {"echo": task}

    run_tasks(["t1", "t2"], fn, store=store, n_jobs=N_JOBS, verbose=False)
    assert calls == ["t1", "t2"]
    run_tasks(["t1", "t2"], fn, store=store, n_jobs=N_JOBS, verbose=False)
    assert calls == ["t1", "t2"], "checkpointed tasks were re-run"


def test_run_tasks_retries_tasks_left_started_but_unfinished(tmp_path):
    """A stale ``.started`` marker with no checkpoint means the worker died,
    so the task must be re-run."""
    store = CheckpointStore("stage", root=tmp_path)
    state = store.state("doomed")
    store.begin(state)
    assert state.was_started() is True

    out = run_tasks(["doomed"], lambda task: {"ok": True},
                    store=store, n_jobs=N_JOBS, verbose=False)
    assert out == [{"ok": True}]


def test_run_tasks_reports_failures_without_aborting(tmp_path):
    store = CheckpointStore("stage", root=tmp_path)

    def fn(task):
        if task == "bad":
            raise RuntimeError("boom")
        return {"ok": True}

    out = run_tasks(["bad", "good"], fn, store=store, n_jobs=N_JOBS,
                    verbose=False)
    assert any("__error__" in row for row in out)
    assert {"ok": True} in out


def test_default_workers_is_bounded_and_positive():
    assert default_workers(3) == 3
    workers = default_workers(None)
    assert 1 <= workers <= 4