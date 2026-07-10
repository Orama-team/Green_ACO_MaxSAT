"""
parallel.py
===========
Parallel execution helpers for running Green ACO across many
(instance, budget, config) combinations — used by the profiling
(n_runs=10, 4 instances x 2 regimes), ablation, and Optuna study
described in the notebook, which would otherwise be prohibitively
slow run sequentially ("Cette cellule est longue (plusieurs minutes)").

Uses `concurrent.futures.ProcessPoolExecutor` to keep the dependency
footprint minimal (no extra parallel-computing library required).
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(fn: Callable[[T], R], tasks: Iterable[T],
                  max_workers: int = None, show_progress: bool = True) -> List[R]:
    """Run `fn(task)` for every task in `tasks`, in parallel processes.
    Preserves the *completion* order (not input order) for progress
    visibility; sort results downstream if strict ordering matters."""
    max_workers = max_workers or min(32, (os.cpu_count() or 4))
    results: List[R] = []
    tasks = list(tasks)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, task): i for i, task in enumerate(tasks)}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if show_progress and (done % max(1, len(tasks) // 10) == 0 or done == len(tasks)):
                print(f"  [parallel] {done}/{len(tasks)} tasks complete")

    return results


def chunk(iterable: List, n_chunks: int) -> List[List]:
    """Split a list into `n_chunks` roughly equal chunks (for batching
    work across processes when tasks are very fine-grained)."""
    k, m = divmod(len(iterable), n_chunks)
    return [iterable[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n_chunks)]
