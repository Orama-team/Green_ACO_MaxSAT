"""
comparison/harness.py
=====================
Runs a comparison method on one instance under energy measurement.

Every method in the comparison is measured the same way as Green ACO: the call
is wrapped in the shared energy meter, and the resulting quality, quality per
Joule and CO2 footprint are computed with the same formulas. That is the point
of running the comparison inside this repository rather than importing
historical numbers: all rows then come from one machine, in one session, under
one measurement regime.

A hard wall-clock limit applies to each call. On expiry the best-so-far state
is harvested rather than discarded, so a slow method still yields a usable
solution; such rows are marked ``timeout=True`` and left out of the aggregate.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

from ..energy import EnergyMeter, compute_green_metrics
from ..wcnf import count_satisfied_clauses
from .registry import method_params


def run_with_timeout(fn, timeout_s: float):
    """Run ``fn`` under a wall-clock limit, returning ``(result, timed_out)``.

    A daemon thread is used so a runaway method cannot block the sweep. The
    shared dict lets a method publish its incumbent as it improves, so a
    timeout still yields a usable solution.
    """
    shared: Dict = {"best_assignment": None, "best_score": -1, "stats": None}
    outcome: Dict = {"result": None, "error": None}

    def _target():
        try:
            outcome["result"] = fn(_shared=shared)
        except TypeError:
            try:
                outcome["result"] = fn()
            except Exception as exc:  # pragma: no cover - surfaced in output
                outcome["error"] = exc
        except Exception as exc:  # pragma: no cover - surfaced in output
            outcome["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout_s)

    if thread.is_alive():
        return {"assignment": shared["best_assignment"],
                "stats": shared["stats"]}, True
    if outcome["error"] is not None:
        # Surface the real failure rather than reporting it as a timeout.
        raise outcome["error"]
    return outcome["result"], False


def run_method(bench, method: str, meter: Optional[EnergyMeter] = None,
               seed: int = 42, timeout_s: float = 300) -> Dict:
    """Run one comparison method on one instance and return its metrics row."""
    from .registry import get_solver

    meter = meter or EnergyMeter()
    fn = get_solver(method, seed=seed, timeout_s=int(timeout_s))
    formula = bench.instance.clauses

    t0 = time.time()

    # The timeout wrapper supplies a shared dict so a method can publish its
    # incumbent as it improves. Solvers that do not accept it are called
    # without. The signature is inspected up front rather than catching
    # TypeError, which would also swallow genuine errors raised inside the
    # solver and make a real failure look like a timeout.
    import inspect

    accepts_shared = "_shared" in inspect.signature(fn).parameters

    if accepts_shared:
        def _call(_shared=None):
            return fn(formula, bench.n_vars, _shared=_shared)
    else:
        def _call(_shared=None):
            return fn(formula, bench.n_vars)

    result, timed_out = run_with_timeout(_call, timeout_s)
    elapsed = time.time() - t0

    assignment = None
    stats: Dict = {}
    energy_j = 0.0

    if result:
        # GA solvers return (assignment, stats); the ACO variants return a
        # dict that already carries its own statistics.
        if isinstance(result, tuple):
            assignment, stats = result[0], (result[1] or {})
        elif isinstance(result, dict):
            assignment = result.get("assignment") or result.get("best_assignment")
            stats = result
        else:
            assignment, stats = result, {}

    if assignment is not None:
        def _count():
            return count_satisfied_clauses(formula, assignment)

        _res, energy_j = meter.measure(_count)

    if assignment is None:
        # Nothing usable: the method failed or produced no incumbent.
        return {
            "method": method,
            "family": None,
            "instance": bench.benchmark,
            "n_vars": bench.n_vars,
            "n_clauses": bench.n_clauses,
            "qualite_solution": None,
            "qualite_pct": None,
            "energie_joules": None,
            "score_per_joule": None,
            "co2_micrograms": None,
            "temps_exec": round(elapsed, 3),
            "timeout": True,
            "error": "no incumbent produced",
            "params": method_params(method),
        }

    best = count_satisfied_clauses(formula, assignment)
    green = compute_green_metrics(best, bench.n_clauses, energy_j,
                                  meter.region)

    # Built from `green` rather than recomputed, so quality and the derived
    # ratio cannot drift apart through rounding.
    return {
        "method": method,
        "family": None,
        "instance": bench.benchmark,
        "n_vars": bench.n_vars,
        "n_clauses": bench.n_clauses,
        "qualite_solution": best,
        "qualite_pct": green["qualite_pct"],
        "energie_joules": green["energie_joules"],
        "score_per_joule": green["score_per_joule"],
        "co2_micrograms": green["co2_micrograms"],
        "region": green["region"],
        "ci_gco2_per_kwh": green["ci_gco2_per_kwh"],
        "temps_exec": round(elapsed, 3),
        "timeout": bool(timed_out),
        "best_known": bench.best_known,
        "params": method_params(method),
    }