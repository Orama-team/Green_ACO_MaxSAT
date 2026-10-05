"""
energy.py
=========
Energy measurement and green-metric computation.

Energy is **measured**, not modelled: a CodeCarbon ``EmissionsTracker`` reports
the power drawn by the CPU while an operator call executes. This means absolute
Joule values are a property of the machine and of how many workers run
concurrently, so they are not bit-reproducible across hardware. Solution
quality, operator choice, and therefore every ranking derived from EI/J *are*
reproducible for a given seed. The machine details and worker count are
recorded in ``results/MANIFEST.json`` so any reported number is interpretable.

One tracker is created per process and reused for every call: constructing a
fresh tracker per call re-triggers hardware detection and dominates the
measurement.

Carbon accounting follows the standard formula
``CO2[g] = E[kWh] * CI[gCO2/kWh]``, with the regional intensity taken from
``REGIONAL_CI``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Tuple

# Grid carbon intensities, in gCO2eq per kWh. The default region (DZ) follows
# the value used throughout the original experiments.
REGIONAL_CI = {
    "FR": 56, "NO": 26, "GB": 233, "DE": 385,
    "US": 386, "DZ": 486, "CN": 581, "DEFAULT": 400,
}

JOULES_PER_KWH = 3_600_000.0


def carbon_intensity_ug_per_joule(region: str = "DZ") -> float:
    """Carbon intensity expressed in micrograms CO2eq per Joule."""
    ci = REGIONAL_CI.get(region.upper(), REGIONAL_CI["DEFAULT"])
    return ci / JOULES_PER_KWH * 1e6


class EnergyMeter:
    """Process-local energy meter backed by CodeCarbon.

    Falls back to a clearly-labelled estimate when CodeCarbon is unavailable,
    so the pipeline still runs (with a warning) on a machine where it cannot be
    installed. The active mode is exposed as :attr:`mode` and recorded in the
    run manifest, so a fallback run is never mistaken for a measured one.
    """

    def __init__(self, region: str = "DZ", log_level: str = "error"):
        self.region = region.upper()
        self._tracker = None
        self._error = None
        self._available = False
        self._task_counter = 0
        self._task_name = None
        try:
            from codecarbon import EmissionsTracker

            self._tracker = EmissionsTracker(log_level=log_level,
                                              save_to_file=False)
            self._tracker.start()
            # CodeCarbon 3.x measures per-task via start_task/stop_task, which
            # is what lets a single tracker be reused across many operator
            # calls. 2.x exposes the running total instead.
            self._task_mode = hasattr(self._tracker, "start_task")
            self._available = True
        except Exception as exc:  # pragma: no cover - environment dependent
            self._task_mode = False
            self._error = str(exc)

    @property
    def available(self) -> bool:
        """True when real hardware measurement is in use."""
        return self._available

    @property
    def mode(self) -> str:
        return "codecarbon" if self._available else "estimated"

    def measure(self, fn: Callable, *args, **kwargs) -> Tuple[object, float]:
        """Run ``fn`` and return ``(result, joules_consumed)``.

        Without CodeCarbon the energy is estimated from the call's wall time at
        a nominal 20 W, which keeps the budget loop finite and comparable in
        structure if not in absolute value.
        """
        if self._available:
            if self._task_mode:
                self._task_counter += 1
                name = f"op{self._task_counter}"
                self._tracker.start_task(name)
                try:
                    result = fn(*args, **kwargs)
                finally:
                    data = self._tracker.stop_task(name)
                joules = max(float(data.energy_consumed) * JOULES_PER_KWH, 1e-9)
                return result, joules

            before = float(self._tracker.final_emissions_data.energy_consumed or 0.0)
            result = fn(*args, **kwargs)
            after = float(self._tracker.final_emissions_data.energy_consumed or 0.0)
            joules = max((after - before) * JOULES_PER_KWH, 1e-9)
            return result, joules

        import time

        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        joules = max((time.perf_counter() - t0) * 20.0, 1e-9)
        return result, joules

    def finalize(self) -> None:
        """Flush and release any hardware-monitoring resources."""
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception:  # pragma: no cover - best effort
                pass


def compute_green_metrics(qualite_solution: int, n_clauses: int,
                          energie_joules: float,
                          region: str = "DZ") -> Dict[str, float]:
    """Compute quality, quality-per-Joule and the CO2 footprint of a run."""
    qualite_pct = 100.0 * qualite_solution / n_clauses if n_clauses > 0 else 0.0
    energy_jwh = energie_joules / JOULES_PER_KWH
    ci = REGIONAL_CI.get(region.upper(), REGIONAL_CI["DEFAULT"])
    co2_g = energy_jwh * ci
    score_per_j = qualite_pct / energie_joules if energie_joules > 0 else 0.0

    return {
        "qualite_pct": round(qualite_pct, 4),
        "energie_joules": round(energie_joules, 6),
        "score_per_joule": round(score_per_j, 6),
        "co2_micrograms": round(co2_g * 1e6, 4),
        "region": region.upper(),
        "ci_gco2_per_kwh": ci,
    }