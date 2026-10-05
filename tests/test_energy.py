"""Energy accounting: the measurement wrapper and the CO2 arithmetic."""

from __future__ import annotations

import pytest

from greenaco.energy import (JOULES_PER_KWH, REGIONAL_CI, EnergyMeter,
                             carbon_intensity_ug_per_joule,
                             compute_green_metrics)


def test_green_metrics_match_hand_computation():
    """Reproduces a known row from the original measurements."""
    metrics = compute_green_metrics(85639, 89879, 6935.669253, region="DZ")
    assert metrics["qualite_pct"] == pytest.approx(95.2825, abs=1e-4)
    assert metrics["score_per_joule"] == pytest.approx(0.013738, abs=1e-6)
    assert metrics["co2_micrograms"] == pytest.approx(936315.3, abs=1.0)
    assert metrics["region"] == "DZ"


def test_score_per_joule_is_quality_over_energy():
    metrics = compute_green_metrics(50, 100, 400.0)
    assert metrics["score_per_joule"] == pytest.approx(50.0 / 400.0, abs=1e-6)


def test_carbon_follows_kwh_times_intensity():
    metrics = compute_green_metrics(1, 1, JOULES_PER_KWH, region="DZ")  # 1 kWh
    assert metrics["co2_micrograms"] == pytest.approx(
        REGIONAL_CI["DZ"] * 1e6, rel=1e-6)


def test_carbon_scales_linearly_with_energy():
    a = compute_green_metrics(1, 1, 1000.0)["co2_micrograms"]
    b = compute_green_metrics(1, 1, 2000.0)["co2_micrograms"]
    assert b == pytest.approx(2 * a, rel=1e-9)


def test_unknown_region_falls_back():
    assert compute_green_metrics(1, 1, 1000.0, region="ZZ")["ci_gco2_per_kwh"] \
        == REGIONAL_CI["DEFAULT"]


def test_zero_clauses_do_not_divide_by_zero():
    metrics = compute_green_metrics(0, 0, 100.0)
    assert metrics["qualite_pct"] == 0.0
    assert metrics["score_per_joule"] == 0.0


def test_carbon_intensity_conversion():
    # 486 gCO2/kWh == 135 ugCO2/J
    assert carbon_intensity_ug_per_joule("DZ") == pytest.approx(135.0, abs=0.01)


def test_meter_reports_its_mode():
    meter = EnergyMeter()
    assert meter.mode in {"codecarbon", "estimated"}


def test_meter_returns_the_function_result():
    meter = EnergyMeter()
    result, joules = meter.measure(lambda a, b: a + b, 2, 3)
    assert result == 5
    assert joules > 0


def test_meter_attributes_energy_per_call():
    """Two calls must not report identical joules from a shared tracker."""
    meter = EnergyMeter()
    if meter.mode != "codecarbon":
        pytest.skip("per-call attribution requires CodeCarbon")
    _, first = meter.measure(sum, range(2_000_000))
    _, second = meter.measure(sum, range(2_000_000))
    assert first > 0 and second > 0