"""Defaults, tuning coherence and serialisation in ``config.py``.

Two facts are load-bearing for the paper's reproducibility story and worth
pinning down: the solver's defaults **are** the tuned values, and a config
rebuilds itself from its own serialisation without losing a field.
"""

from __future__ import annotations

import pytest

from greenaco.config import (
    BUDGETS_JOULES,
    OPERATOR_POOL,
    OPTUNA_SEARCH_SPACE,
    TUNED_PARAMS,
    GreenACOConfig,
)


def test_tuned_values_reproduce_the_shipped_params_file():
    """The in-code defaults must equal ``configs/best_params.json``: if they
    drift, a plain run silently stops reproducing the paper."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "best_params.json"
    recorded = {k: float(v) for k, v in json.loads(path.read_text()).items()}
    for key, value in TUNED_PARAMS.items():
        assert recorded[key] == pytest.approx(value), key


def test_every_tuned_param_lies_inside_its_search_space():
    """A tuned optimum outside its own search space means the search missed."""
    for key, value in TUNED_PARAMS.items():
        low, high = OPTUNA_SEARCH_SPACE[key]
        assert low <= value <= high, key


def test_defaults_use_the_tuned_values():
    cfg = GreenACOConfig()
    for key, value in TUNED_PARAMS.items():
        assert getattr(cfg, key) == pytest.approx(value), key


def test_config_round_trips_through_a_dict():
    cfg = GreenACOConfig(budget_j=400.0, seed=7)
    rebuilt = GreenACOConfig.from_dict(cfg.to_dict())
    assert rebuilt == cfg


def test_from_dict_ignores_unknown_keys():
    """Checkpoints from newer code must still load in older code."""
    cfg = GreenACOConfig.from_dict({"seed": 1, "future_knob": True})
    assert cfg.seed == 1


def test_budgets_are_the_three_reported_values():
    assert BUDGETS_JOULES == [400.0, 1000.0, 2000.0]


def test_operator_pool_lists_only_implemented_operators():
    from greenaco.operators import OPERATOR_FUNCTIONS

    assert set(OPERATOR_POOL) <= set(OPERATOR_FUNCTIONS)