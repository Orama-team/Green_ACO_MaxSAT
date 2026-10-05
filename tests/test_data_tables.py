"""Labelling, subset selection and reference tables.

``short_label`` and ``select_subset`` shape every figure axis and every scope
decision in the pipeline, so their edge behaviour is pinned: a missing core
instance is an error, not a silently smaller benchmark, and a CORE_INSTANCES
token missing from the manifest surfaces as ``ValueError``, not ``KeyError``.
"""

from __future__ import annotations

import pytest

from greenaco.data import (
    CORE_INSTANCES,
    load_manifest,
    select_subset,
    short_label,
)


def test_manifest_has_four_core_instances_marked_mandatory():
    manifest = load_manifest()
    core = manifest.query("mandatory == True")
    assert set(core["benchmark"]) == set(CORE_INSTANCES)
    assert len(core) == 4


def test_core_subset_matches_the_manifest_marks():
    manifest = load_manifest()
    assert select_subset(manifest, "core")["benchmark"].tolist() == \
        manifest.query("mandatory == True")["benchmark"].tolist()


def test_missing_core_instance_is_an_error_not_a_smaller_benchmark():
    manifest = load_manifest()
    first = manifest["benchmark"].iloc[0]
    pruned = manifest[manifest["benchmark"] != first]
    if first in CORE_INSTANCES:
        with pytest.raises(ValueError, match="missing core"):
            select_subset(pruned, "core")
    else:
        out = select_subset(pruned, "core")
        assert set(out["benchmark"]) == set(CORE_INSTANCES)


def test_ad_hoc_subset_by_name():
    manifest = load_manifest()
    name = manifest["benchmark"].iloc[0]
    out = select_subset(manifest, name)
    assert out["benchmark"].tolist() == [name]
    with pytest.raises(ValueError, match="No manifest rows matched"):
        select_subset(manifest, "no-such-instance")


def test_short_label_shortens_known_prefixes():
    assert short_label(
        "decision-tree-car-un-formula_0.8_2021_atleast_15_max-3_reduced_"
        "incomplete_tree") == "dt-car"
    assert short_label("min-fill-MinFill_R0_myciel5") == "minfill-myciel5"


def test_predictor_feature_table_lists_every_extracted_feature():
    """The published Table 2 must cover exactly the features the extractor
    computes. The display names follow the paper, so the check runs both ways:
    nothing computed is undocumented, nothing documented is uncomputed."""
    from greenaco.tables import _predictor_rows, table_predictor_features

    table = table_predictor_features()
    from budget_predictor.feature_extractor import (
        FEATURE_DESCRIPTIONS,
        FEATURE_NAMES,
    )

    rows = _predictor_rows()
    assert [row[0] for row in rows] == FEATURE_NAMES
    assert [row[1] for row in rows] == table["Feature"].tolist()
    assert [row[2] for row in rows] == [
        FEATURE_DESCRIPTIONS[name] for name in FEATURE_NAMES]