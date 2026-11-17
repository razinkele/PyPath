"""Tests for the prey-switching service."""

import numpy as np
import pytest

from pypath_shiny.services.prey_switching import (
    SwitchingStore,
    above_range,
    consumers_from_params,
    effective_values,
    fold_edits,
    on_model_change,
    pending,
    set_override,
    snapshot_source,
    source_key,
    summary,
    summary_line,
    switch_power_with_overrides,
)


def test_consumers_are_pp_type_zero(switching_scenario):
    assert consumers_from_params(switching_scenario.params) == ["Cod", "Seals"]


def test_source_keys():
    assert source_key(None) == "model"
    assert source_key({"path": "/x.eweaccdb"}, 16) == "ewemdb|/x.eweaccdb|16"
    assert (
        source_key({"kind": "ewef_example", "path": "/ex"}, 1) == "ewef_example|/ex|1"
    )


def test_untouched_means_none(switching_scenario):
    assert switch_power_with_overrides(switching_scenario.params, {}) is None


def test_override_written_by_name_on_a_copy(switching_scenario):
    p = switching_scenario.params
    before = p.SwitchPower.copy()
    arr = switch_power_with_overrides(p, {"Cod": 1.0, "Phyto": 1.0})
    i = [str(n).strip() for n in p.spname].index("Cod")
    assert arr[i] == 1.0 and arr.sum() == 1.0  # Phyto (producer) skipped
    assert np.array_equal(p.SwitchPower, before)


def test_set_override_ignores_none():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    assert set_override(s, "model", "Cod", None, 0.0) == s


def test_untouched_rounded_value_writes_no_override():
    assert (
        set_override(SwitchingStore(), "k", "Cod", 0.3, 0.30000001) == SwitchingStore()
    )


def test_untouched_imported_value_above_range_is_kept():
    assert set_override(SwitchingStore(), "k", "Cod", 2.5, 2.5) == SwitchingStore()


def test_editing_back_to_the_displayed_source_value_removes_the_override():
    s = set_override(SwitchingStore(), "k", "Cod", 0.5, 0.30000001)
    s = set_override(s, "k", "Cod", 0.3, 0.30000001)
    assert "Cod" not in s.overrides.get("k", {})


def test_out_of_range_is_refused():
    with pytest.raises(ValueError, match="between 0 and 2; 3.0 was not applied"):
        set_override(SwitchingStore(), "model", "Cod", 3.0, 0.0)


def test_fold_skips_boxes_untouched_since_the_render():
    s = set_override(SwitchingStore(), "model", "Cod", 0.1234, 0.0)  # shown as 0.123
    new, errors = fold_edits(s, "model", {"Cod": 0.123}, {"Cod": 0.0})
    assert new == s and errors == {}


def test_fold_applies_a_revert():
    s, _ = fold_edits(SwitchingStore(), "model", {"Cod": 1.0}, {"Cod": 0.0})
    assert dict(s.overrides) == {"model": {"Cod": 1.0}}
    s, _ = fold_edits(s, "model", {"Cod": 0.0}, {"Cod": 0.0})
    assert s == SwitchingStore()


def test_fold_applies_edits_and_reports_refusals():
    new, errors = fold_edits(
        SwitchingStore(),
        "model",
        {"Cod": 1.0, "Seals": 3.0},
        {"Cod": 0.0, "Seals": 0.0},
    )
    assert dict(new.overrides) == {"model": {"Cod": 1.0}}
    assert errors == {
        "Seals": "Switching power for Seals must be between 0 and 2; 3.0 was not applied."
    }


def test_effective_values_apply_overrides_on_the_base():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    assert effective_values(s, "model", {"Cod": 0.0, "Seals": 0.0}) == {
        "Cod": 1.0,
        "Seals": 0.0,
    }


def test_summary_line():
    assert summary_line(None) == "Prey switching: none"
    assert (
        summary_line({"Cod": 1.0, "Seals": 0.5}) == "Prey switching: Cod 1, Seals 0.5"
    )


def test_summary_and_snapshot_tolerate_missing_or_short_switchpower(switching_scenario):
    p = switching_scenario.params
    p.SwitchPower = None
    assert summary(p) == {}
    assert snapshot_source(p) == {"Cod": 0.0, "Seals": 0.0}
    p.SwitchPower = np.zeros(2)
    assert summary(p) == {}


def test_imported_values_above_range_are_reported():
    assert above_range({"Cod": 2.5, "Seals": 1.0}) == {"Cod": 2.5}


def test_pending(switching_scenario):
    p = switching_scenario.params
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    base = {"Cod": 0.0, "Seals": 0.0}
    assert pending(s, "model", base, p)
    p.SwitchPower = switch_power_with_overrides(p, {"Cod": 1.0})
    assert not pending(s, "model", base, p)


def test_load_clears_and_rebalance_keeps_consumers():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    s = set_override(s, "model", "Seals", 0.5, 0.0)
    s = set_override(s, "ewemdb|/x|1", "Cod", 2.0, 1.0)
    cleared, note = on_model_change(
        s, rebalance=False, consumers=["Cod"], keep_ewe=True
    )
    assert cleared == SwitchingStore() and note.startswith(
        "Prey switching settings cleared (3 group(s))"
    )
    kept, note = on_model_change(s, rebalance=True, consumers=["Cod"], keep_ewe=False)
    assert dict(kept.overrides) == {"model": {"Cod": 1.0}}
    assert note == (
        "Prey switching settings kept for 1 group(s) after re-balancing; "
        "dropped for 1 that are gone or no longer consumers."
    )
    assert on_model_change(
        SwitchingStore(), rebalance=False, consumers=[], keep_ewe=False
    ) == (SwitchingStore(), None)


def test_a_load_right_after_a_rebalance_clears():
    s = set_override(SwitchingStore(), "model", "Cod", 1.0, 0.0)
    s, _ = on_model_change(s, rebalance=True, consumers=["Cod"], keep_ewe=False)
    s, note = on_model_change(s, rebalance=False, consumers=["Cod"], keep_ewe=False)
    assert s == SwitchingStore() and note is not None


def test_consumers_from_model_accepts_balanced_and_params(switching_scenario):
    import pandas as pd

    from pypath_shiny.services.prey_switching import consumers_from_model

    df = pd.DataFrame({"Group": ["A ", "B", "C"], "Type": [0, 1, 2]})
    assert consumers_from_model({"Group": df["Group"], "Type": df["Type"]}) == ["A"]

    class Bal:
        Group = ["Phyto", "Cod ", "Det"]
        type = [1, 0, 2]

    assert consumers_from_model(Bal()) == ["Cod"]
