"""Checks on Ecopath input tables: scenario, diet composition, detritus fate.

The layouts are the EwE-F CSV inputs described in ``pypath.io.schemas``:
scenario rows are groups (-999 = missing), diet rows are prey (plus an
optional ``Import`` row) and columns predators, fate rows are groups and
columns detritus pools plus ``Export``.
"""

import numpy as np
import pandas as pd

from pypath.io.validators import (
    ValidationIssue,
    validate_all_ecopath_files,
    validate_detritus_fate,
    validate_diet_composition,
    validate_ecopath_scenario,
)

GROUPS = ["Cod", "Sprat", "Phyto", "Detritus"]


def _scenario(changes=None):
    df = pd.DataFrame(
        {
            "Biomass": [2.0, 5.0, 10.0, 50.0],
            "PB": [0.5, 1.0, 20.0, -999],
            "QB": [4.0, 6.0, -999, -999],
        },
        index=GROUPS,
    )
    for (group, column), value in (changes or {}).items():
        df[column] = df[column].astype(object)
        df.loc[group, column] = value
    return df


def _diet():
    return pd.DataFrame(
        {"Cod": [0.0, 0.8, 0.0, 0.0, 0.2], "Sprat": [0.0, 0.0, 0.7, 0.3, 0.0]},
        index=GROUPS + ["Import"],
    )


def _fate():
    return pd.DataFrame(
        {"Detritus": [1.0, 1.0, 1.0, 0.0], "Export": [0.0, 0.0, 0.0, 1.0]},
        index=GROUPS,
    )


def _errors(report):
    return [issue.message for issue in report.errors]


def test_consistent_tables_pass():
    report = validate_all_ecopath_files(_scenario(), _diet(), _fate(), GROUPS)
    assert report.passed, _errors(report)


# --- scenario ---


def test_negative_biomass_is_an_error():
    report = validate_ecopath_scenario(_scenario({("Sprat", "Biomass"): -1.0}))
    assert not report.passed
    assert any("biomass" in message.lower() for message in _errors(report))


def test_the_missing_marker_is_not_negative_biomass():
    report = validate_ecopath_scenario(_scenario({("Sprat", "Biomass"): -999}))
    assert report.passed, _errors(report)


def test_a_value_that_is_not_a_number_is_reported():
    report = validate_ecopath_scenario(
        _scenario({("Cod", "PB"): "unknown", ("Sprat", "QB"): "n/a"})
    )
    assert not report.passed
    messages = " ".join(_errors(report))
    assert "unknown" in messages and "n/a" in messages


def test_scenario_groups_other_than_the_expected_ones_are_errors():
    report = validate_ecopath_scenario(
        _scenario(), ["Cod", "Herring", "Phyto", "Detritus"]
    )
    assert not report.passed
    messages = " ".join(_errors(report))
    assert "Herring" in messages and "Sprat" in messages


# --- diet and fate ---


def test_a_missing_diet_fraction_is_an_error():
    diet = _diet()
    diet.loc["Phyto", "Sprat"] = np.nan
    diet.loc["Detritus", "Sprat"] = 1.0
    assert not validate_diet_composition(diet).passed


def test_a_missing_fate_fraction_is_an_error():
    fate = _fate()
    fate.loc["Cod", "Export"] = np.nan
    assert not validate_detritus_fate(fate).passed


def test_diet_labels_must_be_groups_or_import():
    diet = _diet().rename(index={"Phyto": "Algae"}, columns={"Cod": "Whale"})
    report = validate_diet_composition(diet, GROUPS)
    assert not report.passed
    messages = " ".join(_errors(report))
    assert "Algae" in messages and "Whale" in messages
    assert "Import" not in messages


def test_fate_rows_must_be_groups():
    fate = _fate().rename(index={"Sprat": "Herring"})
    report = validate_detritus_fate(fate, GROUPS)
    assert not report.passed
    assert "Herring" in " ".join(_errors(report))


def test_tables_for_other_groups_fail_the_combined_check():
    other = ["A", "B", "C", "D"]
    report = validate_all_ecopath_files(_scenario(), _diet(), _fate(), other)
    assert not report.passed


# --- messages ---


def test_an_issue_without_a_group_shows_its_level():
    assert str(ValidationIssue("error", "data", "x")) == "❌ ERROR: x"


def test_issues_name_the_group_from_a_group_column():
    scenario = _scenario({("Sprat", "Biomass"): np.nan}).reset_index(names="Group")
    report = validate_ecopath_scenario(scenario, GROUPS)
    assert [issue.group for issue in report.warnings] == ["Sprat"]
