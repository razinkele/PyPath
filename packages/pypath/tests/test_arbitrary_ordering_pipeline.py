"""Regression test: full Ecopath+Ecosim pipeline with non-canonical group ordering.

Guards against regressions where code assumes groups are in canonical
[living, dead, fleet] order.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario, rsim_run
from pypath.core.params import create_rpath_params


def _build_detritus_first():
    """Build a model with Detritus at index 0 (non-canonical)."""
    params = create_rpath_params(
        groups=["Detritus", "Phyto", "Zoo", "Fish", "Fleet"],
        types=[2, 1, 0, 0, 3],
    )
    m = params.model
    m.loc[m["Group"] == "Detritus", "Biomass"] = 100.0
    m.loc[m["Group"] == "Phyto", ["Biomass", "PB"]] = [10.0, 50.0]
    m.loc[m["Group"] == "Phyto", "EE"] = 0.8
    m.loc[m["Group"] == "Zoo", ["Biomass", "PB", "QB"]] = [5.0, 10.0, 30.0]
    m.loc[m["Group"] == "Zoo", "EE"] = 0.9
    m.loc[m["Group"] == "Fish", ["Biomass", "PB", "QB"]] = [2.0, 1.0, 5.0]
    m.loc[m["Group"] == "Fish", "EE"] = 0.5
    m["BioAcc"] = 0.0
    m["Unassim"] = 0.2
    m.loc[m["Group"] == "Detritus", "Unassim"] = 0.0
    m.loc[m["Group"] == "Phyto", "Unassim"] = 0.0
    m.loc[m["Group"] == "Fleet", ["BioAcc", "Unassim"]] = np.nan
    m["Detritus"] = 1.0
    m.loc[m["Group"] == "Fleet", "Detritus"] = np.nan

    # Diet: Zoo eats Phyto+Detritus, Fish eats Zoo
    params.diet["Zoo"] = [0.2, 0.8, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 0.0, 1.0, 0.0, 0.0]

    # Fishing: Fleet catches Fish
    m.loc[m["Group"] == "Fish", "Fleet"] = 0.3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = rpath(params)

    return balanced, params


def test_ecopath_balances_with_detritus_first():
    """Ecopath should balance when detritus is at index 0."""
    balanced, _ = _build_detritus_first()
    assert balanced is not None
    living_mask = balanced.type < 2
    assert np.all(np.isfinite(balanced.EE[living_mask]))


def test_ecosim_runs_with_detritus_first():
    """Ecosim should run without crashes when detritus is at index 0."""
    balanced, params = _build_detritus_first()
    scenario = rsim_scenario(balanced, params, years=range(1, 6))
    output = rsim_run(scenario, method="AB")
    bio = output.out_Biomass

    ngroups = balanced.NUM_GROUPS
    for i in range(1, ngroups + 1):
        if bio[0, i] > 0:
            assert bio[-1, i] > 0, (
                f"Group {i} ({scenario.params.spname[i]}) crashed: "
                f"B_init={bio[0, i]:.4f}, B_final={bio[-1, i]:.6e}"
            )


def test_fleet_first_ordering():
    """Ecosim should also work when fleet is at index 0."""
    params = create_rpath_params(
        groups=["Fleet", "Detritus", "Phyto", "Zoo"],
        types=[3, 2, 1, 0],
    )
    m = params.model
    m.loc[m["Group"] == "Detritus", "Biomass"] = 50.0
    m.loc[m["Group"] == "Phyto", ["Biomass", "PB"]] = [10.0, 100.0]
    m.loc[m["Group"] == "Phyto", "EE"] = 0.5
    m.loc[m["Group"] == "Zoo", ["Biomass", "PB", "QB"]] = [3.0, 5.0, 20.0]
    m.loc[m["Group"] == "Zoo", "EE"] = 0.8
    m["BioAcc"] = 0.0
    m["Unassim"] = 0.2
    m.loc[m["Group"] == "Detritus", "Unassim"] = 0.0
    m.loc[m["Group"] == "Phyto", "Unassim"] = 0.0
    m.loc[m["Group"] == "Fleet", ["BioAcc", "Unassim"]] = np.nan
    m["Detritus"] = 1.0
    m.loc[m["Group"] == "Fleet", "Detritus"] = np.nan
    params.diet["Zoo"] = [0.0, 0.1, 0.9, 0.0]
    m.loc[m["Group"] == "Zoo", "Fleet"] = 0.2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        balanced = rpath(params)

    scenario = rsim_scenario(balanced, params, years=range(1, 4))
    output = rsim_run(scenario, method="AB")
    bio = output.out_Biomass

    for i in range(1, balanced.NUM_GROUPS + 1):
        if bio[0, i] > 0:
            assert bio[-1, i] > 0, f"Group {i} crashed with fleet-first ordering"
