"""Diet Import: kept by rpath() and credited by Ecosim (#53).

A predator that takes part of its diet from outside the model ("Import")
must keep its Ecopath balance in Ecosim at baseline effort. Ecopath charges
local prey the raw diet fractions (the import-normalised diet is only for
trophic levels, as in Rpath). Ecosim credits the imported consumption as
``QB * import * B``, proportional to the predator's biomass and independent
of local prey (EwE-F ``derivs``: ``qq = consumption + QBoutside * biomass``).
"""

import numpy as np
import pytest

import pypath.core.ecosim as ecosim_mod
import pypath.core.ecosim_deriv as _ecosim_deriv_mod
from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

YEARS = range(2000, 2004)


def _importer_model(import_frac=0.5):
    """Phyto -> Fish (diet: Phyto and Import) -> Detritus, one fleet on Fish."""
    p = create_rpath_params(["Phyto", "Fish", "Detritus", "Fleet"], [1, 0, 2, 3])
    m = p.model
    m.loc[:, "Biomass"] = [10.0, 1.0, 5.0, np.nan]
    m.loc[:, "PB"] = [1.0, 1.0, np.nan, np.nan]
    m.loc[:, "QB"] = [np.nan, 5.0, np.nan, np.nan]
    m.loc[:, "BioAcc"] = [0.0, 0.0, 0.0, np.nan]
    m.loc[:, "Unassim"] = [0.0, 0.2, 0.0, np.nan]
    m.loc[:, "DetInput"] = [np.nan, np.nan, 0.0, np.nan]
    m.loc[:, "Detritus"] = [1.0, 1.0, 0.0, 0.0]
    m.loc[:, "Fleet"] = [0.0, 0.2, 0.0, np.nan]
    if "Fleet.disc" in m.columns:
        m.loc[:, "Fleet.disc"] = [0.0, 0.0, 0.0, np.nan]
    d = p.diet
    gcol = d.columns[0]
    d.loc[:, "Fish"] = 0.0
    d.loc[d[gcol] == "Phyto", "Fish"] = 1.0 - import_frac
    d.loc[d[gcol] == "Import", "Fish"] = import_frac
    return p


def _year_ratios(balanced, params):
    scenario = rsim_scenario(balanced, params, years=YEARS)
    out = rsim_run(scenario, years=YEARS)
    annual = np.asarray(out.annual_Biomass, dtype=float)
    return annual[-1, 1:3] / annual[0, 1:3]  # (Phyto, Fish)


def _derivative_inputs(scenario):
    """The params, forcing and fishing dicts rsim_run hands deriv_vector."""
    captured = {}
    original = ecosim_mod.deriv_vector

    def spy(state, params, forcing, fishing, *args, **kwargs):
        captured.setdefault("args", (params, forcing, fishing))
        return original(state, params, forcing, fishing, *args, **kwargs)

    ecosim_mod.deriv_vector = spy

    _ecosim_deriv_mod.deriv_vector = spy
    try:
        rsim_run(scenario, years=range(YEARS.start, YEARS.start + 1))
    finally:
        ecosim_mod.deriv_vector = original
        _ecosim_deriv_mod.deriv_vector = original
    return captured["args"]


def test_rpath_keeps_the_import_row_when_fleets_exist():
    balanced = rpath(_importer_model(), eco_name="import")
    fish = list(balanced.Group).index("Fish")
    assert balanced.DC[-1, fish] == pytest.approx(0.5)


def test_ecopath_charges_local_prey_the_raw_diet_fraction():
    """Fish eats 0.5 * QB * B = 2.5 of Phyto, so Phyto's EE is 2.5 / 10."""
    balanced = rpath(_importer_model(), eco_name="import")
    phyto = list(balanced.Group).index("Phyto")
    assert balanced.EE[phyto] == pytest.approx(0.25)


def test_importer_holds_its_baseline_in_ecosim():
    p = _importer_model()
    balanced = rpath(p, eco_name="import")
    np.testing.assert_allclose(_year_ratios(balanced, p), [1.0, 1.0], rtol=1e-6)


def test_import_intake_is_proportional_to_predator_biomass():
    """With a 100% Import diet, intake is QB * B at any biomass."""
    from pypath.core.ecosim_deriv import deriv_vector

    p = _importer_model(import_frac=1.0)
    balanced = rpath(p, eco_name="import")
    scenario = rsim_scenario(balanced, p, years=YEARS)
    params, forcing, fishing = _derivative_inputs(scenario)
    fish = list(balanced.Group).index("Fish") + 1
    qb = float(scenario.params.FtimeQBOpt[fish])

    state = scenario.start_state.Biomass.copy()
    for factor in (1.0, 2.0, 0.25):
        trial = state.copy()
        trial[fish] *= factor
        rates: dict = {}
        deriv_vector(trial, params, forcing, fishing, rates=rates)
        assert rates["FoodGain"][fish] == pytest.approx(qb * trial[fish], rel=1e-9)
