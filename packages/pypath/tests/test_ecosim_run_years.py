"""``rsim_run(years=...)`` runs from the scenario's first year only.

The run reads forcing and effort from the scenario's first month on and
uses only ``len(years)``, so a range that starts later would silently rerun
the first years' forcing. It is refused instead.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run, rsim_scenario
from tests.continuous_support import needs_reco, reco_plain

pytestmark = needs_reco


def _scen(years):
    model, params = reco_plain()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_scenario(model, params, years=years)


@pytest.mark.parametrize("method", ["RK4", "AB", "lsoda"])
@pytest.mark.parametrize("first", [1, 2000])
def test_an_offset_range_is_refused(first, method):
    scen = _scen(range(first, first + 4))
    with pytest.raises(ValueError, match="scenario's first year"):
        rsim_run(scen, method=method, years=range(first + 2, first + 4))


def test_a_range_from_one_is_refused_when_the_scenario_starts_later():
    scen = _scen(range(2000, 2004))
    with pytest.raises(ValueError, match="2000"):
        rsim_run(scen, method="RK4", years=range(1, 3))


@pytest.mark.parametrize("first", [1, 2000])
def test_runs_from_the_first_year_still_work(first):
    scen = _scen(range(first, first + 4))
    part = rsim_run(scen, method="RK4", years=range(first, first + 1))
    full = rsim_run(scen, method="RK4")
    assert part.out_Biomass.shape[0] == 13
    assert full.out_Biomass.shape[0] == 49
    np.testing.assert_array_equal(part.out_Biomass, full.out_Biomass[:13])


def _uniform_ecospace(n_groups_plus_1, n_patches=2):
    from pypath.spatial import EcospaceParams, create_1d_grid

    return EcospaceParams(
        grid=create_1d_grid(n_patches=n_patches, spacing=1.0),
        habitat_preference=np.ones((n_groups_plus_1, n_patches)),
        habitat_capacity=np.ones((n_groups_plus_1, n_patches)),
        dispersal_rate=np.zeros(n_groups_plus_1),
        advection_enabled=np.zeros(n_groups_plus_1, dtype=bool),
        gravity_strength=np.zeros(n_groups_plus_1),
    )


def test_rsim_run_advanced_refuses_an_offset_range():
    from pypath.core.ecosim_advanced import rsim_run_advanced

    scen = _scen(range(2000, 2002))
    with pytest.raises(ValueError, match="scenario's first year"):
        rsim_run_advanced(scen, years=range(2001, 2002))
    assert rsim_run_advanced(scen).out_Biomass.shape[0] == 25
    assert rsim_run_advanced(scen, years=range(2000, 2001)).out_Biomass.shape[0] == 13


def test_rsim_run_spatial_refuses_an_offset_range_with_ecospace():
    from pypath.spatial import rsim_run_spatial

    scen = _scen(range(2000, 2002))
    ng = scen.params.NUM_GROUPS + 1
    with pytest.raises(ValueError, match="scenario's first year"):
        rsim_run_spatial(scen, ecospace=_uniform_ecospace(ng), years=range(2001, 2002))
    full = rsim_run_spatial(scen, ecospace=_uniform_ecospace(ng))
    assert full.out_Biomass.shape[0] == 25
    part = rsim_run_spatial(
        scen, ecospace=_uniform_ecospace(ng), years=range(2000, 2001)
    )
    assert part.out_Biomass.shape[0] == 13
