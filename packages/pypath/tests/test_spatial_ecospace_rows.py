"""EcospaceParams must have one row per Ecosim group.

Row g - 1 is Ecosim group g (``dispersal.calculate_spatial_flux``), fleets
included, with no "Outside" row. A run checks the row count before it starts:
fewer rows would fail inside the run, so they raise; more rows are not read,
so they only warn (callers that pad an extra row still run).
"""

import logging

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import EcospaceParams, create_regular_grid, rsim_run_spatial


@pytest.fixture(scope="module")
def scenario():
    """A 4-group model with one fleet: NUM_GROUPS = 4."""
    params = create_rpath_params(
        ["Fish", "Plankton", "Detritus", "Fleet"], [0, 1, 2, 3]
    )
    params.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 1.0, 5.0, 0.8]
    params.model.loc[1, ["Biomass", "PB", "EE"]] = [5.0, 50.0, 0.6]
    params.model.loc[2, "Biomass"] = 1.0
    params.diet.iloc[1, 1] = 1.0
    params.model.loc[0, "Fleet"] = 0.5
    return rsim_scenario(rpath(params), params, years=range(1, 3))


def _ecospace(n_rows):
    grid = create_regular_grid(bounds=(0, 0, 2, 1), nx=2, ny=1)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n_rows, grid.n_patches)),
        habitat_capacity=np.ones((n_rows, grid.n_patches)),
        dispersal_rate=np.full(n_rows, 1.0),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )


def _run(scenario, n_rows, caplog):
    with caplog.at_level(logging.WARNING, logger="pypath.spatial.integration"):
        return rsim_run_spatial(scenario, ecospace=_ecospace(n_rows), years=range(1, 2))


def test_one_row_per_group_runs_without_a_warning(scenario, caplog):
    _run(scenario, scenario.params.NUM_GROUPS, caplog)
    assert "group rows" not in caplog.text


def test_an_extra_row_warns_about_an_outside_row(scenario, caplog):
    result = _run(scenario, scenario.params.NUM_GROUPS + 1, caplog)
    assert "5 group rows" in caplog.text and "4 groups" in caplog.text
    assert "Outside" in caplog.text
    assert np.all(np.isfinite(result.out_Biomass))


def test_more_extra_rows_warn_without_the_outside_hint(scenario, caplog):
    _run(scenario, scenario.params.NUM_GROUPS + 3, caplog)
    assert "7 group rows" in caplog.text
    assert "Outside" not in caplog.text


def test_too_few_rows_are_rejected_before_the_run(scenario, caplog):
    # e.g. read_ecospace(n_groups=NUM_LIVING + NUM_DEAD) leaves out the fleets
    with pytest.raises(ValueError, match="3 group rows.*4 groups"):
        _run(scenario, scenario.params.NUM_GROUPS - 1, caplog)
