"""``rsim_run_spatial`` can be watched and stopped between months.

``on_month`` works as in ``rsim_run``: it is called after every month with
that month stored, and the run stops with ``RunCancelled`` when it returns
the bool True.
"""

import copy

import numpy as np
import pytest

from pypath.core.ecosim import RunCancelled
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detritus fate of a detritus group to itself:UserWarning"
)

N_PATCHES = 4
YEARS = range(1, 3)  # 24 months


def _ecospace(scenario):
    n_rows = scenario.params.NUM_GROUPS + 1
    capacity = np.ones((n_rows, N_PATCHES))
    capacity[1] = np.linspace(0.5, 1.5, N_PATCHES)  # patches differ
    return EcospaceParams(
        grid=create_1d_grid(n_patches=N_PATCHES, spacing=1.0),
        habitat_preference=np.ones((n_rows, N_PATCHES)),
        habitat_capacity=capacity,
        dispersal_rate=np.full(n_rows, 0.5),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )


def _run(scenario, **kwargs):
    return rsim_run_spatial(
        scenario, ecospace=_ecospace(scenario), years=YEARS, **kwargs
    )


def test_on_month_is_called_once_per_month_in_order(spatial_scenario):
    scenario, _ = spatial_scenario
    calls = []
    _run(scenario, on_month=lambda month, n: calls.append((month, n)))
    assert calls == [(month, 24) for month in range(1, 25)]


def test_watching_a_run_does_not_change_it(spatial_scenario):
    scenario, _ = spatial_scenario
    plain = _run(scenario)
    watched = _run(scenario, on_month=lambda month, n: None)
    np.testing.assert_array_equal(
        watched.out_Biomass_spatial, plain.out_Biomass_spatial
    )
    np.testing.assert_array_equal(watched.out_Biomass, plain.out_Biomass)


def test_returning_true_stops_the_run_after_that_month(spatial_scenario):
    scenario, _ = spatial_scenario
    calls = []

    def stop_at_five(month, n):
        calls.append(month)
        return month == 5

    with pytest.raises(RunCancelled, match="month 5 of 24"):
        _run(scenario, on_month=stop_at_five)
    assert calls == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("answer", [1, "yes", [True], np.True_])
def test_only_the_bool_true_stops_the_run(spatial_scenario, answer):
    scenario, _ = spatial_scenario
    calls = []

    def truthy(month, n):
        calls.append(month)
        return answer

    _run(scenario, on_month=truthy)
    assert len(calls) == 24


def test_a_run_after_a_cancelled_one_equals_a_fresh_run(spatial_scenario):
    scenario, _ = spatial_scenario
    fresh = _run(copy.deepcopy(scenario))
    with pytest.raises(RunCancelled):
        _run(scenario, on_month=lambda month, n: month == 7)
    again = _run(scenario)
    np.testing.assert_array_equal(again.out_Biomass_spatial, fresh.out_Biomass_spatial)
    np.testing.assert_array_equal(again.out_Biomass, fresh.out_Biomass)


def test_without_a_grid_the_hook_reaches_the_plain_run(spatial_scenario):
    scenario, _ = spatial_scenario
    calls = []
    with pytest.raises(RunCancelled):
        rsim_run_spatial(
            scenario,
            years=YEARS,
            on_month=lambda month, n: calls.append(month) or month == 3,
        )
    assert calls == [1, 2, 3]
