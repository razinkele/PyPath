"""``rsim_run_spatial(n_jobs=...)`` runs the patches in worker processes.

The per-patch step is Python, so only processes run patches in parallel.
The default stays serial; with workers the run must give the serial run's
numbers bit for bit, leave no pool behind, and fall back to the serial loop
where workers cannot be used (IBM groups).
"""

import copy
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import pypath.spatial.integration as integ
from pypath.core.ecopath import rpath
from pypath.core.ecosim import RunCancelled, rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.spatial import (
    EcospaceParams,
    SpatialFishing,
    create_1d_grid,
    rsim_run_spatial,
)
from pypath.spatial.mpa import MPAConfig, MPAZone

pytestmark = pytest.mark.filterwarnings(
    "ignore:Detritus fate of a detritus group to itself:UserWarning"
)

N_PATCHES = 6
YEARS = range(1, 2)


def _ecospace(scenario, n_patches=N_PATCHES):
    n_rows = scenario.params.NUM_GROUPS + 1
    capacity = np.ones((n_rows, n_patches))
    capacity[1] = np.linspace(0.4, 1.6, n_patches)  # patches differ
    return EcospaceParams(
        grid=create_1d_grid(n_patches=n_patches, spacing=1.0),
        habitat_preference=np.ones((n_rows, n_patches)),
        habitat_capacity=capacity,
        dispersal_rate=np.full(n_rows, 0.5),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )


def _run(scenario, **kwargs):
    kwargs.setdefault("ecospace", _ecospace(scenario))
    return rsim_run_spatial(scenario, years=YEARS, **kwargs)


class _CountingPool(ProcessPoolExecutor):
    """A process pool that records how it was made and shut down."""

    made: list = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shutdowns = 0
        self.workers = kwargs["max_workers"]
        _CountingPool.made.append(self)

    def shutdown(self, *args, **kwargs):
        self.shutdowns += 1
        super().shutdown(*args, **kwargs)


@pytest.fixture
def pools(monkeypatch):
    _CountingPool.made = []
    monkeypatch.setattr(integ, "ProcessPoolExecutor", _CountingPool)
    return _CountingPool.made


@pytest.mark.parametrize("n_jobs", [2, 4])
def test_workers_give_the_serial_run_bit_for_bit(spatial_scenario, n_jobs):
    scenario, _ = spatial_scenario
    serial = _run(copy.deepcopy(scenario))
    pooled = _run(scenario, n_jobs=n_jobs)
    np.testing.assert_array_equal(
        pooled.out_Biomass_spatial, serial.out_Biomass_spatial
    )
    np.testing.assert_array_equal(pooled.out_Biomass, serial.out_Biomass)


def test_workers_give_the_serial_run_with_explicit_movement(spatial_scenario):
    scenario, _ = spatial_scenario
    serial = _run(copy.deepcopy(scenario), implicit_movement=False)
    pooled = _run(scenario, n_jobs=2, implicit_movement=False)
    np.testing.assert_array_equal(
        pooled.out_Biomass_spatial, serial.out_Biomass_spatial
    )


def _same(pooled, serial):
    np.testing.assert_array_equal(
        pooled.out_Biomass_spatial, serial.out_Biomass_spatial
    )
    np.testing.assert_array_equal(pooled.out_Biomass, serial.out_Biomass)
    if serial.stanza_biomass is not None:
        np.testing.assert_array_equal(pooled.stanza_biomass, serial.stanza_biomass)


@pytest.mark.parametrize(
    "extra",
    [
        # a fleet's effort differs by patch
        {"spatial_fishing": SpatialFishing("gravity", gravity_alpha=2.0)},
        # two patches closed to fishing, with a capacity bonus
        {
            "mpa": MPAConfig(
                zones=[
                    MPAZone(mpa_id=1, name="closed", patches=[1, 2], capacity_bonus=1.5)
                ]
            )
        },
    ],
    ids=["spatial_fishing", "mpa"],
)
def test_workers_give_the_serial_run_with_patch_effort(spatial_scenario, extra):
    scenario, _ = spatial_scenario
    _same(_run(scenario, n_jobs=3, **extra), _run(copy.deepcopy(scenario), **extra))


REF = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"
needs_reco = pytest.mark.skipif(
    not (REF / "stanza_indiv.csv").exists(), reason="Rpath reference data missing"
)


@pytest.fixture(scope="module")
def reco():
    """Rpath's REco model: 4 multi-stanza species."""
    model_df = pd.read_csv(REF / "model_params.csv")
    params = create_rpath_params(model_df["Group"].tolist(), model_df["Type"].tolist())
    params.model = model_df
    params.diet = pd.read_csv(REF / "diet_matrix.csv")
    params.stanzas.stgroups = pd.read_csv(REF / "stanza_groups.csv")
    params.stanzas.stindiv = pd.read_csv(REF / "stanza_indiv.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    return model, params


@needs_reco
@pytest.mark.parametrize("rule", ["ewe", "rpath"])
@pytest.mark.parametrize("new_multi_stanza", [True, False])
def test_workers_give_the_serial_run_with_stanzas(reco, rule, new_multi_stanza):
    """Each patch's stanza consumption index and stage rates cross to a worker."""
    model, params = reco

    def run(**kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scenario = rsim_scenario(
                model, params, years=range(1, 3), fast_group_rule=rule
            )
            scenario.fishing.ForcedEffort[:] = 2.0
            n_rows = scenario.params.NUM_GROUPS + 1
            capacity = np.ones((n_rows, 5))
            capacity[3] = np.linspace(0.5, 1.5, 5)
            grid = create_1d_grid(n_patches=5, spacing=1.0)
            grid.patch_areas = np.array([1.0, 2.0, 1.0, 0.5, 1.5])
            eco = EcospaceParams(
                grid=grid,
                habitat_preference=np.ones((n_rows, 5)),
                habitat_capacity=capacity,
                dispersal_rate=np.full(n_rows, 2.0),
                advection_enabled=np.zeros(n_rows, dtype=bool),
                gravity_strength=np.zeros(n_rows),
                new_multi_stanza=new_multi_stanza,
            )
            return rsim_run_spatial(scenario, ecospace=eco, years=range(1, 2), **kwargs)

    _same(run(n_jobs=3), run())


def test_the_default_starts_no_process(spatial_scenario, pools):
    scenario, _ = spatial_scenario
    _run(scenario)
    _run(scenario, n_jobs=1)
    assert pools == []


def test_the_pool_is_shut_down_after_the_run(spatial_scenario, pools):
    scenario, _ = spatial_scenario
    _run(scenario, n_jobs=2)
    (pool,) = pools
    assert pool.workers == 2 and pool.shutdowns == 1


def test_never_more_workers_than_patches(spatial_scenario, pools):
    scenario, _ = spatial_scenario
    _run(scenario, n_jobs=8, ecospace=_ecospace(scenario, n_patches=3))
    (pool,) = pools
    assert pool.workers == 3


def test_the_pool_is_shut_down_when_the_run_is_cancelled(spatial_scenario, pools):
    scenario, _ = spatial_scenario
    with pytest.raises(RunCancelled):
        _run(scenario, n_jobs=2, on_month=lambda month, n: month == 2)
    (pool,) = pools
    assert pool.shutdowns == 1


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the worker must inherit the patched function (fork)",
)
def test_the_pool_is_shut_down_when_a_patch_fails(spatial_scenario, pools, monkeypatch):
    scenario, _ = spatial_scenario

    def broken(*args, **kwargs):
        raise FloatingPointError("a patch failed")

    # fork: the worker inherits the patched function
    monkeypatch.setattr(integ, "_local_deriv", broken)
    with pytest.raises(FloatingPointError, match="a patch failed"):
        _run(scenario, n_jobs=2, mp_context="fork")
    (pool,) = pools
    assert pool.shutdowns == 1


def test_the_scenario_is_left_as_it_was(spatial_scenario):
    scenario, _ = spatial_scenario
    fresh = _run(copy.deepcopy(scenario))
    _run(scenario, n_jobs=2)
    again = _run(scenario)
    np.testing.assert_array_equal(again.out_Biomass_spatial, fresh.out_Biomass_spatial)


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, True, "2"])
def test_a_bad_n_jobs_is_refused(spatial_scenario, n_jobs):
    scenario, _ = spatial_scenario
    with pytest.raises(ValueError, match="n_jobs"):
        _run(scenario, n_jobs=n_jobs)


def test_an_unknown_start_method_is_refused(spatial_scenario):
    scenario, _ = spatial_scenario
    with pytest.raises(ValueError):
        _run(scenario, n_jobs=2, mp_context="threads")


def test_without_a_grid_n_jobs_is_checked_and_nothing_starts(spatial_scenario, pools):
    scenario, _ = spatial_scenario
    with pytest.raises(ValueError, match="n_jobs"):
        rsim_run_spatial(scenario, years=YEARS, n_jobs=0)
    rsim_run_spatial(scenario, years=YEARS, n_jobs=4)
    assert pools == []


def test_ibm_groups_keep_the_serial_loop(spatial_scenario, caplog):
    """IBM individuals live in the caller's process; a worker's copy is lost."""
    scenario, _ = spatial_scenario
    eco = _ecospace(scenario)
    assert integ._n_patch_workers(scenario, eco, 4) == 4
    scenario.params.ibm_groups = {3: object()}
    with caplog.at_level("INFO", logger="pypath.spatial.integration"):
        assert integ._n_patch_workers(scenario, eco, 4) == 1
    assert "IBM" in caplog.text
