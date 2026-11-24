"""Ecospace's per-patch loop runs on the calling thread.

The patch loop in ``deriv_vector_spatial`` used to hand patches to a
ThreadPoolExecutor whenever numba was installed. The per-patch step is mostly
Python (``deriv_vector``), so the pool never ran in parallel: CPU time equalled
wall time, and the pool's overhead made runs slower (profiling spike,
2026-11-17). The pool is gone, and the serial loop is the only path.

The pool gave each patch its own copy of ``params``, so the caller's dict came
back as it went in. The serial loop writes each patch's values into the
shared dict, so it must put back what was there and remove keys the caller
did not have. Otherwise, for example, the last patch's ``_hden`` (a view into
that patch's Hden column) would stay in ``params``, and a later
``deriv_vector`` on a copy of ``params`` would update that column in place.
"""

import threading

import numpy as np
import pytest

import pypath.spatial.integration as integ
from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial
from pypath.spatial.environmental import EnvironmentalDrivers

# The shared fixture's detritus pool keeps a share of its own surplus, so
# rsim_params warns on every scenario built from it; that is not what these
# tests are about.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Detritus fate of a detritus group to itself:UserWarning"
)

N_PATCHES = 6  # more than 4: the pool used to take over above 4 patches
ZOO = 2  # rsim row of Zoo in the spatial_scenario fixture


def _capacity(n_rows, varied):
    cap = np.ones((n_rows, N_PATCHES))
    if varied:
        cap[ZOO - 1] = np.linspace(0.2, 1.8, N_PATCHES)  # rows are 0-based groups
    return cap


def _ecospace(n_rows, varied=False):
    return EcospaceParams(
        grid=create_1d_grid(n_patches=N_PATCHES, spacing=1.0),
        habitat_preference=np.ones((n_rows, N_PATCHES)),
        habitat_capacity=_capacity(n_rows, varied),
        dispersal_rate=np.zeros(n_rows),
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )


def _first_call(scenario, varied=False):
    """The arguments of a run's first ``deriv_vector_spatial`` call.

    ``params`` is copied as the run hands it over, before any call could
    leave something behind in it.
    """
    captured = {}
    original = integ.deriv_vector_spatial

    def spy(state, params, forcing, fishing, eco, env, *args, **kwargs):
        if "args" not in captured:
            captured["args"] = (state.copy(), dict(params), forcing, fishing, eco, env)
            captured["kwargs"] = {k: v for k, v in kwargs.items() if k != "rates_out"}
        return original(state, params, forcing, fishing, eco, env, *args, **kwargs)

    n_rows = scenario.params.NUM_GROUPS + 1
    integ.deriv_vector_spatial = spy
    try:
        rsim_run_spatial(
            scenario,
            ecospace=_ecospace(n_rows, varied),
            environmental_drivers=EnvironmentalDrivers(),
            years=range(1, 2),
        )
    finally:
        integ.deriv_vector_spatial = original
    return captured["args"], captured["kwargs"]


def test_patches_run_on_the_calling_thread(spatial_scenario, monkeypatch):
    scenario, _ = spatial_scenario
    args, kwargs = _first_call(scenario)
    threads = []
    original = integ._local_deriv

    def spy(*a, **k):
        threads.append(threading.get_ident())
        return original(*a, **k)

    monkeypatch.setattr(integ, "_local_deriv", spy)
    # the per-patch loop (the batched path evaluates no patch on its own)
    monkeypatch.setattr(integ, "USE_BATCHED_DERIV", False)
    integ.deriv_vector_spatial(*args, **kwargs)

    assert len(threads) == N_PATCHES
    assert set(threads) == {threading.get_ident()}


@pytest.mark.parametrize(
    "case",
    [
        "uniform",  # the loop writes each patch's _hden
        "preset",  # ...over a _hden and _eat_eff the caller set
        "calibrated",  # EwE's capacity calibration (_space_cal, _eat_eff)
        "swap",  # each patch's B_BaseRef / Bbase / QQbase
        "swap_without_bbase",  # ...when the caller's dict has no Bbase
    ],
)
def test_the_shared_params_come_back_as_they_went_in(spatial_scenario, case):
    scenario, _ = spatial_scenario
    varied = case != "uniform" and case != "preset"
    (state, params, forcing, fishing, eco, env), kwargs = _first_call(scenario, varied)
    params = dict(params)
    n_rows = state.shape[0]
    # Each patch's own Hden column, as rsim_run_spatial sets it up under EwE's
    # handling time; distinct columns would show a leaked view
    params["_hden_patches"] = np.linspace(0.1, 0.9, n_rows * N_PATCHES).reshape(
        n_rows, N_PATCHES
    )
    params.setdefault("_hden0", np.full(n_rows, 0.5))
    params.setdefault("_cmco_qb", np.zeros(n_rows))
    if case == "preset":
        params["_hden"] = np.full(n_rows, 0.5)
        params["_eat_eff"] = np.ones(n_rows)
    if case == "calibrated":
        assert "_space_cal" in params  # the run set up the calibration
    if case.startswith("swap"):
        # Without the calibration each patch swaps in its own base biomass
        params.pop("_space_cal")
        params.pop("_eat_eff_patches", None)
    if case == "swap_without_bbase":
        params.pop("Bbase")
    before = dict(params)

    integ.deriv_vector_spatial(state, params, forcing, fishing, eco, env, **kwargs)

    assert params.keys() == before.keys()
    changed = [key for key in before if params[key] is not before[key]]
    assert changed == []
