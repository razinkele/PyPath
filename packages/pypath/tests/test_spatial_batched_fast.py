"""The compiled all-patch Ecospace derivative (pypath.spatial.batched_fast)
must give the per-patch deriv_vector loop's results bit for bit, as NumPy's
batched derivative does, on the features EwE's Ecospace runs use: EwE's base
calibration with each patch's EatEff, handling time, multi-stanza predator
ratios, HandleSwitch and the rates the RK4 stages read."""

import copy
from pathlib import Path

import numpy as np
import pytest

import pypath.spatial.integration as si
from pypath.spatial import (
    EcospaceParams,
    batched_fast,
    create_regular_grid,
    rsim_run_spatial,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore"),
    pytest.mark.skipif(not batched_fast.ENABLED, reason="numba not installed"),
]

DB = Path(__file__).resolve().parents[3] / "Data" / "BlackSea_EwEF_testcase.eweaccdb"

#: (USE_BATCHED_DERIV, USE_FAST_BATCHED) of each way of evaluating the patches
PATHS = {"fast": (True, True), "numpy": (True, False), "loop": (False, False)}


def _set_path(monkeypatch, name, month=False):
    """The derivative path; ``month`` the compiled month on top of the
    compiled derivative (its diffusion agrees to round-off only)."""
    batched, fast = PATHS[name]
    monkeypatch.setattr(si, "USE_BATCHED_DERIV", batched)
    monkeypatch.setattr(si, "USE_FAST_BATCHED", fast)
    monkeypatch.setattr(si, "USE_FAST_MONTH", month and name == "fast")


def _count_calls(monkeypatch):
    calls = {"deriv_batched": 0, "deriv_spatial_fast": 0}
    for name in calls:
        real = getattr(si, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            calls[_name] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(si, name, spy)
    return calls


def _calibrated_ecospace(scenario, nx=5, ny=4, seed=3):
    """A map whose habitat capacity varies, so the run takes EwE's base
    calibration (_space_cal, each patch's EatEff)."""
    ng = scenario.params.NUM_GROUPS + 1
    grid = create_regular_grid(bounds=(0, 0, nx, ny), nx=nx, ny=ny)
    n = grid.n_patches
    rng = np.random.default_rng(seed)
    return EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((ng, n)),
        habitat_capacity=rng.uniform(0.2, 1.0, (ng, n)),
        dispersal_rate=np.full(ng, 2.0),
        advection_enabled=np.zeros(ng, dtype=bool),
        gravity_strength=np.zeros(ng),
    )


def _outputs(result):
    return {
        key: np.asarray(getattr(result, key))
        for key in ("out_Biomass_spatial", "out_Biomass", "out_Catch")
    }


def _assert_same(a, b):
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_the_calibrated_run_is_the_patch_loops_on_every_path(
    spatial_scenario, monkeypatch
):
    scenario, _ = spatial_scenario
    ecospace = _calibrated_ecospace(scenario)
    calls = _count_calls(monkeypatch)
    seen = {}
    real = si.deriv_vector_spatial

    def spy(state, params, *args, **kwargs):
        seen.setdefault("space_cal", params.get("_space_cal") is not None)
        return real(state, params, *args, **kwargs)

    monkeypatch.setattr(si, "deriv_vector_spatial", spy)
    out = {}
    for name in PATHS:
        _set_path(monkeypatch, name)
        out[name] = _outputs(
            rsim_run_spatial(
                copy.deepcopy(scenario), ecospace=ecospace, years=range(1, 3)
            )
        )
    assert seen["space_cal"]  # the run did take EwE's base calibration
    assert calls["deriv_spatial_fast"] > 0 and calls["deriv_batched"] > 0
    _assert_same(out["fast"], out["loop"])
    _assert_same(out["numpy"], out["loop"])


def test_threads_give_the_single_threads_run(spatial_scenario, monkeypatch):
    scenario, _ = spatial_scenario
    ecospace = _calibrated_ecospace(scenario, nx=7, ny=5)
    _set_path(monkeypatch, "fast")
    one = _outputs(
        rsim_run_spatial(copy.deepcopy(scenario), ecospace=ecospace, years=range(1, 3))
    )
    two = _outputs(
        rsim_run_spatial(
            copy.deepcopy(scenario), ecospace=ecospace, years=range(1, 3), n_jobs=2
        )
    )
    _assert_same(one, two)


# --- one derivative, with what Ecospace's run does not set itself -------------


def _captured_call(scenario, ecospace, monkeypatch):
    """The arguments of a mid-run deriv_vector_spatial call (a stage that
    asks for rates), with the params dict as it was then."""
    real = si.deriv_vector_spatial
    store = {}

    def spy(state, params, forcing, fishing, *args, **kwargs):
        if kwargs.get("rates_out") is not None and "args" not in store:
            store["args"] = (state.copy(), params, forcing, fishing) + args
            store["kwargs"] = dict(kwargs)
            store["hden"] = (
                None
                if params.get("_hden_patches") is None
                else params["_hden_patches"].copy()
            )
            store["params"] = dict(params)
        return real(state, params, forcing, fishing, *args, **kwargs)

    monkeypatch.setattr(si, "deriv_vector_spatial", spy)
    rsim_run_spatial(copy.deepcopy(scenario), ecospace=ecospace, years=range(1, 2))
    monkeypatch.setattr(si, "deriv_vector_spatial", real)
    return store


def _derivative(store, path, monkeypatch, extra=None, keys=None):
    _set_path(monkeypatch, path)
    state, params, forcing, fishing, *args = store["args"]
    params = dict(store["params"])
    if store["hden"] is not None:
        params["_hden_patches"] = store["hden"].copy()
    params.update(extra or {})
    kwargs = dict(store["kwargs"])
    keys = tuple(kwargs["rates_out"]) if keys is None else keys
    rates = {key: np.zeros_like(state, dtype=float) for key in keys}
    kwargs["rates_out"] = rates
    d = si.deriv_vector_spatial(state.copy(), params, forcing, fishing, *args, **kwargs)
    hden = params.get("_hden_patches")
    return d, rates, None if hden is None else hden.copy()


@pytest.mark.parametrize("exponents", [(0.0, 1.0), (0.0, 1.0, 2.0, 0.5)])
def test_handle_switch_and_rates_match_the_patch_loop(
    spatial_scenario, monkeypatch, exponents
):
    """HandleSwitch, which EwE databases set on almost every link, and all
    the rates the batched kernels fill. The compiled kernel takes the loop's
    scalar power, so even exponents other than 0 and 1 give its bits."""
    scenario, _ = spatial_scenario
    ecospace = _calibrated_ecospace(scenario)
    store = _captured_call(scenario, ecospace, monkeypatch)
    G = store["params"]["NUM_GROUPS"] + 1
    rng = np.random.default_rng(11)
    hs = rng.choice(np.asarray(exponents), size=(G, G))
    extra = {"HandleSwitch": hs}
    keys = (
        "FoodGain",
        "LossPropToB",
        "TotGain",
        "TotLoss",
        "NetGain",
        "NetLoss",
    )
    d_loop, r_loop, h_loop = _derivative(store, "loop", monkeypatch, extra, keys)
    d_fast, r_fast, h_fast = _derivative(store, "fast", monkeypatch, extra, keys)
    np.testing.assert_array_equal(d_fast, d_loop)
    for key in keys:
        np.testing.assert_array_equal(r_fast[key], r_loop[key], err_msg=key)
    if h_loop is not None:
        np.testing.assert_array_equal(h_fast, h_loop)
    # NumPy's batched derivative: exact for exponents 0 and 1, else within
    # the last bits of numpy's array power
    d_np, r_np, _ = _derivative(store, "numpy", monkeypatch, extra, keys)
    if set(exponents) <= {0.0, 1.0}:
        np.testing.assert_array_equal(d_np, d_loop)
    else:
        np.testing.assert_allclose(d_np, d_loop, rtol=1e-12, atol=1e-300)


def test_the_rates_asked_for_are_the_rates_returned(spatial_scenario, monkeypatch):
    scenario, _ = spatial_scenario
    store = _captured_call(scenario, _calibrated_ecospace(scenario), monkeypatch)
    d_all, r_all, _ = _derivative(
        store, "fast", monkeypatch, keys=batched_fast.RATE_KEYS
    )
    d_some, r_some, _ = _derivative(
        store, "fast", monkeypatch, keys=("NetGain", "NetLoss")
    )
    np.testing.assert_array_equal(d_some, d_all)
    for key in ("NetGain", "NetLoss"):
        np.testing.assert_array_equal(r_some[key], r_all[key])


def test_the_environment_variable_turns_it_off():
    assert batched_fast._requested({})
    for value in ("0", "false", "No", " off "):
        assert not batched_fast._requested({batched_fast.FAST_ECOSPACE_ENV: value})
    assert batched_fast._requested({batched_fast.FAST_ECOSPACE_ENV: "1"})


# --- the Black Sea, as EwE 6's maps set it up ------------------------------------


def _black_sea():
    from pypath.io.ewemdb import (
        check_ewemdb_support,
        ecosim_scenario_from_ewemdb,
        read_ecospace,
    )

    if not DB.exists() or not any(check_ewemdb_support().values()):
        pytest.skip("Black Sea database or driver missing")
    scenario = ecosim_scenario_from_ewemdb(str(DB), scenario=1)
    p = scenario.params
    e = read_ecospace(
        str(DB), n_groups=p.NUM_LIVING + p.NUM_DEAD, scenario_id=1
    ).ecospace
    n_rows, n_p = p.NUM_GROUPS, e.grid.n_patches
    hab = np.ones((n_rows, n_p))
    hab[: e.habitat_capacity.shape[0]] = e.habitat_capacity
    rates = np.zeros(n_rows)
    rates[: len(e.dispersal_rate)] = e.dispersal_rate
    eco = EcospaceParams(
        grid=e.grid,
        habitat_preference=np.ones((n_rows, n_p)),
        habitat_capacity=hab,
        dispersal_rate=rates,
        advection_enabled=np.zeros(n_rows, dtype=bool),
        gravity_strength=np.zeros(n_rows),
    )
    return scenario, eco


@pytest.mark.integration
def test_black_sea_compiled_run_is_the_patch_loops(monkeypatch):
    """The run of scripts/benchmark_ecospace.py: 304 cells, EwE's base
    calibration, handling time, stanzas, fast-equilibrium groups."""
    scenario, eco = _black_sea()
    first = int(scenario.start_year)
    years = range(first, first + 2)
    calls = _count_calls(monkeypatch)
    _set_path(monkeypatch, "fast")
    fast = _outputs(
        rsim_run_spatial(copy.deepcopy(scenario), ecospace=eco, years=years)
    )
    assert calls["deriv_spatial_fast"] > 0
    _set_path(monkeypatch, "loop")
    loop = _outputs(
        rsim_run_spatial(copy.deepcopy(scenario), ecospace=eco, years=years)
    )
    _assert_same(fast, loop)


# --- the patch-major entry and the preparation cache ------------------------------


def _captured_batched(scenario, ecospace, monkeypatch):
    """The arguments of a stage's deriv_batched call (NumPy's path)."""
    _set_path(monkeypatch, "numpy")
    real = si.deriv_batched
    store = {}

    def spy(S, params, forcing, fishing, **kwargs):
        if kwargs.get("want_rates") and "args" not in store:
            store["args"] = (S.copy(), dict(params), dict(forcing), fishing)
            store["kwargs"] = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in kwargs.items()
            }
        return real(S, params, forcing, fishing, **kwargs)

    monkeypatch.setattr(si, "deriv_batched", spy)
    rsim_run_spatial(copy.deepcopy(scenario), ecospace=ecospace, years=range(1, 2))
    monkeypatch.setattr(si, "deriv_batched", real)
    return store


def _both(store, params=None):
    from pypath.spatial.batched import deriv_batched

    S, p, forcing, fishing = store["args"]
    p = p if params is None else params
    out = []
    for fn in (deriv_batched, batched_fast.deriv_batched_fast):
        kw = dict(store["kwargs"])
        if kw.get("hden") is not None:
            kw["hden"] = kw["hden"].copy()
        d, r = fn(S.copy(), p, forcing, fishing, **kw)
        out.append((d, r, kw.get("hden")))
    return out


def test_the_patch_major_entry_is_numpys_batched_derivative(
    spatial_scenario, monkeypatch
):
    scenario, _ = spatial_scenario
    store = _captured_batched(scenario, _calibrated_ecospace(scenario), monkeypatch)
    (d_np, r_np, h_np), (d_fast, r_fast, h_fast) = _both(store)
    np.testing.assert_array_equal(d_fast, d_np)
    for key in r_fast:
        np.testing.assert_array_equal(r_fast[key], r_np[key], err_msg=key)
    if h_np is not None:
        np.testing.assert_array_equal(h_fast, h_np)


def test_a_change_in_place_is_not_served_from_the_cache(spatial_scenario, monkeypatch):
    """The preparation is kept by its inputs' values: an array changed in
    place gives a new preparation, not the kept one."""
    scenario, _ = spatial_scenario
    store = _captured_batched(scenario, _calibrated_ecospace(scenario), monkeypatch)
    params = dict(store["args"][1])
    params["PB"] = np.array(params["PB"], dtype=float)
    (_, _, _), (d_before, _, _) = _both(store, params)
    params["PB"][1:] *= 1.25  # in place: the same array object
    (d_np, _, _), (d_fast, _, _) = _both(store, params)
    assert not np.array_equal(d_fast, d_before)
    np.testing.assert_array_equal(d_fast, d_np)


def test_the_preparation_reads_only_the_keys_it_is_kept_by(
    spatial_scenario, monkeypatch
):
    scenario, _ = spatial_scenario
    store = _captured_batched(scenario, _calibrated_ecospace(scenario), monkeypatch)
    S, params, forcing, fishing = store["args"]

    class Recording(dict):
        read = set()

        def __getitem__(self, key):
            Recording.read.add(key)
            return super().__getitem__(key)

        def get(self, key, default=None):
            Recording.read.add(key)
            return super().get(key, default)

    batched_fast._prepare(Recording(params), forcing, fishing, S.shape[0])
    assert Recording.read <= set(batched_fast.PREP_PARAM_KEYS), Recording.read - set(
        batched_fast.PREP_PARAM_KEYS
    )


# --- movement and threads --------------------------------------------------------


@pytest.mark.parametrize("include_diffusion", [True, False])
@pytest.mark.parametrize("advection", [False, True])
@pytest.mark.parametrize("rate", [0.0, 2.0])
def test_the_zero_flux_check_agrees_with_the_flux(
    spatial_scenario, include_diffusion, advection, rate
):
    from pypath.spatial.dispersal import calculate_spatial_flux, spatial_flux_is_zero

    scenario, _ = spatial_scenario
    ng = scenario.params.NUM_GROUPS + 1
    grid = create_regular_grid(bounds=(0, 0, 4, 3), nx=4, ny=3)
    n = grid.n_patches
    rng = np.random.default_rng(5)
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=rng.uniform(0.2, 1.0, (ng, n)),
        habitat_capacity=np.ones((ng, n)),
        dispersal_rate=np.full(ng, rate),
        advection_enabled=np.full(ng, advection),
        gravity_strength=np.full(ng, 0.5),
    )
    state = rng.uniform(0.1, 2.0, (ng, n))
    flux = calculate_spatial_flux(
        state, ecospace, {}, 0.0, include_diffusion=include_diffusion
    )
    if spatial_flux_is_zero(ng, ecospace, {}, include_diffusion):
        assert not np.any(flux)
    else:
        assert np.any(flux)


def test_threads_are_capped_by_the_map():
    per = batched_fast.MIN_PATCHES_PER_THREAD
    assert batched_fast.threads_for(16, 4 * per) == min(
        4, batched_fast.numba.config.NUMBA_NUM_THREADS
    )
    assert batched_fast.threads_for(16, per - 1) == 1
    assert batched_fast.threads_for(1, 100 * per) == 1


# --- the compiled month ------------------------------------------------------------


def _month_run(
    scenario,
    ecospace,
    monkeypatch,
    *,
    month,
    numba_diffusion=False,
    years=range(1, 3),
    n_jobs=1,
):
    """A run with the compiled month, or with the Python month (its diffusion
    SciPy's, or the compiled month's when ``numba_diffusion``)."""
    from pypath.spatial.diffusion_fast import DiffusionBatch

    _set_path(monkeypatch, "fast", month=month)
    if numba_diffusion:
        batches = {}

        def diffuse(biomass, solvers):
            entry = batches.get(id(solvers))
            if entry is None or entry[0] is not solvers:
                entry = batches[id(solvers)] = (
                    solvers,
                    DiffusionBatch(solvers, biomass.shape[1]),
                )
            entry[1].apply(biomass)

        monkeypatch.setattr(si, "_diffuse_rows", diffuse)
    try:
        return _outputs(
            rsim_run_spatial(
                copy.deepcopy(scenario), ecospace=ecospace, years=years, n_jobs=n_jobs
            )
        )
    finally:
        monkeypatch.undo()


def _count_month_calls(monkeypatch):
    calls = {"n": 0}
    real = batched_fast.CellCall.run

    def counting(self, S, out):
        calls["n"] += 1
        return real(self, S, out)

    monkeypatch.setattr(batched_fast.CellCall, "run", counting)
    return calls


@pytest.mark.skipif(not batched_fast.MONTH_ENABLED, reason="compiled month off")
def test_the_compiled_month_is_the_python_month_with_its_diffusion(
    spatial_scenario, monkeypatch
):
    """Only the diffusion differs from the Python month: with the same
    compiled diffusion the two give the same bits."""
    scenario, _ = spatial_scenario
    ecospace = _calibrated_ecospace(scenario)
    compiled = _month_run(scenario, ecospace, monkeypatch, month=True)
    python = _month_run(
        scenario, ecospace, monkeypatch, month=False, numba_diffusion=True
    )
    _assert_same(compiled, python)


@pytest.mark.skipif(not batched_fast.MONTH_ENABLED, reason="compiled month off")
def test_the_compiled_month_agrees_with_the_exact_path(spatial_scenario, monkeypatch):
    scenario, _ = spatial_scenario
    ecospace = _calibrated_ecospace(scenario)
    calls = _count_month_calls(monkeypatch)
    compiled = _month_run(scenario, ecospace, monkeypatch, month=True)
    assert calls["n"] > 0  # the month did run compiled
    exact = _month_run(scenario, ecospace, monkeypatch, month=False)
    for key in compiled:
        np.testing.assert_allclose(
            compiled[key], exact[key], rtol=1e-12, atol=1e-14, err_msg=key
        )


@pytest.mark.skipif(not batched_fast.MONTH_ENABLED, reason="compiled month off")
def test_the_compiled_months_threads_give_its_single_threads_run(
    spatial_scenario, monkeypatch
):
    scenario, _ = spatial_scenario
    ecospace = _calibrated_ecospace(scenario, nx=7, ny=5)
    one = _month_run(scenario, ecospace, monkeypatch, month=True)
    two = _month_run(scenario, ecospace, monkeypatch, month=True, n_jobs=2)
    _assert_same(one, two)


@pytest.mark.skipif(not batched_fast.MONTH_ENABLED, reason="compiled month off")
def test_a_replaced_derivative_keeps_the_python_month(spatial_scenario, monkeypatch):
    """A spy on deriv_vector_spatial sees the run's derivatives: the month
    then runs through it, as ecosim_fast does for Ecosim."""
    scenario, _ = spatial_scenario
    _set_path(monkeypatch, "fast", month=True)
    seen = []
    real = si.deriv_vector_spatial

    def spy(*args, **kwargs):
        seen.append(kwargs.get("rates_out") is not None)
        return real(*args, **kwargs)

    monkeypatch.setattr(si, "deriv_vector_spatial", spy)
    calls = _count_month_calls(monkeypatch)
    rsim_run_spatial(
        copy.deepcopy(scenario),
        ecospace=_calibrated_ecospace(scenario),
        years=range(1, 2),
    )
    assert len(seen) >= 4 * 12  # every stage of every month
    assert calls["n"] == len(seen)  # each one compiled, through the function


def test_the_month_environment_variable_turns_it_off():
    assert batched_fast._month_requested({})
    for value in ("0", "false", "No", " off "):
        assert not batched_fast._month_requested({batched_fast.FAST_MONTH_ENV: value})
    assert batched_fast._month_requested({batched_fast.FAST_MONTH_ENV: "1"})


@pytest.mark.integration
@pytest.mark.skipif(not batched_fast.MONTH_ENABLED, reason="compiled month off")
def test_black_sea_compiled_month_agrees_with_the_exact_path(monkeypatch):
    scenario, eco = _black_sea()
    first = int(scenario.start_year)
    years = range(first, first + 3)
    calls = _count_month_calls(monkeypatch)
    compiled = _month_run(scenario, eco, monkeypatch, month=True, years=years)
    assert calls["n"] > 0
    exact = _month_run(scenario, eco, monkeypatch, month=False, years=years)
    for key in compiled:
        np.testing.assert_allclose(
            compiled[key], exact[key], rtol=1e-12, atol=1e-14, err_msg=key
        )
