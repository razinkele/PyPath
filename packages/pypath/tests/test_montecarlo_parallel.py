"""Monte Carlo runs its samples in worker processes when ``n_jobs`` asks.

``MCConfig.n_jobs`` existed but was never read, so every run was serial. The
samples are drawn up front in the parent, from the config's seed, and each one
is a deterministic Ecopath (and Ecosim) evaluation. Results come back in sample
order, so a parallel run must give exactly the serial run's statistics, stored
runs and progress calls.
"""

import copy
import multiprocessing
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from pypath.core.montecarlo import (
    MCConfig,
    _mp_context,
    _resolve_n_jobs,
    run_montecarlo,
)
from tests.continuous_support import needs_reco, reco_stanzas
from tests.test_mc_integration import _make_mc_model


def _run(n_jobs, model=None, callback=None, **overrides):
    config = MCConfig(
        n_samples=overrides.pop("n_samples", 16),
        method="lhs",
        seed=7,
        ecopath_only=overrides.pop("ecopath_only", True),
        ecosim_years=overrides.pop("ecosim_years", None),
        store_runs=True,
        n_jobs=n_jobs,
        **overrides,
    )
    calls = []

    def progress(i, n):
        calls.append((i, n))
        if callback is not None:
            callback(i, n)

    params = _make_mc_model() if model is None else model
    result = run_montecarlo(params, config, progress_callback=progress)
    return result, calls


def _assert_same(serial, parallel):
    assert parallel.n_total == serial.n_total
    assert parallel.n_feasible == serial.n_feasible
    assert parallel.n_ecosim == serial.n_ecosim
    assert parallel.ecopath_stats.keys() == serial.ecopath_stats.keys()
    for key in serial.ecopath_stats:
        pd.testing.assert_frame_equal(
            parallel.ecopath_stats[key], serial.ecopath_stats[key], check_exact=True
        )
    assert len(parallel.ecopath_runs) == len(serial.ecopath_runs)
    for a, b in zip(parallel.ecopath_runs, serial.ecopath_runs):
        assert a["Biomass"].tobytes() == b["Biomass"].tobytes()
    if serial.ecosim_stats is None:
        assert parallel.ecosim_stats is None
    else:
        assert (
            parallel.ecosim_stats["Biomass"].tobytes()
            == serial.ecosim_stats["Biomass"].tobytes()
        )
    assert len(parallel.ecosim_runs) == len(serial.ecosim_runs)
    for a, b in zip(parallel.ecosim_runs, serial.ecosim_runs):
        assert a.tobytes() == b.tobytes()


def test_parallel_ecopath_matches_serial():
    serial, serial_calls = _run(1)
    parallel, parallel_calls = _run(2)
    assert serial.n_feasible > 0
    _assert_same(serial, parallel)
    assert parallel_calls == serial_calls == [(i, 16) for i in range(1, 17)]


def test_parallel_ecosim_matches_serial():
    kwargs = dict(n_samples=6, ecopath_only=False, ecosim_years=range(1, 3))
    serial, _ = _run(1, **kwargs)
    parallel, _ = _run(3, **kwargs)
    assert serial.n_ecosim > 0
    _assert_same(serial, parallel)


def test_samples_run_in_worker_processes():
    # The parent only collects results: while they arrive, its workers exist
    children = []
    _run(
        2,
        n_samples=6,
        callback=lambda i, n: children.append(len(multiprocessing.active_children())),
    )
    assert children and min(children) >= 1
    # ... and none is left once the run returns
    assert multiprocessing.active_children() == [] or all(
        not p.is_alive() for p in multiprocessing.active_children()
    )


class _FailWhenProducerAbove:
    """Mediation that fails a sample's Ecosim when its producer biomass > 10.

    Module level, so spawned and forkserver workers can import it.
    """

    def compute_group_multipliers(self, BB, Bbase, ActiveLink):
        if Bbase[1] > 10.0:
            raise RuntimeError("sample-dependent Ecosim failure")
        return np.ones((len(BB), len(BB)))

    def compute_fleet_multipliers(self, BB, Bbase, n):
        return np.ones(n)

    def compute_landing_multipliers(self, BB, Bbase, nf, ng):
        return np.ones((nf, ng))


def test_failed_ecosim_samples_match_serial():
    kwargs = dict(
        n_samples=12,
        ecopath_only=False,
        ecosim_years=range(1, 3),
        mediation=_FailWhenProducerAbove(),
    )
    serial, _ = _run(1, **kwargs)
    parallel, _ = _run(3, **kwargs)
    assert 0 < serial.n_ecosim < serial.n_feasible
    _assert_same(serial, parallel)


@needs_reco
def test_the_caller_warning_filters_do_not_change_results():
    # REco's stanza groups warn on every Ecosim scenario. Workers start with
    # their own warning filters, so the evaluation must not depend on the
    # caller's: under "error" a warning would otherwise fail the serial
    # run's Ecosim but not the workers'.
    model = copy.deepcopy(reco_stanzas()[1])
    for col in model.pedigree.columns:
        if col != "Group":
            model.pedigree[col] = 0.1
    kwargs = dict(n_samples=4, ecopath_only=False, ecosim_years=range(1, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        serial, _ = _run(1, model=model, **kwargs)
        parallel, _ = _run(2, model=model, **kwargs)
    assert serial.n_ecosim == serial.n_feasible > 0
    _assert_same(serial, parallel)


class _Stop(Exception):
    pass


def test_an_error_in_the_progress_callback_stops_the_run():
    def stop(i, n):
        if i == 1:
            raise _Stop

    with pytest.raises(_Stop):
        _run(2, n_samples=8, callback=stop)


def test_an_unknown_start_method_is_refused_even_serially():
    with pytest.raises(ValueError):
        _run(1, n_samples=2, mp_context="no-such-method")


def test_workers_are_not_forked_by_default():
    # fork is unsafe in a multi-threaded process. Linux gets forkserver;
    # macOS and Windows get spawn, as CPython's own defaults
    method = _mp_context(None).get_start_method()
    if sys.platform in ("darwin", "win32"):
        assert method == "spawn"
    elif "forkserver" in multiprocessing.get_all_start_methods():
        assert method == "forkserver"
    else:
        assert method == "spawn"


def test_a_default_parallel_run_raises_no_fork_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(2, n_samples=4)
    forks = [w for w in caught if "fork()" in str(w.message)]
    assert forks == []


def test_spawned_workers_match_serial():
    # Windows and macOS start workers by spawning, which pickles the model
    # and the config and imports the worker function afresh
    serial, _ = _run(1, n_samples=4)
    spawned, _ = _run(2, n_samples=4, mp_context="spawn")
    _assert_same(serial, spawned)


@pytest.mark.parametrize(
    "n_jobs, n_samples, expected",
    [
        (1, 10, 1),
        (4, 10, 4),
        (8, 3, 3),  # never more workers than samples
        (-1, 2, min(2, os.cpu_count() or 1)),
        (4, 0, 1),
    ],
)
def test_worker_count(n_jobs, n_samples, expected):
    assert _resolve_n_jobs(n_jobs, n_samples) == expected


def test_all_cpus_means_the_ones_this_process_may_use():
    n = _resolve_n_jobs(-1, 10_000)
    assert 1 <= n <= (os.cpu_count() or 1)
    if hasattr(os, "sched_getaffinity"):
        assert n == len(os.sched_getaffinity(0))
    if sys.platform == "win32":
        assert n <= 61  # ProcessPoolExecutor's limit on Windows


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, True, "4"])
def test_a_meaningless_n_jobs_is_refused(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        _resolve_n_jobs(n_jobs, 10)


def test_run_montecarlo_refuses_a_meaningless_n_jobs():
    with pytest.raises(ValueError, match="n_jobs"):
        _run(0, n_samples=2)


def test_numpy_integers_are_accepted():
    assert _resolve_n_jobs(np.int64(2), 10) == 2
