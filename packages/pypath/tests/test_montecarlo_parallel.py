"""Monte Carlo runs its samples in worker processes when ``n_jobs`` asks.

``MCConfig.n_jobs`` existed but was never read, so every run was serial. The
samples are drawn up front in the parent, from the config's seed, and each one
is a deterministic Ecopath (and Ecosim) evaluation. Results come back in sample
order, so a parallel run must give exactly the serial run's statistics, stored
runs and progress calls.
"""

import multiprocessing
import os
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
from tests.test_mc_integration import _make_mc_model


def _run(n_jobs, **overrides):
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
    result = run_montecarlo(
        _make_mc_model(), config, progress_callback=lambda i, n: calls.append((i, n))
    )
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


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the patched evaluator reaches the workers only through fork",
)
# This test forks on purpose; the pytest process has threads (numpy's BLAS)
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
def test_samples_run_in_worker_processes(tmp_path, monkeypatch):
    import pypath.core.montecarlo as mc

    log = tmp_path / "pids.txt"
    original = mc._run_single_ecopath

    def recording(*args, **kwargs):
        with open(log, "a") as fh:
            fh.write(f"{os.getpid()}\n")
        return original(*args, **kwargs)

    monkeypatch.setattr(mc, "_run_single_ecopath", recording)
    _run(2, n_samples=8, mp_context="fork")

    pids = {int(line) for line in log.read_text().split()}
    assert len(log.read_text().split()) == 8
    assert pids and os.getpid() not in pids


def test_workers_are_not_forked_by_default():
    # fork is unsafe in a multi-threaded process; forkserver and spawn start
    # clean workers on every platform
    method = _mp_context(None).get_start_method()
    assert method in {"forkserver", "spawn"}
    if "forkserver" in multiprocessing.get_all_start_methods():
        assert method == "forkserver"


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
        (-1, 10_000, os.cpu_count() or 1),
        (-1, 2, min(2, os.cpu_count() or 1)),
        (4, 0, 1),
    ],
)
def test_worker_count(n_jobs, n_samples, expected):
    assert _resolve_n_jobs(n_jobs, n_samples) == expected


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, True, "4"])
def test_a_meaningless_n_jobs_is_refused(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        _resolve_n_jobs(n_jobs, 10)


def test_run_montecarlo_refuses_a_meaningless_n_jobs():
    with pytest.raises(ValueError, match="n_jobs"):
        _run(0, n_samples=2)


def test_numpy_integers_are_accepted():
    assert _resolve_n_jobs(np.int64(2), 10) == 2
