"""Sensitivity analysis runs its evaluations in worker processes when asked.

``SensitivityConfig.n_jobs`` existed but was never read, so every run was
serial. The design is built up front in the parent, from the config's seed,
and each point is a deterministic Ecopath evaluation. Outputs come back in
order, so a parallel run must give exactly the serial run's outputs, indices
and progress calls.
"""

import multiprocessing
import warnings

import numpy as np
import pytest

from pypath.core import sensitivity
from pypath.core.sensitivity import SensitivityConfig, run_sensitivity
from tests.test_mc_integration import _make_mc_model


def _run(n_jobs, method="morris", model=None, callback=None, **overrides):
    """One run: ``(result, model outputs, progress calls)``."""
    if method == "morris":
        overrides.setdefault("n_trajectories", 6)
    else:
        pytest.importorskip("SALib")
        overrides.setdefault("n_samples", 8)
    config = SensitivityConfig(method=method, seed=11, n_jobs=n_jobs, **overrides)
    calls = []

    def progress(i, n):
        calls.append((i, n))
        if callback is not None:
            callback(i, n)

    # The outputs reach the analysis in the parent, whichever process
    # computed them: catch them there
    captured = {}
    effects = sensitivity._compute_elementary_effects

    def capture_effects(trajectories, y, *args, **kwargs):
        captured["y"] = np.array(y, copy=True)
        return effects(trajectories, y, *args, **kwargs)

    patches = {"_compute_elementary_effects": capture_effects}
    if sensitivity.HAS_SALIB:
        salib_sobol = sensitivity.salib_sobol

        class _CaptureSobol:
            @staticmethod
            def analyze(problem, Y, **kwargs):
                captured["y"] = np.array(Y, copy=True)
                return salib_sobol.analyze(problem, Y, **kwargs)

        patches["salib_sobol"] = _CaptureSobol
    saved = {name: getattr(sensitivity, name) for name in patches}
    try:
        for name, value in patches.items():
            setattr(sensitivity, name, value)
        params = _make_mc_model() if model is None else model
        result = run_sensitivity(params, config, progress_callback=progress)
    finally:
        for name, value in saved.items():
            setattr(sensitivity, name, value)
    return result, captured["y"], calls


_FIELDS = {
    "morris": ("mu_star", "sigma", "mu"),
    "sobol": ("S1", "ST", "S1_conf", "ST_conf"),
}


def _assert_same(serial, parallel, method):
    (s_result, s_y, s_calls), (p_result, p_y, p_calls) = serial, parallel
    # Outputs worth comparing: finite, and not all alike
    assert np.all(np.isfinite(s_y))
    assert len(np.unique(s_y)) > 3
    assert p_y.tobytes() == s_y.tobytes()
    assert p_result.parameter_names == s_result.parameter_names
    for name in _FIELDS[method]:
        assert getattr(p_result, name).tobytes() == getattr(s_result, name).tobytes()
    assert p_calls == s_calls == [(i, len(s_y)) for i in range(1, len(s_y) + 1)]


def test_parallel_morris_matches_serial():
    _assert_same(_run(1), _run(2), "morris")


def test_parallel_sobol_matches_serial():
    _assert_same(_run(1, "sobol"), _run(3, "sobol"), "sobol")


def test_evaluations_run_in_worker_processes():
    # The parent only collects outputs: while they arrive, its workers exist
    children = []
    _run(
        2,
        n_trajectories=2,
        callback=lambda i, n: children.append(len(multiprocessing.active_children())),
    )
    assert children and min(children) >= 1
    # ... and none is left once the run returns
    assert all(not p.is_alive() for p in multiprocessing.active_children())


def test_sobol_workers_are_counted_by_evaluations_not_base_samples():
    # n_samples is Sobol's base sample size; the design has n_samples *
    # (2k + 2) points, here 2 * 12 = 24, enough for four workers
    children = []
    _run(
        4,
        "sobol",
        n_samples=2,
        callback=lambda i, n: children.append(len(multiprocessing.active_children())),
    )
    assert max(children) == 4


def test_an_evaluation_ignores_the_caller_warning_filters_and_errstate(
    monkeypatch,
):
    # Workers start with their own warning filters and numpy's default
    # floating-point error handling, so an evaluation must not depend on the
    # caller's. No bundled model warns or raises a floating-point event while
    # it is sampled and balanced, so stand-ins do both.
    import pypath.core.ecopath
    import pypath.core.pedigree

    real_apply_sample = pypath.core.pedigree.apply_sample

    def noisy_apply_sample(params, sample):
        warnings.warn("stand-in apply_sample warning", UserWarning)
        np.float64(0.0) / np.float64(0.0)  # an invalid-value event
        return real_apply_sample(params, sample)

    class _Balance:
        Biomass = np.array([1.0, 2.0, 3.0])

    def noisy_rpath(params):
        warnings.warn("stand-in Ecopath warning", RuntimeWarning)
        np.float64(1.0) / np.float64(0.0)  # a divide-by-zero event
        return _Balance()

    monkeypatch.setattr(pypath.core.pedigree, "apply_sample", noisy_apply_sample)
    monkeypatch.setattr(pypath.core.ecopath, "rpath", noisy_rpath)
    point = (_make_mc_model(), [("Biomass", 0)], np.array([10.0]), None, np.arange(3))
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        assert sensitivity._evaluate_point(*point) == 6.0


class _Stop(Exception):
    pass


def test_an_error_in_the_progress_callback_stops_the_run():
    def stop(i, n):
        if i == 1:
            raise _Stop

    with pytest.raises(_Stop):
        _run(2, callback=stop)


def test_an_unknown_start_method_is_refused_even_serially():
    with pytest.raises(ValueError):
        _run(1, n_trajectories=1, mp_context="no-such-method")


def test_a_default_parallel_run_raises_no_fork_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run(2, n_trajectories=2)
    forks = [w for w in caught if "fork()" in str(w.message)]
    assert forks == []


def test_spawned_workers_match_serial():
    # Windows and macOS start workers by spawning, which pickles the model
    # and imports the worker function afresh
    _assert_same(_run(1), _run(2, mp_context="spawn"), "morris")


@pytest.mark.parametrize("n_jobs", [0, -2, 1.5, True, "4"])
def test_a_meaningless_n_jobs_is_refused(n_jobs):
    with pytest.raises(ValueError, match="n_jobs"):
        _run(n_jobs, n_trajectories=1)


def test_only_a_few_evaluations_wait_in_the_pool(monkeypatch):
    # Every task waiting in the pool costs this process about 2 KB, and a
    # default Sobol design on a 54-parameter model has 112,640 of them: keep
    # a small window in flight instead of submitting them all at once
    import pypath.core._parallel as parallel

    submitted = []

    class CountingPool(parallel.ProcessPoolExecutor):
        def submit(self, *args, **kwargs):
            submitted.append(None)
            return super().submit(*args, **kwargs)

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", CountingPool)
    waiting = []
    result, y, _ = _run(
        2, n_trajectories=10, callback=lambda i, n: waiting.append(len(submitted) - i)
    )
    assert len(submitted) == len(y) == 60
    assert max(waiting) <= 4 * 2
