"""rsim_run's per-month hook, ``on_month``, and ``RunCancelled``.

The callback is called as ``on_month(month, n_months)`` at the end of every
simulated month, on every method. Returning True stops the run with
``RunCancelled``; an exception it raises propagates as itself; without it,
or with a callback that returns None or False, the run is unchanged.
"""

import numpy as np
import pytest

from pypath.core.ecosim import RunCancelled, rsim_run
from pypath.core.ecosim_advanced import rsim_run_advanced
from tests.continuous_support import needs_reco, reco_plain, scenario

pytestmark = needs_reco

YEARS = 2
N_MONTHS = 12 * YEARS
METHODS = ("RK4", "AB", "lsoda")


def _scen():
    return scenario(reco_plain(), years=range(1, YEARS + 1))


def _recorder(stop_at=None, result=None):
    calls = []

    def on_month(month, n_months):
        calls.append((month, n_months))
        if stop_at is not None and month == stop_at:
            return True
        return result

    return calls, on_month


def test_run_cancelled_is_a_runtime_error_exported_beside_rsim_run():
    import pypath
    import pypath.core

    assert issubclass(RunCancelled, RuntimeError)
    assert pypath.core.RunCancelled is RunCancelled
    assert pypath.RunCancelled is RunCancelled
    assert "RunCancelled" in pypath.core.__all__
    assert "RunCancelled" in pypath.__all__


@pytest.mark.parametrize("method", METHODS)
def test_called_once_per_month_with_the_total(method):
    calls, on_month = _recorder()
    rsim_run(_scen(), method=method, on_month=on_month)
    assert calls == [(m, N_MONTHS) for m in range(1, N_MONTHS + 1)]


@pytest.mark.parametrize("method", METHODS)
def test_true_stops_the_run_with_run_cancelled(method):
    calls, on_month = _recorder(stop_at=3)
    with pytest.raises(RunCancelled) as excinfo:
        rsim_run(_scen(), method=method, on_month=on_month)
    assert excinfo.type is RunCancelled
    assert str(excinfo.value) == f"run cancelled after month 3 of {N_MONTHS}"
    assert calls == [(1, N_MONTHS), (2, N_MONTHS), (3, N_MONTHS)]


@pytest.mark.parametrize("method", METHODS)
def test_an_exception_in_the_callback_propagates_as_itself(method):
    def on_month(month, n_months):
        if month == 2:
            raise KeyError("from the callback")

    with pytest.raises(KeyError) as excinfo:
        rsim_run(_scen(), method=method, on_month=on_month)
    assert excinfo.type is KeyError
    assert excinfo.value.args == ("from the callback",)


@pytest.mark.parametrize("result", [None, False])
@pytest.mark.parametrize("method", METHODS)
def test_a_callback_that_does_not_stop_changes_nothing(method, result):
    plain = rsim_run(_scen(), method=method)
    calls, on_month = _recorder(result=result)
    hooked = rsim_run(_scen(), method=method, on_month=on_month)
    assert len(calls) == N_MONTHS
    assert np.array_equal(hooked.out_Biomass, plain.out_Biomass)
    assert np.array_equal(hooked.out_Catch, plain.out_Catch)
    for name in ("Biomass", "N", "Ftime"):
        assert np.array_equal(
            getattr(hooked.end_state, name), getattr(plain.end_state, name)
        ), name


def test_rsim_run_advanced_passes_on_month_through():
    calls, on_month = _recorder()
    rsim_run_advanced(_scen(), method="RK4", on_month=on_month)
    assert calls == [(m, N_MONTHS) for m in range(1, N_MONTHS + 1)]


def test_rsim_run_advanced_stops_on_true():
    calls, on_month = _recorder(stop_at=1)
    with pytest.raises(RunCancelled, match=f"after month 1 of {N_MONTHS}"):
        rsim_run_advanced(_scen(), method="AB", on_month=on_month)
    assert calls == [(1, N_MONTHS)]
