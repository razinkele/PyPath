"""The run setup and month forcing that rsim_run's paths share.

``_run_setup`` and ``_month_forcing`` are rsim_run's setup and forcing build,
moved out of its body. They must hand the integration exactly the dicts and
state it got before the move; ``rsim_run``'s method checks are pinned here
too.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecosim import rsim_run
from tests.continuous_support import needs_reco, scenario
from tests.continuous_support import reco_plain as _reco_plain
from tests.continuous_support import reco_stanzas as _reco_stanzas

pytestmark = needs_reco

ONE_YEAR = range(1, 2)
RECO = {"reco_plain": _reco_plain, "reco_stanzas": _reco_stanzas}


class _Captured(Exception):
    def __init__(self, args_):
        super().__init__("captured")
        self.args_ = args_


def _rsim_run_dicts(scen):
    """``(state, params_dict, forcing_dict, fishing_dict)`` as ``rsim_run``
    passes them to its first ``integrate_rk4`` call (deep copies; the run is
    aborted there). This is how ``run_dicts`` captured them before the setup
    was shared, and the reference the helpers are checked against."""
    import pypath.core.ecosim_deriv as deriv

    def stub(*args, **kwargs):
        raise _Captured(copy.deepcopy(args[:4]))

    with pytest.MonkeyPatch.context() as mp:
        # rsim_run imports integrate_rk4 from the module at call time
        mp.setattr(deriv, "integrate_rk4", stub)
        try:
            rsim_run(scen, method="RK4", years=ONE_YEAR)
        except _Captured as captured:
            state, params_dict, forcing_dict, fishing_dict = captured.args_
            return np.array(state), params_dict, forcing_dict, fishing_dict
    raise AssertionError("rsim_run never called integrate_rk4")


def _same(a, b):
    """Equal values: arrays element by element and of the same dtype (NaN
    equal to NaN), containers in the same order, recursively."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        nan_ok = a.dtype.kind in "fc" and b.dtype.kind in "fc"
        return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=nan_ok)
    if isinstance(a, pd.DataFrame):
        return isinstance(b, pd.DataFrame) and a.equals(b)
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and list(a) == list(b)
            and all(_same(a[k], b[k]) for k in a)
        )
    if isinstance(a, (list, tuple)):
        return (
            type(a) is type(b)
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a):
        return bool(np.isnan(b))
    return type(a) is type(b) and a == b


@pytest.mark.parametrize(
    "case, rule",
    [("reco_plain", "ewe"), ("reco_stanzas", "ewe"), ("reco_stanzas", "rpath")],
)
def test_the_setup_reproduces_rsim_runs_dicts(case, rule):
    """The helpers give the integration's month-1 inputs of the old body.

    Same keys in the same order, every value equal, except ``_hden``: the
    setup holds EwE's start value, which the run then advances by one
    derivative at the start biomass before integrating. REco forces no
    biomass, so the setup's state is the month-1 state.
    """
    from pypath.core.ecosim import _month_forcing, _run_setup

    old_state, old_params, old_forcing, old_fishing = _rsim_run_dicts(
        scenario(RECO[case](), rule=rule)
    )
    scen = scenario(RECO[case](), rule=rule)
    setup = _run_setup(scen, ONE_YEAR)
    forcing = _month_forcing(scen, setup, 1, setup.ftime)
    assert not np.any(forcing["ForcedBio"] > 0)

    assert list(setup.params_dict) == list(old_params)
    for key, value in setup.params_dict.items():
        if key != "_hden":
            assert _same(value, old_params[key]), key
    assert list(forcing) == list(old_forcing)
    for key, value in forcing.items():
        assert _same(value, old_forcing[key]), key
    assert list(setup.fishing_dict) == list(old_fishing)
    for key, value in setup.fishing_dict.items():
        assert _same(value, old_fishing[key]), key
    assert _same(np.array(setup.state), old_state)


@pytest.mark.parametrize("method", ["RK4", "AB"])
def test_lowercase_parity_names_run(method):
    upper = rsim_run(scenario(_reco_plain()), method=method, years=ONE_YEAR)
    lower = rsim_run(scenario(_reco_plain()), method=method.lower(), years=ONE_YEAR)
    assert np.array_equal(lower.out_Biomass, upper.out_Biomass)


def test_a_non_string_method_raises():
    with pytest.raises(ValueError, match="method"):
        rsim_run(scenario(_reco_plain()), method=4, years=ONE_YEAR)


def test_a_month_is_named_once(monkeypatch):
    """An error that already names its month is not prefixed again."""
    import pypath.core.ecosim_continuous as cont_mod

    scen = scenario(_reco_plain())

    def integrate_month(*args, **kwargs):
        raise RuntimeError(f"month 1 (year {scen.start_year}): boom")

    # rsim_run imports integrate_month from the module at call time
    monkeypatch.setattr(cont_mod, "integrate_month", integrate_month)
    with pytest.raises(RuntimeError, match="boom") as err:
        rsim_run(scen, method="lsoda", years=ONE_YEAR)
    assert str(err.value).count("month 1") == 1
