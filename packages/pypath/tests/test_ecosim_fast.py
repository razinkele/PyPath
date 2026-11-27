"""The compiled Ecosim RK4 step and loop (pypath.core.ecosim_fast).

The compiled path must give rsim_run's Python results: bit for bit where
no stanza update is involved, and to a few units in the last place where
the stanza update's exp and power round differently in numba and NumPy.
"""

import inspect

import numpy as np
import pytest

pytest.importorskip("numba")

from pypath.core import ecosim_deriv, ecosim_fast  # noqa: E402
from pypath.core.ecosim import RunCancelled, rsim_run  # noqa: E402
from pypath.examples import example_ecosim_scenario  # noqa: E402

KEYS = (
    "out_Biomass",
    "out_Catch",
    "out_Gear_Catch",
    "out_FishMort",
    "annual_Qlink",
    "stanza_biomass",
    "out_Hden",
    "annual_Biomass",
    "annual_Catch",
)
EXAMPLES = ("Curonian_Lagoon", "Central_Baltic", "Lithuanian_Coast")


def _run(scenario, enabled, monkeypatch, **kw):
    monkeypatch.setattr(ecosim_fast, "ENABLED", enabled)
    return rsim_run(scenario, method="RK4", **kw)


def _assert_same(ref, out, rtol):
    for key in KEYS:
        a, b = getattr(ref, key, None), getattr(out, key, None)
        if a is None and b is None:
            continue
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        assert a.shape == b.shape, key
        if rtol == 0:
            assert np.array_equal(a, b, equal_nan=True), key
        else:
            np.testing.assert_allclose(b, a, rtol=rtol, atol=1e-300, err_msg=key)
    np.testing.assert_allclose(
        out.end_state.Ftime, ref.end_state.Ftime, rtol=max(rtol, 0), atol=0
    )
    np.testing.assert_allclose(
        out.end_state.Biomass, ref.end_state.Biomass, rtol=max(rtol, 0), atol=0
    )
    assert out.crash_year == ref.crash_year
    assert out.crashed_groups == ref.crashed_groups


@pytest.mark.parametrize("name", EXAMPLES)
def test_compiled_loop_matches_python(name, monkeypatch):
    sc = example_ecosim_scenario(name, years=range(1, 4))
    ref = _run(sc, False, monkeypatch)
    out = _run(sc, True, monkeypatch)
    # the stanza update's exp and power: round-off only
    _assert_same(ref, out, 1e-13)


def test_curonian_is_bit_identical(monkeypatch):
    sc = example_ecosim_scenario("Curonian_Lagoon", years=range(1, 6))
    _assert_same(_run(sc, False, monkeypatch), _run(sc, True, monkeypatch), 0)


@pytest.mark.parametrize("name", EXAMPLES)
def test_compiled_step_matches_python(name, monkeypatch):
    """The compiled RK4 step inside rsim_run's own loop (fleet dynamics,
    Ecotracer and mediation keep that loop): same results."""
    sc = example_ecosim_scenario(name, years=range(1, 3))
    monkeypatch.setattr(ecosim_fast.CompiledRun, "build", classmethod(lambda *a: None))
    ref = _run(sc, False, monkeypatch)
    out = _run(sc, True, monkeypatch)
    _assert_same(ref, out, 0)


@pytest.mark.parametrize("name", EXAMPLES)
def test_prey_switching_runs_compiled(name, monkeypatch):
    """EwE6 prey switching (SwitchPower) on the compiled path: the switched
    search rates at each derivative call, then the arena; Qlinks too."""
    sc = example_ecosim_scenario(name, years=range(1, 4))
    n = len(sc.params.B_BaseRef)
    nl = sc.params.NUM_LIVING
    sp = np.zeros(n)
    sp[1 : nl + 1 : 2] = np.linspace(0.5, 2.0, len(range(1, nl + 1, 2)))
    sc.params.SwitchPower = sp
    sc.forcing.ForcedSearch[:, 1 : nl + 1] *= 0.9  # off the base: switching acts
    from pypath.core.ecosim import _run_setup

    setup = _run_setup(sc, None)
    assert ecosim_fast.unsupported_reason(setup.params_dict, setup.fishing_dict) is None
    ref = _run(sc, False, monkeypatch)
    out = _run(sc, True, monkeypatch)
    _assert_same(ref, out, 1e-13)


def test_on_month_runs_month_by_month(monkeypatch):
    sc = example_ecosim_scenario("Central_Baltic", years=range(1, 3))
    seen = []
    out = _run(sc, True, monkeypatch, on_month=lambda m, n: seen.append((m, n)))
    assert seen == [(m, 24) for m in range(1, 25)]
    ref = _run(sc, True, monkeypatch)
    _assert_same(ref, out, 0)

    def stop(m, n):
        return m == 5

    with pytest.raises(RunCancelled, match="after month 5 of 24"):
        _run(sc, True, monkeypatch, on_month=stop)


def test_unsupported_options_fall_back():
    sc = example_ecosim_scenario("Curonian_Lagoon", years=range(1, 3))
    from pypath.core.ecosim import _run_setup

    setup = _run_setup(sc, None)
    p, fd = setup.params_dict, setup.fishing_dict
    assert ecosim_fast.unsupported_reason(p, fd) is None
    for key, value in (
        ("_mediation", object()),
        ("UseVarPQ", True),
        ("ibm_groups", {1: object()}),
        ("INSTRUMENT_GROUPS", [1]),
    ):
        q = dict(p)
        q[key] = value
        assert ecosim_fast.unsupported_reason(q, fd), key


def test_pairwise_sum_is_numpys():
    rng = np.random.default_rng(7)
    for n in list(range(0, 400)) + [1000, 1029, 4096]:
        a = rng.standard_normal(n + 3) * 10.0 ** rng.uniform(-8, 8, n + 3)
        assert ecosim_fast._pw_sum(a, 3, n) == np.sum(a[3 : 3 + n])


def test_consumption_copy_matches_deriv_kernel():
    """ecosim_fast keeps its own compiled copy of the consumption loop."""
    src = inspect.getsource(ecosim_deriv._compute_consumption_sparse_flat)
    copy = inspect.getsource(ecosim_fast._consumption_flat.py_func)
    strip = lambda s: s.split(":", 1)[1].split('"""', 2)[2]  # noqa: E731
    assert strip(src) == strip(copy)
