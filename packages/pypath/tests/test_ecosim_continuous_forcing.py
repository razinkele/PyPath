"""Mean-preserving forcing interpolation on the continuous path (opt-in).

``solver_options["forcing_interpolation"]`` is a model option: ``"linear"``
(a continuous tent through each month's midpoint) and ``"smooth"`` (one
quadratic per month) let the forcing vary within each month while keeping
every monthly mean; ``"constant"``, the default, holds each month's forcing.
"""

import copy
import warnings
from functools import lru_cache

import numpy as np
import pytest

import pypath.core.ecosim_continuous as cont
from pypath.core.ecosim import (
    _forcing_interpolant,
    _month_forcing,
    _run_setup,
    rsim_run,
)
from pypath.core.ecosim_continuous import ForcingInterpolant, continuous_options
from tests.continuous_support import (
    needs_reco,
    reco_plain,
    reco_stanzas,
    run_dicts,
    scenario,
)

DT = 1.0 / 12.0
S101 = np.linspace(0.0, 1.0, 101)
# 2-point Gauss-Legendre on each half month (exact for the tent and the
# quadratic, both cubic or lower on each half)
_G = 0.25 / np.sqrt(3.0)
GAUSS_HALVES = (0.25 - _G, 0.25 + _G, 0.75 - _G, 0.75 + _G)


def _n_bio(scen):
    return int(scen.params.NUM_LIVING) + int(scen.params.NUM_DEAD)


def _gauss_mean(interp, k, key):
    return sum(np.asarray(interp.at(k, s)[key], dtype=float) for s in GAUSS_HALVES) / 4


def _check_interpolant(kind):
    """The interpolant's properties for ``kind``: exact means (also by
    quadrature), >= 0 at 101 points per month for a non-negative key,
    continuity at the month boundaries, exact constants, the zero month of
    ``[1, 0, 1]``, and the input arrays left unchanged."""
    rng = np.random.default_rng(0)
    monthly = {
        "ForcedPrey": rng.uniform(0.0, 2.0, size=(12, 5)),
        "NutForcing": rng.uniform(0.0, 2.0, size=12),
        "ForcedMigrate": np.array([1.0, -1.0, 2.0]),
    }
    kept = copy.deepcopy(monthly)
    with pytest.raises(ValueError, match="months"):
        ForcingInterpolant(monthly, kind=kind)  # 12 and 3 months
    for key, series in monthly.items():
        interp = ForcingInterpolant({key: series}, kind=kind)
        assert interp.kind == kind and interp.n_months == len(series)
        assert set(interp.keys) == {key}
        for k in range(len(series)):
            assert np.array_equal(interp.mean(k, key), series[k])
            np.testing.assert_allclose(
                _gauss_mean(interp, k, key), series[k], rtol=1e-14, atol=0
            )
            values = np.array([interp.at(k, s)[key] for s in S101])
            if key != "ForcedMigrate":
                assert np.all(values >= 0.0), (key, k, values.min())
            if k + 1 < len(series):
                np.testing.assert_allclose(
                    interp.at(k, 1.0)[key],
                    interp.at(k + 1, 0.0)[key],
                    rtol=1e-14,
                    atol=0,
                )
    for key in monthly:
        assert np.array_equal(monthly[key], kept[key])  # not written

    zero = ForcingInterpolant({"ForcedPrey": np.array([1.0, 0.0, 1.0])}, kind=kind)
    assert all(zero.at(1, s)["ForcedPrey"] == 0.0 for s in S101)
    const = ForcingInterpolant(
        {
            "ForcedPrey": np.full((4, 3), 0.7),
            "NutForcing": np.full(4, 1.3),
            "ForcedMigrate": np.full(4, -0.3),
        },
        kind=kind,
    )
    for k in range(4):
        for s in S101:
            got = const.at(k, s)
            assert np.all(got["ForcedPrey"] == 0.7)
            assert got["NutForcing"] == 1.3
            assert got["ForcedMigrate"] == -0.3


def test_the_tent_preserves_means():
    _check_interpolant("linear")
    with pytest.raises(ValueError, match="kind"):
        ForcingInterpolant({"ForcedPrey": np.ones((2, 3))}, kind="cubic")


def test_the_split_rule():
    """Under ``"linear"`` the month means split at the tent's kink, ``dt /
    2`` in time (also when a failed solve reached only part of the month):
    exact on a function with a kink there; the 6-point rule when the kink
    is not reached."""

    def kinked(t):
        return np.where(t <= 0.5 * DT, 3.0 + 2.0 * t / DT, 4.0 - 5.0 * (t / DT - 0.5))

    def exact_mean(t_reach):
        """Linear pieces: length x the mean of their end values."""
        cuts = [0.0, min(0.5 * DT, t_reach), t_reach]
        total = sum(
            (b - a) * 0.5 * float(kinked(a) + kinked(b))
            for a, b in zip(cuts[:-1], cuts[1:])
        )
        return total / t_reach

    for frac in (1.0, 0.8):
        nodes, weights = cont._split_nodes(frac * DT, DT)
        assert len(nodes) == 6 and weights.sum() == pytest.approx(1.0, rel=1e-14)
        assert np.all((nodes > 0.0) & (nodes < frac * DT))
        assert np.sum(nodes < 0.5 * DT) == 3
        assert weights @ kinked(nodes) == pytest.approx(
            exact_mean(frac * DT), rel=1e-13
        )
    nodes, weights = cont._split_nodes(0.3 * DT, DT)
    assert np.allclose(nodes, 0.15 * DT * (np.polynomial.legendre.leggauss(6)[0] + 1))
    assert weights @ kinked(nodes) == pytest.approx(exact_mean(0.3 * DT), rel=1e-13)


def _run(scen, method="lsoda", years=range(1, 3), **opts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(
            copy.deepcopy(scen), method=method, years=years, solver_options=opts
        )


def _max_rel(a, b, n):
    return float(
        np.max(np.abs(a.out_Biomass[1:, 1 : n + 1] / b.out_Biomass[1:, 1 : n + 1] - 1))
    )


@needs_reco
def test_constant_forcing_is_unchanged():
    """REco with stanzas has no forcing: ``"linear"`` changes only the
    quadrature of the month means (3 + 3 points against 6), which reach the
    biomass through the stanza update: 1.5e-9 at most (measured; the plan's
    estimate was 1e-9). Two parts, measured: the node sets sample the
    dense output at different times (6 + 6 against 6: 8.3e-10 at rtol 1e-6,
    1.1e-11 at 1e-9) and the 3 + 3 rule is of lower order (3 + 3 against 6
    at rtol 1e-9: 3.4e-10)."""
    scen = scenario(reco_stanzas())
    const = _run(scen)
    linear = _run(scen, forcing_interpolation="linear")
    n = _n_bio(scen)
    diff = _max_rel(linear, const, n)
    print(f"\nREco stanzas, linear against constant: max rel diff {diff:.2e}")
    assert diff <= 1e-8
    assert linear.solver_info["forcing_interpolation"] == "linear"


def _seasonal_prey(scen, rows):
    """ForcedPrey of ``rows`` rows, seasonal on group 13 (Foragefish1)."""
    n1 = scen.params.NUM_GROUPS + 1
    fp = np.ones((rows, n1))
    k = np.arange(rows)
    fp[:, 13] = 1.0 + 0.3 * np.sin(2.0 * np.pi * (k + 0.5) / 12.0)
    scen.forcing.ForcedPrey = fp
    return scen


@needs_reco
@pytest.mark.parametrize("kind", ["linear", "smooth"])
def test_series_ends_and_short_arrays(kind, monkeypatch):
    """A ForcedPrey of 18 rows in a two-year run: every month's mean is
    ``_month_forcing``'s value (ones from month 19 on), the forcing that each
    evaluation of the rates used is the interpolant's at its time, and the
    non-negative keys stay >= 0."""
    scen = _seasonal_prey(scenario(reco_plain(), effort=1.0), 18)
    years = range(1, 3)
    built = copy.deepcopy(scen)
    setup = _run_setup(built, years)
    interp = _forcing_interpolant(built, setup, kind)
    assert interp is not None and interp.n_months == 24
    assert set(interp.keys) == {"ForcedPrey"}  # the only key varying in time
    for k in range(24):
        held = _month_forcing(built, setup, k + 1, setup.ftime)
        for key in interp.keys:
            np.testing.assert_allclose(
                interp.mean(k, key), held[key], rtol=1e-14, atol=0
            )
        values = np.array([interp.at(k, s)["ForcedPrey"] for s in S101])
        assert np.all(values >= 0.0)
        others = np.delete(values, 13, axis=1)
        assert np.all(others == 1.0)
        if k >= 18:  # months 19-24: _month_forcing's default
            assert np.all(interp.mean(k, "ForcedPrey") == 1.0)
        if k >= 19:  # month 19 ramps from the month-18 boundary
            assert np.all(values == 1.0)
    assert np.any(np.array([interp.at(18, s)["ForcedPrey"][13] for s in S101]) != 1)

    seen = []
    month = {"k": -1}
    real_rates = cont.ContinuousRHS.rates
    real_set_month = cont.ContinuousRHS.set_month

    def rates(self, y, t=0.0):
        out = real_rates(self, y, t)
        seen.append((month["k"], t, out["_forcing"]))
        return out

    def set_month(self, *args, **kwargs):
        month["k"] += 1
        return real_set_month(self, *args, **kwargs)

    monkeypatch.setattr(cont.ContinuousRHS, "rates", rates)
    monkeypatch.setattr(cont.ContinuousRHS, "set_month", set_month)
    out = _run(scen, years=years, forcing_interpolation=kind)
    assert month["k"] == 23
    assert all(m["status"] == 0 for m in out.solver_info["months"])
    inside = [0] * 24
    for k, t, used in seen:
        want = interp.at(k, t / DT)
        for key, value in want.items():
            assert np.array_equal(used[key], value), (k, t, key)
        inside[k] += 0.0 < t < DT
    assert min(inside) >= 6  # the quadrature nodes, inside every month


@needs_reco
def test_the_smooth_interpolant():
    _check_interpolant("smooth")
    scen = scenario(reco_stanzas())
    scen.forcing.ForcedPrey[:, 13] = 1.2  # constant in time
    const = _run(scen)
    smooth = _run(scen, forcing_interpolation="smooth")
    n = _n_bio(scen)
    diff = _max_rel(smooth, const, n)
    print(f"\nREco stanzas, smooth against constant: max rel diff {diff:.2e}")
    assert diff <= 1e-12


def _seasonal_budget_scenario(production=True):
    """REco without stanzas: a seasonal migration flux into group 5 and,
    with ``production``, seasonal production (ForcedSearch of the
    producers), both starting from their base value so month 1 starts in
    balance (a start transient, as with a production forcing of 1.3 in
    month 1, is resolved worse by the 3 + 3 point rule than by the 6-point
    one: a budget residual of 1.9e-5 against 3e-7 in month 1, measured)."""
    scen = scenario(reco_plain(), effort=1.0)
    p = scen.params
    k = np.arange(scen.forcing.ForcedSearch.shape[0])
    producers = [i for i in range(1, p.NUM_LIVING + 1) if np.asarray(p.PP_type)[i] > 0]
    for i in producers if production else []:
        scen.forcing.ForcedSearch[:, i] = 1.0 + 0.3 * np.sin(2.0 * np.pi * k / 12.0)
    b5 = float(np.asarray(p.B_BaseRef)[5])
    scen.forcing.ForcedMigrate[:, 5] = 0.2 * b5 * np.sin(2.0 * np.pi * k / 12.0)
    return scen


@needs_reco
@pytest.mark.parametrize("kind", ["constant", "linear", "smooth"])
def test_the_budget_uses_the_applied_forcing(kind):
    """With production and migration varying in time the month budget closes
    at the solver's level: the means of the rates are taken under the forcing
    each quadrature node applied (measured: constant 1.6e-7, linear 2.3e-7,
    smooth 1.1e-8; with the rates under the month's held forcing instead,
    5.6e-5)."""
    scen = _seasonal_budget_scenario()
    out = _run(scen, years=range(1, 2), forcing_interpolation=kind)
    months = out.solver_info["months"]
    worst = max(abs(m["budget_residual"]) for m in months)
    print(f"\nseasonal REco, {kind}: max |budget_residual| {worst:.2e}")
    assert all(m["status"] == 0 for m in months)
    assert worst <= 1e-6


@needs_reco
def test_exchange_reads_the_forcing_of_its_rates():
    """``exchange`` takes ForcedMigrate from the rates it is given: at one
    state, two times differ by the interpolated migration difference (the
    month budget cannot show it: the flux is additive and its mean is
    kept). The caller's forcing dict is not written."""
    scen = _seasonal_budget_scenario(production=False)
    years = range(1, 2)
    built = copy.deepcopy(scen)
    setup = _run_setup(built, years)
    interp = _forcing_interpolant(built, setup, "linear")
    assert set(interp.keys) == {"ForcedMigrate"}
    state, pd, fd, fi = run_dicts(scen, years=years)
    n_bio = _n_bio(scen)
    n1 = pd["NUM_GROUPS"] + 1
    rhs = cont.ContinuousRHS(pd, fi, n_bio, np.array(pd["Bbase"], dtype=float))
    held = copy.deepcopy(fd)
    rhs.set_month(
        fd,
        state,
        np.zeros(n1, dtype=bool),
        np.zeros(n1),
        forcing_at=interp.for_month(2, DT),
    )
    y = np.array(state[1 : n_bio + 1], dtype=float)
    r0, r1 = rhs.rates(y, 0.0), rhs.rates(y, 0.5 * DT)
    mig0 = interp.at(2, 0.0)["ForcedMigrate"]
    mig1 = interp.at(2, 0.5)["ForcedMigrate"]
    assert np.array_equal(r0["_forcing"]["ForcedMigrate"], mig0)
    assert np.array_equal(r1["_forcing"]["ForcedMigrate"], mig1)
    assert mig1[5] != mig0[5]
    nl = int(pd["NUM_LIVING"])
    gap = rhs.exchange(r1, y) - rhs.exchange(r0, y)
    migration = float(mig1[1 : nl + 1].sum() - mig0[1 : nl + 1].sum())
    assert gap == pytest.approx(migration, rel=1e-9)
    for key, value in held.items():
        assert np.array_equal(np.asarray(fd[key]), np.asarray(value)), key


def test_forcing_interpolation_option():
    """``continuous_options`` takes ``forcing_interpolation`` in
    ``{"constant", "linear", "smooth"}`` (default ``"constant"``); anything
    else raises ``ValueError`` naming the key."""
    assert continuous_options("lsoda", None, 5)["forcing_interpolation"] == "constant"
    for kind in ("constant", "linear", "smooth"):
        opts = continuous_options("lsoda", {"forcing_interpolation": kind}, 5)
        assert opts["forcing_interpolation"] == kind
    for bad in ("cubic", "Linear", None, 1, True):
        with pytest.raises(ValueError, match="forcing_interpolation"):
            continuous_options("lsoda", {"forcing_interpolation": bad}, 5)


@needs_reco
@pytest.mark.parametrize("kind", ["constant", "linear", "smooth"])
def test_solver_info_labels_the_model_options(kind):
    scen = _seasonal_prey(scenario(reco_plain(), effort=1.0), 24)
    out = _run(scen, years=range(1, 2), forcing_interpolation=kind)
    info = out.solver_info
    assert info["forcing_interpolation"] == kind
    assert info["positivity"] == "clip"
    assert info["fast_groups"] == "states"
    # the defect corrections stay the defect list
    assert not any("forcing" in c for c in info["corrections"])
    assert np.all(np.isfinite(out.out_Biomass))


@lru_cache(maxsize=None)
def _black_sea_constant():
    from tests.test_ecosim_continuous_validation import BLACK_SEA, _load_ewe

    scen = _load_ewe(BLACK_SEA, scenario=1)
    return scen, _run(scen)


def _black_sea(kind):
    base, const = _black_sea_constant()
    out = _run(base, forcing_interpolation=kind)
    months = out.solver_info["months"]
    nfev = np.mean([m["nfev"] for m in months])
    nfev_c = np.mean([m["nfev"] for m in const.solver_info["months"]])
    ev, ev_c = out.solver_info["rhs_evals"], const.solver_info["rhs_evals"]
    diff = _max_rel(out, const, _n_bio(base))
    print(
        f"\nBlack Sea lsoda, two years, {kind}: nfev/mo {nfev:.1f} (constant "
        f"{nfev_c:.1f}, {nfev / nfev_c - 1:+.0%}); rhs_evals {ev} (constant "
        f"{ev_c}, {ev / ev_c - 1:+.0%}); max rel diff from constant {diff:.2e}"
    )
    assert len(months) == 24
    assert all(m["status"] == 0 for m in months)
    assert sum(m["clips"] for m in months) == 0
    assert out.solver_info["forcing_interpolation"] == kind


@pytest.mark.integration
def test_linear_forcing_runs_the_black_sea():
    """Two years with NutForcing as a tent (spike: 113 nfev/mo against 87.8
    under constant forcing; printed, not asserted)."""
    _black_sea("linear")


@pytest.mark.integration
def test_smooth_forcing_runs_the_black_sea():
    """Two years with NutForcing smooth (spike: 16 % fewer evaluations than
    constant forcing; printed, not asserted)."""
    _black_sea("smooth")
