"""Validation of the continuous integration pathway (design section 5).

The continuous methods have no EwE oracle as truth, so they are checked
against themselves and against RK4 where the two define the same equations:
RK4 converging to DOP853 as its sub-steps double, the SciPy methods agreeing
with each other, a balanced model staying at its base for 50 years, and
10-year runs of REco, LT2022 and the Black Sea staying finite and positive.
The last test prints the differences from the RK4 path and the cost that
``docs/parity/continuous_path.md`` quotes.
"""

import copy
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecosim import rsim_run
from tests.continuous_support import REF, needs_reco, scenario
from tests.continuous_support import reco_plain as _reco_plain
from tests.continuous_support import reco_stanzas as _reco_stanzas

DATA = Path(__file__).resolve().parents[3] / "Data"
BLACK_SEA = DATA / "BlackSea_EwEF_testcase.eweaccdb"
LT2022 = DATA / "LT2022_0.5ST_final7.eweaccdb"

ONE_YEAR = range(1, 2)
TWO_YEARS = range(1, 3)


@pytest.fixture(scope="module")
def reco_plain():
    return _reco_plain()


@pytest.fixture(scope="module")
def reco_stanzas():
    return _reco_stanzas()


def _n_bio(scen):
    """Number of biomass groups (living + detritus) of a scenario."""
    return int(scen.params.NUM_LIVING) + int(scen.params.NUM_DEAD)


def _max_rel(a, b, n):
    """``max |a / b - 1|`` over months 1.. and groups 1..n of two runs."""
    return float(
        np.max(np.abs(a.out_Biomass[1:, 1 : n + 1] / b.out_Biomass[1:, 1 : n + 1] - 1))
    )


@needs_reco
def test_rk4_converges_to_dop853(reco_plain):
    """RK4 approaches DOP853 at fourth order as ``RK4_STEPS`` doubles.

    REco without stanzas under the ``"rpath"`` rule has the same pure
    right-hand side on both paths: no stanzas, its 3 flagged fast groups
    integrated by RK4 under that rule, no handling time, FtimeAdj 0, and the
    continuous corrections (R2, R4, R5, DF once) change nothing on it. So the
    RK4 month-12 biomass must converge to a tight DOP853 solution, the error
    falling by about 2^4 per doubling.

    The reference is DOP853 at rtol 1e-12 and atol 1e-14 x B0, tighter than
    the brief's rtol 1e-10, because that reference is not accurate enough to
    resolve e_16 (measured against rtol 1e-13):

    - at the default atol (1e-8 x B0) it is good to 4.7e-9 only, above e_8,
      and gave e_4 / e_8 = 1.35;
    - at rtol 1e-10 with atol 1e-10 x B0 down to 1e-20, its error (7e-12 to
      1.1e-11) exceeds e_16, and e_8 / e_16 came out 12.3, 5.9 or 5.8
      depending on atol.

    At rtol 1e-12 the reference agrees with rtol 1e-13 to 7.5e-14. The
    thresholds are the brief's. Measured: e_4 = 1.73e-9, e_8 = 6.03e-11,
    e_16 = 2.81e-12 (ratios 28.6 and 21.5). Four sub-steps are stable (one
    and two are not: e = 2.4 and 1.1), so the sequence starts at 4.
    """
    reference_scen = scenario(reco_plain, rule="rpath", effort=2.0)
    p = reference_scen.params
    n_living = p.NUM_LIVING
    b0_bio = np.asarray(p.B_BaseRef, dtype=float)[1 : n_living + p.NUM_DEAD + 1]
    ref = rsim_run(
        reference_scen,
        method="dop853",
        years=ONE_YEAR,
        solver_options={"rtol": 1e-12, "atol": 1e-14 * b0_bio},
    )
    assert all(m["status"] == 0 for m in ref.solver_info["months"])
    b_ref = ref.out_Biomass[12, 1 : n_living + 1]

    errors = {}
    for n in (4, 8, 16):
        scen = scenario(reco_plain, rule="rpath", effort=2.0)
        scen.params.RK4_STEPS = n
        out = rsim_run(scen, method="RK4", years=ONE_YEAR)
        errors[n] = float(
            np.max(np.abs(out.out_Biomass[12, 1 : n_living + 1] / b_ref - 1))
        )
    print(
        "\nRK4 -> DOP853 (rtol 1e-12): "
        + ", ".join(f"e_{n} = {e:.3e}" for n, e in errors.items())
        + f"; e_4/e_8 = {errors[4] / errors[8]:.1f}, "
        f"e_8/e_16 = {errors[8] / errors[16]:.1f}"
    )
    assert errors[4] / errors[8] > 8
    assert errors[8] / errors[16] > 8
    assert errors[16] < 1e-6


@needs_reco
def test_lsoda_agrees_with_dop853(reco_stanzas):
    """LSODA and DOP853 at rtol 1e-8 agree on REco with stanzas (EwE rule).

    Over 2 years at effort 2, living and detritus groups, every month.
    Measured: 3.0e-8 (1.8e-8 over living groups only).
    """
    scen = scenario(reco_stanzas, rule="ewe")
    outs = {
        m: rsim_run(
            scenario(reco_stanzas, rule="ewe"),
            method=m,
            years=TWO_YEARS,
            solver_options={"rtol": 1e-8},
        )
        for m in ("lsoda", "dop853")
    }
    diff = _max_rel(outs["lsoda"], outs["dop853"], _n_bio(scen))
    print(f"\nLSODA vs DOP853 (rtol 1e-8): {diff:.3e}")
    assert diff < 1e-5


@pytest.fixture(scope="module")
def lsoda_plain_tight(reco_plain):
    """A 1-year LSODA run of REco without stanzas at rtol 1e-8 (EwE rule)."""
    return rsim_run(
        scenario(reco_plain, rule="ewe"),
        method="lsoda",
        years=ONE_YEAR,
        solver_options={"rtol": 1e-8},
    )


@needs_reco
@pytest.mark.parametrize("method", ["radau", "bdf"])
def test_radau_and_bdf_agree_with_lsoda(reco_plain, lsoda_plain_tight, method):
    """Radau and BDF at rtol 1e-8 agree with LSODA on REco without stanzas.

    One year at effort 2 under the EwE rule, living and detritus groups,
    every month. Measured: Radau 7.9e-8, BDF 1.0e-6 (2.6e-8 and 3.9e-7 over
    living groups only).
    """
    scen = scenario(reco_plain, rule="ewe")
    out = rsim_run(scen, method=method, years=ONE_YEAR, solver_options={"rtol": 1e-8})
    diff = _max_rel(out, lsoda_plain_tight, _n_bio(scen))
    print(f"\n{method} vs LSODA (rtol 1e-8): {diff:.3e}")
    assert diff < 1e-5


@needs_reco
@pytest.mark.slow
@pytest.mark.parametrize("rule", ["ewe", "rpath"])
def test_a_balanced_model_stays_balanced(reco_plain, rule):
    """REco at base effort, unforced, stays at its base for 50 years (LSODA).

    The start derivative is 4e-14 x B under both rules, so what remains is
    the solver's error at the default rtol 1e-6, which does not grow with
    time. Measured: 3.0e-7 (EwE rule) and 3.1e-7 ("rpath"), Phytoplankton,
    flat from year 1 to year 50; at rtol 1e-8 and 1e-10 a 1-year run stays
    within 2.5e-9 and 1.7e-12.
    """
    scen = scenario(reco_plain, rule=rule, effort=1.0, years=range(1, 51))
    f = scen.forcing
    for name in ("ForcedPrey", "ForcedMort", "ForcedSearch", "ForcedEffort"):
        assert np.all(getattr(f, name) == 1.0), name
    assert np.all(f.ForcedMigrate == 0.0) and np.all(f.ForcedBio <= 0.0)
    assert np.all(scen.fishing.ForcedEffort == 1.0)

    out = rsim_run(scen, method="lsoda")
    n_living = scen.params.NUM_LIVING
    assert out.out_Biomass.shape[0] == 50 * 12 + 1
    b0 = np.asarray(scen.params.B_BaseRef, dtype=float)[1 : n_living + 1]
    drift = float(np.max(np.abs(out.out_Biomass[:, 1 : n_living + 1] / b0 - 1)))
    print(f"\nbalanced REco, {rule} rule, 50 years LSODA: {drift:.3e}")
    assert drift < 1e-6


def _load_ewe(path, **kwargs):
    from pypath.io.ewemdb import check_ewemdb_support, ecosim_scenario_from_ewemdb

    if not path.exists() or not any(check_ewemdb_support().values()):
        pytest.skip(f"{path.name} or a database driver missing")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ecosim_scenario_from_ewemdb(str(path), **kwargs)


def _reco_effort_2():
    if not (REF / "stanza_indiv.csv").exists():
        pytest.skip("Rpath reference data missing")
    return scenario(_reco_stanzas(), rule="ewe", years=range(1, 11))


MODELS = {
    # REco with stanzas, EwE rule, effort 2 (not from Data/; here so the
    # page's three rows come from one printout)
    "reco": _reco_effort_2,
    "black_sea": lambda: _load_ewe(BLACK_SEA, scenario=1),
    "lt2022": lambda: _load_ewe(LT2022, scenario=16, timeseries_dataset=35),
}


def _timed_run(scen, method, years):
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rsim_run(scen, method=method, years=years)
    return out, time.perf_counter() - start


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("model", list(MODELS))
def test_black_sea_and_lt2022_run(model):
    """Ten years of LSODA stay finite and positive; differences from RK4.

    The Black Sea (scenario 1) and LT2022 (scenario 16, dataset 35), plus
    REco with stanzas at effort 2 so that the page's three rows come from
    one printout. Prints, for ``docs/parity/continuous_path.md``: the median
    and maximum of ``|B_lsoda / B_RK4 - 1|`` over months 1-120 (living
    groups, and living plus detritus), the derivative calls per month from
    ``solver_info`` and the wall time per simulated year of both paths.
    The differences are largely definitional (the page attributes them), so
    they are printed, not asserted. The Black Sea takes about 400 s: at its
    CmCo 2 each right-hand-side call solves handling time in up to 132
    derivative evaluations.
    """
    years = range(1, 11)
    scen = MODELS[model]()
    rk4, t_rk4 = _timed_run(copy.deepcopy(scen), "RK4", years)
    lsoda, t_lsoda = _timed_run(scen, "lsoda", years)

    n_living = int(scen.params.NUM_LIVING)
    n_bio = n_living + int(scen.params.NUM_DEAD)
    bio = lsoda.out_Biomass[:, 1 : n_bio + 1]
    assert bio.shape[0] == 121
    assert np.all(np.isfinite(bio)) and np.all(bio > 0)
    months = lsoda.solver_info["months"]
    assert len(months) == 120 and all(m["status"] == 0 for m in months)

    rel = np.abs(bio[1:] / rk4.out_Biomass[1:, 1 : n_bio + 1] - 1)
    living = rel[:, :n_living]
    names = [str(n).strip() for n in scen.params.spname[1 : n_bio + 1]]
    worst = names[int(np.argmax(living.max(axis=0)))]
    worst_all = names[int(np.argmax(rel.max(axis=0)))]
    nfev = np.array([m["nfev"] for m in months])
    print(
        f"\n{model}: lsoda vs RK4, living: median {np.median(living):.3g}, "
        f"max {living.max():.3g} ({worst}); living + detritus: median "
        f"{np.median(rel):.3g}, max {rel.max():.3g} ({worst_all})"
    )
    print(
        f"{model}: nfev per month mean {nfev.mean():.0f}, median "
        f"{np.median(nfev):.0f}, max {nfev.max()}; njev per month mean "
        f"{np.mean([m['njev'] for m in months]):.1f}; Hden evaluations per "
        f"RHS call, max {max(m['hden_evals_max'] for m in months)}; clips "
        f"{sum(m['clips'] for m in months)}"
    )
    print(
        f"{model}: seconds per simulated year: RK4 {t_rk4 / 10:.3f}, "
        f"lsoda {t_lsoda / 10:.3f} ({t_lsoda / t_rk4:.0f} x); "
        f"corrections {lsoda.solver_info['corrections']}"
    )
