"""EwE6 pred/prey mediation (cMediationDataStructures, deviation D3).

A mediation shape holds 1200 points, Medpoints(1..1200). Each derivative
(cEcoSimModel.Derivt, vb 2851) sets every used shape's value from the current
biomass (SetMedFunctions): X = 1e-10 + sum(weight x B) over its mediators,
the point ip = Trunc(IMedBase x X / MedXbase + 0.01) clamped to 1..1200, and
MedVal = Medpoints(ip) / MedYbase. MedXbase is sum(weight x StartBiomass) and
MedYbase = Medpoints(IMedBase) (InitializeMedFunctions, vb 3568-3650), so
MedVal is 1 at the base. A link's mediation function multiplies its search
rate (SearchRate) or its vulnerability (Vulnerability), as a time forcing
shape on that link does (ApplyAVmodifiers, vb 4928-4970).
"""

import math
import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params

# rsim index: 0 Outside, 1 Fish, 2 Zoo, 3 Benthos, 4 Phyto, 5 Det
FISH, ZOO, BENTHOS, PHYTO, DET = 1, 2, 3, 4, 5
SEARCH, VULNERABILITY = 1, 2  # eForcingFunctionApplication


def _points(f, imedbase):
    """1200 points y(ip / IMedBase), written as the oracle patch writes them."""
    return np.array(
        [float(f"{f(ip / imedbase):.7g}") for ip in range(1, 1201)], dtype=np.float32
    )


SHAPE_A = _points(lambda r: 0.4 + 1.2 * r * r / (0.36 + r * r), 300)
SHAPE_B = _points(lambda r: 2.0 - 1.5 * math.exp(-1.5 * r), 500)

# EwE 6.7.144 headless oracle (run 43216917549, mediation_lookup.csv):
# getMedValue at X for the two patch shapes, MedXbase 1.27996302 and
# 11.5459995. 0.99 and 1.01 sit on either side of the base point; shape B is
# clamped at point 1200 from X / MedXbase = 2.4.
EWE_LOOKUP = {
    0: (
        1.27996302,
        [
            (0.0255992599, 0.312965214),
            (0.127996311, 0.337217897),
            (0.422387809, 0.529257357),
            (0.639981508, 0.695442915),
            (1.2671634, 0.996321619),
            (1.27996302, 1.0),
            (1.29276264, 1.00360739),
            (1.91994452, 1.11863351),
            (2.55992603, 1.17043972),
            (4.4798708, 1.22099054),
        ],
    ),
    1: (
        11.5459995,
        [
            (0.230919987, 0.32686609),
            (1.15460002, 0.425710589),
            (3.81017995, 0.651919007),
            (5.77299976, 0.775503576),
            (11.4305401, 0.99696213),
            (11.5459995, 1.0),
            (11.661459, 1.00299227),
            (17.3190002, 1.10604417),
            (23.0919991, 1.15613592),
            (34.6380005, 1.17636955),
            (40.4109993, 1.17636955),
        ],
    ),
}


def _mediation(points, imedbase, weights, links, start):
    from pypath.core.mediation import EweMediation

    med = EweMediation(
        points=np.atleast_2d(points),
        imedbase=np.atleast_1d(imedbase),
        weights=np.atleast_2d(weights),
        links=links,
    )
    med.setup(np.asarray(start, dtype=float))
    return med


@pytest.mark.parametrize("shape", [0, 1])
def test_lookup_is_ewes_get_med_value(shape):
    points = (SHAPE_A, SHAPE_B)[shape]
    imedbase = (300, 500)[shape]
    xbase, table = EWE_LOOKUP[shape]
    # one mediator whose start biomass is the oracle's MedXbase
    med = _mediation(points, imedbase, [0.0, 1.0], [], [1.0, xbase])
    assert med.xbase[0] == pytest.approx(xbase, rel=1e-7)
    for x, ewe in table:
        assert med.lookup(0, x) == pytest.approx(ewe, rel=1e-6), x


def test_values_weigh_the_mediators_against_their_start_biomass():
    weights = [0.0, 1.0, 0.5]  # group 1 weight 1, group 2 weight 0.5
    start = np.array([1.0, 8.0, 4.0])
    med = _mediation(SHAPE_B, 500, weights, [], start)
    assert med.xbase[0] == pytest.approx(10.0)
    assert med.values(start)[0] == pytest.approx(1.0)
    now = np.array([1.0, 2.0, 6.0])  # X = 2 + 3 = 5, half the base
    assert med.values(now)[0] == pytest.approx(med.lookup(0, 5.0))
    assert med.lookup(0, 5.0) < 0.8


def test_a_shape_without_mediators_or_base_is_unused():
    zero_base = SHAPE_A.copy()
    zero_base[299] = 0.0  # Medpoints(IMedBase) = 0
    med = _mediation(
        np.vstack([SHAPE_A, zero_base]),
        [300, 300],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [],
        [1.0, 5.0, 5.0],
    )
    assert not med.used[0] and not med.used[1]
    np.testing.assert_array_equal(med.values(np.array([1.0, 1.0, 9.0])), 1.0)


# --- the mediated link acts as a link forcing shape of the same value ---------


def _scenario():
    params = create_rpath_params(
        groups=["Fish", "Zoo", "Benthos", "Phyto", "Det"], types=[0, 0, 0, 1, 2]
    )
    params.model.loc[0, ["Biomass", "PB", "QB"]] = [2.0, 1.0, 5.0]
    params.model.loc[1, ["Biomass", "PB", "QB"]] = [10.0, 30.0, 100.0]
    params.model.loc[2, ["Biomass", "PB", "QB"]] = [20.0, 2.0, 8.0]
    params.model.loc[3, ["Biomass", "PB"]] = [50.0, 100.0]
    params.model.loc[4, "Biomass"] = 1000.0
    params.model["EE"] = np.nan
    params.diet["Fish"] = [0.0, 0.6, 0.4, 0.0, 0.0, 0.0]
    params.diet["Zoo"] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    params.diet["Benthos"] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    params.diet["Phyto"] = [0.0] * 6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = rsim_scenario(rpath(params), params, years=range(1, 3))
    # Benthos is the mediator, held at half its base biomass
    sc.forcing.ForcedBio[:, BENTHOS] = 0.5 * sc.params.B_BaseRef[BENTHOS]
    return sc


def _flat_shape():
    """0.625 everywhere but the base point: MedVal 0.625 (exact in single
    precision) away from the base."""
    pts = np.full(1200, 0.625, dtype=np.float32)
    pts[399] = 1.0
    return pts


def _run(sc):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rsim_run(sc, method="RK4")


@pytest.mark.parametrize(
    "application, link_key",
    [(SEARCH, "ForcedSearchLink"), (VULNERABILITY, "ForcedVulLink")],
)
def test_a_mediated_link_acts_as_link_forcing(application, link_key):
    mediated = _scenario()
    n = mediated.params.NUM_GROUPS + 1
    weights = np.zeros(n)
    weights[BENTHOS] = 1.0
    mediated.params.EweMediation = _mediation(
        _flat_shape(),
        400,
        weights,
        [(ZOO, FISH, 0, application)],
        mediated.params.B_BaseRef,
    )
    forced = _scenario()
    months = forced.forcing.ForcedSearch.shape[0]
    arr = np.ones((months, n, n))
    arr[:, ZOO, FISH] = 0.625
    setattr(forced.forcing, link_key, arr)
    a, b = _run(mediated), _run(forced)
    assert not np.allclose(b.out_Biomass[-1], b.out_Biomass[0])  # it acts
    np.testing.assert_allclose(a.out_Biomass, b.out_Biomass, rtol=1e-10)


def test_ecospace_uniform_patches_match_rsim_run_with_mediation():
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    sc = _scenario()
    sc.forcing.ForcedBio[:, BENTHOS] = 0  # the mediator moves freely
    n = sc.params.NUM_GROUPS + 1
    weights = np.zeros(n)
    weights[BENTHOS] = 1.0
    sc.params.EweMediation = _mediation(
        SHAPE_A, 300, weights, [(ZOO, FISH, 0, SEARCH)], sc.params.B_BaseRef
    )
    sc.forcing.ForcedPrey[12:, PHYTO] = 0.5  # disturb the base state
    grid = create_1d_grid(n_patches=2, spacing=1.0)
    eco = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n, 2)),
        habitat_capacity=np.ones((n, 2)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    expected = _run(sc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = rsim_run_spatial(sc, ecospace=eco, years=range(1, 3))
    np.testing.assert_allclose(
        got.out_Biomass, expected.out_Biomass, rtol=1e-6, atol=1e-10
    )


# --- EwE databases: purpose-built oracle cases on LT2022 scenario 16 ----------
# The EwE oracle harness applied ewe-oracle patches/lt2022_s16_mediation_{a,b}.sql
# to LT2022 and returned the patched databases. A: shape 900 (IMedBase 300),
# mediator Sprat, on Mobile crustaceans' search rate for Zooplankton. B: shape
# 900 on Herring's search rate for Zooplankton, and shape 901 (IMedBase 500;
# Zooplankton weight 1, Mobile crustaceans 0.5) on the vulnerability of
# Zooplankton to Smelt and to Herring (so Zooplankton -> Herring carries two).

from pathlib import Path  # noqa: E402

DATA = Path(__file__).parents[3] / "Data"
CASES = {
    "a": DATA / "LT2022_s16_mediation_a.eweaccdb",
    "b": DATA / "LT2022_s16_mediation_b.eweaccdb",
}


def _case(name):
    path = CASES[name]
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ecosim_scenario_from_ewemdb(
            str(path), scenario=16, timeseries_dataset=35
        )


@pytest.mark.integration
def test_reader_builds_ewes_mediation(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"):
        sc = _case("b")
    assert "mediation" not in caplog.text.lower()
    med = sc.params.EweMediation
    names = [str(n).strip() for n in sc.params.spname]
    idx = {
        g: names.index(g)
        for g in ("Sprat", "Zooplankton", "Mobile crustaceans", "Herring", "Smelt")
    }
    assert list(med.imedbase) == [300, 500]
    assert med.weights[0, idx["Sprat"]] == 1.0
    assert med.weights[1, idx["Zooplankton"]] == 1.0
    assert med.weights[1, idx["Mobile crustaceans"]] == 0.5
    assert np.count_nonzero(med.weights) == 3
    # oracle MedXbase (mediation_shapes.csv)
    np.testing.assert_allclose(med.xbase, [1.27996302, 11.5459995], rtol=1e-6)
    zoo, her, smelt = idx["Zooplankton"], idx["Herring"], idx["Smelt"]
    assert sorted(med.links) == sorted(
        [
            (zoo, smelt, 1, VULNERABILITY),
            (zoo, her, 0, SEARCH),
            (zoo, her, 1, VULNERABILITY),
        ]
    )


# EwE 6.7.144 oracle, scenario 16 with dataset 35 (runs 43218423464 for A,
# 43216917549 for B): biomass at the end of months 24, 60 and 144. Without
# the mediation the run is the baseline (run 43216863421), e.g. Herring 4.58,
# 4.23 and 1.16, Sprat 0.81, 0.82 and 0.11.
EWE_MEDIATED = {
    "a": {
        "Sprat": {24: 0.83999455, 60: 0.995639384, 144: 0.162098929},
        "Zooplankton": {24: 2.55437565, 60: 1.96012044, 144: 1.7053957},
        "Herring": {24: 4.68199348, 60: 5.09676838, 144: 1.45079303},
        "Mobile crustaceans": {24: 9.06728554, 60: 11.4243584, 144: 0.809769154},
    },
    "b": {
        "Sprat": {24: 0.848743379, 60: 0.986188173, 144: 0.203813002},
        "Zooplankton": {24: 2.54054165, 60: 2.27240658, 144: 1.97005832},
        "Herring": {24: 4.29798603, 60: 3.22801542, 144: 0.851211309},
        "Smelt": {24: 4.45975971, 60: 3.04334712, 144: 1.35549843},
    },
}


@pytest.mark.integration
@pytest.mark.parametrize("name", ["a", "b"])
def test_mediated_runs_follow_the_ewe_oracle(name):
    sc = _case(name)
    out = _run(sc)
    names = [str(n).strip() for n in sc.params.spname]
    for group, months in EWE_MEDIATED[name].items():
        j = names.index(group)
        for month, ewe in months.items():
            assert out.out_Biomass[month, j] == pytest.approx(ewe, rel=2e-3), (
                group,
                month,
            )


@pytest.mark.integration
def test_the_rpath_rule_reports_and_skips_mediation(caplog):
    import logging

    path = CASES["b"]
    if not path.exists():
        pytest.skip(f"{path.name} not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    with (
        caplog.at_level(logging.WARNING, logger="pypath.io.ewemdb"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewemdb(
            str(path), scenario=16, timeseries_dataset=35, fast_group_rule="rpath"
        )
    assert sc.params.EweMediation is None
    assert "3 mediation functions need the EwE rule" in caplog.text


# --- reader edge cases, on stand-in tables (no database) ----------------------


def _fake_tables(monkeypatch, pp_rows, med_rows, weight_rows):
    import pandas as pd

    import pypath.io.ewemdb as ew

    tables = {
        "EcosimShape": pd.DataFrame(
            {"ShapeID": [r[0] for r in med_rows], "ShapeType": 10, "IsSeasonal": 0}
        ),
        "EcosimShapeMediation": pd.DataFrame(
            med_rows, columns=["ShapeID", "zScale", "IMedBase"]
        ),
        "EcosimScenarioPredPreyShape": pd.DataFrame(
            pp_rows,
            columns=["ScenarioID", "PredID", "PreyID", "ShapeID", "FunctionType"],
        ),
        "EcosimScenarioShapeMedWeightsGroup": pd.DataFrame(
            weight_rows, columns=["ScenarioID", "ShapeID", "GroupID", "MedWeights"]
        ),
    }

    def read(filepath, names):
        for name in names:
            if name in tables:
                return tables[name].copy()
        return None

    monkeypatch.setattr(ew, "_try_read_table_variants", read)


def _apply(monkeypatch, pp_rows, med_rows, weight_rows):
    from types import SimpleNamespace

    import pandas as pd

    from pypath.io.ewemdb import _apply_mediation

    _fake_tables(monkeypatch, pp_rows, med_rows, weight_rows)
    names = ["Fish", "Zoo", "Benthos", "Phyto", "Det"]
    selected = {
        "id": 1,
        "scenario_group_df": pd.DataFrame(
            {"GroupID": [11, 12, 13, 14, 15], "EcopathGroupID": [1, 2, 3, 4, 5]}
        ),
        "group_id_to_name": dict(zip([1, 2, 3, 4, 5], names)),
    }
    rsim = SimpleNamespace(
        params=SimpleNamespace(
            PREY_FTIME_ARENA=True,
            spname=["Outside"] + names,
            B_BaseRef=np.array([1.0, 2.0, 10.0, 20.0, 50.0, 1000.0]),
            EweMediation=None,
        )
    )
    params = SimpleNamespace(
        model=pd.DataFrame({"Group": names, "Type": [0, 0, 0, 1, 2]})
    )
    _apply_mediation("stand-in", selected, rsim, params)
    return rsim.params.EweMediation


POINTS = " ".join(["1"] * 1200)


def test_a_cannibal_link_keeps_its_mediation(monkeypatch):
    """Fish eats Fish: EwE applies a link's functions to any arena link,
    a predator's own included (ApplyAVmodifiers)."""
    med = _apply(
        monkeypatch,
        pp_rows=[(1, 11, 11, 900, SEARCH)],
        med_rows=[(900, POINTS, 400)],
        weight_rows=[(1, 900, 13, 1.0)],
    )
    assert med is not None and med.links == [(FISH, FISH, 0, SEARCH)]


def test_an_unreadable_shape_does_not_drop_the_others(monkeypatch):
    med = _apply(
        monkeypatch,
        pp_rows=[(1, 11, 12, 900, SEARCH), (1, 11, 13, 901, VULNERABILITY)],
        med_rows=[(900, "1 x 2", 400), (901, POINTS.replace("1", "0.5", 1), 400)],
        weight_rows=[(1, 900, 13, 1.0), (1, 901, 13, 1.0)],
    )
    assert med is not None and len(med.links) == 2
    np.testing.assert_array_equal(med.points[0], 1.0)  # unreadable: neutral
    assert med.points[1][0] == pytest.approx(0.5)


def test_ecospace_with_ibm_groups_gives_mediation_densities(monkeypatch):
    """IBM groups keep absolute patch biomass in the spatial derivative;
    mediation compares densities with its whole-model base, so a mediator at
    its base density starts at X / MedXbase = 1 in every patch."""
    from types import SimpleNamespace

    from pypath.core.mediation import EweMediation
    from pypath.spatial import EcospaceParams, create_1d_grid, rsim_run_spatial

    sc = _scenario()
    sc.forcing.ForcedBio[:, BENTHOS] = 0
    n = sc.params.NUM_GROUPS + 1
    weights = np.zeros(n)
    weights[BENTHOS] = 1.0
    sc.params.EweMediation = _mediation(
        SHAPE_A, 300, weights, [(ZOO, FISH, 0, SEARCH)], sc.params.B_BaseRef
    )

    class _IBM:
        group_index, n_groups = PHYTO, sc.params.NUM_GROUPS

        def compute_step(self, **kwargs):
            return SimpleNamespace(
                biomass=1000.0, consumption_by_prey=np.zeros(self.n_groups)
            )

    sc.params.ibm_groups = {PHYTO: _IBM()}
    ratios = []
    original = EweMediation.values

    def spy(self, biomass):
        ratios.append(float(biomass[BENTHOS]) / float(self.xbase[0]))
        return original(self, biomass)

    monkeypatch.setattr(EweMediation, "values", spy)
    grid = create_1d_grid(n_patches=2, spacing=1.0)
    eco = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n, 2)),
        habitat_capacity=np.ones((n, 2)),
        dispersal_rate=np.zeros(n),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rsim_run_spatial(sc, ecospace=eco, years=range(1, 3))
    assert ratios[0] == pytest.approx(1.0, rel=1e-9)
