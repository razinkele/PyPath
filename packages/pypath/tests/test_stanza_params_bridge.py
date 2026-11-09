"""Multi-stanza parameters stored as DataFrames must reach Ecosim.

``RpathParams.stanzas`` holds the stanza tables as DataFrames (``stgroups`` /
``stindiv``), which is what ``create_rpath_params``, ``read_rpath_params`` and
the EwE database reader build. ``rpath_stanzas`` / ``rsim_stanzas`` work on
``EcosimStanzaParams`` (lists of dataclasses). Without a bridge between the two,
``rsim_scenario`` failed on every real multi-stanza model and logged it only
at debug level.

rsim_run couples the age structure to group biomass the way Rpath's
SplitUpdate / SplitSetPred do; the coupling tests are at the end of this file
and the trajectory comparison with Rpath is in ``test_rpath_reference.py``.

The reference values come from Rpath's ``rpath.stanzas()`` on the REco model
(``tests/data/rpath_reference/ecopath``).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import RpathStanzaParams, create_rpath_params
from pypath.core.stanzas import (
    EcosimStanzaParams,
    ecosim_stanza_params,
    rpath_stanzas,
)

ECOPATH_DIR = Path(__file__).parent / "data" / "rpath_reference" / "ecopath"

# Columns Rpath fills in itself; strip them so the test feeds inputs only.
_COMPUTED_GROUP_COLS = ["last", "R"]
_COMPUTED_INDIV_COLS = ["Biomass", "Cons", "QB"]


def _reference_stanza_tables():
    stgroups = pd.read_csv(ECOPATH_DIR / "stanza_groups.csv")
    stindiv = pd.read_csv(ECOPATH_DIR / "stanza_indiv.csv")
    return stgroups, stindiv


def _reference_params():
    model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")
    params = create_rpath_params(model_df["Group"].tolist(), model_df["Type"].tolist())
    params.model = model_df
    params.diet = diet_df
    stgroups, stindiv = _reference_stanza_tables()
    params.stanzas = RpathStanzaParams(
        n_stanza_groups=len(stgroups),
        stgroups=stgroups.drop(columns=_COMPUTED_GROUP_COLS),
        stindiv=stindiv.drop(columns=_COMPUTED_INDIV_COLS),
    )
    return params


class TestConversion:
    def test_rpath_style_tables_convert(self):
        params = _reference_params()
        esp = ecosim_stanza_params(params.stanzas, params.model)

        assert isinstance(esp, EcosimStanzaParams)
        assert esp.n_stanza_groups == 4
        assert [sg.n_stanzas for sg in esp.stanza_groups] == [2, 2, 2, 2]
        assert esp.stanza_groups[0].vbgf_ksp == pytest.approx(0.145)
        assert esp.stanza_groups[0].wmat == pytest.approx(0.0577)

        juv = esp.stanza_individuals[0]
        assert (juv.group_name, juv.group_num, juv.first, juv.last) == (
            "JuvRoundfish1",
            4,
            0,
            23,
        )
        assert juv.z == pytest.approx(2.026)
        assert juv.leading is False
        assert esp.stanza_individuals[1].leading is True

    def test_converted_object_is_returned_unchanged(self):
        params = _reference_params()
        esp = ecosim_stanza_params(params.stanzas, params.model)
        assert ecosim_stanza_params(esp, params.model) is esp

    def test_snake_case_tables_convert(self):
        """``example_model_data`` stores the stanza CSVs with snake_case headers."""
        stgroups = pd.DataFrame(
            {
                "stanza_group_num": [1],
                "n_stanzas": [2],
                "vbgf_ksp": [0.145],
                "vbgf_d": [0.66667],
                "wmat": [0.0577],
                "bab": [0.0],
                "rec_power": [1.0],
            }
        )
        stindiv = pd.DataFrame(
            {
                "stanza_group_num": [1, 1],
                "stanza_num": [1, 2],
                "group_num": [4, 5],
                "group_name": ["JuvRoundfish1", "AduRoundfish1"],
                "first": [0, 24],
                "last": [23, 371],
                "z": [2.026, 0.42],
                "leading": [False, True],
            }
        )
        model = pd.read_csv(ECOPATH_DIR / "model_params.csv")
        esp = ecosim_stanza_params(
            RpathStanzaParams(n_stanza_groups=1, stgroups=stgroups, stindiv=stindiv),
            model,
        )
        assert esp.n_stanza_groups == 1
        assert [s.group_name for s in esp.stanza_individuals] == [
            "JuvRoundfish1",
            "AduRoundfish1",
        ]

    def test_group_num_and_stanza_order_are_derived_when_missing(self):
        """The EwE reader gives no GroupNum; create_rpath_params sets StanzaNum=0."""
        params = _reference_params()
        stindiv = params.stanzas.stindiv.drop(columns=["GroupNum"]).copy()
        stindiv["StanzaNum"] = 0
        params.stanzas.stindiv = stindiv.iloc[::-1].reset_index(drop=True)

        esp = ecosim_stanza_params(params.stanzas, params.model)
        rf1 = [s for s in esp.stanza_individuals if s.stanza_group_num == 1]
        rf1.sort(key=lambda s: s.stanza_num)
        assert [(s.group_name, s.group_num, s.stanza_num) for s in rf1] == [
            ("JuvRoundfish1", 4, 1),
            ("AduRoundfish1", 5, 2),
        ]

    def test_ewe_style_group_ids_are_renumbered(self):
        """EwE databases number stanza groups by ID (e.g. 3, 7, 12, 20)."""
        params = _reference_params()
        ids = {1: 3, 2: 7, 3: 12, 4: 20}
        params.stanzas.stgroups["StGroupNum"] = params.stanzas.stgroups[
            "StGroupNum"
        ].map(ids)
        params.stanzas.stindiv["StGroupNum"] = params.stanzas.stindiv["StGroupNum"].map(
            ids
        )
        esp = ecosim_stanza_params(params.stanzas, params.model)
        assert [g.stanza_group_num for g in esp.stanza_groups] == [1, 2, 3, 4]
        flat2 = [
            s.group_name for s in esp.stanza_individuals if s.stanza_group_num == 4
        ]
        assert flat2 == ["JuvFlatfish2", "AduFlatfish2"]

    def test_leading_given_as_text_is_parsed(self):
        params = _reference_params()
        params.stanzas.stindiv["Leading"] = params.stanzas.stindiv["Leading"].map(
            {True: "TRUE", False: "FALSE"}
        )
        esp = ecosim_stanza_params(params.stanzas, params.model)
        assert [s.leading for s in esp.stanza_individuals[:2]] == [False, True]

    @pytest.mark.parametrize("column", ["Z", "First", "Leading"])
    def test_incomplete_stanza_rows_raise(self, column):
        params = _reference_params()
        stindiv = params.stanzas.stindiv
        stindiv[column] = stindiv[column].astype(object)
        stindiv.loc[0, column] = np.nan
        with pytest.raises(ValueError, match="JuvRoundfish1"):
            ecosim_stanza_params(params.stanzas, params.model)

    def test_missing_growth_parameter_raises(self):
        params = _reference_params()
        params.stanzas.stgroups.loc[0, "VBGF_Ksp"] = np.nan
        with pytest.raises(ValueError, match="Roundfish1"):
            ecosim_stanza_params(params.stanzas, params.model)


class TestMatchesRpath:
    """rpath_stanzas on converted tables reproduces Rpath's rpath.stanzas()."""

    @pytest.fixture(scope="class")
    def computed(self):
        params = _reference_params()
        esp = ecosim_stanza_params(params.stanzas, params.model)
        rpath_stanzas(params, stanza_params=esp)
        return params, esp

    def test_last_month_and_recruits(self, computed):
        _, esp = computed
        ref_groups, _ = _reference_stanza_tables()
        for sg, (_, ref) in zip(esp.stanza_groups, ref_groups.iterrows()):
            assert sg.last_month == ref["last"]
            assert sg.recruits == pytest.approx(ref["R"], rel=1e-6)

    def test_stanza_biomass_and_qb(self, computed):
        _, esp = computed
        _, ref_indiv = _reference_stanza_tables()
        got = {s.group_name: s for s in esp.stanza_individuals}
        for _, ref in ref_indiv.iterrows():
            st = got[ref["Group"]]
            assert st.biomass == pytest.approx(ref["Biomass"], rel=1e-6), ref["Group"]
            assert st.qb == pytest.approx(ref["QB"], rel=1e-6), ref["Group"]


class TestEcosimRunsStanzas:
    @pytest.fixture(scope="class")
    def scenario(self):
        params = _reference_params()
        model = rpath(params)
        return rsim_scenario(model, params, years=range(1, 11))

    def test_scenario_has_stanzas(self, scenario):
        assert scenario.stanzas is not None
        assert scenario.stanzas.n_split == 4

    def test_base_age_structure_is_filled(self, scenario):
        st = scenario.stanzas
        for isp in range(1, st.n_split + 1):
            assert np.isfinite(st.base_nage_s[0, isp])
            assert np.isfinite(st.base_wage_s[0, isp])

    def test_baseline_stays_at_equilibrium(self, scenario):
        """The baseline stays flat with the stanza age structure set up."""
        out = rsim_run(scenario, method="RK4", years=range(1, 11))
        assert out.stanza_biomass is not None
        bio = out.out_Biomass
        start = bio[0]
        live = start > 0
        drift = np.max(np.abs(bio[-1, live] / start[live] - 1.0))
        assert drift < 1e-3

    def test_n_stanza_groups_zero_but_tables_present_still_runs(self):
        """The parity fixture used to load the tables but leave the count at 0."""
        params = _reference_params()
        params.stanzas.n_stanza_groups = 0
        sc = rsim_scenario(rpath(params), params, years=range(1, 3))
        assert sc.stanzas is not None and sc.stanzas.n_split == 4


def test_stanza_setup_failure_is_logged_as_warning(caplog):
    params = _reference_params()
    model = rpath(params)
    params.stanzas.stindiv.loc[0, "Z"] = np.nan
    with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim"):
        sc = rsim_scenario(model, params, years=range(1, 3))
    assert sc.stanzas is None
    assert any("stanza" in r.getMessage().lower() for r in caplog.records)


class TestInputChecks:
    def test_youngest_stanza_must_start_at_month_zero(self):
        params = _reference_params()
        params.stanzas.stindiv.loc[0, "First"] = 1
        with pytest.raises(ValueError, match="month 0"):
            ecosim_stanza_params(params.stanzas, params.model)

    def test_first_must_increase_with_stanza_number(self):
        params = _reference_params()
        si = params.stanzas.stindiv
        # Both Roundfish1 stanzas start at month 0: passes the month-0 check,
        # fails the ordering one.
        si.loc[1, "First"] = 0
        with pytest.raises(ValueError, match="First must increase"):
            ecosim_stanza_params(params.stanzas, params.model)

    def test_duplicate_stanza_group_ids_are_rejected(self):
        params = _reference_params()
        params.stanzas.stgroups.loc[1, "StGroupNum"] = 1
        with pytest.raises(ValueError, match="appears twice"):
            ecosim_stanza_params(params.stanzas, params.model)

    def test_non_positive_growth_rate_is_rejected(self):
        params = _reference_params()
        params.stanzas.stgroups.loc[0, "VBGF_Ksp"] = 0.0
        with pytest.raises(ValueError, match="positive"):
            ecosim_stanza_params(params.stanzas, params.model)

    def test_group_num_must_match_the_named_group(self):
        params = _reference_params()
        params.stanzas.stindiv.loc[0, "GroupNum"] = 5  # AduRoundfish1's row
        with pytest.raises(ValueError, match="GroupNum 5"):
            ecosim_stanza_params(params.stanzas, params.model)

    def test_stgroups_alone_counts_as_stanza_tables(self):
        from pypath.core.stanzas import has_stanza_tables

        stgroups, _ = _reference_stanza_tables()
        st = RpathStanzaParams(stgroups=stgroups, stindiv=pd.DataFrame())
        assert has_stanza_tables(st)
        with pytest.raises(ValueError, match="no stanzas"):
            ecosim_stanza_params(st, None)


def test_last_month_never_precedes_the_plus_group_start():
    """With very high adult Z the 99.999% cutoff falls at month 24, which the
    year rounding turned into 23, before the adults start."""
    params = _reference_params()
    params.stanzas.stindiv.loc[1, "Z"] = 500.0
    esp = ecosim_stanza_params(params.stanzas, params.model)
    rpath_stanzas(params, stanza_params=esp)
    assert esp.stanza_groups[0].last_month >= 24


def test_scenario_does_not_edit_the_params_table():
    params = _reference_params()
    model = rpath(params)
    before = params.model.copy()
    sc = rsim_scenario(model, params, years=range(1, 3))
    assert sc.stanzas is not None
    pd.testing.assert_frame_equal(params.model, before)


def test_stanza_update_uses_each_stanzas_own_group():
    """group_num is the 1-based model row and the Rsim arrays have Outside
    at 0, so a stanza's mortality is LossPropToB[group_num] / B[group_num]."""
    from pypath.core.stanzas import RsimStanzaState, split_update

    params = _reference_params()
    sc = rsim_scenario(rpath(params), params, years=range(1, 3))
    st = sc.stanzas
    biomass = sc.start_state.Biomass.copy()
    juv_row = params.model.index[params.model["Group"] == "JuvRoundfish1"][0] + 1
    loss = np.zeros_like(biomass)
    loss[juv_row] = 12.0 * biomass[juv_row]  # monthly survival exp(-1)
    sstate = RsimStanzaState.from_base(st, sc.params.NUM_GROUPS)
    before = st.base_nage_s[:, 1].copy()
    split_update(st, sstate, biomass, np.zeros_like(biomass), loss)
    after = sstate.nage_s[:, 1]
    assert after[5] == pytest.approx(before[4] * np.exp(-1.0))  # juvenile age
    assert after[30] == pytest.approx(before[29])  # adult age, no loss
    # The base age structure is a parameter, not run state
    np.testing.assert_array_equal(st.base_nage_s[:, 1], before)


class TestCoupling:
    """The age structure drives the stanza groups' biomass (Rpath
    SplitSetPred), and their consumption drives the age structure."""

    @pytest.fixture(scope="class")
    def model_and_params(self):
        params = _reference_params()
        return rpath(params), params

    @staticmethod
    def _run(model, params, effort, stanzas=True, years=5):
        sc = rsim_scenario(model, params, years=range(1, years + 1))
        if not stanzas:
            sc.stanzas = None
        sc.fishing.ForcedEffort = sc.fishing.ForcedEffort * effort
        return sc, rsim_run(sc, method="RK4", years=range(1, years + 1))

    def test_stanza_groups_respond_to_the_age_structure(self, model_and_params):
        model, params = model_and_params
        sc, coupled = self._run(model, params, 2.0)
        _, uncoupled = self._run(model, params, 2.0, stanzas=False)
        from pypath.core.stanzas import stanza_group_indices

        idx = stanza_group_indices(sc.stanzas)
        diff = np.abs(coupled.out_Biomass[-1, idx] / uncoupled.out_Biomass[-1, idx] - 1)
        assert diff.max() > 0.01

    def test_stanza_biomass_output_is_the_coupled_biomass(self, model_and_params):
        model, params = model_and_params
        sc, out = self._run(model, params, 2.0, years=2)
        from pypath.core.stanzas import stanza_group_indices

        idx = stanza_group_indices(sc.stanzas)
        np.testing.assert_allclose(
            out.stanza_biomass[:, idx], out.out_Biomass[:, idx], rtol=1e-12
        )

    def test_a_run_leaves_the_base_age_structure_alone(self, model_and_params):
        """The age structure evolves in run state; a second run starts from
        the same base again. (rsim_run does keep its M0 adjustments in
        scenario.params, so later months may differ slightly.)"""
        model, params = model_and_params
        sc = rsim_scenario(model, params, years=range(1, 3))
        sc.fishing.ForcedEffort = sc.fishing.ForcedEffort * 2.0
        base = {
            name: getattr(sc.stanzas, name).copy()
            for name in (
                "base_nage_s",
                "base_wage_s",
                "base_qage_s",
                "base_stanza_pred",
            )
        }
        first = rsim_run(sc, method="RK4", years=range(1, 3))
        for name, before in base.items():
            np.testing.assert_array_equal(getattr(sc.stanzas, name), before, name)
        second = rsim_run(sc, method="RK4", years=range(1, 3))
        np.testing.assert_array_equal(first.stanza_biomass[0], second.stanza_biomass[0])

    def test_incomplete_stanza_arrays_fall_back_with_a_warning(
        self, model_and_params, caplog
    ):
        model, params = model_and_params
        sc = rsim_scenario(model, params, years=range(1, 3))
        sc.stanzas.split_alpha = np.zeros((2, 2))
        with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim"):
            out = rsim_run(sc, method="RK4", years=range(1, 3))
        assert out.stanza_biomass is None
        assert any("split_alpha" in r.getMessage() for r in caplog.records)

    def test_consistent_stanza_biomass_does_not_warn(self, model_and_params, caplog):
        model, params = model_and_params
        sc = rsim_scenario(model, params, years=range(1, 3))
        with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim"):
            rsim_run(sc, method="RK4", years=range(1, 3))
        assert not any(
            "differs from its age structure" in r.getMessage() for r in caplog.records
        )

    def test_hand_typed_stanza_biomass_warns(self, model_and_params, caplog):
        """A stanza biomass that is not sum(N*W) is reset at month 0; say so."""
        model, params = model_and_params
        sc = rsim_scenario(model, params, years=range(1, 3))
        from pypath.core.stanzas import stanza_group_indices

        g = int(stanza_group_indices(sc.stanzas)[0])
        sc.start_state.Biomass[g] *= 2.0
        with caplog.at_level(logging.WARNING, logger="pypath.core.ecosim"):
            out = rsim_run(sc, method="RK4", years=range(1, 3))
        msgs = [
            r.getMessage()
            for r in caplog.records
            if "differs from its age structure" in r.getMessage()
        ]
        assert len(msgs) == 1
        assert sc.params.spname[g] in msgs[0]
        # the run still starts from the age structure, as in Rpath
        assert out.out_Biomass[0, g] == pytest.approx(
            sc.start_state.Biomass[g] / 2.0, rel=0.01
        )

    def test_split_update_gets_ecopath_rates_at_baseline(
        self, model_and_params, monkeypatch
    ):
        """At the Ecopath baseline a stanza group eats Q/B * B and loses
        P/B * B to predation + M0 + fishing (production balance)."""
        import pypath.core.ecosim as ecosim_mod

        model, params = model_and_params
        sc = rsim_scenario(model, params, years=range(1, 3))
        seen = []
        real = ecosim_mod.split_update

        def spy(stanzas, sstate, biomass, food_gain, loss, forced_recs=None):
            seen.append((biomass.copy(), food_gain.copy(), loss.copy()))
            return real(stanzas, sstate, biomass, food_gain, loss, forced_recs)

        monkeypatch.setattr(ecosim_mod, "split_update", spy)
        rsim_run(sc, method="RK4", years=range(1, 3))
        biomass, food_gain, loss = seen[0]
        from pypath.core.stanzas import stanza_group_indices

        idx = stanza_group_indices(sc.stanzas)
        qb = np.asarray(model.QB, dtype=float)[idx - 1]
        pb = np.asarray(model.PB, dtype=float)[idx - 1]
        np.testing.assert_allclose(food_gain[idx] / biomass[idx], qb, rtol=1e-3)
        np.testing.assert_allclose(loss[idx] / biomass[idx], pb, rtol=1e-3)


def test_tables_are_saved_even_when_the_count_was_left_at_zero(tmp_path):
    from pypath.core.params import read_rpath_params, write_rpath_params

    params = _reference_params()
    params.stanzas.n_stanza_groups = 0
    write_rpath_params(params, "m", tmp_path)
    back = read_rpath_params(
        tmp_path / "m_model.csv",
        tmp_path / "m_diet.csv",
        stanza_group_file=tmp_path / "m_stanza_groups.csv",
        stanza_file=tmp_path / "m_stanzas.csv",
    )
    assert len(back.stanzas.stindiv) == 8
