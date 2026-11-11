"""Integration test: LT2022 model with scenario 16 should not crash Zooplankton."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = str(DATA_DIR / "LT2022_0.5ST_final7.eweaccdb")

# EwE 6.7.144's own sum of squares for scenario 16 against dataset 35
# (headless oracle run 42863343066, ecosim_ss.csv): sum over series of
# WtType x (ln(obs / B at the start of month 6) - ln q)^2, q estimated for
# relative series. Until 2026-11-11 this test pinned PyPath's own linear SS
# (sum of (annual mean B/B0 - obs)^2; 617.0 at the end, its history is in git),
# which was not EwE's statistic (audit C3). PyPath's run gives 75.211.
EWE_SS16 = 75.2231369
EWE_SS16_BY_GROUP = {
    "Turbot": 3.07261205,
    "Flounder": 5.24761868,
    "Cod": 1.68064594,
    "Smelt": 2.30133533,
    "Perciformes": 17.7172451,
    "Cypriniformes": 5.27536535,
    "Blue mussel (juv.)": 14.9818096,
    "Blue mussel (ad.)": 13.3068247,
    "Sand bivalves": 0.605426371,
    "Worms": 3.24376297,
    "Zooplankton": 5.25244427,
    "Phytoplankton": 2.53803778,
}
# The same with dataset 22 (oracle run 43079493003).
EWE_SS16_DS22 = 42447.1445
EWE_SS16_DS22_BY_GROUP = {
    "Turbot": 12.2528095,
    "Flounder": 27.5206299,
    "Cod": 18.2156181,
    "Herring": 6.97605467,
    "Smelt": 0.0107403072,
    "Perciformes": 19.7464447,
    "Cypriniformes": 5.63677883,
    "Round goby": 110.498817,
    "Blue mussel (juv.)": 42185.9844,
    "Sand bivalves": 39.2457161,
    "Worms": 8.15091324,
    "Zooplankton": 6.36616802,
    "Phytoplankton": 6.54104805,
}


@pytest.mark.integration
class TestLT2022Parity:
    @pytest.fixture
    def lt2022_scenario(self):
        if not Path(DB_PATH).exists():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

        return ecosim_scenario_from_ewemdb(DB_PATH, scenario=16, timeseries_dataset=35)

    def test_zooplankton_does_not_crash(self, lt2022_scenario):
        """Zooplankton (group 2, ecosim idx 3) should not collapse to zero."""
        from pypath.core.ecosim import rsim_run

        out = rsim_run(lt2022_scenario, method="RK4")

        zoo_final = out.annual_Biomass[-1, 3]
        zoo_initial = out.annual_Biomass[0, 3]

        assert zoo_final > zoo_initial * 0.01, (
            f"Zooplankton crashed: initial={zoo_initial:.4f}, final={zoo_final:.4f}"
        )

    def test_few_crashed_groups(self, lt2022_scenario):
        """At most 2 groups may crash; with M0 nudging disabled some
        low-biomass groups (e.g. indices 3, 5) can still hit epsilon."""
        from pypath.core.ecosim import rsim_run

        out = rsim_run(lt2022_scenario, method="RK4")

        assert len(out.crashed_groups) <= 2, (
            f"Too many groups crashed: {out.crashed_groups}"
        )

    def test_ewe_sum_of_squares_matches_the_oracle(self, lt2022_scenario):
        """EwE's SS of scenario 16 against dataset 35, in total and by group
        (the other groups' series carry weight 0)."""
        from pypath.core.calibration import ewe_sum_of_squares
        from pypath.core.ecosim import rsim_run

        out = rsim_run(lt2022_scenario, method="RK4")
        res = ewe_sum_of_squares(out, lt2022_scenario._timeseries)
        assert res.total == pytest.approx(EWE_SS16, rel=1e-3)
        names = [str(n).strip() for n in lt2022_scenario.params.spname]
        by_name = {names[g + 1]: v for g, v in res.by_group.items() if v > 0}
        assert set(by_name) == set(EWE_SS16_BY_GROUP)
        for group, ewe in EWE_SS16_BY_GROUP.items():
            assert by_name[group] == pytest.approx(ewe, rel=2e-3), group

    def test_ewe_sum_of_squares_with_catches_matches_the_oracle(self):
        """Dataset 22 ("fittingdata24A") forces seven groups' catches (type -6,
        weighted, compared as B x FishTime of the month before), adds a catch
        series, absolute biomass and a weight of 100 (oracle run 43079493003,
        EwE SS 42447.14). On EwE's own biomass and FishTime the function
        reproduces it to 1e-7; PyPath's run follows to 0.2 %."""
        if not Path(DB_PATH).exists():
            pytest.skip("LT2022 database not found")
        from pypath.core.calibration import ewe_sum_of_squares
        from pypath.core.ecosim import rsim_run
        from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

        sc = ecosim_scenario_from_ewemdb(DB_PATH, scenario=16, timeseries_dataset=22)
        res = ewe_sum_of_squares(rsim_run(sc, method="RK4"), sc._timeseries)
        assert res.total == pytest.approx(EWE_SS16_DS22, rel=5e-3)
        names = [str(n).strip() for n in sc.params.spname]
        by_name = {names[g + 1]: v for g, v in res.by_group.items() if v > 0}
        assert set(by_name) == set(EWE_SS16_DS22_BY_GROUP)
        for group, ewe in EWE_SS16_DS22_BY_GROUP.items():
            assert by_name[group] == pytest.approx(ewe, rel=5e-3), group


# EwE 6.7.144's own scenario-16 run (headless EwECore oracle, runs 42860867155
# and 42863343066, identical): biomass at the end of months 1 and 12. The birds
# carry biomass accumulation, so they test the BA path and the variable P/Q
# setting together. EwE computes in single precision.
EWE_ORACLE_BIRDS = {
    "Scoters": (0.038677871, 0.035548966),
    "Long-tailed duck": (0.005950296, 0.005438729),
}


@pytest.mark.integration
def test_birds_follow_the_ewe_oracle(lt2022_scenario_module):
    from pypath.core.ecosim import rsim_run

    out = rsim_run(lt2022_scenario_module, method="RK4")
    names = [str(n).strip() for n in lt2022_scenario_module.params.spname]
    for group, (month1, month12) in EWE_ORACLE_BIRDS.items():
        i = names.index(group)
        assert out.out_Biomass[1, i] == pytest.approx(month1, rel=1e-5), group
        assert out.out_Biomass[12, i] == pytest.approx(month12, rel=2e-3), group


@pytest.fixture(scope="module")
def lt2022_scenario_module():
    if not Path(DB_PATH).exists():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    return ecosim_scenario_from_ewemdb(DB_PATH, scenario=16, timeseries_dataset=35)


ORACLE_DIR = Path(__file__).parent / "data" / "ewe_oracle" / "lt2022"


@pytest.mark.integration
def test_mussels_follow_the_ewe_oracle(lt2022_scenario_module):
    """EwE's adult mussels step +0.19 % in month 1 (27.000 -> 27.050) and the
    juveniles hold: EwE's growth coefficient at the month before the plus
    group comes from the corrected plus-group weight, so the plus group's
    weight steps once (test_stanza_plus_group.py). With Rpath's age table the
    adults moved +0.004 % and the juveniles +0.04 %."""
    import pandas as pd

    from pypath.core.ecosim import rsim_run

    oracle = pd.read_csv(ORACLE_DIR / "ecosim_biomass_monthly.csv", index_col=0)
    oracle.columns = [c.strip() for c in oracle.columns]
    out = rsim_run(lt2022_scenario_module, method="RK4", years=range(1, 3))
    names = [str(n).strip() for n in lt2022_scenario_module.params.spname]
    for group in ("Blue mussel (juv.)", "Blue mussel (ad.)"):
        i = names.index(group)
        expected = float(oracle.loc[1, group])
        assert out.out_Biomass[1, i] == pytest.approx(expected, rel=2e-5), group
        # After the year-2 production step they still follow EwE (3e-4;
        # Rpath's age table left them 3-4 % off by then).
        for t in (15, 17, 19):
            expected = float(oracle.loc[t, group])
            assert out.out_Biomass[t, i] == pytest.approx(expected, rel=1e-3), (
                group,
                t,
            )


# EwE 6.7.144 oracle, month 13: the year-2 step of the "PTOT_Baltic" production
# forcing on Phytoplankton (x0.875) reaches the fast groups in the same month.
# EwE integrates with RK4 only (cEcoSimModel.vb:758).
EWE_ORACLE_MONTH13 = {"Zooplankton": 3.5051415, "Phytoplankton": 24.5586262}


@pytest.mark.integration
def test_fast_groups_follow_the_ewe_oracle_at_the_forcing_step(lt2022_scenario_module):
    from pypath.core.ecosim import rsim_run

    out = rsim_run(lt2022_scenario_module, method="RK4")
    names = [str(n).strip() for n in lt2022_scenario_module.params.spname]
    for group, month13 in EWE_ORACLE_MONTH13.items():
        i = names.index(group)
        assert out.out_Biomass[13, i] == pytest.approx(month13, rel=1e-3), group
