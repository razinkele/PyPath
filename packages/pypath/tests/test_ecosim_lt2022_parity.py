"""Integration test: LT2022 model with scenario 16 should not crash Zooplankton."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = str(DATA_DIR / "LT2022_0.5ST_final7.eweaccdb")

# Regression pin, NOT EwE's sum of squares (audit 2026-11-10, C3): this is a
# linear sum of (annual mean B/B0 - obs)^2 over dataset 35's biomass series,
# with no q, no weights and no catch series. EwE's SS is weighted and
# logarithmic, sum W*(ln(obs/B_month6) - q)^2, so the pin guards PyPath against
# regressions and says nothing about agreement with EwE.
# History: ~725 was quoted as an "old broken baseline";
# the limit was relaxed to 1100 on 2026-10-24 (message left at 800); SS was
# 1084.9 before the detritus/discard fate reading fix (#29) and 1101.2 after
# it. Every value up to there was computed with the scenario's vulnerabilities
# and FtimeAdjust on the wrong groups, the wrong forcing shape applied as prey
# availability, and a flat stanza (importer defects fixed 2026-11-07); with
# the settings EwE6 actually uses, SS was 881.2 on main's dynamics and is
# 642.0 with the stanza coupling, the closed detritus budget, EwE's fast-group
# rule (Zooplankton, Phytoplankton and Detritus at equilibrium, under RK4 as
# under AB) and donor control at v = 1, and 643.5 with dataset 35 applied as
# EwE applies a loaded dataset (Round goby's biomass forced from 0.03 to 7.8
# and back to 2.4; the dataset is also the fit's observations), 637.0
# with the diet import of Piscivorous birds read (EcopathGroup.ImpVar), and
# 631.7 with biomass accumulation and detritus import read (A1/A2: Scoters and
# Long-tailed duck BiomAcc, Detritus DtImports 1000; zeroing those three
# reproduces 637.0), and 618.9 with UseVarPQ off as EwE 6.7.144 runs it (the
# stored True is lost to a cEwEDatabase.ReadSafe bug; forcing it on gives 631.7).
# This is still PyPath's own linear SS. EwE's own scenario-16 run (headless
# oracle) gives EwE SS 75.22, a different metric; its trajectory is checked by
# test_birds_follow_the_ewe_oracle below.
# 624.6 under RK4, the only integrator EwE has (cEcoSimModel.vb:758), with
# EwE's fast-group biomeq (net production over predation, other mortality and
# fishing, cEcoSimModel.vb:3112). RK4 with Rpath's gross TotGain/TotLoss biomeq
# gives 622.9; the AB runs above were 618.9. 624.8 with the feeding-time
# update from the start-of-month consumption, as EwE's rk4 (end-of-month: 624.6).
# 615.6 with that update taking the derivative's consumption (prey Ftime
# included, as EwE's Derivt), and 617.6 with prey Ftime and vulnerability
# forcing saturating in the arena as in EwE (D1; linear prey term: 615.6).
# 618.2 with EwE's multi-stanza age table and Ford-Walford plus group (P2;
# the blue mussels follow the oracle to 0.2 % instead of 15-20 %). EwE's
# stanza feeding-time denominator leaves it there (no FtimeAdj on the
# mussels in scenario 16).
# 617.0 without Rpath's per-link handling term DD / (DD - 1 + preyYY), which
# EwE6 does not have (Cod, Turbot and Perciformes through the year-8 Round goby
# boom; test_link_handling.py).
# A change beyond 2 % needs a reason.
LT2022_SS16 = 617.0


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

    def test_ss_regression_pin(self, lt2022_scenario):
        """PyPath's own linear SS on dataset 35 stays at LT2022_SS16 (not EwE's SS)."""
        from pypath.core.ecopath import rpath
        from pypath.core.ecosim import rsim_run
        from pypath.io.ewemdb import read_ewemdb, read_ewemdb_table

        out = rsim_run(lt2022_scenario, method="RK4")

        params = read_ewemdb(DB_PATH)
        model = rpath(params)
        ts_all = read_ewemdb_table(DB_PATH, "EcosimTimeSeries")
        ds35 = ts_all[ts_all["DatasetID"] == 35].copy()
        tsg = read_ewemdb_table(DB_PATH, "EcosimTimeSeriesGroup")
        groups_db = read_ewemdb_table(DB_PATH, "EcopathGroup")
        gid_to_name = dict(zip(groups_db["GroupID"], groups_db["GroupName"]))

        ts_to_group = {}
        for _, r in tsg.iterrows():
            db_gname = gid_to_name.get(r["GroupID"], "").strip()
            for g in range(model.NUM_GROUPS):
                if str(model.Group[g]).strip() == db_gname:
                    ts_to_group[int(r["TimeSeriesID"])] = g
                    break

        total_ss = 0.0
        for _, row in ds35.iterrows():
            dtype = int(row["DatType"])
            tsid = int(row["TimeSeriesID"])
            g = ts_to_group.get(tsid)
            if dtype == 0 and g is not None:
                vals = np.array([float(x) for x in str(row["TimeValues"]).split()])
                sim = out.annual_Biomass[:, g + 1] / model.Biomass[g]
                n = min(len(vals), len(sim))
                total_ss += np.sum((sim[:n] - vals[:n]) ** 2)

        assert abs(total_ss - LT2022_SS16) <= 0.02 * LT2022_SS16, (
            f"SS={total_ss:.1f}, pinned at {LT2022_SS16} (+/- 2 %)"
        )


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
