"""Integration test: LT2022 model with scenario 16 should not crash Zooplankton."""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = str(DATA_DIR / "LT2022_0.5ST_final7.eweaccdb")

# Regression pin, NOT EwE's sum of squares (audit 2026-11-10, C3): this is a
# linear sum of (annual mean B/B0 - obs)^2 over dataset 35's biomass series,
# with no q, no weights and no catch series. EwE's SS is weighted and
# logarithmic, sum W*(ln(obs/B_month6) - q)^2, so 637.0 guards PyPath against
# regressions and says nothing about agreement with EwE.
# No EwE 6 reference SS for this calibrated
# scenario is available. History: ~725 was quoted as an "old broken baseline";
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
# reproduces 637.0). 631.7 is PyPath's own output, not checked against EwE.
# The real reference is EwE6's own scenario-16 trajectory.
# A change beyond 2 % needs a reason.
LT2022_SS16 = 631.7


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

        out = rsim_run(lt2022_scenario, method="AB")

        zoo_final = out.annual_Biomass[-1, 3]
        zoo_initial = out.annual_Biomass[0, 3]

        assert zoo_final > zoo_initial * 0.01, (
            f"Zooplankton crashed: initial={zoo_initial:.4f}, final={zoo_final:.4f}"
        )

    def test_few_crashed_groups(self, lt2022_scenario):
        """At most 2 groups may crash; with M0 nudging disabled some
        low-biomass groups (e.g. indices 3, 5) can still hit epsilon."""
        from pypath.core.ecosim import rsim_run

        out = rsim_run(lt2022_scenario, method="AB")

        assert len(out.crashed_groups) <= 2, (
            f"Too many groups crashed: {out.crashed_groups}"
        )

    def test_ss_regression_pin(self, lt2022_scenario):
        """PyPath's own linear SS on dataset 35 stays at LT2022_SS16 (not EwE's SS)."""
        from pypath.core.ecopath import rpath
        from pypath.core.ecosim import rsim_run
        from pypath.io.ewemdb import read_ewemdb, read_ewemdb_table

        out = rsim_run(lt2022_scenario, method="AB")

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
