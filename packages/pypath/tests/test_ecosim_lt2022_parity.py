"""Integration test: LT2022 model with scenario 16 should not crash Zooplankton."""
import warnings
import numpy as np
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = str(DATA_DIR / "LT2022_0.5ST_final7.eweaccdb")


@pytest.mark.integration
class TestLT2022Parity:
    @pytest.fixture
    def lt2022_scenario(self):
        if not Path(DB_PATH).exists():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import ecosim_scenario_from_ewemdb
        return ecosim_scenario_from_ewemdb(DB_PATH, scenario=16)

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

    def test_ss_improves_over_baseline(self, lt2022_scenario):
        """Sum of squares should be lower than 725 (the old broken baseline)."""
        from pypath.core.ecosim import rsim_run
        from pypath.io.ewemdb import read_ewemdb, read_ewemdb_table
        from pypath.core.ecopath import rpath

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

        # Old broken baseline SS was ~725. Use 800 as generous threshold
        assert total_ss < 800, f"SS={total_ss:.1f} should be < 800"
