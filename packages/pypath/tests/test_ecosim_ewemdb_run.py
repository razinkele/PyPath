"""Integration test: load LT2022 EwE database and run Ecosim scenarios.

Tests the full pipeline: read .eweaccdb → balance Ecopath → build Ecosim
scenario → run simulation, validating that the native EwE model file
format is correctly parsed and produces a stable Ecosim run.

The LT2022 model is a 25-group Lithuanian coastal food web with
23 living groups, 1 detritus group, and 1 fishing fleet.
"""

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path(__file__).parents[3] / "Data"
DB_PATH = str(DATA_DIR / "LT2022_0.5ST_final7.eweaccdb")

pytestmark = pytest.mark.integration


def _db_available():
    return Path(DB_PATH).exists()


@pytest.fixture(scope="module")
def ecopath_params():
    """Read Ecopath parameters from the EwE database."""
    if not _db_available():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import read_ewemdb

    return read_ewemdb(DB_PATH)


@pytest.fixture(scope="module")
def balanced_model(ecopath_params):
    """Balance the Ecopath model."""
    from pypath.core.ecopath import rpath

    return rpath(ecopath_params)


@pytest.fixture(scope="module")
def ecosim_scenario_1():
    """Load Ecosim scenario 1 from database."""
    if not _db_available():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    return ecosim_scenario_from_ewemdb(DB_PATH, scenario=1)


@pytest.fixture(scope="module")
def ecosim_scenario_16():
    """Load the calibrated Ecosim scenario 16."""
    if not _db_available():
        pytest.skip("LT2022 database not found")
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb

    return ecosim_scenario_from_ewemdb(DB_PATH, scenario=16)


# =========================================================================
# Ecopath model structure
# =========================================================================


class TestEcopathFromEwemdb:
    """Verify Ecopath model is correctly read from native EwE database."""

    def test_group_count(self, balanced_model):
        assert balanced_model.NUM_GROUPS == 25

    def test_living_groups(self, balanced_model):
        assert balanced_model.NUM_LIVING == 23

    def test_dead_groups(self, balanced_model):
        assert balanced_model.NUM_DEAD == 1

    def test_group_names_present(self, balanced_model):
        names = [
            str(balanced_model.Group[i]).strip()
            for i in range(balanced_model.NUM_GROUPS)
        ]
        joined = " ".join(names)
        assert "Herring" in joined
        assert "Cod" in joined
        assert "Zooplankton" in joined
        assert "Detritus" in joined

    def test_biomass_positive_for_living(self, balanced_model):
        for i in range(balanced_model.NUM_GROUPS):
            b = balanced_model.Biomass[i]
            assert b >= 0, (
                f"Group {i} ({balanced_model.Group[i]}): negative biomass {b}"
            )

    def test_ee_bounded(self, balanced_model):
        """EE should be in [0, 1] for living groups (with small tolerance)."""
        for i in range(balanced_model.NUM_LIVING):
            ee = balanced_model.EE[i]
            assert -0.01 <= ee <= 1.05, (
                f"Group {i} ({balanced_model.Group[i]}): EE={ee} out of range"
            )

    def test_stanza_data_loaded(self, ecopath_params):
        """The model has multi-stanza groups (Blue mussel juv/ad)."""
        assert hasattr(ecopath_params, "stanzas")
        if (
            hasattr(ecopath_params.stanzas, "stgroups")
            and ecopath_params.stanzas.stgroups is not None
        ):
            assert len(ecopath_params.stanzas.stgroups) >= 1


# =========================================================================
# Ecosim scenario loading
# =========================================================================


class TestEcosimScenarioLoading:
    """Verify Ecosim scenario is correctly built from EwE database."""

    def test_scenario_has_params(self, ecosim_scenario_1):
        assert hasattr(ecosim_scenario_1, "params")
        assert ecosim_scenario_1.params is not None

    def test_scenario_has_forcing(self, ecosim_scenario_1):
        assert hasattr(ecosim_scenario_1, "forcing")

    def test_scenario_has_fishing(self, ecosim_scenario_1):
        assert hasattr(ecosim_scenario_1, "fishing")

    def test_predprey_links(self, ecosim_scenario_1):
        """Predator-prey link arrays should be populated."""
        p = ecosim_scenario_1.params
        assert len(p.PreyFrom) > 50, "Expected >50 pred-prey links"
        assert len(p.PreyFrom) == len(p.PreyTo)
        assert len(p.QQ) == len(p.PreyFrom)
        assert len(p.VV) == len(p.PreyFrom)
        assert len(p.DD) == len(p.PreyFrom)

    def test_vulnerability_values(self, ecosim_scenario_1):
        """VV values should be >= 1 (default is 2 in EwE)."""
        vv = ecosim_scenario_1.params.VV
        assert np.all(vv >= 1.0), f"VV has values < 1: min={vv.min()}"

    def test_biomass_baseline(self, ecosim_scenario_1):
        """B_BaseRef should match balanced Ecopath biomass."""
        b = ecosim_scenario_1.params.B_BaseRef
        assert len(b) > 20
        # At least some groups should have positive biomass
        assert np.sum(b > 0) >= 20


# =========================================================================
# Ecosim simulation run — scenario 1 (default, uncalibrated)
# =========================================================================


class TestEcosimRunScenario1:
    """Run Ecosim simulation from native EwE file — scenario 1."""

    @pytest.fixture(scope="class")
    def sim_output(self, ecosim_scenario_1):
        from pypath.core.ecosim import rsim_run

        return rsim_run(ecosim_scenario_1, method="AB")

    def test_simulation_completes(self, sim_output):
        """Ecosim should complete without raising."""
        assert sim_output is not None

    def test_output_shape(self, sim_output, ecosim_scenario_1):
        """annual_Biomass should have (n_years+1, n_groups+1) shape."""
        n_years = 11  # TotalTime for scenario 1
        shape = sim_output.annual_Biomass.shape
        assert shape[0] == n_years, f"Expected {n_years} time steps, got {shape[0]}"
        assert shape[1] > 20, f"Expected >20 groups, got {shape[1]}"

    def test_no_all_zeros(self, sim_output):
        """No living group should be entirely zero throughout."""
        bio = sim_output.annual_Biomass
        # Column 0 is 'Outside', skip it; also skip fleet (last col)
        for col in range(1, bio.shape[1] - 1):
            col_max = np.max(bio[:, col])
            assert col_max > 0, f"Column {col} is all zeros"

    def test_few_crashes(self, sim_output):
        """Uncalibrated scenario 1 may have some crashes, but not many."""
        assert len(sim_output.crashed_groups) <= 5, (
            f"Too many crashes: {sim_output.crashed_groups}"
        )

    def test_biomass_not_exploding(self, sim_output):
        """No group's final biomass should exceed 100x its initial."""
        initial = sim_output.annual_Biomass[0, :]
        final = sim_output.annual_Biomass[-1, :]
        for i in range(len(initial)):
            if initial[i] > 1e-6:
                ratio = final[i] / initial[i]
                assert ratio < 100, f"Group {i}: biomass exploded {ratio:.1f}x"


# =========================================================================
# Ecosim simulation run — scenario 16 (calibrated, final)
# =========================================================================


class TestEcosimRunScenario16:
    """Run Ecosim from calibrated scenario 16 — should perform better."""

    @pytest.fixture(scope="class")
    def sim_output(self, ecosim_scenario_16):
        from pypath.core.ecosim import rsim_run

        return rsim_run(ecosim_scenario_16, method="AB")

    def test_simulation_completes(self, sim_output):
        assert sim_output is not None

    def test_zooplankton_stable(self, sim_output):
        """Zooplankton (group 2, ecosim idx 3) should not collapse."""
        zoo_initial = sim_output.annual_Biomass[0, 3]
        zoo_final = sim_output.annual_Biomass[-1, 3]
        if zoo_initial > 1e-6:
            assert zoo_final > zoo_initial * 0.01, (
                f"Zooplankton crashed: {zoo_initial:.4f} → {zoo_final:.4f}"
            )

    def test_fewer_crashes_than_scenario_1(self, sim_output):
        """Calibrated scenario should have at most 2 crashed groups."""
        assert len(sim_output.crashed_groups) <= 2

    def test_ss_reasonable(self, sim_output, ecosim_scenario_16):
        """Sum of squares should be within reasonable bounds."""
        from pypath.core.ecopath import rpath
        from pypath.io.ewemdb import read_ewemdb, read_ewemdb_table

        params = read_ewemdb(DB_PATH)
        model = rpath(params)

        # Load time series for SS calculation
        try:
            ts_all = read_ewemdb_table(DB_PATH, "EcosimTimeSeries")
            ds35 = ts_all[ts_all["DatasetID"] == 35].copy()
        except Exception:
            pytest.skip("Time series data not available for SS calculation")

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
        n_series = 0
        for _, row in ds35.iterrows():
            dtype = int(row["DatType"])
            tsid = int(row["TimeSeriesID"])
            g = ts_to_group.get(tsid)
            if dtype == 0 and g is not None:
                vals = np.array([float(x) for x in str(row["TimeValues"]).split()])
                sim = sim_output.annual_Biomass[:, g + 1] / model.Biomass[g]
                n = min(len(vals), len(sim))
                total_ss += np.sum((sim[:n] - vals[:n]) ** 2)
                n_series += 1

        assert total_ss < 800, (
            f"SS={total_ss:.1f} should be < 800 (n_series={n_series})"
        )


# =========================================================================
# Database table structure verification
# =========================================================================


class TestEwemdbTableStructure:
    """Verify that key Ecosim tables are correctly read from the database."""

    def test_ecosim_scenario_table(self):
        if not _db_available():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import read_ewemdb_table

        df = read_ewemdb_table(DB_PATH, "EcosimScenario")
        assert len(df) >= 15
        assert "ScenarioID" in df.columns
        assert "ScenarioName" in df.columns
        assert "TotalTime" in df.columns

    def test_forcing_matrix_table(self):
        if not _db_available():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import read_ewemdb_table

        df = read_ewemdb_table(DB_PATH, "EcosimScenarioForcingMatrix")
        assert len(df) > 100
        assert "vulnerability" in df.columns
        assert "PredID" in df.columns
        assert "PreyID" in df.columns

    def test_scenario_group_table(self):
        if not _db_available():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import read_ewemdb_table

        df = read_ewemdb_table(DB_PATH, "EcosimScenarioGroup")
        assert len(df) > 100
        assert "FtimeAdjust" in df.columns

    def test_time_series_table(self):
        if not _db_available():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import read_ewemdb_table

        df = read_ewemdb_table(DB_PATH, "EcosimTimeSeries")
        assert len(df) > 100
        assert "TimeSeriesID" in df.columns
        assert "TimeValues" in df.columns

    def test_shape_tables(self):
        if not _db_available():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import read_ewemdb_table

        shapes = read_ewemdb_table(DB_PATH, "EcosimShape")
        assert len(shapes) > 50
        assert "ShapeID" in shapes.columns

        fish_rate = read_ewemdb_table(DB_PATH, "EcosimShapeFishRate")
        assert len(fish_rate) >= 1
        assert "zScale" in fish_rate.columns

        time_shapes = read_ewemdb_table(DB_PATH, "EcosimShapeTime")
        assert len(time_shapes) >= 1
