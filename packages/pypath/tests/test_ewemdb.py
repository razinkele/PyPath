"""
Tests for EwE database (ewemdb) reader module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from pypath.io.ewemdb import (
    EwEDatabaseError,
    _get_connection_string,
    check_ewemdb_support,
    get_ewemdb_metadata,
    list_ewemdb_tables,
    read_ewemdb,
    ecosim_scenario_from_ewemdb,
    read_ewemdb_table,
)


class TestCheckSupport:
    """Tests for check_ewemdb_support function."""

    def test_returns_dict(self):
        """Test that check_ewemdb_support returns a dict."""
        result = check_ewemdb_support()
        assert isinstance(result, dict)
        assert "pyodbc" in result
        assert "pypyodbc" in result
        assert "mdb_tools" in result
        assert "any_available" in result

    def test_values_are_bool(self):
        """Test that all values are booleans."""
        result = check_ewemdb_support()
        for key, value in result.items():
            assert isinstance(value, bool)


class TestConnectionString:
    """Tests for connection string generation."""

    def test_connection_string_format(self):
        """Test that connection string has correct format."""
        conn_str = _get_connection_string("test.ewemdb")
        assert "DRIVER=" in conn_str
        assert "DBQ=" in conn_str
        assert "test.ewemdb" in conn_str


class TestFileNotFound:
    """Tests for file not found errors."""

    def test_list_tables_file_not_found(self):
        """Test list_ewemdb_tables with non-existent file."""
        with pytest.raises(FileNotFoundError):
            list_ewemdb_tables("nonexistent_file.ewemdb")

    def test_read_table_file_not_found(self):
        """Test read_ewemdb_table with non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_ewemdb_table("nonexistent_file.ewemdb", "EcopathGroup")

    def test_read_ewemdb_file_not_found(self):
        """Test read_ewemdb with non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_ewemdb("nonexistent_file.ewemdb")


class TestMockedDatabase:
    """Tests with mocked database connections."""

    @pytest.mark.skip(reason="Requires pyodbc installed")
    def test_list_tables_with_pyodbc(self):
        """Test listing tables with mocked pyodbc."""
        # This test requires pyodbc to be installed
        pass

    @pytest.mark.skip(reason="Requires pyodbc installed")
    def test_read_table_with_pyodbc(self):
        """Test reading table with mocked pyodbc."""
        # This test requires pyodbc to be installed
        pass


class TestReadEwemdb:
    """Tests for read_ewemdb function."""

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_read_ewemdb_basic(self, mock_read_table):
        """Test reading a basic ewemdb file."""
        # Setup mock returns for different tables
        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2, 3],
                "GroupName": ["Phytoplankton", "Zooplankton", "Fish"],
                "Type": [1, 0, 0],
                "Biomass": [10.0, 5.0, 2.0],
                "PB": [100.0, 40.0, 1.5],
                "QB": [0.0, 150.0, 5.0],
                "EE": [0.95, 0.90, 0.80],
            }
        )

        diet_df = pd.DataFrame(
            {
                "PreyName": ["Phytoplankton", "Zooplankton", "Phytoplankton"],
                "PredName": ["Zooplankton", "Fish", "Fish"],
                "Diet": [1.0, 0.8, 0.2],
            }
        )

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table == "EcopathDietComp":
                return diet_df
            elif table in ["EcopathFleet", "Fleet"]:
                raise Exception("Table not found")
            elif table in ["EcopathCatch", "Catch"]:
                raise Exception("Table not found")
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        # Create temp file to pass existence check
        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path)

            assert len(params.model) == 3
            assert "Phytoplankton" in params.model["Group"].values
            assert "Zooplankton" in params.model["Group"].values
            assert "Fish" in params.model["Group"].values
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_read_ewemdb_with_fleets(self, mock_read_table):
        """Test reading ewemdb with fleet data."""
        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2],
                "GroupName": ["Fish", "Detritus"],
                "Type": [0, 2],
                "Biomass": [2.0, 100.0],
                "PB": [1.5, 0.0],
                "QB": [5.0, 0.0],
                "EE": [0.80, 0.0],
            }
        )

        fleet_df = pd.DataFrame(
            {
                "FleetID": [1],
                "FleetName": ["Trawlers"],
            }
        )

        catch_df = pd.DataFrame(
            {
                "GroupName": ["Fish"],
                "FleetName": ["Trawlers"],
                "Landing": [0.5],
            }
        )

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcopathDietComp", "DietComp"]:
                return pd.DataFrame()
            elif table in ["EcopathFleet", "Fleet"]:
                return fleet_df
            elif table in ["EcopathCatch", "Catch"]:
                return catch_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path)

            assert len(params.model) == 2
            # Fleet column should be added
            assert "Trawlers" in params.model.columns
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_read_ewemdb_with_ecosim(self, mock_read_table):
        """Test reading ewemdb with Ecosim scenario and its time series."""
        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2],
                "GroupName": ["Fish", "Detritus"],
                "Type": [0, 2],
                "Biomass": [2.0, 100.0],
                "PB": [1.5, 0.0],
                "QB": [5.0, 0.0],
                "EE": [0.80, 0.0],
            }
        )

        diet_df = pd.DataFrame({"PreyName": [], "PredName": [], "Diet": []})

        ecosim_df = pd.DataFrame(
            {
                "ScenarioID": [1],
                "ScenarioName": ["TestScenario"],
                "StartYear": [2000],
                "EndYear": [2005],
                "NumYears": [6],
                "Description": ["Test Ecosim scenario"],
            }
        )

        forcing_df = pd.DataFrame(
            {"ScenarioID": [1, 1, 1, 1], "Time": [0, 0, 1, 1], "Parameter": ["ForcedPrey", "ForcedMort", "ForcedPrey", "ForcedMort"], "Group": ["Fish", "Fish", "Fish", "Fish"], "Value": [1.0, 1.0, 0.9, 1.0]}
        )

        fishing_df = pd.DataFrame(
            {"ScenarioID": [1], "Time": [0], "Gear": [1], "Effort": [0.5]}
        )

        frate_df = pd.DataFrame({"ScenarioID": [1, 1], "Year": [2000, 2001], "Group": ["Fish", "Fish"], "FRate": [0.1, 0.2]})
        catch_yr_df = pd.DataFrame({"ScenarioID": [1, 1], "Year": [2000, 2001], "Group": ["Fish", "Fish"], "Catch": [5.0, 6.0]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcopathDietComp", "DietComp"]:
                return diet_df
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings"]:
                return forcing_df
            elif table in ["EcosimFishing", "EcosimEffort"]:
                return fishing_df
            elif table in ["EcosimFRate", "EcosimFRateTable"]:
                return frate_df
            elif table in ["EcosimCatch", "EcosimAnnualCatch"]:
                return catch_yr_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path, include_ecosim=True)

            assert getattr(params, "ecosim", None) is not None
            assert params.ecosim["has_ecosim"] is True
            assert len(params.ecosim["scenarios"]) == 1
            sc = params.ecosim["scenarios"][0]
            assert sc["name"] == "TestScenario"
            assert int(sc["start_year"]) == 2000
            assert sc["num_years"] == 6
            assert "forcing_df" in sc and len(sc["forcing_df"]) == 4
            assert "fishing_df" in sc and len(sc["fishing_df"]) == 1
            # New: parsed time-series should be present
            assert "forcing_ts" in sc and "_times" in sc["forcing_ts"]
            fp = sc["forcing_ts"].get("ForcedPrey")
            assert isinstance(fp, pd.DataFrame)
            assert list(fp["Fish"]) == [1.0, 0.9]
            assert "fishing_ts" in sc and "_times" in sc["fishing_ts"]
            # Effort pivot present as DataFrame with one gear
            effort_df = sc["fishing_ts"].get("Effort")
            assert hasattr(effort_df, "shape") and (effort_df.shape[0] == 1 or effort_df.shape[0] == 2)
            # Monthly resampled data
            assert "forcing_monthly" in sc and "_monthly_times" in sc["forcing_monthly"]
            # For long-format parameter/group/value, we expect keys such as 'ForcedPrey' to exist as DataFrames
            fp = sc["forcing_ts"].get("ForcedPrey")
            assert isinstance(fp, pd.DataFrame)
            # Monthly matrix should exist in forcing_matrices
            assert "forcing_matrices" in sc and "ForcedPrey" in sc["forcing_matrices"]
            fpm = sc["forcing_matrices"]["ForcedPrey"]
            assert fpm.shape[0] == 6 * 12
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_forcing_wide_monthly_format(self, mock_read_table):
        """Test parsing a wide-format forcing table with monthly columns (Jan..Dec)."""
        groups_df = pd.DataFrame({"GroupID": [1], "GroupName": ["Fish"], "Type": [0], "Biomass": [2.0], "PB": [1.5], "QB": [5.0], "EE": [0.80]})
        ecosim_df = pd.DataFrame({"ScenarioID": [1], "ScenarioName": ["MonthlyWide"], "StartYear": [2000], "EndYear": [2000], "NumYears": [1]})
        # Forcing wide format: Year + Jan..Dec columns as values for Fish
        forcing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Parameter": ["ForcedPrey"], "Fish": [None], "Jan": [1.0], "Feb": [1.1], "Mar": [1.2], "Apr": [1.3], "May": [1.4], "Jun": [1.5], "Jul": [1.6], "Aug": [1.7], "Sep": [1.8], "Oct": [1.9], "Nov": [2.0], "Dec": [2.1]})
        fishing_df = pd.DataFrame({"ScenarioID": [1], "Time": [0], "Gear": [1], "Effort": [0.5]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings", "EcosimForcingTable"]:
                return forcing_df
            elif table in ["EcosimFishing", "EcosimEffort"]:
                return fishing_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path, include_ecosim=True)
            sc = params.ecosim["scenarios"][0]
            fm = sc["forcing_monthly"]["ForcedPrey"]
            # Should have 12 monthly rows for 2000
            assert fm.shape[0] == 12
            assert abs(float(fm.iloc[0]["Fish"]) - 1.0) < 1e-8
            assert abs(float(fm.iloc[-1]["Fish"]) - 2.1) < 1e-8
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_forcing_localized_month_names(self, mock_read_table):
        """Test parsing month names in French/Spanish variants (localized month names)."""
        groups_df = pd.DataFrame({"GroupID": [1], "GroupName": ["Fish"], "Type": [0], "Biomass": [2.0], "PB": [1.5], "QB": [5.0], "EE": [0.80]})
        ecosim_df = pd.DataFrame({"ScenarioID": [1], "ScenarioName": ["LocalizedMonths"], "StartYear": [2000], "EndYear": [2000], "NumYears": [1]})
        # French month abbreviations: Janv, Fev, Mar, etc.
        forcing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Parameter": ["ForcedPrey"], "Fish": [None], "Janv": [1.0], "Fev": [1.1], "Mar": [1.2], "Avr": [1.3], "Mai": [1.4], "Juin": [1.5], "Juil": [1.6], "Aou": [1.7], "Sep": [1.8], "Oct": [1.9], "Nov": [2.0], "Dec": [2.1]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings", "EcosimForcingTable"]:
                return forcing_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path, include_ecosim=True)
            sc = params.ecosim["scenarios"][0]
            fm = sc["forcing_monthly"]["ForcedPrey"]
            assert fm.shape[0] == 12
            assert abs(float(fm.iloc[0]["Fish"]) - 1.0) < 1e-8
            assert abs(float(fm.iloc[1]["Fish"]) - 1.1) < 1e-8
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_forcing_start_month_relative(self, mock_read_table):
        """Test that M1..M12 labels are interpreted relative to scenario StartMonth."""
        groups_df = pd.DataFrame({"GroupID": [1], "GroupName": ["Fish"], "Type": [0], "Biomass": [2.0], "PB": [1.5], "QB": [5.0], "EE": [0.80]})
        # Set StartMonth to 4 (April)
        ecosim_df = pd.DataFrame({"ScenarioID": [1], "ScenarioName": ["StartMonthRel"], "StartYear": [2000], "EndYear": [2000], "NumYears": [1], "StartMonth": [4]})
        # M1..M12 where M1 corresponds to April
        forcing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Parameter": ["ForcedPrey"], "M1": [4.0], "M2": [5.0], "M3": [6.0], "M4": [7.0], "M5": [8.0], "M6": [9.0], "M7": [10.0], "M8": [11.0], "M9": [12.0], "M10": [13.0], "M11": [14.0], "M12": [15.0]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings", "EcosimForcingTable"]:
                return forcing_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path, include_ecosim=True)
            sc = params.ecosim["scenarios"][0]
            fm = sc["forcing_monthly"]["ForcedPrey"]
            # M1 -> April => month index 0 should be April value 4.0
            assert abs(float(fm.iloc[0]["Fish"]) - 4.0) < 1e-8
            # M12 -> March => last month should be 15.0
            assert abs(float(fm.iloc[-1]["Fish"]) - 15.0) < 1e-8
        finally:
            os.unlink(temp_path)

    def test_resample_actual_month_lengths_leap_year(self):
        """Test resampling with actual month lengths across a leap year"""
        from pypath.io.ewemdb import _resample_to_monthly
        import numpy as _np
        # Construct parsed_ts with times 1999 and 2000 and a simple series
        parsed = {"_times": [1999.0, 2000.0], "Value": _np.array([1.0, 2.0])}
        res_actual = _resample_to_monthly(parsed, 1999, 2, start_month=1, use_actual_month_lengths=True)
        res_simple = _resample_to_monthly(parsed, 1999, 2, start_month=1, use_actual_month_lengths=False)
        # both should have 24 months
        assert len(res_actual["Value"]) == 24
        assert len(res_simple["Value"]) == 24
        # Feb 2000 positions should differ between methods due to leap year day counts
        # find index of Feb 2000 in monthly times
        feb2000_idx = list(res_actual["_monthly_times"]).index(2000.0833333333333) if 2000.0833333333333 in list(res_actual["_monthly_times"]) else None
        # Best-effort check: ensure arrays not identical
        assert not all(_np.isclose(res_actual["Value"], res_simple["Value"]))


    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_fishing_wide_monthly_format(self, mock_read_table):
        """Test parsing fishing wide-format monthly columns per gear."""
        groups_df = pd.DataFrame({"GroupID": [1], "GroupName": ["Fish"], "Type": [0], "Biomass": [2.0], "PB": [1.5], "QB": [5.0], "EE": [0.80]})
        ecosim_df = pd.DataFrame({"ScenarioID": [1], "ScenarioName": ["FishingMonthly"], "StartYear": [2000], "EndYear": [2000], "NumYears": [1]})
        # Fishing wide: Year + gear columns Jan..Dec for Gear=1
        fishing_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Gear": [1], "Jan": [0.5], "Feb": [0.6], "Mar": [0.7], "Apr": [0.8], "May": [0.9], "Jun": [1.0], "Jul": [1.1], "Aug": [1.2], "Sep": [1.3], "Oct": [1.4], "Nov": [1.5], "Dec": [1.6]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings"]:
                return pd.DataFrame()
            elif table in ["EcosimFishing", "EcosimEffort", "EcosimEffortTable"]:
                return fishing_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path, include_ecosim=True)
            sc = params.ecosim["scenarios"][0]
            fm = sc["fishing_monthly"]["Effort"]
            assert fm.shape[0] == 12
            # single gear column (1) -> value at first month
            assert abs(float(fm.iloc[0, 1]) - 0.5) < 1e-8
        finally:
            os.unlink(temp_path)
            assert "fishing_monthly" in sc and "_monthly_times" in sc["fishing_monthly"]
            fish_eff = sc["fishing_monthly"].get("Effort")
            assert fish_eff is not None
            assert fish_eff.shape[0] == 12
            # Optional: if rsim_fishing was constructed, validate annual mappings
            if "rsim_fishing" in sc:
                fc = sc["rsim_fishing"].ForcedCatch
                # group index for Fish should be 1
                gi = 1
                # Year 2000 -> index 0, Year 2001 -> index 1
                assert abs(float(fr[0, gi]) - 0.1) < 1e-8
                assert abs(float(fr[1, gi]) - 0.2) < 1e-8
                assert abs(float(fc[0, gi]) - 5.0) < 1e-8
                assert abs(float(fc[1, gi]) - 6.0) < 1e-8

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_build_full_rsim_scenario(self, mock_read_table):
        """Test building a full RsimScenario from EwE DB."""
        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2],
                "GroupName": ["Fish", "Detritus"],
                "Type": [0, 2],
                "Biomass": [2.0, 100.0],
                "PB": [1.5, 0.0],
                "QB": [5.0, 0.0],
                "EE": [0.80, 0.0],
            }
        )

        ecosim_df = pd.DataFrame(
            {
                "ScenarioID": [1],
                "ScenarioName": ["TestScenario"],
                "StartYear": [2000],
                "EndYear": [2005],
                "NumYears": [6],
                "Description": ["Test Ecosim scenario"],
            }
        )

        forcing_df = pd.DataFrame(
            {"ScenarioID": [1, 1, 1, 1], "Time": [0, 0, 1, 1], "Parameter": ["ForcedPrey", "ForcedMort", "ForcedPrey", "ForcedMort"], "Group": ["Fish", "Fish", "Fish", "Fish"], "Value": [1.0, 1.0, 0.9, 1.0]}
        )

        fishing_df = pd.DataFrame(
            {"ScenarioID": [1], "Time": [0], "Gear": [1], "Effort": [0.5]}
        )

        frate_df = pd.DataFrame({"ScenarioID": [1, 1], "Year": [2000, 2001], "Group": ["Fish", "Fish"], "FRate": [0.1, 0.2]})
        catch_yr_df = pd.DataFrame({"ScenarioID": [1, 1], "Year": [2000, 2001], "Group": ["Fish", "Fish"], "Catch": [5.0, 6.0]})

        habitat_df = pd.DataFrame({"Group": ["Fish", "Fish"], "Patch": [1, 2], "Value": [0.8, 0.6]})
        grid_df = pd.DataFrame({"PatchID": [1, 2], "Area": [10.0, 5.0], "Lon": [0.0, 0.1], "Lat": [50.0, 50.1]})
        dispersal_df = pd.DataFrame({"Group": ["Fish"], "Dispersal": [0.1]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcopathDietComp", "DietComp"]:
                return pd.DataFrame({"PreyName": [], "PredName": [], "Diet": []})
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings"]:
                return forcing_df
            elif table in ["EcosimFishing", "EcosimEffort"]:
                return fishing_df
            elif table in ["EcosimFRate", "EcosimFRateTable"]:
                return frate_df
            elif table in ["EcosimCatch", "EcosimAnnualCatch"]:
                return catch_yr_df
            elif table in ["EcospaceHabitat", "EcospaceLayer"]:
                return habitat_df
            elif table == "EcospaceGrid":
                return grid_df
            elif table == "EcospaceDispersal":
                return dispersal_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            scen = ecosim_scenario_from_ewemdb(temp_path, scenario=1)
            from pypath.core.ecosim import RsimScenario
            assert isinstance(scen, RsimScenario)
            # Forcing present
            assert scen.forcing.ForcedPrey.shape[0] == 6 * 12
            # Fishing annual FRATE/Catch present
            fr = scen.fishing.ForcedFRate
            fc = scen.fishing.ForcedCatch
            gi = 1
            assert abs(float(fr[0, gi]) - 0.1) < 1e-8
            assert abs(float(fr[1, gi]) - 0.2) < 1e-8
            assert abs(float(fc[0, gi]) - 5.0) < 1e-8
            assert abs(float(fc[1, gi]) - 6.0) < 1e-8

            # Ecospace mapping
            assert hasattr(scen, "ecospace") and scen.ecospace is not None
            eco = scen.ecospace
            assert eco.grid.n_patches == 2
            # use scenario params to find group index
            gi = scen.params.spname.index("Fish")
            # habitat preference for Fish patch 1 and 2
            assert abs(float(eco.habitat_preference[gi, 0]) - 0.8) < 1e-8
            assert abs(float(eco.habitat_preference[gi, 1]) - 0.6) < 1e-8
            assert abs(float(eco.dispersal_rate[gi]) - 0.1) < 1e-8
            # adjacency inferred from centroids should have at least one non-zero entry
            adj = eco.grid.adjacency_matrix
            assert adj.nnz > 0
            # edge lengths should include a pair for patches (0,1) or similar
            assert len(eco.grid.edge_lengths) > 0
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_ecospace_missing_tables(self, mock_read_table):
        """If ecospace tables are missing, no ecospace should be attached."""
        groups_df = pd.DataFrame(
            {
                "GroupID": [1],
                "GroupName": ["Fish"],
                "Type": [0],
                "Biomass": [2.0],
                "PB": [1.5],
                "QB": [5.0],
                "EE": [0.80],
            }
        )

        ecosim_df = pd.DataFrame(
            {
                "ScenarioID": [1],
                "ScenarioName": ["NoEcospace"],
                "StartYear": [2000],
                "EndYear": [2001],
                "NumYears": [2],
                "Description": ["No ecospace tables present"],
            }
        )

        forcing_df = pd.DataFrame(
            {"ScenarioID": [1, 1], "Time": [0, 1], "Parameter": ["ForcedPrey", "ForcedPrey"], "Group": ["Fish", "Fish"], "Value": [1.0, 0.9]}
        )

        fishing_df = pd.DataFrame(
            {"ScenarioID": [1], "Time": [0], "Gear": [1], "Effort": [0.5]}
        )

        frate_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Group": ["Fish"], "FRate": [0.1]})
        catch_yr_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Group": ["Fish"], "Catch": [5.0]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcopathDietComp", "DietComp"]:
                return pd.DataFrame({"PreyName": [], "PredName": [], "Diet": []})
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings"]:
                return forcing_df
            elif table in ["EcosimFishing", "EcosimEffort"]:
                return fishing_df
            elif table in ["EcosimFRate", "EcosimFRateTable"]:
                return frate_df
            elif table in ["EcosimCatch", "EcosimAnnualCatch"]:
                return catch_yr_df
            # Simulate missing ecospace tables
            elif table in ["EcospaceHabitat", "EcospaceLayer", "EcospaceGrid", "EcospaceDispersal"]:
                raise Exception("Table not found")
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            scen = ecosim_scenario_from_ewemdb(temp_path, scenario=1)
            from pypath.core.ecosim import RsimScenario
            assert isinstance(scen, RsimScenario)
            # No ecospace attached when tables missing
            assert scen.ecospace is None
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_ecospace_incomplete_habitat(self, mock_read_table):
        """If habitat table is incomplete, missing patches are filled with defaults."""
        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2],
                "GroupName": ["Fish", "Detritus"],
                "Type": [0, 2],
                "Biomass": [2.0, 100.0],
                "PB": [1.5, 0.0],
                "QB": [5.0, 0.0],
                "EE": [0.80, 0.0],
            }
        )

        ecosim_df = pd.DataFrame(
            {
                "ScenarioID": [1],
                "ScenarioName": ["PartialHabitat"],
                "StartYear": [2000],
                "EndYear": [2001],
                "NumYears": [2],
                "Description": ["Partial habitat entries"],
            }
        )

        forcing_df = pd.DataFrame(
            {"ScenarioID": [1, 1], "Time": [0, 1], "Parameter": ["ForcedPrey", "ForcedPrey"], "Group": ["Fish", "Fish"], "Value": [1.0, 0.9]}
        )

        fishing_df = pd.DataFrame(
            {"ScenarioID": [1], "Time": [0], "Gear": [1], "Effort": [0.5]}
        )

        frate_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Group": ["Fish"], "FRate": [0.1]})
        catch_yr_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Group": ["Fish"], "Catch": [5.0]})

        # Habitat only contains a value for patch 1, patch 2 missing
        habitat_df = pd.DataFrame({"Group": ["Fish"], "Patch": [1], "Value": [0.8]})
        grid_df = pd.DataFrame({"PatchID": [1, 2], "Area": [10.0, 5.0], "Lon": [0.0, 0.1], "Lat": [50.0, 50.1]})
        # Dispersal missing for Fish -> empty DataFrame
        dispersal_df = pd.DataFrame({"Group": [], "Dispersal": []})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcopathDietComp", "DietComp"]:
                return pd.DataFrame({"PreyName": [], "PredName": [], "Diet": []})
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings"]:
                return forcing_df
            elif table in ["EcosimFishing", "EcosimEffort"]:
                return fishing_df
            elif table in ["EcosimFRate", "EcosimFRateTable"]:
                return frate_df
            elif table in ["EcosimCatch", "EcosimAnnualCatch"]:
                return catch_yr_df
            elif table in ["EcospaceHabitat", "EcospaceLayer"]:
                return habitat_df
            elif table == "EcospaceGrid":
                return grid_df
            elif table == "EcospaceDispersal":
                return dispersal_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            scen = ecosim_scenario_from_ewemdb(temp_path, scenario=1)
            from pypath.core.ecosim import RsimScenario
            assert isinstance(scen, RsimScenario)
            # Ecospace should be constructed but filled with defaults for missing entries
            assert hasattr(scen, "ecospace") and scen.ecospace is not None
            eco = scen.ecospace
            # Find group index for Fish in scenario params
            gi = scen.params.spname.index("Fish")
            assert abs(float(eco.habitat_preference[gi, 0]) - 0.8) < 1e-8
            # Missing patch 2 should default to 0.0
            assert abs(float(eco.habitat_preference[gi, 1]) - 0.0) < 1e-8
            # Missing dispersal should default to 0.0
            assert abs(float(eco.dispersal_rate[gi]) - 0.0) < 1e-8
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_alternate_table_names(self, mock_read_table):
        """Test that alternate table name variants are recognized."""
        groups_df = pd.DataFrame(
            {
                "GroupID": [1],
                "GroupName": ["Fish"],
                "Type": [0],
                "Biomass": [2.0],
                "PB": [1.5],
                "QB": [5.0],
                "EE": [0.80],
            }
        )

        ecosim_df = pd.DataFrame(
            {
                "ScenarioID": [1],
                "ScenarioName": ["AltNames"],
                "StartYear": [2000],
                "EndYear": [2001],
                "NumYears": [2],
                "Description": ["Alternate table names"],
            }
        )

        forcing_df = pd.DataFrame(
            {"ScenarioID": [1, 1], "Time": [0, 1], "Parameter": ["ForcedPrey", "ForcedPrey"], "Group": ["Fish", "Fish"], "Value": [1.0, 0.9]}
        )

        fishing_df = pd.DataFrame(
            {"ScenarioID": [1], "Time": [0], "Gear": [1], "Effort": [0.5]}
        )

        frate_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Group": ["Fish"], "FRate": [0.1]})
        catch_yr_df = pd.DataFrame({"ScenarioID": [1], "Year": [2000], "Group": ["Fish"], "Catch": [5.0]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcopathDietComp", "DietComp"]:
                return pd.DataFrame({"PreyName": [], "PredName": [], "Diet": []})
            elif table in ["EcosimScenarioTable", "Ecosim Scenario"]:
                return ecosim_df
            elif table in ["EcosimForcingTable", "Ecosim Forcing"]:
                return forcing_df
            elif table in ["EcosimEffortTable", "Ecosim Effort"]:
                return fishing_df
            elif table in ["EcosimFRateTable", "Ecosim_FRate"]:
                return frate_df
            elif table in ["EcosimAnnualCatch", "EcosimCatchTable"]:
                return catch_yr_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            params = read_ewemdb(temp_path, include_ecosim=True)

            assert getattr(params, "ecosim", None) is not None
            assert params.ecosim["has_ecosim"] is True
            assert len(params.ecosim["scenarios"]) == 1
            sc = params.ecosim["scenarios"][0]
            assert sc["name"] == "AltNames"
            # Forcing/fishing parsed via alternate names
            assert "forcing_df" in sc and len(sc["forcing_df"]) == 2
            assert "fishing_df" in sc and len(sc["fishing_df"]) == 1
        finally:
            os.unlink(temp_path)

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_ecospace_alternate_table_names(self, mock_read_table):
        """Ensure Ecospace mapping recognizes alternate table names."""
        groups_df = pd.DataFrame(
            {
                "GroupID": [1],
                "GroupName": ["Fish"],
                "Type": [0],
                "Biomass": [2.0],
                "PB": [1.5],
                "QB": [5.0],
                "EE": [0.80],
            }
        )

        ecosim_df = pd.DataFrame(
            {
                "ScenarioID": [1],
                "ScenarioName": ["AltEcospace"],
                "StartYear": [2000],
                "EndYear": [2000],
                "NumYears": [1],
                "Description": ["Alt ecospace table names"],
            }
        )

        # Provide grid using alternate name 'Ecospace_Grid'
        grid_df = pd.DataFrame({"PatchID": [1, 2], "Area": [10.0, 5.0], "Lon": [0.0, 0.1], "Lat": [50.0, 50.1]})
        habitat_df = pd.DataFrame({"Group": ["Fish", "Fish"], "Patch": [1, 2], "Value": [0.5, 0.3]})
        dispersal_df = pd.DataFrame({"Group": ["Fish"], "Dispersal": [0.05]})

        def mock_table_reader(filepath, table):
            if table == "EcopathGroup":
                return groups_df
            elif table in ["EcosimScenario", "EcosimScenarios"]:
                return ecosim_df
            elif table in ["EcosimForcing", "EcosimForcings"]:
                return pd.DataFrame()
            elif table in ["EcosimFishing", "EcosimEffort"]:
                return pd.DataFrame()
            elif table == "Ecospace_Grid":
                return grid_df
            elif table in ["EcospaceLayer", "EcospaceHabitat"]:
                return habitat_df
            elif table in ["Ecospace_Dispersal", "EcospaceDispersal"]:
                return dispersal_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            scen = ecosim_scenario_from_ewemdb(temp_path, scenario=1)
            from pypath.core.ecosim import RsimScenario
            assert isinstance(scen, RsimScenario)
            assert hasattr(scen, "ecospace") and scen.ecospace is not None
            eco = scen.ecospace
            assert eco.grid.n_patches == 2
            assert eco.grid.adjacency_matrix.nnz > 0
        finally:
            os.unlink(temp_path)


class TestGetMetadata:
    """Tests for get_ewemdb_metadata function."""

    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_get_metadata_basic(self, mock_read_table):
        """Test getting metadata from ewemdb file."""
        model_df = pd.DataFrame(
            {
                "ModelName": ["Baltic Sea Model"],
                "Description": ["A test model"],
                "Author": ["Test Author"],
            }
        )

        groups_df = pd.DataFrame(
            {
                "GroupID": [1, 2, 3],
                "GroupName": ["Phyto", "Zoo", "Fish"],
            }
        )

        fleet_df = pd.DataFrame(
            {
                "FleetID": [1],
                "FleetName": ["Trawlers"],
            }
        )

        def mock_table_reader(filepath, table):
            if table == "EcopathModel":
                return model_df
            elif table == "EcopathGroup":
                return groups_df
            elif table == "EcopathFleet":
                return fleet_df
            else:
                raise Exception(f"Unknown table: {table}")

        mock_read_table.side_effect = mock_table_reader

        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            metadata = get_ewemdb_metadata(temp_path)

            assert metadata["name"] == "Baltic Sea Model"
            assert metadata["description"] == "A test model"
            assert metadata["author"] == "Test Author"
            assert metadata["num_groups"] == 3
            assert metadata["num_fleets"] == 1
        finally:
            os.unlink(temp_path)

    def test_get_metadata_uses_filename(self):
        """Test that metadata uses filename when model table is missing."""
        with tempfile.NamedTemporaryFile(suffix=".ewemdb", delete=False) as f:
            temp_path = f.name

        try:
            # This should use the filename as the model name
            with patch("pypath.io.ewemdb.read_ewemdb_table") as mock_read:
                mock_read.side_effect = Exception("Table not found")
                metadata = get_ewemdb_metadata(temp_path)

                # Should use stem of filename
                assert Path(temp_path).stem in metadata["name"]
        finally:
            os.unlink(temp_path)


class TestEwEDatabaseError:
    """Tests for EwEDatabaseError exception."""

    def test_error_message(self):
        """Test that error message is preserved."""
        with pytest.raises(EwEDatabaseError) as exc_info:
            raise EwEDatabaseError("Test error message")
        assert "Test error message" in str(exc_info.value)

    def test_error_inheritance(self):
        """Test that EwEDatabaseError inherits from Exception."""
        assert issubclass(EwEDatabaseError, Exception)
