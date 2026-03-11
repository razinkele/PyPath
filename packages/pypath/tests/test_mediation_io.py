"""I/O tests for mediation functions."""

from unittest.mock import patch

import pandas as pd
import pytest


class TestReadMediation:
    """Test read_mediation() with mocked database access."""

    def test_reads_shapes(self):
        """Shape rows are converted to MediationShape objects."""
        from pypath.io.ewemdb import read_mediation

        shape_df = pd.DataFrame(
            [
                {
                    "ShapeID": 1,
                    "Title": "Positive",
                    "nPoints": 9,
                    "YY1": 0.5,
                    "YY2": 0.6,
                    "YY3": 0.7,
                    "YY4": 0.8,
                    "YY5": 1.0,
                    "YY6": 1.2,
                    "YY7": 1.3,
                    "YY8": 1.4,
                    "YY9": 1.5,
                }
            ]
        )
        group_df = pd.DataFrame(
            columns=[
                "ScenarioID",
                "ShapeID",
                "GroupID",
                "PredID",
                "PreyID",
                "AppliedWeight",
            ]
        )
        fleet_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )
        landing_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )

        table_map = {
            "EcosimShapeMediation": shape_df,
            "EcosimScenarioshapeMedWeightsGroup": group_df,
            "EcosimScenarioshapeMedWeightsFleet": fleet_df,
            "EcosimScenarioshapeMedWeightsLandings": landing_df,
        }

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                coll = read_mediation("fake.eweaccdb")

        assert len(coll.shapes) == 1
        s = coll.shapes[0]
        assert s.shape_id == 1
        assert s.name == "Positive"
        assert len(s.y_points) == 9
        assert s.y_points[0] == pytest.approx(0.5)
        assert s.y_points[8] == pytest.approx(1.5)

    def test_reads_group_links(self):
        """Group weight rows become MediationLink with prey/pred."""
        from pypath.io.ewemdb import read_mediation

        shape_df = pd.DataFrame(
            [
                {
                    "ShapeID": 1,
                    "Title": "Test",
                    "nPoints": 9,
                    "YY1": 1.0,
                    "YY2": 1.0,
                    "YY3": 1.0,
                    "YY4": 1.0,
                    "YY5": 1.0,
                    "YY6": 1.0,
                    "YY7": 1.0,
                    "YY8": 1.0,
                    "YY9": 1.0,
                }
            ]
        )
        group_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "ShapeID": 1,
                    "GroupID": 3,
                    "PredID": 2,
                    "PreyID": 1,
                    "AppliedWeight": 0.8,
                }
            ]
        )
        fleet_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )
        landing_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )

        table_map = {
            "EcosimShapeMediation": shape_df,
            "EcosimScenarioshapeMedWeightsGroup": group_df,
            "EcosimScenarioshapeMedWeightsFleet": fleet_df,
            "EcosimScenarioshapeMedWeightsLandings": landing_df,
        }

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                coll = read_mediation("fake.eweaccdb")

        assert len(coll.group_links) == 1
        link = coll.group_links[0]
        assert link.mediator_idx == 2  # GroupID 3 -> 0-based 2
        assert link.pred_idx == 1  # PredID 2 -> 0-based 1
        assert link.prey_idx == 0  # PreyID 1 -> 0-based 0
        assert link.weight == pytest.approx(0.8)

    def test_missing_tables_returns_empty(self):
        """Missing mediation tables return empty collection."""
        from pypath.io.ewemdb import read_mediation

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=["SomeOtherTable"]
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=Exception("Table not found"),
            ):
                coll = read_mediation("fake.eweaccdb")

        assert len(coll.shapes) == 0
        assert len(coll.links) == 0

    def test_reads_fleet_links(self):
        """Fleet weight rows become MediationLink with fleet_idx set."""
        from pypath.io.ewemdb import read_mediation

        shape_df = pd.DataFrame(
            [
                {
                    "ShapeID": 2,
                    "Title": "FleetShape",
                    "nPoints": 9,
                    "YY1": 1.0,
                    "YY2": 1.0,
                    "YY3": 1.0,
                    "YY4": 1.0,
                    "YY5": 1.0,
                    "YY6": 1.0,
                    "YY7": 1.0,
                    "YY8": 1.0,
                    "YY9": 1.0,
                }
            ]
        )
        group_df = pd.DataFrame(
            columns=[
                "ScenarioID",
                "ShapeID",
                "GroupID",
                "PredID",
                "PreyID",
                "AppliedWeight",
            ]
        )
        fleet_df = pd.DataFrame(
            [
                {
                    "ScenarioID": 1,
                    "ShapeID": 2,
                    "GroupID": 4,
                    "FleetID": 2,
                    "AppliedWeight": 0.5,
                }
            ]
        )
        landing_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )

        table_map = {
            "EcosimShapeMediation": shape_df,
            "EcosimScenarioshapeMedWeightsGroup": group_df,
            "EcosimScenarioshapeMedWeightsFleet": fleet_df,
            "EcosimScenarioshapeMedWeightsLandings": landing_df,
        }

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                coll = read_mediation("fake.eweaccdb")

        assert len(coll.fleet_links) == 1
        link = coll.fleet_links[0]
        assert link.mediator_idx == 3  # GroupID 4 -> 0-based 3
        assert link.fleet_idx == 1  # FleetID 2 -> 0-based 1
        assert link.weight == pytest.approx(0.5)

    def test_shape_with_fewer_than_9_points(self):
        """nPoints < 9 trims the y_points array."""
        from pypath.io.ewemdb import read_mediation

        shape_df = pd.DataFrame(
            [
                {
                    "ShapeID": 1,
                    "Title": "Short",
                    "nPoints": 5,
                    "YY1": 0.8,
                    "YY2": 0.9,
                    "YY3": 1.0,
                    "YY4": 1.1,
                    "YY5": 1.2,
                    "YY6": 1.0,
                    "YY7": 1.0,
                    "YY8": 1.0,
                    "YY9": 1.0,
                }
            ]
        )
        group_df = pd.DataFrame(
            columns=[
                "ScenarioID",
                "ShapeID",
                "GroupID",
                "PredID",
                "PreyID",
                "AppliedWeight",
            ]
        )
        fleet_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )
        landing_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )

        table_map = {
            "EcosimShapeMediation": shape_df,
            "EcosimScenarioshapeMedWeightsGroup": group_df,
            "EcosimScenarioshapeMedWeightsFleet": fleet_df,
            "EcosimScenarioshapeMedWeightsLandings": landing_df,
        }

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                coll = read_mediation("fake.eweaccdb")

        assert len(coll.shapes) == 1
        assert len(coll.shapes[0].y_points) == 5

    def test_link_referencing_unknown_shape_skipped(self):
        """Links referencing a shape not in shapes list are silently skipped."""
        from pypath.io.ewemdb import read_mediation

        shape_df = pd.DataFrame(
            [
                {
                    "ShapeID": 1,
                    "Title": "Known",
                    "nPoints": 9,
                    "YY1": 1.0,
                    "YY2": 1.0,
                    "YY3": 1.0,
                    "YY4": 1.0,
                    "YY5": 1.0,
                    "YY6": 1.0,
                    "YY7": 1.0,
                    "YY8": 1.0,
                    "YY9": 1.0,
                }
            ]
        )
        group_df = pd.DataFrame(
            [
                {
                    # ShapeID 99 does not exist in shapes
                    "ScenarioID": 1,
                    "ShapeID": 99,
                    "GroupID": 1,
                    "PredID": 2,
                    "PreyID": 1,
                    "AppliedWeight": 1.0,
                }
            ]
        )
        fleet_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )
        landing_df = pd.DataFrame(
            columns=["ScenarioID", "ShapeID", "GroupID", "FleetID", "AppliedWeight"]
        )

        table_map = {
            "EcosimShapeMediation": shape_df,
            "EcosimScenarioshapeMedWeightsGroup": group_df,
            "EcosimScenarioshapeMedWeightsFleet": fleet_df,
            "EcosimScenarioshapeMedWeightsLandings": landing_df,
        }

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", return_value=list(table_map.keys())
        ):
            with patch(
                "pypath.io.ewemdb.read_ewemdb_table",
                side_effect=lambda path, tbl: table_map[tbl],
            ):
                coll = read_mediation("fake.eweaccdb")

        assert len(coll.shapes) == 1
        assert len(coll.links) == 0

    def test_list_tables_exception_returns_empty(self):
        """Exception from list_ewemdb_tables returns empty collection."""
        from pypath.io.ewemdb import read_mediation

        with patch(
            "pypath.io.ewemdb.list_ewemdb_tables", side_effect=Exception("DB not found")
        ):
            coll = read_mediation("nonexistent.eweaccdb")

        assert len(coll.shapes) == 0
        assert len(coll.links) == 0


class TestMediationSchema:
    def test_shape_table_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        tbl = EWE_TABLES["EcosimShapeMediation"]
        assert tbl["ShapeID"] == "INTEGER"
        assert tbl["Title"] == "TEXT"
        assert tbl["nPoints"] == "INTEGER"
        for i in range(1, 10):
            assert tbl[f"YY{i}"] == "DOUBLE"

    def test_group_weights_table(self):
        from pypath.io._ewe_schema import EWE_TABLES

        tbl = EWE_TABLES["EcosimScenarioshapeMedWeightsGroup"]
        assert "PredID" in tbl
        assert "PreyID" in tbl
        assert tbl["AppliedWeight"] == "DOUBLE"

    def test_fleet_weights_table(self):
        from pypath.io._ewe_schema import EWE_TABLES

        tbl = EWE_TABLES["EcosimScenarioshapeMedWeightsFleet"]
        assert "FleetID" in tbl

    def test_landings_weights_table(self):
        from pypath.io._ewe_schema import EWE_TABLES

        tbl = EWE_TABLES["EcosimScenarioshapeMedWeightsLandings"]
        assert "FleetID" in tbl

    def test_all_mediation_tables_present(self):
        from pypath.io._ewe_schema import EWE_TABLES

        expected = [
            "EcosimShapeMediation",
            "EcosimScenarioshapeMedWeightsGroup",
            "EcosimScenarioshapeMedWeightsFleet",
            "EcosimScenarioshapeMedWeightsLandings",
        ]
        for tbl_name in expected:
            assert tbl_name in EWE_TABLES, f"Missing table: {tbl_name}"

    def test_group_weights_table_has_scenario_and_shape(self):
        from pypath.io._ewe_schema import EWE_TABLES

        tbl = EWE_TABLES["EcosimScenarioshapeMedWeightsGroup"]
        assert tbl["ScenarioID"] == "INTEGER"
        assert tbl["ShapeID"] == "INTEGER"
        assert tbl["GroupID"] == "INTEGER"

    def test_shape_table_has_nine_yy_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        tbl = EWE_TABLES["EcosimShapeMediation"]
        yy_cols = [k for k in tbl if k.startswith("YY")]
        assert len(yy_cols) == 9
