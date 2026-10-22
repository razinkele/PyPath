# EwE Export Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export PyPath models (Ecopath + Ecosim scenarios + Ecospace) back to native EwE `.eweaccdb` format, with a CSV bundle fallback for platforms without ODBC.

**Architecture:** `write_ewemdb()` entry point delegates to AccessWriter (pyodbc, copies blank template + INSERTs) or CsvBundleWriter (zip of CSVs). Auto-detects backend. New files in `pypath/io/`.

**Tech Stack:** pyodbc (optional), zipfile, json, pandas, numpy, shutil, tempfile

---

### Task 1: EwE Schema Constants Module

**Files:**
- Create: `packages/pypath/src/pypath/io/_ewe_schema.py`
- Test: `packages/pypath/tests/test_ewe_writer.py`

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ewe_writer.py`:

```python
"""Tests for EwE database export (writer) infrastructure."""

import pytest


class TestEweSchema:
    """Test the EwE 6 table/column schema definitions."""

    def test_ecopath_group_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcopathGroup" in EWE_TABLES
        cols = EWE_TABLES["EcopathGroup"]
        assert "GroupName" in cols
        assert "Biomass" in cols
        assert "PB" in cols
        assert "QB" in cols

    def test_ecopath_diet_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcopathDietComp" in cols
        cols = EWE_TABLES["EcopathDietComp"]
        assert "PreyID" in cols
        assert "PredID" in cols
        assert "Diet" in cols

    def test_ecosim_scenario_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcosimScenario" in EWE_TABLES

    def test_ecospace_scenario_columns_defined(self):
        from pypath.io._ewe_schema import EWE_TABLES

        assert "EcospaceScenario" in EWE_TABLES

    def test_table_count_minimum(self):
        """EwE 6 has ~89 tables; we must define at least the core ones."""
        from pypath.io._ewe_schema import EWE_TABLES

        assert len(EWE_TABLES) >= 15

    def test_rpath_to_ewe_mapping_exists(self):
        from pypath.io._ewe_schema import RPATH_TO_EWE_COLUMNS

        assert "Biomass" in RPATH_TO_EWE_COLUMNS
        assert "PB" in RPATH_TO_EWE_COLUMNS
        assert "QB" in RPATH_TO_EWE_COLUMNS
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py::TestEweSchema -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.io._ewe_schema'`

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/io/_ewe_schema.py`:

```python
"""EwE 6 database schema definitions.

Column names and types for the core EwE 6 Access database tables.
Used by both AccessWriter and CsvBundleWriter to ensure correct output schema.
"""

from __future__ import annotations

from collections import OrderedDict

# Column definitions per table: OrderedDict of {column_name: sql_type}.
# sql_type is "INTEGER", "DOUBLE", "TEXT", or "YESNO".
# Only core tables needed for Ecopath + Ecosim + Ecospace export are defined.

EWE_TABLES: dict[str, OrderedDict[str, str]] = {}

# ── Ecopath tables ──────────────────────────────────────────────────

EWE_TABLES["EcopathModel"] = OrderedDict(
    [
        ("ModelID", "INTEGER"),
        ("ModelName", "TEXT"),
        ("Description", "TEXT"),
        ("Author", "TEXT"),
        ("NumGroups", "INTEGER"),
        ("NumFleets", "INTEGER"),
        ("NumLiving", "INTEGER"),
        ("NumDetritus", "INTEGER"),
        ("FirstYear", "INTEGER"),
        ("LastYear", "INTEGER"),
        ("EwEVersion", "TEXT"),
    ]
)

EWE_TABLES["EcopathGroup"] = OrderedDict(
    [
        ("GroupID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("GroupName", "TEXT"),
        ("Sequence", "INTEGER"),
        ("Type", "INTEGER"),
        ("PP", "INTEGER"),
        ("Biomass", "DOUBLE"),
        ("PB", "DOUBLE"),
        ("QB", "DOUBLE"),
        ("EE", "DOUBLE"),
        ("GE", "DOUBLE"),
        ("GS", "DOUBLE"),
        ("BA", "DOUBLE"),
        ("BiomassAreaInput", "DOUBLE"),
        ("ImmigEmig", "DOUBLE"),
        ("Area", "DOUBLE"),
        ("AreaHabitat", "DOUBLE"),
    ]
)

EWE_TABLES["EcopathDietComp"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("PreyID", "INTEGER"),
        ("PredID", "INTEGER"),
        ("Diet", "DOUBLE"),
    ]
)

EWE_TABLES["EcopathFleet"] = OrderedDict(
    [
        ("FleetID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("FleetName", "TEXT"),
    ]
)

EWE_TABLES["EcopathCatch"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("Landing", "DOUBLE"),
        ("Discard", "DOUBLE"),
    ]
)

EWE_TABLES["Stanza"] = OrderedDict(
    [
        ("StanzaID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("StanzaName", "TEXT"),
        ("NStanzas", "INTEGER"),
    ]
)

EWE_TABLES["StanzaLifeStage"] = OrderedDict(
    [
        ("StanzaID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("Sequence", "INTEGER"),
        ("First", "DOUBLE"),
        ("Last", "DOUBLE"),
        ("Z", "DOUBLE"),
        ("Leading", "YESNO"),
    ]
)

# ── Ecosim tables ───────────────────────────────────────────────────

EWE_TABLES["EcosimScenario"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("ScenarioName", "TEXT"),
        ("ScenarioDescription", "TEXT"),
        ("NumYears", "INTEGER"),
        ("StartYear", "INTEGER"),
    ]
)

EWE_TABLES["EcosimGroupInfo"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("VV", "DOUBLE"),
        ("DD", "DOUBLE"),
        ("HandleTime", "DOUBLE"),
        ("MaxRelPB", "DOUBLE"),
        ("MaxRelFeedingTime", "DOUBLE"),
        ("SwitchingPower", "DOUBLE"),
    ]
)

EWE_TABLES["EcosimScenarioGroup"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("FtimeAdj", "DOUBLE"),
    ]
)

EWE_TABLES["EcosimForcing"] = OrderedDict(
    [
        ("ForcingID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("ForcingName", "TEXT"),
        ("ForcingType", "INTEGER"),
    ]
)

EWE_TABLES["EcosimShapeTime"] = OrderedDict(
    [
        ("ForcingID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("TimeStep", "INTEGER"),
        ("Value", "DOUBLE"),
    ]
)

EWE_TABLES["EcosimScenarioForcingMatrix"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("ForcingID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("Value", "DOUBLE"),
    ]
)

EWE_TABLES["EcosimShapeFishRate"] = OrderedDict(
    [
        ("ShapeID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("FleetID", "INTEGER"),
        ("TimeStep", "INTEGER"),
        ("Value", "DOUBLE"),
    ]
)

# ── Ecospace tables ─────────────────────────────────────────────────

EWE_TABLES["EcospaceScenario"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("ScenarioName", "TEXT"),
        ("NRows", "INTEGER"),
        ("NCols", "INTEGER"),
        ("CellSize", "DOUBLE"),
        ("OriginLat", "DOUBLE"),
        ("OriginLon", "DOUBLE"),
    ]
)

EWE_TABLES["EcospaceGroup"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("GroupID", "INTEGER"),
        ("DispersalRate", "DOUBLE"),
        ("MoveInBadHabitat", "YESNO"),
        ("AdvectionEnabled", "YESNO"),
    ]
)

EWE_TABLES["EcospaceHabitat"] = OrderedDict(
    [
        ("HabitatID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("HabitatName", "TEXT"),
    ]
)

EWE_TABLES["EcospaceMap"] = OrderedDict(
    [
        ("ScenarioID", "INTEGER"),
        ("Row", "INTEGER"),
        ("Col", "INTEGER"),
        ("Depth", "DOUBLE"),
        ("HabitatID", "INTEGER"),
        ("InModel", "YESNO"),
    ]
)

EWE_TABLES["EcospaceMPA"] = OrderedDict(
    [
        ("MPAID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("MPAName", "TEXT"),
    ]
)

EWE_TABLES["EcospaceRegion"] = OrderedDict(
    [
        ("RegionID", "INTEGER"),
        ("ScenarioID", "INTEGER"),
        ("RegionName", "TEXT"),
    ]
)

# ── Mapping: RpathParams column name → EwE column name ──────────────

RPATH_TO_EWE_COLUMNS: dict[str, str] = {
    "Group": "GroupName",
    "Type": "Type",
    "Biomass": "Biomass",
    "PB": "PB",
    "QB": "QB",
    "EE": "EE",
    "ProdCons": "GE",
    "Unassim": "GS",
    "BioAcc": "BA",
    "DetInput": "ImmigEmig",
}

# Type mapping: RpathParams Type → EwE PP code
TYPE_TO_PP: dict[int, int] = {
    0: 0,   # consumer
    1: 1,   # primary producer
    2: 2,   # detritus
    3: 0,   # fleet (handled separately in EcopathFleet)
}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py::TestEweSchema -v`
Expected: PASS (fix the typo in test_ecopath_diet_columns_defined: `assert "EcopathDietComp" in cols` should be `in EWE_TABLES`)

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_ewe_writer.py
git commit -m "feat(io): add EwE 6 database schema constants for export writer"
```

---

### Task 2: CSV Bundle Writer

**Files:**
- Create: `packages/pypath/src/pypath/io/_csv_bundle_writer.py`
- Modify: `packages/pypath/tests/test_ewe_writer.py`

**Step 1: Write the failing tests**

Append to `packages/pypath/tests/test_ewe_writer.py`:

```python
import json
import os
import tempfile
import zipfile

import numpy as np
import pandas as pd

from pypath.core.params import create_rpath_params


def _make_simple_model():
    """Create a minimal 5-group model for testing."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Detritus", "Fleet1"],
        types=[1, 0, 0, 2, 3],
    )
    params.model["Biomass"] = [10.0, 5.0, 2.0, 100.0, np.nan]
    params.model["PB"] = [100.0, 40.0, 1.5, np.nan, np.nan]
    params.model["QB"] = [0.0, 100.0, 5.0, np.nan, np.nan]
    params.model["EE"] = [0.9, 0.8, 0.7, np.nan, np.nan]
    params.model["Unassim"] = [0.0, 0.2, 0.2, np.nan, np.nan]
    # Set some diet values
    diet_groups = params.diet["Group"].tolist()
    phyto_idx = diet_groups.index("Phyto")
    zoo_idx = diet_groups.index("Zoo")
    params.diet.iloc[phyto_idx, params.diet.columns.get_loc("Zoo")] = 1.0
    params.diet.iloc[zoo_idx, params.diet.columns.get_loc("Fish")] = 0.8
    params.diet.iloc[phyto_idx, params.diet.columns.get_loc("Fish")] = 0.2
    return params


class TestCsvBundleWriter:
    """Test the CSV bundle fallback writer."""

    def test_creates_zip_file(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        assert outpath.exists()
        assert zipfile.is_zipfile(outpath)

    def test_zip_contains_manifest(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "manifest.json" in zf.namelist()
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["ewe_version"] == "6.6"

    def test_zip_contains_ecopath_group_csv(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "EcopathGroup.csv" in zf.namelist()
            df = pd.read_csv(zf.open("EcopathGroup.csv"))
            # 4 biological groups (fleet handled separately)
            assert len(df) == 4
            assert "GroupName" in df.columns
            assert df.iloc[0]["GroupName"] == "Phyto"

    def test_zip_contains_diet_csv(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "EcopathDietComp.csv" in zf.namelist()
            df = pd.read_csv(zf.open("EcopathDietComp.csv"))
            # Non-zero diet entries only
            assert len(df) >= 3
            assert "PreyID" in df.columns
            assert "Diet" in df.columns

    def test_zip_contains_fleet_csv(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            assert "EcopathFleet.csv" in zf.namelist()
            df = pd.read_csv(zf.open("EcopathFleet.csv"))
            assert len(df) == 1
            assert df.iloc[0]["FleetName"] == "Fleet1"

    def test_biomass_values_roundtrip(self, tmp_path):
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        with zipfile.ZipFile(outpath) as zf:
            df = pd.read_csv(zf.open("EcopathGroup.csv"))
            assert abs(df.iloc[0]["Biomass"] - 10.0) < 1e-6
            assert abs(df.iloc[1]["Biomass"] - 5.0) < 1e-6

    def test_atomic_write_no_partial_file(self, tmp_path):
        """If writer is not closed, no file at target path."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.ewecsv.zip"
        writer = CsvBundleWriter(params, str(outpath))
        writer.write_ecopath()
        # NOT calling writer.close()
        # The final file should not exist yet (written to temp)
        # (Implementation detail: file appears at close())
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py::TestCsvBundleWriter -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.io._csv_bundle_writer'`

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/io/_csv_bundle_writer.py`:

```python
"""CSV bundle writer for EwE export.

Fallback writer that creates a .ewecsv.zip file containing one CSV per EwE table
plus a manifest.json. This works on all platforms without ODBC drivers.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from pypath.io._ewe_schema import (
    EWE_TABLES,
    RPATH_TO_EWE_COLUMNS,
    TYPE_TO_PP,
)

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario
    from pypath.core.params import RpathParams

logger = logging.getLogger(__name__)


class CsvBundleWriter:
    """Write EwE model data as a zip of CSV files.

    Parameters
    ----------
    params : RpathParams
        Ecopath model parameters.
    path : str
        Output path for the .ewecsv.zip file.
    scenario_id : int
        ScenarioID to stamp in all tables (default 1).
    """

    def __init__(self, params: "RpathParams", path: str, scenario_id: int = 1):
        self.params = params
        self.path = str(Path(path).resolve())
        self.scenario_id = scenario_id
        self._tables: dict[str, pd.DataFrame] = {}
        # Write to a temp file first for atomic output
        fd, self._tmp_path = tempfile.mkstemp(
            suffix=".ewecsv.zip.tmp", dir=str(Path(path).parent)
        )
        os.close(fd)

    def write_ecopath(self) -> None:
        """Convert RpathParams to EwE Ecopath table DataFrames."""
        model = self.params.model
        sid = self.scenario_id

        # Separate biological groups from fleets
        bio_mask = model["Type"] != 3
        fleet_mask = model["Type"] == 3
        bio_df = model[bio_mask].reset_index(drop=True)
        fleet_df = model[fleet_mask].reset_index(drop=True)

        # ── EcopathGroup ─────────────────────────────────────
        rows = []
        for i, (_, row) in enumerate(bio_df.iterrows()):
            rows.append(
                {
                    "GroupID": i + 1,
                    "ScenarioID": sid,
                    "GroupName": row["Group"],
                    "Sequence": i + 1,
                    "Type": int(row["Type"]),
                    "PP": TYPE_TO_PP.get(int(row["Type"]), 0),
                    "Biomass": row.get("Biomass", np.nan),
                    "PB": row.get("PB", np.nan),
                    "QB": row.get("QB", np.nan),
                    "EE": row.get("EE", np.nan),
                    "GE": row.get("ProdCons", np.nan),
                    "GS": row.get("Unassim", np.nan),
                    "BA": row.get("BioAcc", np.nan),
                    "BiomassAreaInput": row.get("Biomass", np.nan),
                    "ImmigEmig": row.get("DetInput", np.nan),
                    "Area": row.get("Area", 1.0) if "Area" in row.index else 1.0,
                    "AreaHabitat": 1.0,
                }
            )
        self._tables["EcopathGroup"] = pd.DataFrame(rows)

        # ── EcopathFleet ─────────────────────────────────────
        fleet_rows = []
        for i, (_, row) in enumerate(fleet_df.iterrows()):
            fleet_rows.append(
                {
                    "FleetID": i + 1,
                    "ScenarioID": sid,
                    "FleetName": row["Group"],
                }
            )
        self._tables["EcopathFleet"] = pd.DataFrame(fleet_rows)

        # ── EcopathDietComp ──────────────────────────────────
        diet = self.params.diet
        bio_group_names = bio_df["Group"].tolist()
        diet_rows = []
        for pred_name in bio_group_names:
            if pred_name not in diet.columns:
                continue
            pred_id = bio_group_names.index(pred_name) + 1
            for prey_idx, prey_name in enumerate(diet["Group"]):
                val = diet.iloc[prey_idx, diet.columns.get_loc(pred_name)]
                if pd.notna(val) and float(val) > 0:
                    # Import row has prey_id = n_bio + 1 by EwE convention
                    if prey_name == "Import":
                        prey_id = len(bio_group_names) + 1
                    elif prey_name in bio_group_names:
                        prey_id = bio_group_names.index(prey_name) + 1
                    else:
                        continue
                    diet_rows.append(
                        {
                            "ScenarioID": sid,
                            "PreyID": prey_id,
                            "PredID": pred_id,
                            "Diet": float(val),
                        }
                    )
        self._tables["EcopathDietComp"] = pd.DataFrame(diet_rows)

        # ── EcopathCatch ─────────────────────────────────────
        catch_rows = []
        for fi, (_, frow) in enumerate(fleet_df.iterrows()):
            fleet_name = frow["Group"]
            land_col = f"Landings.{fleet_name}"
            disc_col = f"Discards.{fleet_name}"
            for gi, (_, grow) in enumerate(bio_df.iterrows()):
                landing = grow.get(land_col, 0.0) if land_col in bio_df.columns else 0.0
                discard = grow.get(disc_col, 0.0) if disc_col in bio_df.columns else 0.0
                if pd.notna(landing) and landing > 0 or pd.notna(discard) and discard > 0:
                    catch_rows.append(
                        {
                            "ScenarioID": sid,
                            "FleetID": fi + 1,
                            "GroupID": gi + 1,
                            "Landing": float(landing) if pd.notna(landing) else 0.0,
                            "Discard": float(discard) if pd.notna(discard) else 0.0,
                        }
                    )
        self._tables["EcopathCatch"] = pd.DataFrame(catch_rows)

        # ── Stanzas ──────────────────────────────────────────
        stanzas = self.params.stanzas
        if stanzas and stanzas.n_stanza_groups > 0 and stanzas.stgroups is not None:
            st_rows = []
            for i, (_, row) in enumerate(stanzas.stgroups.iterrows()):
                st_rows.append(
                    {
                        "StanzaID": i + 1,
                        "ScenarioID": sid,
                        "StanzaName": row.get("StGroupName", f"Stanza{i+1}"),
                        "NStanzas": int(row.get("nstanzas", 2)),
                    }
                )
            self._tables["Stanza"] = pd.DataFrame(st_rows)

            if stanzas.stindiv is not None:
                sl_rows = []
                for _, row in stanzas.stindiv.iterrows():
                    gid = int(row.get("GroupNum", 0))
                    sl_rows.append(
                        {
                            "StanzaID": int(row.get("StGroupNum", 1)),
                            "ScenarioID": sid,
                            "GroupID": gid,
                            "Sequence": int(row.get("Sequence", 0)),
                            "First": float(row.get("First", 0)),
                            "Last": float(row.get("Last", 0)),
                            "Z": float(row.get("Z", 0)),
                            "Leading": bool(row.get("Leading", False)),
                        }
                    )
                self._tables["StanzaLifeStage"] = pd.DataFrame(sl_rows)

        # ── EcopathModel ─────────────────────────────────────
        n_living = len(bio_df[bio_df["Type"].isin([0, 1])])
        n_detritus = len(bio_df[bio_df["Type"] == 2])
        self._tables["EcopathModel"] = pd.DataFrame(
            [
                {
                    "ModelID": 1,
                    "ModelName": "PyPath Export",
                    "Description": "Exported from PyPath",
                    "Author": "",
                    "NumGroups": len(bio_df),
                    "NumFleets": len(fleet_df),
                    "NumLiving": n_living,
                    "NumDetritus": n_detritus,
                    "FirstYear": 0,
                    "LastYear": 0,
                    "EwEVersion": "6.6",
                }
            ]
        )

    def write_ecosim(self, scenarios: list[Any] | None = None) -> None:
        """Convert RsimScenario objects to EwE Ecosim table DataFrames.

        Parameters
        ----------
        scenarios : list of RsimScenario, optional
            Ecosim scenarios to export. If None, no Ecosim tables are written.
        """
        if not scenarios:
            return

        sid = self.scenario_id
        scen_rows = []
        group_info_rows = []
        forcing_rows = []
        shape_time_rows = []
        forcing_matrix_rows = []
        fish_rate_rows = []

        for si, scen in enumerate(scenarios):
            scen_id = si + 1
            scen_rows.append(
                {
                    "ScenarioID": scen_id,
                    "ScenarioName": getattr(scen, "eco_name", f"Scenario {scen_id}"),
                    "ScenarioDescription": "",
                    "NumYears": len(getattr(scen, "forcing", None).ForcedBio[0])
                    // 12
                    if hasattr(scen, "forcing")
                    and hasattr(scen.forcing, "ForcedBio")
                    else 0,
                    "StartYear": getattr(scen, "start_year", 1),
                }
            )

            # Group-level Ecosim parameters (VV, DD, etc.)
            if hasattr(scen, "params") and hasattr(scen.params, "VV"):
                for gi in range(len(scen.params.VV)):
                    group_info_rows.append(
                        {
                            "ScenarioID": scen_id,
                            "GroupID": gi + 1,
                            "VV": float(scen.params.VV[gi])
                            if gi < len(scen.params.VV)
                            else 2.0,
                            "DD": float(scen.params.DD[gi])
                            if hasattr(scen.params, "DD")
                            and gi < len(scen.params.DD)
                            else 0.0,
                            "HandleTime": float(scen.params.HandleTime[gi])
                            if hasattr(scen.params, "HandleTime")
                            and gi < len(scen.params.HandleTime)
                            else 0.0,
                            "MaxRelPB": float(scen.params.MaxRelPB[gi])
                            if hasattr(scen.params, "MaxRelPB")
                            and gi < len(scen.params.MaxRelPB)
                            else 2.0,
                            "MaxRelFeedingTime": float(
                                scen.params.MaxRelFeedingTime[gi]
                            )
                            if hasattr(scen.params, "MaxRelFeedingTime")
                            and gi < len(scen.params.MaxRelFeedingTime)
                            else 2.0,
                            "SwitchingPower": float(
                                scen.params.SwitchingPower[gi]
                            )
                            if hasattr(scen.params, "SwitchingPower")
                            and gi < len(scen.params.SwitchingPower)
                            else 0.0,
                        }
                    )

            # Fishing effort shapes
            if hasattr(scen, "fishing") and hasattr(scen.fishing, "FishingEffort"):
                effort = scen.fishing.FishingEffort
                for fi in range(effort.shape[0]):
                    for t in range(effort.shape[1]):
                        if abs(effort[fi, t] - 1.0) > 1e-9:
                            fish_rate_rows.append(
                                {
                                    "ShapeID": fi + 1,
                                    "ScenarioID": scen_id,
                                    "FleetID": fi + 1,
                                    "TimeStep": t,
                                    "Value": float(effort[fi, t]),
                                }
                            )

        self._tables["EcosimScenario"] = pd.DataFrame(scen_rows)
        if group_info_rows:
            self._tables["EcosimGroupInfo"] = pd.DataFrame(group_info_rows)
        if forcing_rows:
            self._tables["EcosimForcing"] = pd.DataFrame(forcing_rows)
        if shape_time_rows:
            self._tables["EcosimShapeTime"] = pd.DataFrame(shape_time_rows)
        if forcing_matrix_rows:
            self._tables["EcosimScenarioForcingMatrix"] = pd.DataFrame(
                forcing_matrix_rows
            )
        if fish_rate_rows:
            self._tables["EcosimShapeFishRate"] = pd.DataFrame(fish_rate_rows)

    def write_ecospace(self, ecospace: Any | None = None) -> None:
        """Convert EcospaceParams to EwE Ecospace table DataFrames.

        Parameters
        ----------
        ecospace : EcospaceParams, optional
            Ecospace spatial parameters to export.
        """
        if ecospace is None:
            return

        sid = self.scenario_id
        grid = ecospace.grid

        self._tables["EcospaceScenario"] = pd.DataFrame(
            [
                {
                    "ScenarioID": sid,
                    "ScenarioName": "PyPath Ecospace",
                    "NRows": grid.n_rows,
                    "NCols": grid.n_cols,
                    "CellSize": grid.cell_size,
                    "OriginLat": getattr(grid, "origin_lat", 0.0),
                    "OriginLon": getattr(grid, "origin_lon", 0.0),
                }
            ]
        )

        # Dispersal rates per group
        if hasattr(ecospace, "dispersal_rate"):
            group_rows = []
            for gi in range(len(ecospace.dispersal_rate)):
                group_rows.append(
                    {
                        "ScenarioID": sid,
                        "GroupID": gi + 1,
                        "DispersalRate": float(ecospace.dispersal_rate[gi]),
                        "MoveInBadHabitat": bool(
                            ecospace.advection_enabled[gi]
                        )
                        if hasattr(ecospace, "advection_enabled")
                        else False,
                        "AdvectionEnabled": bool(
                            ecospace.advection_enabled[gi]
                        )
                        if hasattr(ecospace, "advection_enabled")
                        else False,
                    }
                )
            self._tables["EcospaceGroup"] = pd.DataFrame(group_rows)

    def close(self) -> None:
        """Write all accumulated tables to the zip file and rename to final path."""
        with zipfile.ZipFile(
            self._tmp_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as zf:
            # Write manifest
            manifest = {
                "ewe_version": "6.6",
                "created": datetime.now(timezone.utc).isoformat(),
                "source": "PyPath",
                "tables": list(self._tables.keys()),
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

            # Write each table as CSV
            for table_name, df in self._tables.items():
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                zf.writestr(f"{table_name}.csv", buf.getvalue())

        # Atomic rename
        os.replace(self._tmp_path, self.path)
        logger.info("Wrote CSV bundle: %s (%d tables)", self.path, len(self._tables))
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py::TestCsvBundleWriter -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/tests/test_ewe_writer.py
git commit -m "feat(io): add CSV bundle writer for EwE export fallback"
```

---

### Task 3: Access Writer

**Files:**
- Create: `packages/pypath/src/pypath/io/_access_writer.py`
- Modify: `packages/pypath/tests/test_ewe_writer.py`

**Step 1: Write the failing tests**

Append to `packages/pypath/tests/test_ewe_writer.py`:

```python
class TestAccessWriter:
    """Test the Access database writer (requires pyodbc + Access driver)."""

    @pytest.fixture(autouse=True)
    def _skip_no_odbc(self):
        """Skip tests if ODBC Access driver is not available."""
        try:
            from pypath.io._access_writer import AccessWriter

            AccessWriter._check_odbc()
        except (ImportError, RuntimeError):
            pytest.skip("Access ODBC driver not available")

    def test_creates_accdb_file(self, tmp_path):
        from pypath.io._access_writer import AccessWriter

        params = _make_simple_model()
        outpath = tmp_path / "test_model.eweaccdb"
        writer = AccessWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        assert outpath.exists()
        assert outpath.stat().st_size > 0

    def test_accdb_has_ecopath_group_table(self, tmp_path):
        from pypath.io._access_writer import AccessWriter
        from pypath.io.ewemdb import read_ewemdb_table

        params = _make_simple_model()
        outpath = tmp_path / "test_model.eweaccdb"
        writer = AccessWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        groups = read_ewemdb_table(str(outpath), "EcopathGroup")
        assert len(groups) == 4  # 4 bio groups (fleet is separate)
        assert groups.iloc[0]["GroupName"] == "Phyto"

    def test_accdb_diet_roundtrip(self, tmp_path):
        from pypath.io._access_writer import AccessWriter
        from pypath.io.ewemdb import read_ewemdb_table

        params = _make_simple_model()
        outpath = tmp_path / "test_model.eweaccdb"
        writer = AccessWriter(params, str(outpath))
        writer.write_ecopath()
        writer.close()
        diet = read_ewemdb_table(str(outpath), "EcopathDietComp")
        assert len(diet) >= 3
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py::TestAccessWriter -v`
Expected: FAIL with `ModuleNotFoundError` (or skip if no ODBC)

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/io/_access_writer.py`:

```python
"""Access database writer for EwE export.

Primary writer that copies a blank .eweaccdb template and INSERTs rows
using pyodbc. Requires an ODBC Access driver to be installed.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

from pypath.io._ewe_schema import EWE_TABLES, RPATH_TO_EWE_COLUMNS, TYPE_TO_PP

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario
    from pypath.core.params import RpathParams

logger = logging.getLogger(__name__)

# Path to the blank EwE 6 template shipped with the package
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "blank_ewe6.eweaccdb"


def _find_access_driver() -> str:
    """Find an available Access ODBC driver name.

    Returns
    -------
    str
        Driver name suitable for pyodbc connection string.

    Raises
    ------
    RuntimeError
        If no Access ODBC driver is found.
    """
    import pyodbc

    candidates = [
        "Microsoft Access Driver (*.mdb, *.accdb)",
        "Microsoft Access Driver (*.mdb)",
    ]
    available = pyodbc.drivers()
    for drv in candidates:
        if drv in available:
            return drv
    raise RuntimeError(
        "No Microsoft Access ODBC driver found. "
        "Install 'Microsoft Access Database Engine' or use backend='csv'."
    )


class AccessWriter:
    """Write EwE model data to a Microsoft Access .eweaccdb file.

    Parameters
    ----------
    params : RpathParams
        Ecopath model parameters.
    path : str
        Output path for the .eweaccdb file.
    scenario_id : int
        ScenarioID to stamp in all tables (default 1).
    """

    def __init__(self, params: "RpathParams", path: str, scenario_id: int = 1):
        self.params = params
        self.path = str(Path(path).resolve())
        self.scenario_id = scenario_id
        self._conn = None

        # Copy template to a temp location
        if not _TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                f"Blank EwE template not found at {_TEMPLATE_PATH}. "
                "Reinstall pypath-ewe or use backend='csv'."
            )
        fd, self._tmp_path = tempfile.mkstemp(
            suffix=".eweaccdb", dir=str(Path(path).parent)
        )
        os.close(fd)
        shutil.copy2(_TEMPLATE_PATH, self._tmp_path)

        # Open ODBC connection
        import pyodbc

        driver = _find_access_driver()
        conn_str = f"DRIVER={{{driver}}};DBQ={self._tmp_path};"
        self._conn = pyodbc.connect(conn_str)

    @staticmethod
    def _check_odbc() -> None:
        """Check if ODBC Access driver is available. Raises RuntimeError if not."""
        import pyodbc  # noqa: F401

        _find_access_driver()

    def _insert_rows(self, table: str, df: pd.DataFrame) -> None:
        """INSERT rows from a DataFrame into an Access table."""
        if df is None or len(df) == 0:
            return
        cursor = self._conn.cursor()
        cols = list(df.columns)
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(f"[{c}]" for c in cols)
        sql = f"INSERT INTO [{table}] ({col_names}) VALUES ({placeholders})"
        for _, row in df.iterrows():
            values = []
            for v in row:
                if isinstance(v, float) and np.isnan(v):
                    values.append(None)
                else:
                    values.append(v)
            cursor.execute(sql, values)
        self._conn.commit()

    def write_ecopath(self) -> None:
        """Write Ecopath tables to the Access database.

        Reuses CsvBundleWriter's table-building logic, then INSERTs.
        """
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        # Build DataFrames using CSV writer's logic (DRY)
        csv_writer = CsvBundleWriter.__new__(CsvBundleWriter)
        csv_writer.params = self.params
        csv_writer.scenario_id = self.scenario_id
        csv_writer._tables = {}
        csv_writer.write_ecopath()

        for table_name, df in csv_writer._tables.items():
            self._insert_rows(table_name, df)

    def write_ecosim(self, scenarios: list[Any] | None = None) -> None:
        """Write Ecosim tables to the Access database."""
        if not scenarios:
            return
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        csv_writer = CsvBundleWriter.__new__(CsvBundleWriter)
        csv_writer.params = self.params
        csv_writer.scenario_id = self.scenario_id
        csv_writer._tables = {}
        csv_writer.write_ecosim(scenarios)

        for table_name, df in csv_writer._tables.items():
            self._insert_rows(table_name, df)

    def write_ecospace(self, ecospace: Any | None = None) -> None:
        """Write Ecospace tables to the Access database."""
        if ecospace is None:
            return
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        csv_writer = CsvBundleWriter.__new__(CsvBundleWriter)
        csv_writer.params = self.params
        csv_writer.scenario_id = self.scenario_id
        csv_writer._tables = {}
        csv_writer.write_ecospace(ecospace)

        for table_name, df in csv_writer._tables.items():
            self._insert_rows(table_name, df)

    def close(self) -> None:
        """Close connection and rename temp file to final path."""
        if self._conn:
            self._conn.close()
            self._conn = None
        os.replace(self._tmp_path, self.path)
        logger.info("Wrote Access database: %s", self.path)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py::TestAccessWriter -v`
Expected: PASS (or skip if no ODBC driver)

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_access_writer.py packages/pypath/tests/test_ewe_writer.py
git commit -m "feat(io): add Access database writer for EwE export"
```

---

### Task 4: Create Blank EwE 6 Template

**Files:**
- Create: `packages/pypath/src/pypath/io/templates/blank_ewe6.eweaccdb`
- Create: `packages/pypath/src/pypath/io/templates/__init__.py` (empty)

**Step 1: Create the blank template**

The blank template is a real Access database with all tables created but empty. Since we need the ODBC driver to create it, we generate it programmatically and ship the result.

Create a script `packages/pypath/scripts/create_blank_template.py`:

```python
"""Create a blank EwE 6 template database with all required tables.

This script generates the blank_ewe6.eweaccdb template file.
Requires pyodbc and Microsoft Access ODBC driver.

Run once on Windows to generate the template, then commit the binary file.
"""

import os
import sys
from pathlib import Path

import pyodbc

# Target path
template_dir = Path(__file__).parent.parent / "src" / "pypath" / "io" / "templates"
template_dir.mkdir(exist_ok=True)
template_path = template_dir / "blank_ewe6.eweaccdb"

# Import schema
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pypath.io._ewe_schema import EWE_TABLES

# SQL type mapping
TYPE_MAP = {
    "INTEGER": "LONG",
    "DOUBLE": "DOUBLE",
    "TEXT": "TEXT(255)",
    "YESNO": "YESNO",
}

# Find Access driver
driver = None
for d in pyodbc.drivers():
    if "Access" in d:
        driver = d
        break
if not driver:
    print("ERROR: No Access ODBC driver found")
    sys.exit(1)

# Create new empty database using ADOX (via pyodbc)
# Actually, the simplest way is to create via connection with CREATE TABLE
# But Access requires the file to exist first. Use win32com if available.
try:
    import win32com.client

    catalog = win32com.client.Dispatch("ADOX.Catalog")
    conn_str = f"Provider=Microsoft.ACE.OLEDB.12.0;Data Source={template_path};"
    catalog.Create(conn_str)
    catalog = None
    print(f"Created blank database: {template_path}")
except ImportError:
    # Alternative: copy a minimal blank accdb from the system
    # Or use a pre-existing blank file
    print("win32com not available. Please provide a blank .accdb file.")
    sys.exit(1)

# Connect and create tables
conn_str = f"DRIVER={{{driver}}};DBQ={template_path};"
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

for table_name, columns in EWE_TABLES.items():
    col_defs = []
    for col_name, col_type in columns.items():
        sql_type = TYPE_MAP.get(col_type, "TEXT(255)")
        col_defs.append(f"[{col_name}] {sql_type}")
    sql = f"CREATE TABLE [{table_name}] ({', '.join(col_defs)})"
    try:
        cursor.execute(sql)
        print(f"  Created table: {table_name}")
    except Exception as e:
        print(f"  WARN: Failed to create {table_name}: {e}")

conn.commit()
conn.close()
print(f"\nTemplate saved: {template_path} ({template_path.stat().st_size} bytes)")
```

**Step 2: Run the script to generate the template**

Run: `python packages/pypath/scripts/create_blank_template.py`
Expected: Creates `packages/pypath/src/pypath/io/templates/blank_ewe6.eweaccdb`

**Step 3: Create the __init__.py**

Create empty `packages/pypath/src/pypath/io/templates/__init__.py`.

**Step 4: Add template to package-data in pyproject.toml**

In `packages/pypath/pyproject.toml`, add after `[tool.setuptools.packages.find]`:

```toml
[tool.setuptools.package-data]
pypath = ["io/templates/*.eweaccdb"]
```

**Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/templates/ packages/pypath/scripts/create_blank_template.py packages/pypath/pyproject.toml
git commit -m "feat(io): add blank EwE 6 template database for Access writer"
```

---

### Task 5: Public Entry Point — write_ewemdb()

**Files:**
- Create: `packages/pypath/src/pypath/io/ewe_writer.py`
- Modify: `packages/pypath/src/pypath/io/__init__.py`
- Modify: `packages/pypath/tests/test_ewe_writer.py`

**Step 1: Write the failing tests**

Append to `packages/pypath/tests/test_ewe_writer.py`:

```python
class TestWriteEwemdb:
    """Test the public write_ewemdb() entry point."""

    def test_write_csv_bundle(self, tmp_path):
        from pypath.io.ewe_writer import write_ewemdb

        params = _make_simple_model()
        outpath = tmp_path / "model.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")
        assert outpath.exists()
        assert zipfile.is_zipfile(outpath)

    def test_write_auto_detects_backend(self, tmp_path):
        from pypath.io.ewe_writer import write_ewemdb

        params = _make_simple_model()
        outpath = tmp_path / "model_auto.ewecsv.zip"
        # auto should always succeed (falls back to csv)
        write_ewemdb(params, str(outpath), backend="auto")
        assert outpath.exists()

    def test_write_access_backend_raises_without_odbc(self, tmp_path):
        """If ODBC not available, backend='access' should raise RuntimeError."""
        try:
            import pyodbc

            from pypath.io._access_writer import _find_access_driver

            _find_access_driver()
            pytest.skip("Access ODBC driver IS available")
        except (ImportError, RuntimeError):
            pass

        from pypath.io.ewe_writer import write_ewemdb

        params = _make_simple_model()
        outpath = tmp_path / "model.eweaccdb"
        with pytest.raises(RuntimeError):
            write_ewemdb(params, str(outpath), backend="access")

    def test_write_rejects_empty_model(self, tmp_path):
        from pypath.io.ewe_writer import write_ewemdb
        from pypath.core.params import create_rpath_params

        params = create_rpath_params(groups=[], types=[])
        with pytest.raises(ValueError, match="empty"):
            write_ewemdb(params, str(tmp_path / "empty.ewecsv.zip"), backend="csv")

    def test_csv_bundle_ecopath_roundtrip_values(self, tmp_path):
        """Write and read back CSV bundle, verify numeric values survive."""
        from pypath.io.ewe_writer import write_ewemdb

        params = _make_simple_model()
        outpath = tmp_path / "rt.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        with zipfile.ZipFile(outpath) as zf:
            groups = pd.read_csv(zf.open("EcopathGroup.csv"))
            diet = pd.read_csv(zf.open("EcopathDietComp.csv"))

        # Check biomass
        phyto = groups[groups["GroupName"] == "Phyto"].iloc[0]
        assert abs(phyto["Biomass"] - 10.0) < 1e-6
        assert abs(phyto["PB"] - 100.0) < 1e-6

        # Check diet
        assert len(diet) >= 3
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py::TestWriteEwemdb -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pypath.io.ewe_writer'`

**Step 3: Write minimal implementation**

Create `packages/pypath/src/pypath/io/ewe_writer.py`:

```python
"""EwE database writer for PyPath.

Export PyPath models back to native EwE format:
- .eweaccdb (Access database) via pyodbc when ODBC driver is available
- .ewecsv.zip (CSV bundle) as cross-platform fallback

Functions:
- write_ewemdb(params, path, ...): Write an EwE database file from RpathParams

Example:
    >>> from pypath.io.ewe_writer import write_ewemdb
    >>> write_ewemdb(params, "my_model.eweaccdb")

    >>> # With Ecosim scenarios
    >>> write_ewemdb(params, "model.eweaccdb", scenarios=[scenario1])

    >>> # Force CSV fallback
    >>> write_ewemdb(params, "model.ewecsv.zip", backend="csv")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pypath.core.ecosim import RsimScenario
    from pypath.core.params import RpathParams

logger = logging.getLogger(__name__)


def _odbc_available() -> bool:
    """Check if an Access ODBC driver is available."""
    try:
        from pypath.io._access_writer import _find_access_driver

        _find_access_driver()
        return True
    except (ImportError, RuntimeError):
        return False


def write_ewemdb(
    params: "RpathParams",
    path: str,
    *,
    scenarios: list[Any] | None = None,
    ecospace: Any | None = None,
    backend: str = "auto",
    scenario_id: int = 1,
) -> None:
    """Write an EwE database file from PyPath model parameters.

    Parameters
    ----------
    params : RpathParams
        Ecopath model parameters (from create_rpath_params or read_ewemdb).
    path : str
        Output file path. Extension determines format suggestion but
        ``backend`` takes precedence.
    scenarios : list of RsimScenario, optional
        Ecosim scenarios to include. If None, only Ecopath data is written.
    ecospace : EcospaceParams, optional
        Ecospace spatial parameters to include.
    backend : str
        Writer backend: ``"auto"`` (detect ODBC, fall back to CSV),
        ``"access"`` (require ODBC), or ``"csv"`` (always CSV bundle).
    scenario_id : int
        ScenarioID to use in all tables (default 1).

    Raises
    ------
    ValueError
        If params.model is empty.
    RuntimeError
        If ``backend="access"`` but no ODBC driver is available.
    FileNotFoundError
        If the blank template is missing (Access backend).

    Examples
    --------
    >>> from pypath.io.ewe_writer import write_ewemdb
    >>> write_ewemdb(params, "output.eweaccdb")

    >>> write_ewemdb(params, "output.ewecsv.zip", backend="csv")

    >>> write_ewemdb(params, "output.eweaccdb", scenarios=[scen1, scen2])
    """
    # Validate
    if params.model is None or len(params.model) == 0:
        raise ValueError("Cannot export empty model (params.model is empty)")

    path = str(Path(path).resolve())

    # Select backend
    if backend == "auto":
        backend = "access" if _odbc_available() else "csv"
        logger.info("Auto-detected backend: %s", backend)

    if backend == "access":
        from pypath.io._access_writer import AccessWriter

        writer = AccessWriter(params, path, scenario_id=scenario_id)
    elif backend == "csv":
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        writer = CsvBundleWriter(params, path, scenario_id=scenario_id)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'auto', 'access', or 'csv'.")

    try:
        writer.write_ecopath()
        writer.write_ecosim(scenarios)
        writer.write_ecospace(ecospace)
        writer.close()
    except Exception:
        # Clean up temp file on failure
        import os

        tmp = getattr(writer, "_tmp_path", None)
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        raise
```

**Step 4: Update `__init__.py` to export write_ewemdb**

In `packages/pypath/src/pypath/io/__init__.py`, add:

```python
from pypath.io.ewe_writer import write_ewemdb
```

And add `"write_ewemdb"` to the `__all__` list under the `# EwE database` section.

**Step 5: Run tests to verify they pass**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py::TestWriteEwemdb -v`
Expected: PASS

**Step 6: Commit**

```bash
git add packages/pypath/src/pypath/io/ewe_writer.py packages/pypath/src/pypath/io/__init__.py packages/pypath/tests/test_ewe_writer.py
git commit -m "feat(io): add write_ewemdb() public entry point for EwE export"
```

---

### Task 6: Integration Round-Trip Test

**Files:**
- Create: `packages/pypath/tests/test_ewe_writer_roundtrip.py`

**Step 1: Write the failing test**

Create `packages/pypath/tests/test_ewe_writer_roundtrip.py`:

```python
"""Integration tests for EwE export round-trip.

These tests read a real EwE database, write it back, and verify values match.
Requires: Access ODBC driver + LT2022 test database.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def lt2022_path():
    """Path to the LT2022 test database."""
    from pathlib import Path

    p = Path(__file__).parent / "data" / "LT2022.eweaccdb"
    if not p.exists():
        pytest.skip("LT2022.eweaccdb not found in test data")
    return str(p)


class TestEcopathRoundTrip:
    """Read LT2022 → write → read back → compare Ecopath values."""

    def test_group_names_survive(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_test.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        import json
        import zipfile

        import pandas as pd

        with zipfile.ZipFile(outpath) as zf:
            groups = pd.read_csv(zf.open("EcopathGroup.csv"))

        original_names = params.model[params.model["Type"] != 3]["Group"].tolist()
        exported_names = groups["GroupName"].tolist()
        assert exported_names == original_names

    def test_biomass_values_match(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        import zipfile

        import pandas as pd

        params = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_test.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        with zipfile.ZipFile(outpath) as zf:
            groups = pd.read_csv(zf.open("EcopathGroup.csv"))

        bio_mask = params.model["Type"] != 3
        original_biomass = params.model[bio_mask]["Biomass"].values
        exported_biomass = groups["Biomass"].values

        # Compare non-NaN values
        for orig, exp in zip(original_biomass, exported_biomass):
            if np.isnan(orig):
                assert np.isnan(exp)
            else:
                np.testing.assert_allclose(exp, orig, rtol=1e-6)

    def test_diet_values_match(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        import zipfile

        import pandas as pd

        params = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_test.ewecsv.zip"
        write_ewemdb(params, str(outpath), backend="csv")

        with zipfile.ZipFile(outpath) as zf:
            diet = pd.read_csv(zf.open("EcopathDietComp.csv"))

        # Count non-zero diet entries in original
        original_nonzero = 0
        for col in params.diet.columns[1:]:  # skip 'Group' column
            original_nonzero += (params.diet[col] > 0).sum()

        assert len(diet) == original_nonzero

    @pytest.fixture(autouse=True)
    def _skip_no_odbc_for_access(self, request):
        """For Access-specific tests, skip if no ODBC."""
        if "access" in request.node.name:
            try:
                from pypath.io._access_writer import AccessWriter

                AccessWriter._check_odbc()
            except (ImportError, RuntimeError):
                pytest.skip("Access ODBC driver not available")


class TestAccessRoundTrip:
    """Full round-trip: read LT2022 → write Access → read back."""

    @pytest.fixture(autouse=True)
    def _skip_no_odbc(self):
        try:
            from pypath.io._access_writer import AccessWriter

            AccessWriter._check_odbc()
        except (ImportError, RuntimeError):
            pytest.skip("Access ODBC driver not available")

    def test_access_full_roundtrip(self, lt2022_path, tmp_path):
        from pypath.io.ewemdb import read_ewemdb
        from pypath.io.ewe_writer import write_ewemdb

        params_orig = read_ewemdb(lt2022_path)
        outpath = tmp_path / "rt_access.eweaccdb"
        write_ewemdb(params_orig, str(outpath), backend="access")

        params_back = read_ewemdb(str(outpath))

        # Compare group names
        orig_names = params_orig.model[params_orig.model["Type"] != 3][
            "Group"
        ].tolist()
        back_names = params_back.model[params_back.model["Type"] != 3][
            "Group"
        ].tolist()
        assert back_names == orig_names

        # Compare biomass
        bio_orig = params_orig.model[params_orig.model["Type"] != 3][
            "Biomass"
        ].values
        bio_back = params_back.model[params_back.model["Type"] != 3][
            "Biomass"
        ].values
        for o, b in zip(bio_orig, bio_back):
            if np.isnan(o):
                assert np.isnan(b)
            else:
                np.testing.assert_allclose(b, o, rtol=1e-6)
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer_roundtrip.py -v -m integration`
Expected: FAIL (modules not yet wired, or pass if Tasks 1-5 are done)

**Step 3: No implementation needed** (all code was written in Tasks 1-5)

**Step 4: Run tests**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer_roundtrip.py -v -m integration`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/pypath/tests/test_ewe_writer_roundtrip.py
git commit -m "test(io): add integration round-trip tests for EwE export"
```

---

### Task 7: Add pyodbc Optional Dependency + Update Docs

**Files:**
- Modify: `packages/pypath/pyproject.toml`
- Modify: `packages/pypath/CHANGELOG.md`
- Modify: `packages/pypath/docs/docs/api/io.md`

**Step 1: Update pyproject.toml**

Add a new optional dependency group in `packages/pypath/pyproject.toml`:

```toml
ewedb = [
    "pyodbc>=4.0",
]
```

Update the `all` extra to include `ewedb`:

```toml
all = [
    "pypath-ewe[interactive,spatial,spatial-data,biodata,numba,ewedb,docs,dev]",
]
```

**Step 2: Update CHANGELOG.md**

Add under the v0.3.1 section in `packages/pypath/CHANGELOG.md`:

```markdown
- **EwE Export**: New `write_ewemdb()` function exports PyPath models back to
  native EwE `.eweaccdb` format (Access via pyodbc) or `.ewecsv.zip` (CSV
  bundle fallback for cross-platform use). Supports Ecopath groups, diet,
  fleets, stanzas, Ecosim scenarios, and Ecospace spatial data.
```

**Step 3: Update API docs**

Add to `packages/pypath/docs/docs/api/io.md`:

```markdown
### EwE Database Export

Export PyPath models back to native EwE format:

- `write_ewemdb(params, path)` — Auto-detects best backend
- `write_ewemdb(params, path, backend="csv")` — Force CSV bundle
- `write_ewemdb(params, path, scenarios=[scen1])` — Include Ecosim
- `write_ewemdb(params, path, ecospace=ecospace)` — Include Ecospace

```python
from pypath.io.ewe_writer import write_ewemdb

# Export Ecopath model
write_ewemdb(params, "my_model.eweaccdb")

# With Ecosim scenarios
write_ewemdb(params, "my_model.eweaccdb", scenarios=[scenario1])

# Cross-platform CSV fallback
write_ewemdb(params, "my_model.ewecsv.zip", backend="csv")
```
```

**Step 4: Update conda recipe test imports**

In `conda-recipes/pypath-ewe/meta.yaml`, add under test imports:

```yaml
    - pypath.io.ewe_writer
```

**Step 5: Commit**

```bash
git add packages/pypath/pyproject.toml packages/pypath/CHANGELOG.md packages/pypath/docs/docs/api/io.md conda-recipes/pypath-ewe/meta.yaml
git commit -m "docs(io): add EwE export docs, ewedb optional dependency, update changelog"
```

---

### Task 8: Run Full Test Suite

**Step 1: Run all writer tests**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer.py -v --tb=short`
Expected: All TestEweSchema, TestCsvBundleWriter, TestWriteEwemdb tests PASS; TestAccessWriter tests PASS or SKIP

**Step 2: Run integration round-trip tests**

Run: `python -m pytest packages/pypath/tests/test_ewe_writer_roundtrip.py -v -m integration --tb=short`
Expected: PASS or SKIP depending on ODBC availability

**Step 3: Run existing test suite to check for regressions**

Run: `python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts --tb=short`
Expected: 937+ passed (same as before, no regressions)

**Step 4: Commit** (only if any test fixes were needed)

```bash
git commit -m "fix(io): address test feedback for EwE export"
```
