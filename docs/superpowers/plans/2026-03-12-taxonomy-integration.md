# Taxonomy & External Database Integration — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read, write, and auto-populate EwE taxonomy tables (EcopathTaxon, EcopathGroupTaxon, EcopathStanzaTaxon) with external database keys and species-to-group mappings.

**Architecture:** Extend 6 existing IO modules — schema in `_ewe_schema.py`, dataclasses + reader in `ewemdb.py`, writer logic in `_csv_bundle_writer.py` + `_access_writer.py` + `ewe_writer.py`, auto-populate in `biodata.py`, exports in `__init__.py`. One new test file.

**Tech Stack:** Python dataclasses, pandas DataFrames, pyodbc (Access), pyworms (WoRMS API)

**Spec:** `docs/superpowers/specs/2026-03-12-taxonomy-integration-design.md`

---

## File Structure

### New files
| File | Purpose |
|------|---------|
| `packages/pypath/tests/test_taxonomy.py` | ~15 unit tests |

### Modified files
| File | Change |
|------|--------|
| `packages/pypath/src/pypath/io/_ewe_schema.py` | Add 3 taxonomy tables to `EWE_TABLES` |
| `packages/pypath/src/pypath/io/ewemdb.py` | Add `TaxonomyRecord`, `TaxonomyData` dataclasses + `read_taxonomy()` |
| `packages/pypath/src/pypath/io/_csv_bundle_writer.py` | Add `write_taxonomy()` method |
| `packages/pypath/src/pypath/io/_access_writer.py` | Add `write_taxonomy()` method, update `_ECOPATH_TABLES` |
| `packages/pypath/src/pypath/io/ewe_writer.py` | Add `taxonomy` parameter to `write_ewemdb()` |
| `packages/pypath/src/pypath/io/biodata.py` | Add `auto_populate_taxonomy()` |
| `packages/pypath/src/pypath/io/__init__.py` | Export new types and functions |

---

## Chunk 1: Schema, Data Types, Reader

### Task 1: Schema additions (`_ewe_schema.py`)

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py` (after line 536, before `RPATH_TO_EWE_COLUMNS`)
- Test: `packages/pypath/tests/test_taxonomy.py`

- [ ] **Step 1: Write failing tests for schema**

Create `packages/pypath/tests/test_taxonomy.py`:

```python
"""Tests for taxonomy table read/write/auto-populate."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import field

from pypath.io._ewe_schema import EWE_TABLES


class TestSchema:
    """Schema definition tests."""

    def test_taxonomy_tables_exist(self):
        """All 3 taxonomy tables exist in EWE_TABLES."""
        assert "EcopathTaxon" in EWE_TABLES
        assert "EcopathGroupTaxon" in EWE_TABLES
        assert "EcopathStanzaTaxon" in EWE_TABLES

    def test_ecopath_taxon_columns(self):
        """EcopathTaxon has all 26 expected columns with correct types."""
        cols = EWE_TABLES["EcopathTaxon"]
        assert cols["TaxonID"] == "INTEGER"
        assert cols["ClassName"] == "TEXT"
        assert cols["OrderName"] == "TEXT"
        assert cols["FamilyName"] == "TEXT"
        assert cols["GenusName"] == "TEXT"
        assert cols["SpeciesName"] == "TEXT"
        assert cols["CommonName"] == "TEXT"
        assert cols["CodeAphia"] == "INTEGER"
        assert cols["CodeFB"] == "INTEGER"
        assert cols["CodeOBIS"] == "INTEGER"
        assert cols["VulnerabiltyIndex"] == "DOUBLE"  # EwE typo preserved
        assert cols["Winf"] == "DOUBLE"
        assert cols["vbgfK"] == "DOUBLE"
        assert len(cols) == 31
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/pypath && python -m pytest tests/test_taxonomy.py::TestSchema -v`
Expected: FAIL — `KeyError: 'EcopathTaxon'`

- [ ] **Step 3: Add 3 taxonomy tables to `_ewe_schema.py`**

Insert before the closing `}` of `EWE_TABLES` (after the Ecotracer tables, before line 536):

```python
    # -------------------------------------------------------------------
    # Taxonomy tables
    # -------------------------------------------------------------------
    "EcopathTaxon": OrderedDict([
        ("TaxonID", "INTEGER"),
        ("ClassName", "TEXT"),
        ("OrderName", "TEXT"),
        ("FamilyName", "TEXT"),
        ("GenusName", "TEXT"),
        ("SpeciesName", "TEXT"),
        ("CommonName", "TEXT"),
        ("SourceName", "TEXT"),
        ("SourceKey", "TEXT"),
        ("LastUpdated", "DOUBLE"),
        ("EcologyType", "INTEGER"),
        ("OrganismType", "INTEGER"),
        ("Exploited", "INTEGER"),
        ("ConservationStatus", "INTEGER"),
        ("OccurrenceStatus", "INTEGER"),
        ("MeanWeight", "DOUBLE"),
        ("MeanLength", "DOUBLE"),
        ("MaxLength", "DOUBLE"),
        ("MeanLifeSpan", "DOUBLE"),
        ("VulnerabiltyIndex", "DOUBLE"),
        ("CodeSAUP", "INTEGER"),
        ("CodeFB", "INTEGER"),
        ("CodeSLB", "INTEGER"),
        ("CodeLCID", "TEXT"),
        ("CodeFAO", "TEXT"),
        ("Winf", "DOUBLE"),
        ("vbgfK", "DOUBLE"),
        ("ExploitationStatus", "TEXT"),
        ("CodeAquaMaps", "TEXT"),
        ("CodeAphia", "INTEGER"),
        ("CodeOBIS", "INTEGER"),
    ]),
    "EcopathGroupTaxon": OrderedDict([
        ("TaxonID", "INTEGER"),
        ("EcopathGroupID", "INTEGER"),
        ("Proportion", "DOUBLE"),
        ("PropCatch", "DOUBLE"),
    ]),
    "EcopathStanzaTaxon": OrderedDict([
        ("TaxonID", "INTEGER"),
        ("StanzaID", "INTEGER"),
    ]),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/pypath && python -m pytest tests/test_taxonomy.py::TestSchema -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_taxonomy.py
git commit -m "feat(schema): add EcopathTaxon, EcopathGroupTaxon, EcopathStanzaTaxon tables"
```

---

### Task 2: Data types and reader (`ewemdb.py`)

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py` (add dataclasses near top, function at end)
- Test: `packages/pypath/tests/test_taxonomy.py`

- [ ] **Step 1: Write failing tests for dataclasses and reader**

Append to `test_taxonomy.py`:

```python
from pypath.io.ewemdb import TaxonomyRecord, TaxonomyData, read_taxonomy


def _make_taxon_row():
    """Build a dict mimicking one row from EcopathTaxon table."""
    return {
        "TaxonID": 1,
        "ClassName": "Actinopteri",
        "OrderName": "Gadiformes",
        "FamilyName": "Gadidae",
        "GenusName": "Gadus",
        "SpeciesName": "morhua",
        "CommonName": "Atlantic cod",
        "SourceName": "PyPath-biodata",
        "SourceKey": "126436",
        "LastUpdated": 0.0,
        "EcologyType": -9999,
        "OrganismType": -9999,
        "Exploited": -9999,
        "ConservationStatus": -9999,
        "OccurrenceStatus": -9999,
        "MeanWeight": -9999.0,
        "MeanLength": 50.0,
        "MaxLength": 200.0,
        "MeanLifeSpan": -9999.0,
        "VulnerabiltyIndex": -9999.0,
        "CodeSAUP": -9999,
        "CodeFB": 69,
        "CodeSLB": -9999,
        "CodeLCID": "",
        "CodeFAO": "",
        "Winf": 15000.0,
        "vbgfK": 0.15,
        "ExploitationStatus": "",
        "CodeAquaMaps": "",
        "CodeAphia": 126436,
        "CodeOBIS": -9999,
    }


class TestReader:
    """read_taxonomy() tests."""

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_reads_taxon_records(self, mock_read, mock_tables):
        """Reads EcopathTaxon rows into TaxonomyRecord list."""
        mock_tables.return_value = [
            "EcopathTaxon", "EcopathGroupTaxon", "EcopathStanzaTaxon"
        ]
        row = _make_taxon_row()
        mock_read.side_effect = lambda db, table: {
            "EcopathTaxon": pd.DataFrame([row]),
            "EcopathGroupTaxon": pd.DataFrame(
                columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
            ),
            "EcopathStanzaTaxon": pd.DataFrame(
                columns=["TaxonID", "StanzaID"]
            ),
        }[table]

        result = read_taxonomy("fake.eweaccdb")
        assert len(result.taxa) == 1
        t = result.taxa[0]
        assert t.taxon_id == 1
        assert t.scientific_name == "Gadus morhua"
        assert t.common_name == "Atlantic cod"
        assert t.taxonomy["class_name"] == "Actinopteri"
        assert t.taxonomy["genus_name"] == "Gadus"
        assert t.external_keys["aphia_id"] == 126436
        assert t.external_keys["fishbase_code"] == 69
        assert t.source_name == "PyPath-biodata"

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_reads_group_taxon_dataframe(self, mock_read, mock_tables):
        """Reads EcopathGroupTaxon into DataFrame."""
        mock_tables.return_value = [
            "EcopathTaxon", "EcopathGroupTaxon", "EcopathStanzaTaxon"
        ]
        gt_df = pd.DataFrame([
            {"TaxonID": 1, "EcopathGroupID": 3, "Proportion": 0.5, "PropCatch": 0.5},
            {"TaxonID": 2, "EcopathGroupID": 3, "Proportion": 0.5, "PropCatch": 0.5},
        ])
        mock_read.side_effect = lambda db, table: {
            "EcopathTaxon": pd.DataFrame(columns=list(_make_taxon_row().keys())),
            "EcopathGroupTaxon": gt_df,
            "EcopathStanzaTaxon": pd.DataFrame(columns=["TaxonID", "StanzaID"]),
        }[table]

        result = read_taxonomy("fake.eweaccdb")
        assert len(result.group_assignments) == 2
        assert list(result.group_assignments.columns) == [
            "TaxonID", "EcopathGroupID", "Proportion", "PropCatch"
        ]

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_reads_stanza_taxon_dataframe(self, mock_read, mock_tables):
        """Reads EcopathStanzaTaxon into DataFrame."""
        mock_tables.return_value = [
            "EcopathTaxon", "EcopathGroupTaxon", "EcopathStanzaTaxon"
        ]
        st_df = pd.DataFrame([{"TaxonID": 1, "StanzaID": 1}])
        mock_read.side_effect = lambda db, table: {
            "EcopathTaxon": pd.DataFrame(columns=list(_make_taxon_row().keys())),
            "EcopathGroupTaxon": pd.DataFrame(
                columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
            ),
            "EcopathStanzaTaxon": st_df,
        }[table]

        result = read_taxonomy("fake.eweaccdb")
        assert len(result.stanza_assignments) == 1

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    def test_missing_tables_return_empty(self, mock_tables):
        """Missing tables return empty defaults, not errors."""
        mock_tables.return_value = ["EcopathGroup"]  # no taxonomy tables
        result = read_taxonomy("fake.eweaccdb")
        assert result.taxa == []
        assert len(result.group_assignments) == 0
        assert list(result.group_assignments.columns) == [
            "TaxonID", "EcopathGroupID", "Proportion", "PropCatch"
        ]
        assert len(result.stanza_assignments) == 0

    @patch("pypath.io.ewemdb.list_ewemdb_tables")
    @patch("pypath.io.ewemdb.read_ewemdb_table")
    def test_sentinel_values_converted_to_none(self, mock_read, mock_tables):
        """-9999 sentinel values are converted to None in traits and metadata."""
        mock_tables.return_value = ["EcopathTaxon"]
        row = _make_taxon_row()
        mock_read.side_effect = lambda db, table: pd.DataFrame([row])

        result = read_taxonomy("fake.eweaccdb")
        t = result.taxa[0]
        # Traits with -9999 should be None
        assert t.traits["mean_weight"] is None
        assert t.traits["vulnerability_index"] is None
        # Traits with real values should be kept
        assert t.traits["mean_length"] == 50.0
        assert t.traits["winf"] == 15000.0
        # Metadata with -9999 should be None
        assert t.metadata["ecology_type"] is None
        assert t.metadata["organism_type"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/pypath && python -m pytest tests/test_taxonomy.py::TestReader -v`
Expected: FAIL — `ImportError: cannot import name 'TaxonomyRecord'`

- [ ] **Step 3: Implement dataclasses and `read_taxonomy()`**

Add to `ewemdb.py`, after the imports (around line 46):

```python
dataclasses import field` (modify existing line 35: `from dataclasses import dataclass` → `from dataclasses import dataclass, field`)
```

Add the dataclasses after the `_validate_sql_identifier` function (around line 90, after the module-level setup):

```python
@dataclass
class TaxonomyRecord:
    """A single species/taxon entry from EcopathTaxon."""

    taxon_id: int
    scientific_name: str
    common_name: str
    taxonomy: dict
    external_keys: dict
    traits: dict
    metadata: dict = field(default_factory=dict)
    source_name: str = ""
    source_key: str = ""


@dataclass
class TaxonomyData:
    """Complete taxonomy data from an EwE model."""

    taxa: list
    group_assignments: "pd.DataFrame"
    stanza_assignments: "pd.DataFrame"
```

Add at the end of the file:

```python
# Column name → (dict_name, key) mapping for TaxonomyRecord construction
_TAXON_EXTERNAL_KEYS = {
    "CodeAphia": "aphia_id",
    "CodeFB": "fishbase_code",
    "CodeSLB": "sealifebase_code",
    "CodeOBIS": "obis_code",
    "CodeSAUP": "saup_code",
    "CodeFAO": "fao_code",
    "CodeAquaMaps": "aquamaps_code",
    "CodeLCID": "lsid",
}

_TAXON_TRAITS = {
    "Winf": "winf",
    "vbgfK": "vbgf_k",
    "MeanWeight": "mean_weight",
    "MeanLength": "mean_length",
    "MaxLength": "max_length",
    "MeanLifeSpan": "mean_lifespan",
    "VulnerabiltyIndex": "vulnerability_index",
}

_TAXON_METADATA = {
    "EcologyType": "ecology_type",
    "OrganismType": "organism_type",
    "Exploited": "exploited",
    "ConservationStatus": "conservation_status",
    "OccurrenceStatus": "occurrence_status",
    "ExploitationStatus": "exploitation_status",
    "LastUpdated": "last_updated",
}


def _sentinel_to_none(value, sentinel=-9999):
    """Convert EwE sentinel values to None."""
    if isinstance(value, (int, float)) and value == sentinel:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _none_to_sentinel(value, sql_type, sentinel=-9999):
    """Convert None back to EwE sentinel for writing."""
    if value is None:
        return "" if sql_type == "TEXT" else sentinel
    return value


def read_taxonomy(db_path: str) -> TaxonomyData:
    """Read taxonomy tables from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the .eweaccdb database file.

    Returns
    -------
    TaxonomyData
        Taxonomy records, group assignments, and stanza assignments.
        Empty defaults if tables are missing.
    """
    try:
        tables = list_ewemdb_tables(db_path)
    except Exception:
        return TaxonomyData(
            taxa=[],
            group_assignments=pd.DataFrame(
                columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
            ),
            stanza_assignments=pd.DataFrame(columns=["TaxonID", "StanzaID"]),
        )

    # Read EcopathTaxon
    taxa = []
    if "EcopathTaxon" in tables:
        try:
            df = read_ewemdb_table(db_path, "EcopathTaxon")
            for _, row in df.iterrows():
                genus = str(row.get("GenusName", "") or "").strip()
                species = str(row.get("SpeciesName", "") or "").strip()
                sci_name = f"{genus} {species}".strip()

                taxonomy = {
                    "class_name": str(row.get("ClassName", "") or "").strip(),
                    "order_name": str(row.get("OrderName", "") or "").strip(),
                    "family_name": str(row.get("FamilyName", "") or "").strip(),
                    "genus_name": genus,
                    "species_name": species,
                }

                external_keys = {}
                for col, key in _TAXON_EXTERNAL_KEYS.items():
                    val = row.get(col)
                    external_keys[key] = _sentinel_to_none(val)

                traits = {}
                for col, key in _TAXON_TRAITS.items():
                    val = row.get(col)
                    traits[key] = _sentinel_to_none(val)

                metadata = {}
                for col, key in _TAXON_METADATA.items():
                    val = row.get(col)
                    metadata[key] = _sentinel_to_none(val)

                taxa.append(TaxonomyRecord(
                    taxon_id=int(row["TaxonID"]),
                    scientific_name=sci_name,
                    common_name=str(row.get("CommonName", "") or "").strip(),
                    taxonomy=taxonomy,
                    external_keys=external_keys,
                    traits=traits,
                    metadata=metadata,
                    source_name=str(row.get("SourceName", "") or "").strip(),
                    source_key=str(row.get("SourceKey", "") or "").strip(),
                ))
        except Exception as e:
            logger.warning("Failed to read EcopathTaxon: %s", e)

    # Read EcopathGroupTaxon
    group_cols = ["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
    if "EcopathGroupTaxon" in tables:
        try:
            group_assignments = read_ewemdb_table(db_path, "EcopathGroupTaxon")
        except Exception:
            group_assignments = pd.DataFrame(columns=group_cols)
    else:
        group_assignments = pd.DataFrame(columns=group_cols)

    # Read EcopathStanzaTaxon
    stanza_cols = ["TaxonID", "StanzaID"]
    if "EcopathStanzaTaxon" in tables:
        try:
            stanza_assignments = read_ewemdb_table(db_path, "EcopathStanzaTaxon")
        except Exception:
            stanza_assignments = pd.DataFrame(columns=stanza_cols)
    else:
        stanza_assignments = pd.DataFrame(columns=stanza_cols)

    return TaxonomyData(
        taxa=taxa,
        group_assignments=group_assignments,
        stanza_assignments=stanza_assignments,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/pypath && python -m pytest tests/test_taxonomy.py -v`
Expected: PASS (7 tests — 2 schema + 5 reader)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py packages/pypath/tests/test_taxonomy.py
git commit -m "feat(io): add TaxonomyRecord, TaxonomyData, read_taxonomy()"
```

---

## Chunk 2: Writer

### Task 3: CsvBundleWriter.write_taxonomy() + AccessWriter.write_taxonomy()

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py` (add method at end of class)
- Modify: `packages/pypath/src/pypath/io/_access_writer.py` (add method + update `_ECOPATH_TABLES`)
- Modify: `packages/pypath/src/pypath/io/ewe_writer.py` (add `taxonomy` param)
- Test: `packages/pypath/tests/test_taxonomy.py`

- [ ] **Step 1: Write failing tests for writer**

Append to `test_taxonomy.py`:

```python
class TestWriter:
    """write_taxonomy() tests."""

    def _make_taxonomy_data(self):
        """Build a small TaxonomyData for testing."""
        taxa = [
            TaxonomyRecord(
                taxon_id=1,
                scientific_name="Gadus morhua",
                common_name="Atlantic cod",
                taxonomy={
                    "class_name": "Actinopteri",
                    "order_name": "Gadiformes",
                    "family_name": "Gadidae",
                    "genus_name": "Gadus",
                    "species_name": "morhua",
                },
                external_keys={"aphia_id": 126436, "fishbase_code": 69},
                traits={"winf": 15000.0, "vbgf_k": 0.15, "mean_weight": None},
                metadata={"ecology_type": None},
                source_name="PyPath-biodata",
                source_key="126436",
            ),
        ]
        group_assignments = pd.DataFrame([
            {"TaxonID": 1, "EcopathGroupID": 3, "Proportion": 1.0, "PropCatch": 1.0},
        ])
        stanza_assignments = pd.DataFrame(columns=["TaxonID", "StanzaID"])
        return TaxonomyData(taxa, group_assignments, stanza_assignments)

    def test_csv_writer_builds_tables(self):
        """CsvBundleWriter.write_taxonomy() builds correct table dicts."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        writer = CsvBundleWriter.__new__(CsvBundleWriter)
        writer._params = None
        writer._scenario_id = 1
        writer._tables = {}

        taxonomy = self._make_taxonomy_data()
        writer.write_taxonomy(taxonomy=taxonomy)

        assert "EcopathTaxon" in writer._tables
        assert "EcopathGroupTaxon" in writer._tables
        assert "EcopathStanzaTaxon" in writer._tables

        taxon_df = writer._tables["EcopathTaxon"]
        assert len(taxon_df) == 1
        assert taxon_df.iloc[0]["GenusName"] == "Gadus"
        assert taxon_df.iloc[0]["SpeciesName"] == "morhua"
        assert taxon_df.iloc[0]["CodeAphia"] == 126436
        # None traits should be written as -9999
        assert taxon_df.iloc[0]["MeanWeight"] == -9999

    def test_round_trip(self):
        """Write then read back produces equivalent data."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        taxonomy = self._make_taxonomy_data()

        # Write
        writer = CsvBundleWriter.__new__(CsvBundleWriter)
        writer._params = None
        writer._scenario_id = 1
        writer._tables = {}
        writer.write_taxonomy(taxonomy=taxonomy)

        # Simulate read from the written tables
        with patch("pypath.io.ewemdb.list_ewemdb_tables") as mock_tables, \
             patch("pypath.io.ewemdb.read_ewemdb_table") as mock_read:
            mock_tables.return_value = [
                "EcopathTaxon", "EcopathGroupTaxon", "EcopathStanzaTaxon"
            ]
            mock_read.side_effect = lambda db, table: writer._tables[table]

            result = read_taxonomy("fake.eweaccdb")

        assert len(result.taxa) == 1
        t = result.taxa[0]
        assert t.scientific_name == "Gadus morhua"
        assert t.external_keys["aphia_id"] == 126436
        assert t.traits["winf"] == 15000.0
        assert t.traits["mean_weight"] is None  # -9999 → None round-trip

    def test_empty_taxonomy_writes_empty_tables(self):
        """Empty TaxonomyData writes empty tables without error."""
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        writer = CsvBundleWriter.__new__(CsvBundleWriter)
        writer._params = None
        writer._scenario_id = 1
        writer._tables = {}

        empty = TaxonomyData(
            taxa=[],
            group_assignments=pd.DataFrame(
                columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
            ),
            stanza_assignments=pd.DataFrame(columns=["TaxonID", "StanzaID"]),
        )
        writer.write_taxonomy(taxonomy=empty)

        assert len(writer._tables["EcopathTaxon"]) == 0
        assert len(writer._tables["EcopathGroupTaxon"]) == 0
        assert len(writer._tables["EcopathStanzaTaxon"]) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/pypath && python -m pytest tests/test_taxonomy.py::TestWriter -v`
Expected: FAIL — `AttributeError: 'CsvBundleWriter' object has no attribute 'write_taxonomy'`

- [ ] **Step 3: Implement `CsvBundleWriter.write_taxonomy()`**

Add to `_csv_bundle_writer.py` at end of `CsvBundleWriter` class:

```python
    def write_taxonomy(self, taxonomy=None) -> None:
        """Write taxonomy tables to the CSV bundle.

        Parameters
        ----------
        taxonomy : TaxonomyData, optional
            Taxonomy data to write.
        """
        if taxonomy is None:
            return

        from pypath.io._ewe_schema import EWE_TABLES
        from pypath.io.ewemdb import (
            _TAXON_EXTERNAL_KEYS, _TAXON_TRAITS, _TAXON_METADATA,
            _none_to_sentinel,
        )

        taxon_schema = EWE_TABLES["EcopathTaxon"]
        key_to_col = {v: k for k, v in _TAXON_EXTERNAL_KEYS.items()}
        trait_to_col = {v: k for k, v in _TAXON_TRAITS.items()}
        meta_to_col = {v: k for k, v in _TAXON_METADATA.items()}

        # Build EcopathTaxon rows
        taxon_rows = []
        for t in taxonomy.taxa:
            row = {
                "TaxonID": t.taxon_id,
                "GenusName": t.taxonomy.get("genus_name", ""),
                "SpeciesName": t.taxonomy.get("species_name", ""),
                "ClassName": t.taxonomy.get("class_name", ""),
                "OrderName": t.taxonomy.get("order_name", ""),
                "FamilyName": t.taxonomy.get("family_name", ""),
                "CommonName": t.common_name,
                "SourceName": t.source_name,
                "SourceKey": t.source_key,
            }
            # External keys → columns
            for key, col in key_to_col.items():
                val = t.external_keys.get(key)
                sql_type = taxon_schema.get(col, "INTEGER")
                row[col] = _none_to_sentinel(val, sql_type)

            # Traits → columns
            for key, col in trait_to_col.items():
                val = t.traits.get(key)
                row[col] = _none_to_sentinel(val, "DOUBLE")

            # Metadata → columns
            for key, col in meta_to_col.items():
                val = t.metadata.get(key)
                sql_type = taxon_schema.get(col, "INTEGER")
                row[col] = _none_to_sentinel(val, sql_type)

            taxon_rows.append(row)

        self._tables["EcopathTaxon"] = pd.DataFrame(
            taxon_rows,
            columns=list(taxon_schema.keys()),
        ) if taxon_rows else pd.DataFrame(
            columns=list(taxon_schema.keys())
        )

        # Group and stanza assignment tables (already DataFrames)
        self._tables["EcopathGroupTaxon"] = taxonomy.group_assignments.copy()
        self._tables["EcopathStanzaTaxon"] = taxonomy.stanza_assignments.copy()
```

- [ ] **Step 4: Update `_ECOPATH_TABLES` in `_access_writer.py`**

In `_access_writer.py`, add `"EcopathGroupTaxon"` and `"EcopathTaxon"` to `_ECOPATH_TABLES` after the existing `"EcopathStanzaTaxon"` entry (line 99). The list becomes:

```python
    _ECOPATH_TABLES = [
        # Children first
        "EcopathGroupSample",
        "EcopathGroupCatchSample",
        "EcopathStanzaTaxon",
        "EcopathGroupTaxon",   # NEW
        "EcopathTaxon",        # NEW
        "EcopathDietComp",
        ...
    ]
```

- [ ] **Step 5: Add `AccessWriter.write_taxonomy()` method**

In `_access_writer.py`, add after `write_mediation()` (around line 710):

```python
    def write_taxonomy(self, taxonomy=None) -> None:
        """Write taxonomy tables to the Access database."""
        if taxonomy is None:
            return
        self._build_tables_via_csv_writer("write_taxonomy", taxonomy=taxonomy)
```

- [ ] **Step 6: Update `write_ewemdb()` in `ewe_writer.py`**

Add `taxonomy=None` parameter to function signature and add `writer.write_taxonomy(taxonomy)` call after `writer.write_mediation(mediation)` and before `writer.close()`.

In function signature (line 31), add after `mediation`:
```python
    taxonomy: Any | None = None,
```

In the docstring Parameters section, add after the `mediation` entry:
```
    taxonomy : TaxonomyData, optional
        Taxonomy species records and group assignments to include.
```

In dispatch block (after line 98), add:
```python
        writer.write_taxonomy(taxonomy)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd packages/pypath && python -m pytest tests/test_taxonomy.py -v`
Expected: PASS (10 tests — 2 schema + 5 reader + 3 writer)

- [ ] **Step 8: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py \
       packages/pypath/src/pypath/io/_access_writer.py \
       packages/pypath/src/pypath/io/ewe_writer.py \
       packages/pypath/tests/test_taxonomy.py
git commit -m "feat(io): add write_taxonomy() to CsvBundleWriter and AccessWriter"
```

---

## Chunk 3: Auto-populate and Exports

### Task 4: `auto_populate_taxonomy()` in `biodata.py`

**Files:**
- Modify: `packages/pypath/src/pypath/io/biodata.py` (add function at end)
- Test: `packages/pypath/tests/test_taxonomy.py`

- [ ] **Step 1: Write failing tests for auto-populate**

Append to `test_taxonomy.py`:

```python
from pypath.io.biodata import auto_populate_taxonomy, SpeciesInfo


def _mock_species_info(name="Atlantic cod"):
    """Build a mock SpeciesInfo."""
    return SpeciesInfo(
        common_name=name,
        scientific_name="Gadus morhua",
        aphia_id=126436,
        authority="Linnaeus, 1758",
        trophic_level=4.0,
        max_length=200.0,
        growth_params={"K": 0.15, "Loo": 132.0},
    )


def _mock_worms_record():
    """Build a mock WoRMS API record."""
    return {
        "AphiaID": 126436,
        "scientificname": "Gadus morhua",
        "class": "Actinopteri",
        "order": "Gadiformes",
        "family": "Gadidae",
        "genus": "Gadus",
    }


def _mock_rpath(group_names):
    """Build a minimal mock Rpath with Group array."""
    rpath = MagicMock()
    rpath.Group = np.array(group_names)
    return rpath


class TestAutoPopulate:
    """auto_populate_taxonomy() tests."""

    @patch("pypath.io.biodata._fetch_worms_accepted")
    @patch("pypath.io.biodata.get_species_info")
    def test_builds_taxonomy_data(self, mock_get, mock_worms):
        """Builds TaxonomyData with correct fields from species map."""
        mock_get.return_value = _mock_species_info()
        mock_worms.return_value = _mock_worms_record()
        rpath = _mock_rpath(["Phyto", "Zoo", "Cod", "Detritus"])

        result = auto_populate_taxonomy(
            rpath, {"Cod": ["Atlantic cod"]}
        )

        assert len(result.taxa) == 1
        t = result.taxa[0]
        assert t.taxon_id == 1
        assert t.scientific_name == "Gadus morhua"
        assert t.taxonomy["class_name"] == "Actinopteri"
        assert t.taxonomy["family_name"] == "Gadidae"
        assert t.external_keys["aphia_id"] == 126436
        assert t.traits["winf"] == 132.0
        assert t.traits["vbgf_k"] == 0.15
        assert t.source_name == "PyPath-biodata"

    @patch("pypath.io.biodata._fetch_worms_accepted")
    @patch("pypath.io.biodata.get_species_info")
    def test_multi_species_equal_proportion(self, mock_get, mock_worms):
        """Multi-species groups get equal Proportion (1/n)."""
        info1 = _mock_species_info("Atlantic cod")
        info2 = SpeciesInfo(
            common_name="Herring",
            scientific_name="Clupea harengus",
            aphia_id=126417,
            authority="Linnaeus, 1758",
        )
        mock_get.side_effect = lambda name, **kw: {
            "Atlantic cod": info1, "Herring": info2
        }[name]
        mock_worms.side_effect = lambda aid, **kw: {
            126436: _mock_worms_record(),
            126417: {"AphiaID": 126417, "scientificname": "Clupea harengus",
                     "class": "Actinopteri", "order": "Clupeiformes",
                     "family": "Clupeidae", "genus": "Clupea"},
        }[aid]
        rpath = _mock_rpath(["Fish", "Detritus"])

        result = auto_populate_taxonomy(
            rpath, {"Fish": ["Atlantic cod", "Herring"]}
        )

        assert len(result.group_assignments) == 2
        props = result.group_assignments["Proportion"].tolist()
        assert all(abs(p - 0.5) < 1e-10 for p in props)
        # EcopathGroupID should be 1 (Fish is at index 0, + 1)
        assert all(result.group_assignments["EcopathGroupID"] == 1)

    @patch("pypath.io.biodata._fetch_worms_accepted")
    @patch("pypath.io.biodata.get_species_info")
    def test_custom_proportions(self, mock_get, mock_worms):
        """Custom proportions are respected."""
        info1 = _mock_species_info("Atlantic cod")
        info2 = SpeciesInfo(
            common_name="Herring",
            scientific_name="Clupea harengus",
            aphia_id=126417,
            authority="Linnaeus, 1758",
        )
        mock_get.side_effect = lambda name, **kw: {
            "Atlantic cod": info1, "Herring": info2
        }[name]
        mock_worms.side_effect = lambda aid, **kw: {
            126436: _mock_worms_record(),
            126417: {"AphiaID": 126417, "scientificname": "Clupea harengus",
                     "class": "Actinopteri", "order": "Clupeiformes",
                     "family": "Clupeidae", "genus": "Clupea"},
        }[aid]
        rpath = _mock_rpath(["Fish", "Detritus"])

        result = auto_populate_taxonomy(
            rpath,
            {"Fish": ["Atlantic cod", "Herring"]},
            proportions={"Fish": [0.7, 0.3]},
        )

        props = result.group_assignments["Proportion"].tolist()
        assert abs(props[0] - 0.7) < 1e-10
        assert abs(props[1] - 0.3) < 1e-10

    @patch("pypath.io.biodata.get_species_info")
    def test_lookup_failure_logged_as_warning(self, mock_get, caplog):
        """Species lookup failure is logged as warning, others still processed."""
        mock_get.side_effect = Exception("API error")
        rpath = _mock_rpath(["Fish", "Detritus"])

        import logging
        with caplog.at_level(logging.WARNING, logger="pypath.io.biodata"):
            result = auto_populate_taxonomy(
                rpath, {"Fish": ["Unknown species"]}
            )

        assert len(result.taxa) == 0
        assert len(result.group_assignments) == 0
        assert "Species lookup failed" in caplog.text
        assert "Unknown species" in caplog.text

    @patch("pypath.io.biodata._fetch_worms_accepted")
    @patch("pypath.io.biodata.get_species_info")
    def test_group_id_lookup_correct(self, mock_get, mock_worms):
        """Group names mapped to correct 1-based EcopathGroupID."""
        mock_get.return_value = _mock_species_info()
        mock_worms.return_value = _mock_worms_record()
        # "Cod" is at index 2 → EcopathGroupID = 3
        rpath = _mock_rpath(["Phyto", "Zoo", "Cod", "Detritus"])

        result = auto_populate_taxonomy(
            rpath, {"Cod": ["Atlantic cod"]}
        )

        assert result.group_assignments.iloc[0]["EcopathGroupID"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/pypath && python -m pytest tests/test_taxonomy.py::TestAutoPopulate -v`
Expected: FAIL — `ImportError: cannot import name 'auto_populate_taxonomy'`

- [ ] **Step 3: Implement `auto_populate_taxonomy()`**

Add to end of `biodata.py`:

```python
def auto_populate_taxonomy(
    rpath,
    group_species_map: dict,
    proportions: dict = None,
) -> "TaxonomyData":
    """Auto-populate taxonomy data from species names using WoRMS/FishBase.

    Parameters
    ----------
    rpath : Rpath
        Balanced Ecopath model output (pypath.core.ecopath.Rpath).
        Used to look up group indices by name (rpath.Group).
    group_species_map : dict[str, list[str]]
        {group_name: [species_name, ...]}. Names can be common or scientific.
    proportions : dict[str, list[float]], optional
        {group_name: [proportion, ...]}. Must sum to 1.0 per group.
        If None, defaults to equal split (1/n).

    Returns
    -------
    TaxonomyData
        Populated taxonomy data with species records and group assignments.
    """
    from pypath.io.ewemdb import TaxonomyData, TaxonomyRecord

    from concurrent.futures import ThreadPoolExecutor

    # Collect unique species across all groups
    species_to_info = {}
    species_to_worms = {}

    all_species = []
    for names in group_species_map.values():
        all_species.extend(names)
    unique_species = list(dict.fromkeys(all_species))  # preserve order, dedup

    # Look up each species (parallel via ThreadPoolExecutor)
    def _fetch_one(name):
        try:
            info = get_species_info(name, include_occurrences=False)
        except Exception as e:
            logger.warning("Species lookup failed for %r: %s", name, e)
            return name, None, {}
        try:
            worms_record = _fetch_worms_accepted(info.aphia_id)
        except Exception as e:
            logger.warning("WoRMS classification lookup failed for %r: %s", name, e)
            worms_record = {}
        return name, info, worms_record

    with ThreadPoolExecutor(max_workers=5) as executor:
        for name, info, worms in executor.map(_fetch_one, unique_species):
            if info is not None:
                species_to_info[name] = info
                species_to_worms[name] = worms

    # Build TaxonomyRecords
    taxa = []
    name_to_taxon_id = {}
    taxon_counter = 0

    for name in unique_species:
        if name not in species_to_info:
            continue
        info = species_to_info[name]
        worms = species_to_worms.get(name, {})

        taxon_counter += 1
        name_to_taxon_id[name] = taxon_counter

        sci_name = info.scientific_name
        species_part = sci_name.split(" ", 1)[1] if " " in sci_name else ""

        taxonomy = {
            "class_name": str(worms.get("class", "") or ""),
            "order_name": str(worms.get("order", "") or ""),
            "family_name": str(worms.get("family", "") or ""),
            "genus_name": str(worms.get("genus", "") or ""),
            "species_name": species_part,
        }

        external_keys = {"aphia_id": info.aphia_id}

        traits = {
            "max_length": info.max_length,
            "winf": (
                info.growth_params.get("Loo")
                if info.growth_params else None
            ),
            "vbgf_k": (
                info.growth_params.get("K")
                if info.growth_params else None
            ),
            "mean_weight": None,
            "mean_length": None,
            "mean_lifespan": None,
            "vulnerability_index": None,
        }

        taxa.append(TaxonomyRecord(
            taxon_id=taxon_counter,
            scientific_name=sci_name,
            common_name=info.common_name,
            taxonomy=taxonomy,
            external_keys=external_keys,
            traits=traits,
            metadata={},
            source_name="PyPath-biodata",
            source_key=str(info.aphia_id),
        ))

    # Build group_assignments
    group_rows = []
    for group_name, species_names in group_species_map.items():
        # Look up EcopathGroupID (0-based index + 1)
        matches = np.where(rpath.Group == group_name)[0]
        if len(matches) == 0:
            logger.warning("Group %r not found in rpath.Group, skipping", group_name)
            continue
        group_id = int(matches[0]) + 1  # 0-based → 1-based

        # Filter to successfully looked-up species
        valid_names = [n for n in species_names if n in name_to_taxon_id]
        if not valid_names:
            continue

        n = len(valid_names)
        group_props = None
        if proportions and group_name in proportions:
            group_props = proportions[group_name]
        else:
            group_props = [1.0 / n] * n

        for i, name in enumerate(valid_names):
            prop = group_props[i] if i < len(group_props) else 1.0 / n
            group_rows.append({
                "TaxonID": name_to_taxon_id[name],
                "EcopathGroupID": group_id,
                "Proportion": prop,
                "PropCatch": prop,
            })

    group_assignments = pd.DataFrame(
        group_rows,
        columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"],
    ) if group_rows else pd.DataFrame(
        columns=["TaxonID", "EcopathGroupID", "Proportion", "PropCatch"]
    )

    stanza_assignments = pd.DataFrame(columns=["TaxonID", "StanzaID"])

    return TaxonomyData(
        taxa=taxa,
        group_assignments=group_assignments,
        stanza_assignments=stanza_assignments,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/pypath && python -m pytest tests/test_taxonomy.py -v`
Expected: PASS (15 tests — 2 schema + 5 reader + 3 writer + 5 auto-populate)

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/biodata.py packages/pypath/tests/test_taxonomy.py
git commit -m "feat(io): add auto_populate_taxonomy() with WoRMS classification"
```

---

### Task 5: Exports (`__init__.py`)

**Files:**
- Modify: `packages/pypath/src/pypath/io/__init__.py`

- [ ] **Step 1: Add imports and __all__ entries**

In `io/__init__.py`, add to the `ewemdb` import block (around line 35):

```python
    read_taxonomy,
    TaxonomyRecord,
    TaxonomyData,
```

Add to the `biodata` import block (around line 12):

```python
    auto_populate_taxonomy,
```

Add to `__all__` list — in the EwE database section (after line 94):

```python
    "read_taxonomy",
    "TaxonomyRecord",
    "TaxonomyData",
```

And in the Biodiversity databases section (after line 106):

```python
    "auto_populate_taxonomy",
```

- [ ] **Step 2: Run full test suite to verify no regressions**

Run: `cd packages/pypath && python -m pytest tests/ -q -m "not integration and not slow" --ignore=tests/scripts`
Expected: All tests pass (~1200+)

- [ ] **Step 3: Commit**

```bash
git add packages/pypath/src/pypath/io/__init__.py
git commit -m "feat(api): export taxonomy types and functions from io package"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run full test suite**

Run: `cd packages/pypath && python -m pytest tests/ -q -m "not integration and not slow" --ignore=tests/scripts`
Expected: All tests pass, no regressions

- [ ] **Step 2: Verify imports work**

Run: `cd packages/pypath && python -c "from pypath.io import TaxonomyRecord, TaxonomyData, read_taxonomy, auto_populate_taxonomy; print('All exports OK')" `
Expected: "All exports OK"
