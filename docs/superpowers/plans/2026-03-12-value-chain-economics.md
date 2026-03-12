# Value Chain Economics I/O Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add round-trip read/write support for EwE Value Chain Economics (21 `c`-prefix tables) using DataFrame passthrough.

**Architecture:** `ValueChainData` dataclass holds raw DataFrames per table. `read_value_chain()` reads from EwE database; writers pass DataFrames through unchanged. No runtime economics.

**Tech Stack:** Python 3.10+, pandas, dataclasses, pytest

**Spec:** `docs/superpowers/specs/2026-03-12-value-chain-economics-design.md`

---

## Chunk 1: Schema + DataClass + Reader

### Task 1: Add 21 value chain table schemas to `EWE_TABLES`

**Files:**
- Modify: `packages/pypath/src/pypath/io/_ewe_schema.py` (append after line 629, before closing `}`)

- [ ] **Step 1: Write the failing test**

Create `packages/pypath/tests/test_value_chain_io.py`:

```python
"""Tests for Value Chain Economics I/O."""

import pandas as pd
import pytest


class TestValueChainSchema:
    """Test that all 21 c-prefix tables are in the schema."""

    def test_all_value_chain_tables_in_schema(self):
        from pypath.io._ewe_schema import EWE_TABLES

        expected_tables = [
            "cOOPStorable", "cParameters", "cUnit", "cEconomicUnit",
            "cProducerUnit", "cProcessingUnit", "cDistributionUnit",
            "cWholesalerUnit", "cRetailerUnit", "cConsumerUnit",
            "cProducerDefault", "cProcessingDefault", "cDistributionDefault",
            "cWholesalerDefault", "cRetailerDefault", "cConsumerDefault",
            "cLink", "cLinkDefault", "cLinkLandings",
            "cFlowDiagram", "cFlowPosition",
        ]
        for table in expected_tables:
            assert table in EWE_TABLES, f"Missing table: {table}"

    def test_ceconomicunit_has_revenue_columns(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["cEconomicUnit"]
        assert "RevenueLocalDomestic" in cols
        assert "CostOperatingEquil" in cols
        assert "DBID" in cols

    def test_cproducerunit_has_fleet_link(self):
        from pypath.io._ewe_schema import EWE_TABLES

        cols = EWE_TABLES["cProducerUnit"]
        assert "EcopathFleetID" in cols
        assert "ObserverCost" in cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_value_chain_io.py -v`
Expected: FAIL — tables not in EWE_TABLES yet

- [ ] **Step 3: Add all 21 table definitions to `_ewe_schema.py`**

Append before the closing `}` of `EWE_TABLES` (after `EcopathStanzaTaxon`):

```python
    # -----------------------------------------------------------------------
    # Value Chain Economics tables (c-prefix, EwE Value Chain plugin)
    # -----------------------------------------------------------------------
    "cOOPStorable": OrderedDict([
        ("xCLASS_NAMEx", "TEXT"),
        ("DBID", "INTEGER"),
        ("AllowEvents", "YESNO"),
    ]),
    "cParameters": OrderedDict([
        ("EquilibriumEffortMin", "DOUBLE"),
        ("EquilibriumEffortMax", "DOUBLE"),
        ("EquilibriumEffortIncrement", "DOUBLE"),
        ("RunWithEcopath", "YESNO"),
        ("RunWithEcosim", "YESNO"),
        ("RunWithSearches", "YESNO"),
    ]),
    "cUnit": OrderedDict([
        ("Sequence", "INTEGER"),
        ("Name", "TEXT"),
        ("Nationality", "TEXT"),
        ("NameLocal", "TEXT"),
        ("DBID", "INTEGER"),
    ]),
    "cEconomicUnit": OrderedDict([
        ("DBID", "INTEGER"),
        ("RevenueLocalDomestic", "DOUBLE"),
        ("RevenueLocalExport", "DOUBLE"),
        ("RevenueForeignDomestic", "DOUBLE"),
        ("RevenueForeignExport", "DOUBLE"),
        ("CostOperating", "DOUBLE"),
        ("CostCapital", "DOUBLE"),
        ("CostLabour", "DOUBLE"),
        ("CostLabourForeign", "DOUBLE"),
        ("CostRawMaterial", "DOUBLE"),
        ("CostRawMaterialForeign", "DOUBLE"),
        ("CostIntermediate", "DOUBLE"),
        ("CostIntermediateForeign", "DOUBLE"),
        ("TaxDirect", "DOUBLE"),
        ("TaxIndirect", "DOUBLE"),
        ("TaxExport", "DOUBLE"),
        ("TaxImport", "DOUBLE"),
        ("SubsidyDirect", "DOUBLE"),
        ("SubsidyIndirect", "DOUBLE"),
        ("EmploymentDirect", "DOUBLE"),
        ("EmploymentIndirect", "DOUBLE"),
        ("DependentsDirect", "DOUBLE"),
        ("DependentsIndirect", "DOUBLE"),
        ("EmploymentDirectForeign", "DOUBLE"),
        ("EmploymentIndirectForeign", "DOUBLE"),
        ("DependentsDirectForeign", "DOUBLE"),
        ("DependentsIndirectForeign", "DOUBLE"),
        ("RevenueLocalDomesticEquil", "DOUBLE"),
        ("RevenueLocalExportEquil", "DOUBLE"),
        ("RevenueForeignDomesticEquil", "DOUBLE"),
        ("RevenueForeignExportEquil", "DOUBLE"),
        ("CostOperatingEquil", "DOUBLE"),
        ("CostCapitalEquil", "DOUBLE"),
        ("CostLabourEquil", "DOUBLE"),
        ("CostLabourForeignEquil", "DOUBLE"),
        ("CostRawMaterialEquil", "DOUBLE"),
        ("CostRawMaterialForeignEquil", "DOUBLE"),
        ("CostIntermediateEquil", "DOUBLE"),
        ("CostIntermediateForeignEquil", "DOUBLE"),
    ]),
    "cProducerUnit": OrderedDict([
        ("DBID", "INTEGER"),
        ("ObserverCost", "DOUBLE"),
        ("ObserverRate", "DOUBLE"),
        ("TicketProducts", "TEXT"),
        ("EcopathFleetID", "INTEGER"),
    ]),
    "cProcessingUnit": OrderedDict([
        ("DBID", "INTEGER"),
        ("AgriculturalProducts", "TEXT"),
        ("AgriculturalInput", "TEXT"),
    ]),
    "cDistributionUnit": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cWholesalerUnit": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cRetailerUnit": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cConsumerUnit": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cProducerDefault": OrderedDict([
        ("DBID", "INTEGER"),
        ("ObserverCost", "DOUBLE"),
        ("ObserverRate", "DOUBLE"),
        ("TicketProducts", "TEXT"),
        ("EcopathFleetID", "INTEGER"),
    ]),
    "cProcessingDefault": OrderedDict([
        ("DBID", "INTEGER"),
        ("AgriculturalProducts", "TEXT"),
        ("AgriculturalInput", "TEXT"),
    ]),
    "cDistributionDefault": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cWholesalerDefault": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cRetailerDefault": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cConsumerDefault": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cLink": OrderedDict([
        ("DBID", "INTEGER"),
    ]),
    "cLinkDefault": OrderedDict([
        ("LinkType", "INTEGER"),
        ("BiomassRatio", "DOUBLE"),
        ("ValuePerTon", "DOUBLE"),
        ("ValueRatio", "DOUBLE"),
    ]),
    "cLinkLandings": OrderedDict([
        ("EcopathGroupID", "INTEGER"),
        ("ValuePerTon", "DOUBLE"),
    ]),
    "cFlowDiagram": OrderedDict([
        ("DBID", "INTEGER"),
        ("Name", "TEXT"),
        ("Description", "TEXT"),
    ]),
    "cFlowPosition": OrderedDict([
        ("DBID", "INTEGER"),
        ("DiagramDBID", "INTEGER"),
        ("UnitDBID", "INTEGER"),
        ("X", "DOUBLE"),
        ("Y", "DOUBLE"),
        ("Width", "DOUBLE"),
        ("Height", "DOUBLE"),
    ]),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_value_chain_io.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_ewe_schema.py packages/pypath/tests/test_value_chain_io.py
git commit -m "feat(io): add 21 value chain table schemas to EWE_TABLES"
```

---

### Task 2: Add `ValueChainData` dataclass and `read_value_chain()` function

**Files:**
- Modify: `packages/pypath/src/pypath/io/ewemdb.py` (add dataclass near top, function at end)
- Modify: `packages/pypath/tests/test_value_chain_io.py` (add reader tests)

- [ ] **Step 1: Write the failing tests**

Append to `test_value_chain_io.py`:

```python
from unittest.mock import patch, MagicMock


def _make_sample_value_chain_dfs():
    """Create minimal sample DataFrames for all 21 tables."""
    return {
        "cOOPStorable": pd.DataFrame({
            "xCLASS_NAMEx": ["cProducerUnit", "cProcessingUnit"],
            "DBID": [1, 2],
            "AllowEvents": [True, False],
        }),
        "cParameters": pd.DataFrame({
            "EquilibriumEffortMin": [0.5],
            "EquilibriumEffortMax": [2.0],
            "EquilibriumEffortIncrement": [0.1],
            "RunWithEcopath": [True],
            "RunWithEcosim": [False],
            "RunWithSearches": [False],
        }),
        "cUnit": pd.DataFrame({
            "Sequence": [1, 2],
            "Name": ["Trawler Fleet", "Fish Processor"],
            "Nationality": ["LT", "LT"],
            "NameLocal": ["", ""],
            "DBID": [1, 2],
        }),
        "cEconomicUnit": pd.DataFrame({
            "DBID": [1, 2],
            "RevenueLocalDomestic": [100.0, 200.0],
            "CostOperating": [50.0, 80.0],
        }),
        "cProducerUnit": pd.DataFrame({
            "DBID": [1],
            "ObserverCost": [10.0],
            "ObserverRate": [0.5],
            "TicketProducts": [""],
            "EcopathFleetID": [1],
        }),
        "cProcessingUnit": pd.DataFrame({
            "DBID": [2],
            "AgriculturalProducts": [""],
            "AgriculturalInput": [""],
        }),
        "cLink": pd.DataFrame({"DBID": [1]}),
        "cLinkDefault": pd.DataFrame({
            "LinkType": [0],
            "BiomassRatio": [1.0],
            "ValuePerTon": [500.0],
            "ValueRatio": [1.0],
        }),
        "cLinkLandings": pd.DataFrame({
            "EcopathGroupID": [3],
            "ValuePerTon": [250.0],
        }),
    }


class TestValueChainReader:
    """Test read_value_chain() function."""

    def test_read_value_chain_returns_dataclass(self):
        from pypath.io.ewemdb import read_value_chain, ValueChainData, _VALUE_CHAIN_TABLES

        sample = _make_sample_value_chain_dfs()

        def _mock_read(db, tbl):
            if tbl in sample:
                return sample[tbl]
            return pd.DataFrame()

        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read):
            result = read_value_chain("fake.ewemdb")

        assert isinstance(result, ValueChainData)
        assert result.oop_storables is not None
        assert len(result.oop_storables) == 2
        assert result.producers is not None
        assert result.producers.iloc[0]["EcopathFleetID"] == 1

    def test_read_value_chain_empty_db_returns_none(self):
        from pypath.io.ewemdb import read_value_chain, EwEDatabaseError

        def _mock_read(db, tbl):
            raise EwEDatabaseError(f"Table {tbl} not found")

        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read):
            result = read_value_chain("fake.ewemdb")

        assert result is None

    def test_read_value_chain_partial_tables(self):
        from pypath.io.ewemdb import read_value_chain

        # Only cOOPStorable exists
        sample = {
            "cOOPStorable": pd.DataFrame({
                "xCLASS_NAMEx": ["cProducerUnit"],
                "DBID": [1],
                "AllowEvents": [True],
            }),
        }

        def _mock_read(db, tbl):
            if tbl in sample:
                return sample[tbl]
            return pd.DataFrame()

        with patch("pypath.io.ewemdb.read_ewemdb_table", side_effect=_mock_read):
            result = read_value_chain("fake.ewemdb")

        assert result is not None
        assert result.oop_storables is not None
        # Other fields should be None (empty DFs are treated as None)
        assert result.producers is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_value_chain_io.py::TestValueChainReader -v`
Expected: FAIL — `read_value_chain` not yet defined

- [ ] **Step 3: Add `ValueChainData` dataclass and `_VALUE_CHAIN_TABLES` to `ewemdb.py`**

Add near the top of `ewemdb.py`, after existing imports and before functions (around line 78, after `_MONTH_NAME_MAP`):

```python
@dataclass
class ValueChainData:
    """Raw EwE value chain tables as DataFrames for round-trip I/O."""

    parameters: pd.DataFrame | None = None
    units: pd.DataFrame | None = None
    economic_units: pd.DataFrame | None = None
    producers: pd.DataFrame | None = None
    processors: pd.DataFrame | None = None
    distributors: pd.DataFrame | None = None
    wholesalers: pd.DataFrame | None = None
    retailers: pd.DataFrame | None = None
    consumers: pd.DataFrame | None = None
    links: pd.DataFrame | None = None
    link_defaults: pd.DataFrame | None = None
    link_landings: pd.DataFrame | None = None
    oop_storables: pd.DataFrame | None = None
    producer_defaults: pd.DataFrame | None = None
    processing_defaults: pd.DataFrame | None = None
    distribution_defaults: pd.DataFrame | None = None
    wholesaler_defaults: pd.DataFrame | None = None
    retailer_defaults: pd.DataFrame | None = None
    consumer_defaults: pd.DataFrame | None = None
    flow_diagram: pd.DataFrame | None = None
    flow_positions: pd.DataFrame | None = None


_VALUE_CHAIN_TABLES = {
    "oop_storables": "cOOPStorable",
    "parameters": "cParameters",
    "units": "cUnit",
    "economic_units": "cEconomicUnit",
    "producers": "cProducerUnit",
    "processors": "cProcessingUnit",
    "distributors": "cDistributionUnit",
    "wholesalers": "cWholesalerUnit",
    "retailers": "cRetailerUnit",
    "consumers": "cConsumerUnit",
    "producer_defaults": "cProducerDefault",
    "processing_defaults": "cProcessingDefault",
    "distribution_defaults": "cDistributionDefault",
    "wholesaler_defaults": "cWholesalerDefault",
    "retailer_defaults": "cRetailerDefault",
    "consumer_defaults": "cConsumerDefault",
    "links": "cLink",
    "link_defaults": "cLinkDefault",
    "link_landings": "cLinkLandings",
    "flow_diagram": "cFlowDiagram",
    "flow_positions": "cFlowPosition",
}
```

- [ ] **Step 4: Add `read_value_chain()` function to `ewemdb.py`**

Add at the end of the file (after `read_taxonomy`):

```python
def read_value_chain(db_path: str) -> ValueChainData | None:
    """Read value chain economics tables from an EwE database.

    Parameters
    ----------
    db_path : str
        Path to the EwE database file.

    Returns
    -------
    ValueChainData or None
        Populated dataclass if value chain tables exist, None otherwise.
    """
    # Check if cOOPStorable exists (sentinel for value chain data)
    try:
        oop_df = read_ewemdb_table(db_path, "cOOPStorable")
        if oop_df is None or len(oop_df) == 0:
            return None
    except (EwEDatabaseError, Exception):
        return None

    fields: dict[str, pd.DataFrame | None] = {}
    for attr_name, table_name in _VALUE_CHAIN_TABLES.items():
        try:
            df = read_ewemdb_table(db_path, table_name)
            fields[attr_name] = df if df is not None and len(df) > 0 else None
        except (EwEDatabaseError, Exception):
            fields[attr_name] = None

    return ValueChainData(**fields)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_value_chain_io.py -v`
Expected: 6 PASS (3 schema + 3 reader)

- [ ] **Step 6: Commit**

```bash
git add packages/pypath/src/pypath/io/ewemdb.py packages/pypath/tests/test_value_chain_io.py
git commit -m "feat(io): add ValueChainData dataclass and read_value_chain() reader"
```

---

## Chunk 2: Writer + Integration

### Task 3: Add `write_value_chain()` to CSV bundle writer

**Files:**
- Modify: `packages/pypath/src/pypath/io/_csv_bundle_writer.py` (add method)
- Modify: `packages/pypath/tests/test_value_chain_io.py` (add writer tests)

- [ ] **Step 1: Write the failing test**

Append to `test_value_chain_io.py`:

```python
class TestValueChainWriter:
    """Test write_value_chain() on CSV bundle writer."""

    def _make_value_chain_data(self):
        from pypath.io.ewemdb import ValueChainData

        sample = _make_sample_value_chain_dfs()
        return ValueChainData(
            oop_storables=sample["cOOPStorable"],
            parameters=sample["cParameters"],
            units=sample["cUnit"],
            economic_units=sample["cEconomicUnit"],
            producers=sample["cProducerUnit"],
            processors=sample["cProcessingUnit"],
            links=sample["cLink"],
            link_defaults=sample["cLinkDefault"],
            link_landings=sample["cLinkLandings"],
        )

    def test_csv_writer_produces_value_chain_tables(self, tmp_path):
        import numpy as np
        from pypath.core.params import create_rpath_params
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Detritus", "Fleet"],
            types=[1, 0, 2, 3],
        )
        params.model["Biomass"] = [10.0, 5.0, 100.0, np.nan]
        params.model["PB"] = [50.0, 10.0, np.nan, np.nan]
        params.model["QB"] = [0.0, 30.0, np.nan, np.nan]

        out = str(tmp_path / "test_vc.csv.zip")
        writer = CsvBundleWriter(params, out, scenario_id=1)
        writer.write_ecopath()

        vc = self._make_value_chain_data()
        writer.write_value_chain(vc)
        writer.close()

        # Check that the tables were written
        assert "cOOPStorable" in writer._tables
        assert "cProducerUnit" in writer._tables
        assert "cLinkLandings" in writer._tables
        assert len(writer._tables["cOOPStorable"]) == 2

    def test_csv_writer_none_value_chain_no_tables(self, tmp_path):
        import numpy as np
        from pypath.core.params import create_rpath_params
        from pypath.io._csv_bundle_writer import CsvBundleWriter

        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Detritus", "Fleet"],
            types=[1, 0, 2, 3],
        )
        params.model["Biomass"] = [10.0, 5.0, 100.0, np.nan]
        params.model["PB"] = [50.0, 10.0, np.nan, np.nan]
        params.model["QB"] = [0.0, 30.0, np.nan, np.nan]

        out = str(tmp_path / "test_no_vc.csv.zip")
        writer = CsvBundleWriter(params, out, scenario_id=1)
        writer.write_ecopath()
        writer.write_value_chain(None)
        writer.close()

        c_tables = [t for t in writer._tables if t.startswith("c")]
        assert len(c_tables) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_value_chain_io.py::TestValueChainWriter -v`
Expected: FAIL — `write_value_chain` not yet defined

- [ ] **Step 3: Add `write_value_chain()` to `CsvBundleWriter`**

Add method to `CsvBundleWriter` class in `_csv_bundle_writer.py` (after `write_taxonomy`):

```python
    def write_value_chain(self, value_chain=None) -> None:
        """Write value chain economics tables to the CSV bundle."""
        if value_chain is None:
            return

        from pypath.io.ewemdb import _VALUE_CHAIN_TABLES

        for attr_name, table_name in _VALUE_CHAIN_TABLES.items():
            df = getattr(value_chain, attr_name, None)
            if df is not None and len(df) > 0:
                self._tables[table_name] = df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_value_chain_io.py -v`
Expected: 8 PASS

- [ ] **Step 5: Commit**

```bash
git add packages/pypath/src/pypath/io/_csv_bundle_writer.py packages/pypath/tests/test_value_chain_io.py
git commit -m "feat(io): add write_value_chain() to CsvBundleWriter"
```

---

### Task 4: Add `write_value_chain()` to Access writer and integrate with `write_ewemdb()`

**Files:**
- Modify: `packages/pypath/src/pypath/io/_access_writer.py` (add method)
- Modify: `packages/pypath/src/pypath/io/ewe_writer.py` (add parameter and dispatch)
- Modify: `packages/pypath/src/pypath/io/__init__.py` (export new symbols)
- Modify: `packages/pypath/tests/test_value_chain_io.py` (add integration test)

- [ ] **Step 1: Write the failing test**

Append to `test_value_chain_io.py`:

```python
class TestValueChainIntegration:
    """Test write_ewemdb() integration with value_chain parameter."""

    def test_write_ewemdb_with_value_chain(self, tmp_path):
        import numpy as np
        from pypath.core.params import create_rpath_params
        from pypath.io.ewe_writer import write_ewemdb
        from pypath.io.ewemdb import ValueChainData

        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Detritus", "Fleet"],
            types=[1, 0, 2, 3],
        )
        params.model["Biomass"] = [10.0, 5.0, 100.0, np.nan]
        params.model["PB"] = [50.0, 10.0, np.nan, np.nan]
        params.model["QB"] = [0.0, 30.0, np.nan, np.nan]

        sample = _make_sample_value_chain_dfs()
        vc = ValueChainData(
            oop_storables=sample["cOOPStorable"],
            parameters=sample["cParameters"],
            units=sample["cUnit"],
            producers=sample["cProducerUnit"],
            link_landings=sample["cLinkLandings"],
        )

        out = str(tmp_path / "test_vc_full.csv.zip")
        write_ewemdb(params, out, backend="csv", value_chain=vc)

        import zipfile
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "cOOPStorable.csv" in names
            assert "cProducerUnit.csv" in names
            assert "cLinkLandings.csv" in names

    def test_io_exports_value_chain_symbols(self):
        from pypath.io import read_value_chain, ValueChainData
        assert read_value_chain is not None
        assert ValueChainData is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_value_chain_io.py::TestValueChainIntegration -v`
Expected: FAIL — `value_chain` parameter not accepted yet

- [ ] **Step 3: Add `write_value_chain()` to `AccessWriter`**

Add method to `AccessWriter` class in `_access_writer.py` (after `write_taxonomy`):

```python
    def write_value_chain(self, value_chain=None) -> None:
        """Write value chain economics tables."""
        if value_chain is None:
            return
        self._build_tables_via_csv_writer(
            "write_value_chain", value_chain=value_chain
        )
```

- [ ] **Step 4: Add `value_chain` parameter to `write_ewemdb()`**

In `ewe_writer.py`, add to function signature:

```python
def write_ewemdb(
    params: "RpathParams",
    path: str,
    *,
    scenarios: list[Any] | None = None,
    ecospace: Any | None = None,
    mpa_config: Any | None = None,
    timeseries: Any | None = None,
    mediation: Any | None = None,
    taxonomy: Any | None = None,
    value_chain: Any | None = None,  # NEW
    backend: str = "auto",
    scenario_id: int = 1,
    source_db: str | None = None,
) -> None:
```

Add to docstring:
```
    value_chain : ValueChainData, optional
        Value chain economics data (21 c-prefix tables) to include.
```

Add dispatch call before `writer.close()`:
```python
        writer.write_value_chain(value_chain)
```

- [ ] **Step 5: Add exports to `io/__init__.py`**

Add to imports from `pypath.io.ewemdb`:
```python
    read_value_chain,
    ValueChainData,
```

Add to `__all__`:
```python
    # Value chain
    "read_value_chain",
    "ValueChainData",
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/test_value_chain_io.py -v`
Expected: 10 PASS

- [ ] **Step 7: Run full test suite to check for regressions**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`
Expected: All pass, no regressions

- [ ] **Step 8: Commit**

```bash
git add packages/pypath/src/pypath/io/_access_writer.py packages/pypath/src/pypath/io/ewe_writer.py packages/pypath/src/pypath/io/__init__.py packages/pypath/tests/test_value_chain_io.py
git commit -m "feat(io): integrate value chain I/O into write_ewemdb() pipeline"
```

---

### Task 5: Update roadmap and verify

**Files:**
- Modify: `docs/superpowers/plans/2026-03-11-pypath-ewe-feature-roadmap.md`

- [ ] **Step 1: Update Phase 8 status in roadmap**

In the implementation priority matrix, change Phase 8 row:
```
| 8 | Value chain economics | Low | Very High | Defer | Not started |
```
to:
```
| 8 | Value chain economics (I/O) | Low | Low | Done | ✅ Done |
```

Update the "Remaining work" section to reflect completion.

- [ ] **Step 2: Run full test suite one final time**

Run: `conda run -n shiny python -m pytest packages/pypath/tests/ -q -m "not integration and not slow" --ignore=packages/pypath/tests/scripts`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-03-11-pypath-ewe-feature-roadmap.md
git commit -m "docs: update roadmap - Phase 8 Value Chain I/O complete"
```
