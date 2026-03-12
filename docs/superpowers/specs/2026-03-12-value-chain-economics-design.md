# Value Chain Economics I/O — Design Spec

**Date:** 2026-03-12
**Phase:** 8 (PyPath EwE Feature Roadmap)
**Scope:** I/O parity only — schema definitions, reader, writer. No runtime economic calculations.

---

## Goal

Enable round-trip read/write of EwE Value Chain Economics tables (21 `c`-prefix tables) so that models containing value chain data can be imported and re-exported without data loss.

## Background

EwE 6.x includes an optional Value Chain plugin that models the economics of seafood supply chains: producer (fleet) → processor → distributor → wholesaler → retailer → consumer. The data is stored in 21 database tables using an OOP inheritance pattern where objects span multiple tables joined by DBID.

PyPath does not need to compute value chain economics at runtime. The goal is purely I/O fidelity: read these tables from an EwE database and write them back unchanged.

## Architecture

### Storage Pattern: DataFrame Passthrough

The same pattern used for Ecospace's 16 optional tables. Raw DataFrames are read from the database and stored in a container dataclass. On write, the DataFrames are passed through directly to output tables with no transformation.

This approach:
- Minimizes code (~250 lines total across reader/writer)
- Guarantees perfect round-trip fidelity
- Requires no runtime data model for the OOP inheritance hierarchy

### EwE OOP Inheritance Pattern (reference only)

The value chain uses a multi-table inheritance storage:

```
cOOPStorable (DBID, xCLASS_NAMEx)
  └─ cUnit (DBID, Name, Sequence, Nationality, NameLocal)
       ├─ cConsumerUnit (DBID only — no extra columns)
       └─ cEconomicUnit (DBID + ~40 economic columns)
            ├─ cProducerUnit (DBID + ObserverCost, ObserverRate, TicketProducts, EcopathFleetID)
            ├─ cProcessingUnit (DBID + AgriculturalProducts, AgriculturalInput)
            ├─ cDistributionUnit (DBID only)
            ├─ cWholesalerUnit (DBID only)
            └─ cRetailerUnit (DBID only)
```

We do NOT reconstruct this hierarchy. Each table is read/written as an independent DataFrame.

---

## Schema: 21 Tables

All tables are added to `EWE_TABLES` in `_ewe_schema.py` (the shared schema module imported by both writer backends).

### Master Registry

| Table | Columns | Notes |
|-------|---------|-------|
| `cOOPStorable` | xCLASS_NAMEx (TEXT), DBID (INTEGER), AllowEvents (YESNO) | Registers every value chain object |

### Configuration

| Table | Columns | Notes |
|-------|---------|-------|
| `cParameters` | EquilibriumEffortMin (DOUBLE), EquilibriumEffortMax (DOUBLE), EquilibriumEffortIncrement (DOUBLE), RunWithEcopath (YESNO), RunWithEcosim (YESNO), RunWithSearches (YESNO) | Plugin settings |

### Unit Tables

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `cUnit` | Sequence (INTEGER), Name (TEXT), Nationality (TEXT), NameLocal (TEXT), DBID (INTEGER) | Base for all units |
| `cEconomicUnit` | DBID (INTEGER) + ~40 DOUBLE columns (revenue, cost, tax, employment per-tonne) | Base for economic units |
| `cProducerUnit` | DBID (INTEGER), ObserverCost (DOUBLE), ObserverRate (DOUBLE), TicketProducts (TEXT), EcopathFleetID (INTEGER) | Links to Ecopath fleet |
| `cProcessingUnit` | DBID (INTEGER), AgriculturalProducts (TEXT), AgriculturalInput (TEXT) | Processor stage |
| `cDistributionUnit` | DBID (INTEGER) | Distributor stage |
| `cWholesalerUnit` | DBID (INTEGER) | Wholesaler stage |
| `cRetailerUnit` | DBID (INTEGER) | Retailer stage |
| `cConsumerUnit` | DBID (INTEGER) | End consumer (extends cUnit, not cEconomicUnit) |

### Default Templates (6 tables)

| Table | Notes |
|-------|-------|
| `cProducerDefault` | Same schema as cProducerUnit |
| `cProcessingDefault` | Same schema as cProcessingUnit |
| `cDistributionDefault` | Same schema as cDistributionUnit |
| `cWholesalerDefault` | Same schema as cWholesalerUnit |
| `cRetailerDefault` | Same schema as cRetailerUnit |
| `cConsumerDefault` | Same schema as cConsumerUnit |

### Link Tables

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `cLink` | DBID (INTEGER) — Source and Target joined via cOOPStorable | Connects units in chain |
| `cLinkDefault` | LinkType (INTEGER), BiomassRatio (DOUBLE), ValuePerTon (DOUBLE), ValueRatio (DOUBLE) | Default link parameters |
| `cLinkLandings` | EcopathGroupID (INTEGER), ValuePerTon (DOUBLE) | Species-to-producer mapping |

### UI Layout Tables

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `cFlowDiagram` | DBID (INTEGER), Name (TEXT), Description (TEXT) | Flow diagram definition |
| `cFlowPosition` | DBID (INTEGER), DiagramDBID (INTEGER), UnitDBID (INTEGER), X (DOUBLE), Y (DOUBLE), Width (DOUBLE), Height (DOUBLE) | Node positions in diagram |

### cEconomicUnit Full Column List

All DOUBLE type, per-tonne values:

```
DBID, RevenueLocalDomestic, RevenueLocalExport, RevenueForeignDomestic,
RevenueForeignExport, CostOperating, CostCapital, CostLabour,
CostLabourForeign, CostRawMaterial, CostRawMaterialForeign,
CostIntermediate, CostIntermediateForeign, TaxDirect, TaxIndirect,
TaxExport, TaxImport, SubsidyDirect, SubsidyIndirect,
EmploymentDirect, EmploymentIndirect, DependentsDirect,
DependentsIndirect, EmploymentDirectForeign, EmploymentIndirectForeign,
DependentsDirectForeign, DependentsIndirectForeign,
RevenueLocalDomesticEquil, RevenueLocalExportEquil,
RevenueForeignDomesticEquil, RevenueForeignExportEquil,
CostOperatingEquil, CostCapitalEquil, CostLabourEquil,
CostLabourForeignEquil, CostRawMaterialEquil, CostRawMaterialForeignEquil,
CostIntermediateEquil, CostIntermediateForeignEquil
```

**Note:** This column list is a best-effort approximation from EwE source code. Some EwE versions may include additional Equil-suffix columns for employment/dependents. The DataFrame passthrough design handles this gracefully — any extra columns in the source database are preserved automatically. The schema in `EWE_TABLES` defines the minimum set; `_ensure_table` in AccessWriter handles divergence.

---

## Data Model

### ValueChainData Dataclass

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
```

Location: `pypath/io/ewemdb.py` (near top of file, alongside other dataclasses like `TaxonomyData`)

---

## Reader

### `read_value_chain(db: str) -> ValueChainData | None`

Location: `pypath/io/ewemdb.py`

Behavior:
1. Try to read `cOOPStorable` table — if missing or empty, return `None` (no value chain data)
2. For each of the 21 tables, try `read_ewemdb_table(db, table_name)`; store DataFrame on success, `None` on failure
3. Return populated `ValueChainData`

This is a **standalone function** (same pattern as `read_taxonomy()`, `read_ecospace()`, `read_mediation()`). It is NOT called inside `read_ewemdb()`. Callers invoke it explicitly when they need value chain data.

### Field-to-table mapping

```python
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

---

## Writer

### `_VALUE_CHAIN_TABLES` mapping

Defined in `ewemdb.py` (alongside `read_value_chain()`). Imported by the writer modules that need it. This dict is the single source of truth for field↔table mapping.

### CSV Bundle Writer

`write_value_chain(self, value_chain)` method on `CsvBundleWriter`:

```python
def write_value_chain(self, value_chain):
    if value_chain is None:
        return
    from pypath.io.ewemdb import _VALUE_CHAIN_TABLES
    for attr_name, table_name in _VALUE_CHAIN_TABLES.items():
        df = getattr(value_chain, attr_name, None)
        if df is not None and len(df) > 0:
            self._tables[table_name] = df
```

### Access Writer

`write_value_chain(self, value_chain)` method on `AccessWriter`:
- Delegates to `self._build_tables_via_csv_writer("write_value_chain", value_chain=value_chain)`
- Value chain tables are **insert-only** (no clearing pass). Since the writer always starts with a fresh or cleared database, duplicate rows are not a concern. The DBID foreign key chains make deletion-order complex and unnecessary for the passthrough pattern.

### write_ewemdb() Integration

New parameter: `value_chain: ValueChainData | None = None`

The parameter must be added to the function signature, docstring, and the dispatch sequence inside the existing `try` block (before `writer.close()`).

Dispatch order updated:
```python
writer.write_ecopath()
writer.write_ecosim(scenarios)
writer.write_ecospace(ecospace)
writer.write_mpa(mpa_config)
writer.write_timeseries(timeseries)
writer.write_mediation(mediation)
writer.write_taxonomy(taxonomy)
writer.write_value_chain(value_chain)  # NEW
writer.close()
```

---

## Testing

### test_value_chain_io.py

1. **Schema test**: All 21 c-prefix tables present in `EWE_TABLES`
2. **Reader test**: Mock database with sample value chain data → `read_value_chain()` returns correct `ValueChainData`
3. **Reader empty test**: Database without c-prefix tables → returns `None`
4. **Writer CSV test**: `ValueChainData` → CSV bundle writer → all 21 tables in output
5. **Writer empty test**: `None` value chain → writer produces no c-prefix tables
6. **Round-trip test**: Create `ValueChainData` → write → read back → DataFrames match

---

## Scope Boundaries

**In scope:**
- 21 table schema definitions in `EWE_TABLES` (`_ewe_schema.py`)
- `ValueChainData` dataclass
- `read_value_chain()` function
- Writer support in both CSV and Access backends
- `write_ewemdb()` integration via `value_chain=` parameter
- Unit tests

**Out of scope:**
- Runtime economic calculations (equilibrium solver, price elasticity)
- Supply chain graph reconstruction from DBID joins
- Value chain visualization
- Shiny app integration
- Economic optimization

---

## File Changes

| File | Change |
|------|--------|
| `packages/pypath/src/pypath/io/_ewe_schema.py` | Add 21 `c`-prefix table definitions to `EWE_TABLES` |
| `packages/pypath/src/pypath/io/ewemdb.py` | Add `ValueChainData` dataclass, `_VALUE_CHAIN_TABLES` mapping, `read_value_chain()` function |
| `packages/pypath/src/pypath/io/_csv_bundle_writer.py` | Add `write_value_chain()` method |
| `packages/pypath/src/pypath/io/_access_writer.py` | Add `write_value_chain()` method |
| `packages/pypath/src/pypath/io/ewe_writer.py` | Add `value_chain` parameter to `write_ewemdb()` signature, docstring, and dispatch |
| `packages/pypath/src/pypath/io/__init__.py` | Export `read_value_chain` and `ValueChainData` |
| `packages/pypath/tests/test_value_chain_io.py` | New: 6 tests |
| `docs/superpowers/plans/2026-03-11-pypath-ewe-feature-roadmap.md` | Update Phase 8 status |
