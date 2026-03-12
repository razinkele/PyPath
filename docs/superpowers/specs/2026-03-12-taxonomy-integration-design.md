# Taxonomy & External Database Integration Design Spec

**Goal:** Read, write, and auto-populate EwE taxonomy tables (`EcopathTaxon`, `EcopathGroupTaxon`, `EcopathStanzaTaxon`), enabling round-trip of species-to-group mappings with external database keys (WoRMS, FishBase, SeaLifeBase, OBIS).

**Approach:** Extend existing modules — reader in `ewemdb.py`, writer in `ewe_writer.py`/`_access_writer.py`, auto-populate in `biodata.py`. New dataclasses in `ewemdb.py`. Schema additions in `_ewe_schema.py`. No new files except tests.

---

## 1. Schema Additions

Add 3 tables to `_ewe_schema.py` matching real EwE 6.6+ databases (verified against LT2022):

```python
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

Note: `VulnerabiltyIndex` matches the real EwE spelling (typo in original EwE schema, preserved for compatibility).

---

## 2. Data Types

Defined in `ewemdb.py` alongside the reader:

```python
@dataclass
class TaxonomyRecord:
    """A single species/taxon entry from EcopathTaxon.

    Attributes
    ----------
    taxon_id : int
        Unique identifier (1-based).
    scientific_name : str
        "Genus species" (derived from GenusName + SpeciesName).
    common_name : str
        Vernacular name.
    taxonomy : dict
        Taxonomic hierarchy: {"class_name", "order_name", "family_name",
        "genus_name", "species_name"}.
    external_keys : dict
        Database cross-references: {"aphia_id", "fishbase_code",
        "sealifebase_code", "obis_code", "saup_code", "fao_code",
        "aquamaps_code", "lsid"}.
    traits : dict
        Ecological traits: {"winf", "vbgf_k", "mean_weight", "mean_length",
        "max_length", "mean_lifespan", "vulnerability_index"}.
    metadata : dict
        Status/classification fields: {"ecology_type", "organism_type",
        "exploited", "conservation_status", "occurrence_status",
        "exploitation_status", "last_updated"}.
    source_name : str
        Data source identifier (e.g., "EwEWoRMSPlugin!...").
    source_key : str
        Source-specific key (e.g., WoRMS AphiaID as string).
    """

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
    """Complete taxonomy data from an EwE model.

    Attributes
    ----------
    taxa : list[TaxonomyRecord]
        All species/taxon entries.
    group_assignments : pd.DataFrame
        Columns: TaxonID, EcopathGroupID, Proportion, PropCatch.
        Maps species to Ecopath groups with proportional weights.
    stanza_assignments : pd.DataFrame
        Columns: TaxonID, StanzaID.
        Maps species to multi-stanza groups.
    """

    taxa: list
    group_assignments: "pd.DataFrame"
    stanza_assignments: "pd.DataFrame"
```

### Mapping EcopathTaxon columns to TaxonomyRecord fields

| EcopathTaxon column | TaxonomyRecord field | Notes |
|---------------------|---------------------|-------|
| TaxonID | taxon_id | Direct |
| GenusName + " " + SpeciesName | scientific_name | Concatenated |
| CommonName | common_name | Direct |
| ClassName, OrderName, FamilyName, GenusName, SpeciesName | taxonomy dict | All 5 hierarchy levels |
| CodeAphia | external_keys["aphia_id"] | WoRMS AphiaID |
| CodeFB | external_keys["fishbase_code"] | FishBase species code |
| CodeSLB | external_keys["sealifebase_code"] | SeaLifeBase code |
| CodeOBIS | external_keys["obis_code"] | OBIS identifier |
| CodeSAUP | external_keys["saup_code"] | Sea Around Us |
| CodeFAO | external_keys["fao_code"] | FAO code |
| CodeAquaMaps | external_keys["aquamaps_code"] | AquaMaps |
| CodeLCID | external_keys["lsid"] | LSID URN |
| EcologyType, OrganismType, Exploited, ConservationStatus, OccurrenceStatus, ExploitationStatus, LastUpdated | metadata dict | Integer/text status fields not in traits |
| Winf, vbgfK, MeanWeight, MeanLength, MaxLength, MeanLifeSpan, VulnerabiltyIndex | traits dict | -9999 → None |
| SourceName | source_name | Direct |
| SourceKey | source_key | Direct |

EwE uses `-9999` as a sentinel for missing numeric values. The reader converts these to `None`.

### Unmapped columns

The EcopathTaxon table has several status/classification columns (`EcologyType`, `OrganismType`, `Exploited`, `ConservationStatus`, `OccurrenceStatus`, `ExploitationStatus`, `LastUpdated`) that don't fit neatly into taxonomy, external_keys, or traits. These are stored in a `metadata: dict` field on `TaxonomyRecord` with snake_case keys (`ecology_type`, `organism_type`, etc.). The reader populates them from the database; the writer converts them back. Default value for missing metadata keys is `None`.

---

## 3. Reader: `read_taxonomy()`

Added to `ewemdb.py`:

```python
def read_taxonomy(db_path: str) -> TaxonomyData:
```

### Reading sequence

1. **EcopathTaxon** → build `TaxonomyRecord` per row. Convert `-9999` sentinel values to `None` in traits. Concatenate `GenusName + " " + SpeciesName` for `scientific_name`. Map column names to dict keys.
2. **EcopathGroupTaxon** → read as DataFrame. Empty DataFrame if table missing.
3. **EcopathStanzaTaxon** → read as DataFrame. Empty DataFrame if table missing.
4. Return `TaxonomyData(taxa, group_assignments, stanza_assignments)`.

### Missing table handling

- Missing `EcopathTaxon` → empty `taxa` list (not an error — model may have no taxonomy)
- Missing `EcopathGroupTaxon` → empty DataFrame with columns `[TaxonID, EcopathGroupID, Proportion, PropCatch]`
- Missing `EcopathStanzaTaxon` → empty DataFrame with columns `[TaxonID, StanzaID]`

---

## 4. Writer: `write_taxonomy()`

Added to `ewe_writer.py` (delegates to `_access_writer.py` for Access format):

### Public API in `ewe_writer.py`

The existing `write_ewemdb()` gains a `taxonomy: TaxonomyData | None = None` parameter. When provided, taxonomy tables are written after Ecopath tables.

### Writer logic in `_access_writer.py`

A new `write_taxonomy(self, taxonomy: TaxonomyData)` method on `AccessWriter`:

1. Convert `TaxonomyRecord` list → rows for `EcopathTaxon` table. Convert `None` traits/metadata back to `-9999` (numeric) or `""` (text). Use stored `taxonomy["genus_name"]` and `taxonomy["species_name"]` for the database columns.
2. Write `EcopathGroupTaxon` DataFrame directly.
3. Write `EcopathStanzaTaxon` DataFrame directly.
4. Uses existing `_insert_rows()` method for writing.

**`_ECOPATH_TABLES` update:** `"EcopathStanzaTaxon"` already exists in the list (line 99). Add only `"EcopathGroupTaxon"` and `"EcopathTaxon"` (children first: GroupTaxon before Taxon, both after the existing StanzaTaxon entry).

### Writer pattern

Follow the existing DRY pattern: `AccessWriter.write_taxonomy()` delegates to `_build_tables_via_csv_writer("write_taxonomy", taxonomy=taxonomy)`. This requires a corresponding `CsvBundleWriter.write_taxonomy()` method that builds the table dicts. This is the same pattern used by `write_timeseries()`, `write_mediation()`, etc.

### Writer dispatch sequence

In `write_ewemdb()` (`ewe_writer.py`), add `taxonomy: TaxonomyData | None = None` parameter. Insert `writer.write_taxonomy(taxonomy)` after `writer.write_mediation(mediation)` and before `writer.close()`. The full dispatch order becomes:

1. `writer.write_ecopath()`
2. `writer.write_ecosim(scenarios)`
3. `writer.write_ecospace(ecospace)`
4. `writer.write_timeseries(timeseries)`
5. `writer.write_mediation(mediation)`
6. `writer.write_taxonomy(taxonomy)` — **new**
7. `writer.close()`

### CSV backend

The `CsvBundleWriter.write_taxonomy()` method is needed for the DRY pattern but will produce taxonomy tables in CSV format. This is acceptable — CSV export of taxonomy is low-priority but comes for free with the existing architecture.

---

## 5. Auto-populate: `auto_populate_taxonomy()`

Added to `biodata.py`:

```python
def auto_populate_taxonomy(
    rpath: "Rpath",
    group_species_map: dict[str, list[str]],
    proportions: dict[str, list[float]] | None = None,
) -> "TaxonomyData":
```

### Parameters

- `rpath`: Balanced Ecopath model output (`pypath.core.ecopath.Rpath`, not `RpathParams`). Used to look up group indices by name (`rpath.Group` is a 0-indexed numpy array of group name strings).
- `group_species_map`: `{group_name: [species_name, ...]}`. Each group maps to one or more species. Names can be either scientific names (e.g., `"Gadus morhua"`) or common names (e.g., `"Atlantic cod"`). The auto-populate function handles both: it first tries `get_species_info(name)` which does a WoRMS vernacular search; if that fails (no vernacular match), it falls back to `_fetch_worms_match(name)` for scientific/fuzzy name matching, then builds the `SpeciesInfo` manually from the WoRMS record.
- `proportions`: Optional `{group_name: [proportion, ...]}`. Must sum to 1.0 per group. If `None`, defaults to equal split (`1/n`).

### Logic

1. Collect all unique species names from `group_species_map`.
2. Call `get_species_info(name)` per species (not `batch_get_species_info()` which returns a DataFrame). `get_species_info()` returns a `SpeciesInfo` object with `aphia_id`, `scientific_name`, `common_name`, `trophic_level`, `max_length`, `k`, `loo` etc. Use a ThreadPoolExecutor for parallelism (same pattern as `batch_get_species_info()`).
3. **Fetch taxonomy classification**: For each species with a successful lookup, call `_fetch_worms_accepted(aphia_id)` to get the full WoRMS record containing `class`, `order`, `family`, `genus` fields. This reuses the existing caching and synonym resolution logic. This extra call is needed because `SpeciesInfo` does not store taxonomy hierarchy — `_merge_species_data()` discards these fields from the WoRMS response.
4. For each species, build a `TaxonomyRecord`:
   - `taxon_id`: auto-incremented starting from 1
   - `scientific_name`: from `SpeciesInfo.scientific_name`
   - `common_name`: from `SpeciesInfo.common_name`
   - `taxonomy`: `{"class_name": record["class"], "order_name": record["order"], "family_name": record["family"], "genus_name": record["genus"], "species_name": scientific_name.split(" ", 1)[1] if " " in scientific_name else ""}` from WoRMS record
   - `external_keys`: `{"aphia_id": species_info.aphia_id}`. FishBase species code is not available via the public API (`SpeciesInfo` does not expose it), so `fishbase_code` is omitted. Other external keys default to `None`.
   - `traits`: `{"max_length": species_info.max_length, "winf": species_info.growth_params.get("Loo") if species_info.growth_params else None, "vbgf_k": species_info.growth_params.get("K") if species_info.growth_params else None}` from SpeciesInfo; all others default to `None`
   - `metadata`: all default to `None` (auto-populate does not set status fields)
   - `source_name`: `"PyPath-biodata"`
   - `source_key`: str(aphia_id)
5. Build `group_assignments` DataFrame: for each group in `group_species_map`, look up the group's 1-based `EcopathGroupID` from `rpath.Group` (0-based index + 1), and create rows mapping `TaxonID → EcopathGroupID` with `Proportion` and `PropCatch` (both set to the same value — equal split `1/n` by default, or user-provided proportions). Note: `PropCatch` represents catch proportion, not biomass proportion. Setting it equal to `Proportion` assumes catch is proportional to biomass within a group — a simplification that can be overridden by the user after generation.
6. Species that fail lookup are logged as warnings and skipped (not errors).
7. `stanza_assignments` is left as empty DataFrame (user can add manually).
8. Return `TaxonomyData`.

### Group index lookup

`rpath.Group` is a 0-indexed numpy array of group names (from `pypath.core.ecopath.Rpath`, not `RpathParams`). EwE databases use 1-based `EcopathGroupID`. To find the group ID for a group name:
```python
matches = np.where(rpath.Group == group_name)[0]
if len(matches) > 0:
    group_id = matches[0] + 1  # convert 0-based numpy index to 1-based EcopathGroupID
```

Groups not found in `rpath.Group` are logged as warnings and skipped.

---

## 6. Exports

### `io/__init__.py`

Add: `TaxonomyRecord`, `TaxonomyData`, `read_taxonomy`, `auto_populate_taxonomy`

Note: `write_taxonomy` is not exported as a standalone function. Taxonomy writing is accessed via the existing `write_ewemdb(taxonomy=...)` parameter. The `write_taxonomy()` method is internal to `AccessWriter`/`CsvBundleWriter`.

### `_access_writer.py`

Add `"EcopathGroupTaxon"` and `"EcopathTaxon"` to `_ECOPATH_TABLES` list. Note: `"EcopathStanzaTaxon"` is already present (line 99). Insert GroupTaxon before Taxon (children first), both after the existing StanzaTaxon entry.

---

## 7. Testing Strategy

### Unit tests (`test_taxonomy.py`, ~15 tests)

**Schema (2):**
- 3 new tables exist in `EWE_TABLES`
- Column names and types match spec

**Reader (5):**
- Reads `EcopathTaxon` rows into `TaxonomyRecord` list with correct field mapping
- Reads `EcopathGroupTaxon` into DataFrame with correct columns
- Reads `EcopathStanzaTaxon` into DataFrame
- Missing tables return empty defaults (no error)
- `-9999` sentinel values converted to `None` in traits

**Writer (3):**
- Writes all 3 tables to mock database
- Round-trip: write then read back produces equivalent data
- Empty `TaxonomyData` writes empty tables without error

**Auto-populate (5):**
- Builds `TaxonomyData` from species map with correct `TaxonomyRecord` fields
- Multi-species groups get equal `Proportion` (1/n)
- Custom proportions respected when provided
- Species lookup failure logged as warning, other species still processed
- Group names matched to correct 1-based `EcopathGroupID` from `rpath.Group`

All tests use mocked database connections and mocked `get_species_info()` / `_fetch_worms_accepted()`.

---

## 8. File Structure

### New files
| File | Purpose |
|------|---------|
| `tests/test_taxonomy.py` | ~15 unit tests |

### Modified files
| File | Change |
|------|--------|
| `io/_ewe_schema.py` | Add 3 taxonomy tables |
| `io/ewemdb.py` | Add `TaxonomyRecord`, `TaxonomyData`, `read_taxonomy()` |
| `io/ewe_writer.py` | Add `taxonomy` parameter to `write_ewemdb()` |
| `io/_access_writer.py` | Add `write_taxonomy()` method, update `_ECOPATH_TABLES` |
| `io/biodata.py` | Add `auto_populate_taxonomy()` |
| `io/__init__.py` | Export new types and functions |
