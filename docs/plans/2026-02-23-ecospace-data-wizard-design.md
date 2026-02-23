# Ecospace Data Wizard — Design Document

**Date:** 2026-02-23
**Status:** Approved

## Overview

Add infrastructure to pypath-shiny for creating ecospace models from real-world marine data. Users draw a study area on a map, download EMODnet seabed habitats (EUNIS level 3+) and bathymetry, optionally upload salinity data, then are guided through a 7-step wizard to build a complete ecospace model.

## Architecture

Two layers:

1. **Core data layer** — `pypath.io.marine_data` module in `pypath-ewe`
   - EMODnet WFS/WCS clients for habitats and bathymetry
   - Grid sampling and rasterization
   - Habitat preference builder with semi-automatic suggestions
   - Local file cache (`~/.pypath/cache/marine_data/`)

2. **Wizard UI** — `pypath_shiny.pages.ecospace_wizard` page in `pypath-shiny`
   - 7-step wizard with progress indicator
   - Each step builds on the previous, feeding into `EcospaceParams`

This separation ensures data fetching works from both the Shiny app and Python scripts/notebooks.

**Dependency direction:** `pypath-shiny` calls `pypath.io.marine_data`, never the reverse.

## Data Sources

| Data | Protocol | Endpoint | Output | Auth |
|------|----------|----------|--------|------|
| Seabed habitats | WFS | `ows.emodnet-seabedhabitats.eu/geoserver/emodnet_view/wfs` | EUNIS polygons (GeoJSON) | None |
| Bathymetry | WCS | `ows.emodnet-bathymetry.eu/wcs` | Depth raster (GeoTIFF) | None |
| Salinity | File upload | User-provided NetCDF/CSV | Gridded salinity | N/A |

Salinity is **optional** — user can upload a file or skip it entirely. EMODnet habitats and bathymetry are fetched via open APIs (no authentication required).

## Wizard Steps

### Step 1: Select Area

- Leaflet map centered on European seas
- Leaflet.draw polygon drawing tool (polygon mode only)
- Drawn polygon displayed with area estimate in km²
- User can redraw or clear
- Coordinates captured as WGS84 GeoJSON geometry

### Step 2: Configure Grid

- Choose grid type:
  - **Regular rectangular**: specify cell size in km or rows x columns
  - **Hexagonal**: specify hex diameter in km
- Preview: grid overlay on map with patch count
- Uses existing `EcospaceGrid.from_regular_grid()` and hex grid generation
- Grid stored as `EcospaceGrid` instance

### Step 3: Download Data

Single "Download" button triggers parallel fetches:

**EMODnet Seabed Habitats (WFS):**
- GetFeature request for EUSeaMap layer within drawn polygon bbox
- Filter by polygon intersection
- Returns EUNIS 2022 classified polygons as GeoDataFrame
- Cache key: SHA256 of bbox coordinates + layer name

**EMODnet Bathymetry (WCS):**
- GetCoverage request for bbox area
- Native resolution ~115m, averaged into ecospace patches
- Returns depth values per patch as ndarray
- Cache key: SHA256 of bbox + requested resolution

**Salinity (optional):**
- File upload widget (NetCDF or CSV)
- Resampled/interpolated onto ecospace grid patches
- Stored as `EnvironmentalLayer`

Progress bar shows download status per layer. Cache checked before network requests. Cache location: `~/.pypath/cache/marine_data/`.

### Step 4: Review Habitats

EUNIS polygons rasterized onto ecospace grid (majority habitat class per patch).

**UI elements:**
- Map view: patches colored by EUNIS level 3 type
- Legend: all EUNIS types found in area with patch counts
- Merge tool: combine rare/similar types to reduce complexity (e.g., merge all A5.x subtypes)
- Depth overlay toggle: bathymetry values per patch
- Table view: habitat types, area coverage, depth range per type

**Output:** `habitat_types[n_patches]` array mapping each patch to a habitat category.

### Step 5: Assign Habitat Preferences

Semi-automatic habitat preference assignment per species group:

1. **Auto-suggest**: query group name against biodata (WoRMS/FishBase) for depth range and substrate preferences; propose initial preference values per EUNIS habitat type
2. **Preference matrix**: editable table — rows = groups, columns = habitat types, values = 0-1
3. **Quick presets**: "Pelagic" (all habitats equal), "Demersal" (substrate-weighted), "Benthic" (strong substrate preference)
4. **Depth response**: optional Gaussian or threshold depth response per group (using existing `habitat.py` response functions)

**Output:** `habitat_preference[n_groups, n_patches]` and `habitat_capacity[n_groups, n_patches]` arrays, computed from preference matrix combined with environmental suitability.

### Step 6: Set Dispersal

Per-group dispersal configuration:

- Default dispersal rate (km²/month) slider
- Per-group override table: group name + dispersal rate + advection toggle
- Size-based auto-suggest: larger/more mobile species get higher rates
- Gravity strength slider for habitat-directed movement

**Output:** `dispersal_rate[n_groups]`, `advection_enabled[n_groups]`, `gravity_strength[n_groups]`.

### Step 7: Review & Launch

Summary dashboard:

- Grid: patch count, total area, grid type
- Habitats: number of types, coverage map thumbnail
- Environment: depth range, salinity range (if uploaded)
- Species: number of groups with preferences assigned
- Dispersal: min/max/mean dispersal rates

**"Create Ecospace Model"** button: populates `EcospaceParams` and transitions to the existing Ecospace simulation page.

## Core Module Design

### File: `packages/pypath/src/pypath/io/marine_data.py`

```
MarineDataCache
├── __init__(cache_dir="~/.pypath/cache/marine_data/")
├── get(key: str) -> bytes | None
├── put(key: str, data: bytes) -> None
└── cache_key(bbox, layer, **kwargs) -> str

EMODnetHabitatsClient
├── __init__(cache: MarineDataCache)
├── fetch_euseamap(bbox: tuple, eunis_level: int = 3) -> GeoDataFrame
│   # WFS GetFeature request, returns EUNIS-classified polygons
├── rasterize_habitats(gdf: GeoDataFrame, grid: EcospaceGrid) -> ndarray
│   # Assign majority EUNIS class to each patch via spatial join
└── get_habitat_types(gdf: GeoDataFrame, level: int) -> list[str]
    # Extract unique EUNIS codes at requested level

EMODnetBathymetryClient
├── __init__(cache: MarineDataCache)
├── fetch_depth(bbox: tuple, resolution: float = 0.002) -> ndarray
│   # WCS GetCoverage request, returns depth raster
└── sample_to_grid(raster: ndarray, transform, grid: EcospaceGrid) -> ndarray
    # Average raster values within each patch polygon

SalinityLoader
└── load_from_file(filepath: str, grid: EcospaceGrid) -> EnvironmentalLayer
    # Read NetCDF/CSV, resample onto grid, return EnvironmentalLayer

HabitatPreferenceBuilder
├── suggest_preferences(group_names: list[str], habitat_types: list[str],
│       depth_per_patch: ndarray | None) -> DataFrame
│   # Auto-suggest preference matrix using biodata + depth
├── apply_preset(n_groups: int, habitat_types: list[str],
│       preset: str) -> DataFrame
│   # "pelagic", "demersal", "benthic" presets
└── build_preference_matrix(preferences: DataFrame,
        habitat_map: ndarray, grid: EcospaceGrid) -> ndarray
    # Convert habitat-type preferences to per-patch preferences [n_groups, n_patches]
```

### Dependencies

New dependencies for `pypath-ewe`:
- `requests` (already optional via `[biodata]`) — for WFS/WCS HTTP requests
- `rasterio` — for reading WCS GeoTIFF responses (new optional dependency)

Group as a new extra: `[spatial-data]` containing `requests`, `rasterio`.

The `geopandas` and `shapely` dependencies are already in the `[spatial]` extra.

## Shiny Page Design

### File: `packages/pypath-shiny/src/pypath_shiny/pages/ecospace_wizard.py`

**Navigation:** New sidebar entry "Ecospace Wizard" between "Ecospace" and existing pages.

**Layout:**
- Top: step progress bar (Steps 1-7 with labels, current step highlighted)
- Center: step content area (changes per step)
- Bottom: "Back" / "Next" navigation buttons

**Reactive state:**
- `wizard_step`: current step (1-7)
- `drawn_polygon`: GeoJSON geometry from map
- `ecospace_grid`: EcospaceGrid instance
- `habitat_gdf`: downloaded habitat GeoDataFrame
- `depth_per_patch`: ndarray of depth values
- `salinity_layer`: optional EnvironmentalLayer
- `habitat_types`: ndarray of EUNIS codes per patch
- `preference_matrix`: DataFrame of group-habitat preferences
- `dispersal_params`: dict of dispersal settings

**Step transitions:** "Next" button validates current step is complete before advancing. "Back" preserves state.

## Data Flow

```
Step 1: User draws polygon
    ↓ GeoJSON geometry
Step 2: User configures grid
    ↓ EcospaceGrid
Step 3: Download EMODnet data
    ↓ GeoDataFrame (habitats) + ndarray (depth) + optional EnvironmentalLayer (salinity)
Step 4: Review & merge habitats
    ↓ habitat_types[n_patches]
Step 5: Assign preferences
    ↓ habitat_preference[n_groups, n_patches] + habitat_capacity[n_groups, n_patches]
Step 6: Set dispersal
    ↓ dispersal_rate[n_groups] + advection_enabled[n_groups] + gravity_strength[n_groups]
Step 7: Review → Build EcospaceParams
    ↓ Complete EcospaceParams object → Ecospace simulation page
```

## Caching Strategy

- Cache directory: `~/.pypath/cache/marine_data/`
- Cache key: SHA256 hash of (endpoint + bbox coordinates + layer name + resolution)
- Cached files: raw WFS/WCS responses (GeoJSON, GeoTIFF)
- No expiry (marine habitat maps change infrequently)
- Cache can be cleared manually by deleting the directory

## Error Handling

- Network failures: show error message with retry button, do not block wizard
- Empty WFS response (no habitats in area): warn user, suggest expanding area
- WCS resolution too fine for large areas: auto-coarsen and inform user
- Missing biodata for auto-suggest: fall back to uniform preferences with warning
- Invalid uploaded salinity file: show format error, allow re-upload

## Testing Strategy

**Core module (`pypath-ewe`):**
- Unit tests for rasterization, grid sampling, preference building (mock HTTP responses)
- Integration tests for actual API calls (marked `@pytest.mark.integration`)
- Cache tests (hit, miss, invalidation)

**Shiny wizard (`pypath-shiny`):**
- Step navigation tests (forward, backward, validation)
- Mock data tests (pre-loaded GeoDataFrame, test grid)
- Reactive state consistency tests
