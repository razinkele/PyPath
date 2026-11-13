"""EwE-F data format specifications and schemas.

Documents the structure of EwE-F input and output file formats,
including column definitions, data types, and conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ============================================================================
# FORTRAN NAMELIST FORMAT
# ============================================================================

NAMELIST_FORMAT_DOC = """
Fortran List-Directed Namelist Format (&filenames ... /)

EwE-F uses Fortran namelist format for configuration files (*_filenames.nml).

Key characteristics:
- Starts with &filenames and ends with /
- Variable name = value pairs
- String values quoted ("path/to/file")
- Boolean values: .true. or .false. (or .t., .f.)
- Numbers: integers or floats (d/D notation for exponent)
- Comments: ! starts inline comment to end of line
- Missing values: empty or omitted

Example:
    &filenames
        InputData_fname = "path/to/scenario.csv"
        DietComp_fname = "path/to/diet.csv"
        DetFate_fname = "path/to/detritus.csv"
        Results_fname = "results.h5"
        isASCIIinputFile = .true.
    /
"""

# ============================================================================
# ECOPATH SCENARIO FILE (CSV)
# ============================================================================


@dataclass
class EcopathScenarioColumn:
    """Definition of a column in Ecopath scenario CSV."""

    ewef_label: str  # Header in EwE-F file
    pypath_name: str  # Name in PyPath RpathParams
    data_type: str  # numpy/pandas dtype
    required: bool  # Is this column required?
    description: str
    units: Optional[str] = None
    valid_range: Optional[tuple] = None  # (min, max)
    na_value: float = -999.0  # EwE-F missing value marker


ECOPATH_SCENARIO_COLUMNS = [
    EcopathScenarioColumn(
        ewef_label="B",
        pypath_name="Biomass",
        data_type="float64",
        required=False,
        description="Group biomass",
        units="t/km²",
        valid_range=(0, 1e6),
    ),
    EcopathScenarioColumn(
        ewef_label="P/B",
        pypath_name="PB",
        data_type="float64",
        required=False,
        description="Production / Biomass ratio (instantaneous)",
        units="year⁻¹",
        valid_range=(0.001, 100),
    ),
    EcopathScenarioColumn(
        ewef_label="Q/B",
        pypath_name="QB",
        data_type="float64",
        required=False,
        description="Consumption / Biomass ratio",
        units="year⁻¹",
        valid_range=(0.01, 100),
    ),
    EcopathScenarioColumn(
        ewef_label="EE",
        pypath_name="EE",
        data_type="float64",
        required=False,
        description="Ecotrophic efficiency (proportion predated/dead)",
        units="dimensionless",
        valid_range=(0, 1),
    ),
    EcopathScenarioColumn(
        ewef_label="P/Q",
        pypath_name="ProdCons",
        data_type="float64",
        required=False,
        description="Production / Consumption (assimilation efficiency)",
        units="dimensionless",
        valid_range=(0, 1),
    ),
    EcopathScenarioColumn(
        ewef_label="Unass. Q",
        pypath_name="Unassim",
        data_type="float64",
        required=False,
        description="Unassimilated diet fraction",
        units="dimensionless",
        valid_range=(0, 1),
    ),
    EcopathScenarioColumn(
        ewef_label="Detritus Import",
        pypath_name="DetInput",
        data_type="float64",
        required=False,
        description="Detritus import flow",
        units="t/km²/year",
    ),
    EcopathScenarioColumn(
        ewef_label="Landings",
        pypath_name="landings",
        data_type="float64",
        required=False,
        description="Fishery landings (catch retained)",
        units="t/km²/year",
    ),
    EcopathScenarioColumn(
        ewef_label="Discards",
        pypath_name="discards",
        data_type="float64",
        required=False,
        description="Fishery discards (catch discarded)",
        units="t/km²/year",
    ),
    EcopathScenarioColumn(
        ewef_label="Biomass Accumulation",
        pypath_name="BioAcc",
        data_type="float64",
        required=False,
        description="Biomass accumulation rate",
        units="t/km²/year",
    ),
    EcopathScenarioColumn(
        ewef_label="Organism Type",
        pypath_name="org_type",
        data_type="int32",
        required=False,
        description="Group type: 0=detritus, 1=producer, 2=consumer",
        units="dimensionless",
        valid_range=(0, 2),
    ),
    EcopathScenarioColumn(
        ewef_label="Split Group?",
        pypath_name="split",
        data_type="int32",
        required=False,
        description="Is group split into stanzas? (0=no, 1=yes)",
        units="dimensionless",
    ),
    EcopathScenarioColumn(
        ewef_label="Stanza No",
        pypath_name="stanza_no",
        data_type="int32",
        required=False,
        description="Stanza (age class) number for multistanza groups",
        units="dimensionless",
    ),
    EcopathScenarioColumn(
        ewef_label="Age Start",
        pypath_name="age_start",
        data_type="float64",
        required=False,
        description="Starting age for this stanza",
        units="years",
    ),
    EcopathScenarioColumn(
        ewef_label="Leading stanza?",
        pypath_name="leading",
        data_type="int32",
        required=False,
        description="Is this the leading (adult) stanza? (0=no, 1=yes)",
        units="dimensionless",
    ),
]

ECOPATH_SCENARIO_FORMAT_DOC = """
Ecopath Scenario CSV Format (*_Scenario.csv)

Structure:
1. Two metadata rows with group count and parameter count
2. Header row with column names
3. Data rows with one row per group

Example:
    "Number of groups",22
    "Number of parameters",15
    Groups,B,P/B,Q/B,EE,P/Q,"Unass. Q",...
    Dolphins,0.0118645,0.43613,5.77288,-999,-999,0.2,...
    Bonito,-999,0.5038866,5.289608,0.6950969,-999,0.2,...
    ...

Notes:
- Missing values are represented as -999
- All numeric values use comma as decimal separator in some regions
- String values (group names) should be quoted if containing commas
- The order of columns can vary between files
"""

# ============================================================================
# DIET COMPOSITION MATRIX (CSV)
# ============================================================================

DIET_COMPOSITION_FORMAT_DOC = """
Diet Composition Matrix CSV Format (*_DC.csv)

Structure:
1. Two metadata rows with group counts
2. Header row: Prey/Pred, [predator_1, predator_2, ...]
3. Data rows: prey_group, diet_fraction_from_pred1, diet_fraction_from_pred2, ...
4. Optional: Import row (external diet source)

Dimensions:
- Rows: All groups + optional Import row
- Columns: Consumer groups (typically excludes detritus)

Example:
    "Number of groups incl. import",23
    "Number of predators",19
    Prey/Pred,Dolphins,Bonito,Bluefish,...
    Dolphins,0,0,0,...
    Bonito,0.05,0,0,...
    ...
    Import,0,0.2,0.2,...

Constraints:
- Values must be in range [0, 1]
- For each column (predator), sum should be ≈ 1.0 (±5%)
- Diagonal typically 0 (no cannibalism in basic models)
- Import row provides external energy input
"""

# ============================================================================
# DETRITUS FATE MATRIX (CSV)
# ============================================================================

DETRITUS_FATE_FORMAT_DOC = """
Detritus Fate Matrix CSV Format (*_DetFate.csv)

Structure:
1. Three metadata rows with group counts
2. Header row: Groups, [detritus_pool_1, detritus_pool_2, ..., Export]
3. Data rows: group_name, fate_fraction_to_detritus_1, ..., fate_fraction_export

Dimensions:
- Rows: All ecosystem groups
- Columns: Detritus groups + Export

Example:
    "Number of groups",22
    "Number of detrital groups incl. export",3
    "Number of detrital groups",2
    Groups,POM,Detritus,Export
    Dolphins,0,1,0
    Bonito,0,1,0
    ...
    Detritus,0,0,1
    Phytoplankton,0,0,1

Notes:
- Each row should sum to 1.0 (all mortality goes somewhere)
- Typically one "main" detritus pool and one "export" sink
- Detritus groups send their production to Export (row sums to 1)
"""

# ============================================================================
# MULTI-STANZA GROWTH PARAMETERS (CSV)
# ============================================================================

GROWTH_PARAMETERS_FORMAT_DOC = """
Multi-Stanza Growth Parameters CSV Format (*_GrowthParameters.csv)

Structure:
1. Metadata row with number of stanzas
2. Header row with parameter names
3. Data rows: one per stanza (age class)

Parameters:
- WInf: Asymptotic weight (g or reference unit)
- K: von Bertalanffy growth rate (year⁻¹)
- t0: Age at length=0 (years)
- WL50: Weight at 50% maturity (g)
- WAM: Weight at recruitment (g)
- Fec: Fecundity (eggs per unit weight per year)
- RecMode: Recruitment mode (integer code)

Example:
    "Number of stanzas",6
    Groups,WInf,K,t0,WL50,WAM,Fec,RecMode
    Anchovy 0-0+,250,0.8,-0.1,150,25,5000,0
    Anchovy 1-1+,250,0.8,-0.1,150,200,50000,0
    ...

Notes:
- vonBertalanffy: L(t) = Linf * (1 - exp(-K*(t-t0)))
- Weight conversion: W = a * L^b (stored in model separately)
- RecMode: 0=none, 1=beverton-holt, etc. (see EwE-F docs)
"""

# ============================================================================
# HDF5 RESULTS FILE
# ============================================================================

HDF5_RESULTS_FORMAT_DOC = """
EwE-F HDF5 Results File Format (*.h5)

Structure (typical - may vary by EwE-F version):

/
├── Biomass                  (time_steps, n_groups) float64
│   └── Biomass over time for each group
├── Catches                  (time_steps, n_groups) float64
│   └── Catch/landings over time for each group
├── GroupNames              (n_groups,) string/bytes
│   └── Names of ecosystem groups
├── Attributes
│   ├── Version             (string)
│   ├── Model               (string)
│   ├── Timestamp           (string)
│   └── ...other metadata
└── [Optional]
    ├── Diagnostics
    ├── EnergyBalance
    └── TimeSteps

Data Types:
- Floating-point arrays: typically float64 (double precision)
- Time-series: indexed by time_step (0 to N-1)
- Group identifiers: either index or group name string

Example Access (h5py):
    import h5py
    with h5py.File('model.h5', 'r') as f:
        biomass = f['Biomass'][:]          # (time, groups)
        catches = f['Catches'][:]
        groups = f['GroupNames'][:]
        print(biomass.shape)  # (100, 22) = 100 timesteps, 22 groups

Notes:
- Missing or invalid data typically represented as NaN
- Time-series indexed from 0, not year 1
- For Ecopath: usually 1 time step (t=0)
- For Ecosim: N time steps over simulation period
"""

# ============================================================================
# ECOSIM GROUP INFO (CSV)
# ============================================================================

ECOSIM_GROUPINFO_FORMAT_DOC = """
Ecosim Group Info CSV Format (*_GroupInfo.csv)

Per-group Ecosim dynamic parameters.

Typical Columns:
- Name: Group name
- Max.rel.P/B: Maximum relative P/B increase
- Feeding.time.adjust: Feeding time adjustment factor
- Max.rel.feeding.time: Maximum relative feeding time
- Feeding.time: Initial feeding time
- Fraction.of.other.mortality: Other (non-predation) mortality fraction
- Predator.effect.on.feeding.time: Vulnerability to predation feedback
- Density.dep.catchability: Density-dependent catchability factor
- Switching.power: Diet switching response exponent
- Advected: Is group advected/mobile? (0=no, 1=yes)

Example:
    Name,Max.rel.P.B,Feeding.time.adjust,...
    Dolphins,1.5,1.0,...
    Bonito,2.0,1.2,...
    ...

Notes:
- Parameters control individual group dynamics
- Most are optional (use defaults if not specified)
- Units vary by parameter (see EwE-F documentation)
"""

# ============================================================================
# ECOSPACE SPATIAL FORMATS
# ============================================================================

ECOSPACE_GRID_FORMAT_DOC = """
Ecospace Grid Definition Format (*.grid)

Defines spatial domain for Ecospace simulations.

Structure (typical):
- Grid dimensions: nx × ny cells
- Cell coordinates: longitude, latitude (geographic)
- Sea/land mask: binary grid (1=sea, 0=land)
- Optional: depth, temperature, current data

Format may be:
- ASCII matrix (simple text format)
- HDF5 (structured format)
- NetCDF (climate/oceanographic standard)

Content:
    nrows=94
    ncols=51
    xllcorner=20.5
    yllcorner=55.0
    cellsize=0.05
    NODATA_value=-9999
    [data matrix]

Example Metadata:
- Domain: Curonian Lagoon (94×51 cells)
- Extent: ~4.7 km × 2.55 km
- Sea cells: 1,576 out of 4,794 total
"""

ECOSPACE_HABITAT_FORMAT_DOC = """
Ecospace Habitat Suitability Format (*.habitat)

Defines spatial habitat preference/suitability for each group.

Structure:
- Rows: Groups (same order as ecosystem groups)
- Columns: Grid cells (nx × ny, linearized or 2D)
- Values: Habitat suitability index (0-1)
  - 1.0: Optimal habitat
  - 0.5: Suitable
  - 0.0: Unsuitable/no habitat

Usage:
- Affects predator-prey encounter rates
- Determines available prey in each cell
- Can vary with depth, temperature, season
"""

ECOSPACE_ADVECTION_FORMAT_DOC = """
Ecospace Advection/Flow Format (*.advection)

Defines water movement and biomass transport between grid cells.

Structure:
- Rows & Columns: Grid cells (one per cell, symmetric if flow is reversible)
- Values: Flow rate (proportion of biomass transferred per time step)
  - Positive: flow from row to column
  - Negative: flow from column to row
  - 0: no connection

Usage:
- Simulates ocean currents, river flow
- Transports biomass between spatial patches
- Usually derived from hydrodynamic models
"""

# ============================================================================
# SUMMARY DOCUMENTATION
# ============================================================================


def print_format_guide() -> str:
    """Return formatted guide to all EwE-F file formats."""
    sections = [
        ("Fortran Namelist Format", NAMELIST_FORMAT_DOC),
        ("Ecopath Scenario CSV", ECOPATH_SCENARIO_FORMAT_DOC),
        ("Diet Composition Matrix", DIET_COMPOSITION_FORMAT_DOC),
        ("Detritus Fate Matrix", DETRITUS_FATE_FORMAT_DOC),
        ("Multi-Stanza Growth Parameters", GROWTH_PARAMETERS_FORMAT_DOC),
        ("HDF5 Results File", HDF5_RESULTS_FORMAT_DOC),
        ("Ecosim GroupInfo CSV", ECOSIM_GROUPINFO_FORMAT_DOC),
        ("Ecospace Grid Format", ECOSPACE_GRID_FORMAT_DOC),
        ("Ecospace Habitat Suitability", ECOSPACE_HABITAT_FORMAT_DOC),
        ("Ecospace Advection Flows", ECOSPACE_ADVECTION_FORMAT_DOC),
    ]

    lines = ["=" * 80]
    lines.append("EwE-F DATA FORMAT GUIDE")
    lines.append("=" * 80)

    for title, doc in sections:
        lines.append(f"\n{title}")
        lines.append("-" * len(title))
        lines.append(doc.strip())
        lines.append("")

    return "\n".join(lines)
