# Monte Carlo / Pedigree Uncertainty — Design Spec

**Phase:** 3 of PyPath EwE Feature Roadmap
**Date:** 2026-03-11
**Status:** Approved

## Overview

Add pedigree-based Monte Carlo uncertainty analysis and sensitivity screening to PyPath. Pedigree values (coefficients of variation) define parameter distributions; Monte Carlo sampling propagates uncertainty through Ecopath mass balance and optionally Ecosim dynamics. Morris screening and optional Sobol analysis identify which parameters most influence model outputs.

## Architecture

Three new modules, split by concern:

```
core/pedigree.py      — Pedigree-to-CV mapping, distributions, sampling
core/montecarlo.py    — MC runner, ensemble management, result aggregation
core/sensitivity.py   — Morris screening, optional Sobol (SALib)
```

Dependency flow: `sensitivity.py` → `montecarlo.py` → `pedigree.py` → `params.py`

---

## Section 1: Data Structures — `core/pedigree.py`

### ParameterDistribution

Two distribution types: scalar (for Biomass, PB, QB, Catch) and diet (for diet composition vectors).

```python
@dataclass
class ScalarDistribution:
    """A single scalar parameter's sampling distribution (log-normal)."""
    param_name: str          # e.g. "Biomass", "PB", "QB", "Catch"
    group_idx: int           # 0-based group index
    base_value: float        # current Ecopath value
    cv: float                # coefficient of variation from pedigree
    bounds: tuple[float, float] | None = None  # optional hard bounds

@dataclass
class DietDistribution:
    """A predator's diet composition distribution (Dirichlet)."""
    pred_idx: int            # 0-based predator group index
    base_proportions: np.ndarray  # current diet column (prey proportions, sum=1)
    cv: float                # controls Dirichlet concentration (higher CV = more spread)

ParameterDistribution = ScalarDistribution | DietDistribution
```

**Log-normal parameterization (mean-preserving):**
For a parameter with base value `v` and coefficient of variation `CV`:
- `sigma = sqrt(ln(1 + CV^2))`
- `mu = ln(v) - sigma^2/2`
- This ensures `E[X] = v` (the mean of the sampled distribution equals the base value).

**Dirichlet parameterization:**
For a diet column with base proportions `p` and CV:
- Concentration `alpha = p / CV^2` (higher CV → flatter distribution, lower CV → concentrated near base)
- Each sample is drawn from `Dirichlet(alpha)` and automatically sums to 1.0.

### PedigreeConfig

```python
@dataclass
class PedigreeConfig:
    """Configuration for pedigree-to-CV mapping.

    EwE 6 stores pedigree as (VarName, LevelID) pairs in the Pedigree table,
    where each VarName has its own set of levels with IndexValue (confidence)
    and Confidence (%) columns. The IndexValue is treated as the CV.

    In the Python API, params.pedigree values are treated as CVs directly.
    PedigreeConfig is only needed when importing from EwE databases.
    """
    # VarName -> {LevelID -> CV} mapping (populated from Pedigree table)
    level_to_cv: dict[str, dict[int, float]] = field(default_factory=dict)
```

### Key Functions

- `build_distributions(params: RpathParams, config: PedigreeConfig | None = None) -> list[ParameterDistribution]`
  - Reads `params.pedigree` DataFrame columns (Biomass, PB, QB, Diet, Fleet1..N)
  - Maps each cell to a distribution:
    - Biomass, PB, QB, Catch → **ScalarDistribution** (log-normal, strictly positive)
    - Diet → **DietDistribution** (Dirichlet, per-consumer column)
  - Skips parameters with CV = 0 (known exactly)
  - If all pedigree values are 1.0 (the default from `create_rpath_params`), issue `warnings.warn("All pedigree values are 1.0 (default = 100% CV). Consider setting pedigree values before MC analysis.", UserWarning)`. A CV of 1.0 is technically valid but produces very wide distributions and low feasibility rates.
  - Skips stanza groups (multi-stanza parameters are internally derived; sampling lead stanza Biomass propagates through stanza calculations — warn via `logger.info`)
  - If `config` provided and `params.pedigree` contains LevelID integers, converts to CVs using `config.level_to_cv` mapping
  - Fleet pedigree CVs apply to **landings only** (discards are not independently sampled — they scale proportionally)

- `sample_parameters(distributions: list[ParameterDistribution], n_samples: int, method: str = "lhs", rng: np.random.Generator | None = None) -> list[dict]`
  - Returns N parameter sets as list of dicts
  - For `ScalarDistribution`: dict key = `(param_name, group_idx)`, value = float
  - For `DietDistribution`: dict key = `("Diet", pred_idx)`, value = np.ndarray (prey proportions)
  - `method="lhs"`: Uses `scipy.stats.qmc.LatinHypercube` for stratified coverage. LHS operates in the unit hypercube; samples are mapped to parameter space via inverse CDF of the target distribution.
  - `method="random"`: Direct sampling from distributions via `rng`

- `apply_sample(params: RpathParams, sample: dict) -> RpathParams`
  - Creates a targeted copy: `params.model.copy()`, `params.diet.copy()`, and `copy.deepcopy(params.stanzas)` if stanza groups exist (stanza calculations can modify parameters in-place)
  - Applies sampled scalar values to `model` DataFrame
  - Applies sampled diet vectors to `diet` DataFrame columns
  - Original params object is never modified

**Edge cases:**
- **Dirichlet with zero prey proportions:** Prey items with proportion 0.0 are excluded from Dirichlet sampling (alpha must be > 0). Sampled vector is placed back into the full prey vector with zeros preserved.
- **LHS + Dirichlet:** LHS applies only to scalar parameters. Diet distributions are always sampled directly from Dirichlet (no inverse CDF for multivariate distributions).
- **Failed Ecosim runs:** If `rsim_run()` crashes mid-simulation (shorter output), the run is counted as failed and excluded from streaming statistics. Only complete runs contribute to `ecosim_stats`.

---

## Section 2: Monte Carlo Runner — `core/montecarlo.py`

### MCConfig

```python
@dataclass
class MCConfig:
    """Monte Carlo run configuration."""
    n_samples: int = 1000
    method: str = "lhs"             # "lhs" or "random"
    seed: int | None = None         # for reproducibility
    ecopath_only: bool = False      # skip Ecosim propagation
    ecosim_years: range | None = None  # years for Ecosim; defaults to range(1, 11) if None and not ecopath_only
    store_runs: bool = False        # keep individual run outputs
    n_jobs: int = 1                 # parallelism (joblib → futures → sequential)
    # Ecosim pass-through options
    mediation: "MediationCollection | None" = None
    ecosim_method: str = "RK4"      # "RK4" or "AB" integration method
    eco_area: float = 1.0           # Ecopath model area
```

### MCResult

```python
@dataclass
class MCResult:
    """Monte Carlo ensemble results."""
    n_total: int
    n_feasible: int
    n_ecosim: int

    # Streaming statistics (always available)
    ecopath_stats: dict[str, pd.DataFrame]  # key=param, df has columns: mean, std, p5, p25, p50, p75, p95
    ecosim_stats: dict[str, np.ndarray] | None  # key=output, array shape: (timesteps, n_groups, 7_stats)
    # Note: ecosim_stats excludes the 1-based padding column 0 from out_Biomass.
    # Indices align with 0-based group_idx. timesteps = monthly resolution from Ecosim.

    # Optional raw storage
    ecopath_runs: list[dict] | None
    ecosim_runs: list[np.ndarray] | None

    # Diagnostics
    feasibility_rate: float         # n_feasible / n_total
    parameter_samples: pd.DataFrame | None  # sampled values for reproducibility

    def to_dataframe(self) -> pd.DataFrame: ...  # summary stats as flat DataFrame
    def to_dict(self) -> dict: ...                # JSON-serializable dict
```

### run_montecarlo

```python
def run_montecarlo(
    params: RpathParams,
    config: MCConfig | None = None,
    *,
    pedigree_config: PedigreeConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MCResult:
```

**Execution flow:**
1. `build_distributions(params, pedigree_config)` → parameter distributions
2. `sample_parameters(distributions, config.n_samples, config.method, rng)` → N samples
3. For each sample:
   a. `apply_sample(params, sample)` → sampled_params
   b. `rpath(sampled_params, eco_area=config.eco_area)` → keep if mass balance succeeds (no exception)
   c. Collect Ecopath outputs (Biomass, TL, etc.)
4. If not `ecopath_only`, for each feasible result:
   a. `rsim_scenario(rpath_result, sampled_params, years=config.ecosim_years)`
   b. `rsim_run(scenario, method=config.ecosim_method, mediation=config.mediation)`
   c. Collect `out_Biomass` trajectories
5. Compute streaming statistics incrementally (Welford's algorithm for mean/variance)
6. Return `MCResult`

**Parallelism strategy:**
- `n_jobs=1`: sequential loop
- `n_jobs>1`: try `joblib.Parallel(n_jobs=n_jobs)(delayed(worker)(sample) for sample in samples)`
- If joblib unavailable: fall back to `concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs)`
- If that fails: fall back to sequential with a warning

**Error handling:**
- Individual `rpath()` failures (mass balance infeasible) are counted but don't stop the run
- Individual `rsim_run()` failures (numerical instability) are counted and skipped
- If `n_feasible == 0`, return result with `feasibility_rate=0` and empty stats

---

## Section 3: Sensitivity Analysis — `core/sensitivity.py`

### Morris Screening (native, no external deps)

```python
@dataclass
class MorrisResult:
    parameter_names: list[str]       # e.g. ["Biomass_0", "PB_1", "QB_1"]
    mu_star: np.ndarray              # |mean| of elementary effects (importance)
    sigma: np.ndarray                # std of elementary effects (interaction/nonlinearity)
    mu: np.ndarray                   # signed mean (direction)
    output_name: str
```

**Morris algorithm (Campolongo et al. 2007):**
1. Build parameter list from `build_distributions()` — k parameters with bounds
2. Generate `n_trajectories` random base points in the unit hypercube, grid-discretized to `n_levels`
3. For each trajectory: perturb one parameter at a time (k+1 evaluations per trajectory)
4. Map unit hypercube points to actual parameter values via inverse CDF
5. Run model (rpath + optional rsim_run) at each point
6. Compute elementary effects: `EE_i = (y(x+delta_i) - y(x)) / delta_i`
7. Aggregate: `mu_star_i = mean(|EE_i|)`, `sigma_i = std(EE_i)`, `mu_i = mean(EE_i)`

**Total runs:** `n_trajectories * (k + 1)`. For 30 params, 10 trajectories → 310 runs.

### Sobol Analysis (requires SALib)

```python
@dataclass
class SobolResult:
    parameter_names: list[str]
    S1: np.ndarray                   # first-order indices
    ST: np.ndarray                   # total-order indices
    S1_conf: np.ndarray
    ST_conf: np.ndarray
    output_name: str
```

**Implementation:** Delegates to SALib for sampling and analysis.
- Check `HAS_SALIB` flag; raise `ImportError("Install SALib for Sobol analysis: pip install SALib")` if missing
- Use `SALib.sample.saltelli.sample()` for quasi-random design
- Run model at all sample points (same pipeline as MC)
- Use `SALib.analyze.sobol.analyze()` for index computation

**Total runs:** `n_samples * (2k + 2)`. For 30 params, 1024 base → 63,488 runs. If total runs exceed 10,000, issue `warnings.warn(f"Sobol analysis requires {n_runs} model evaluations", UserWarning)` before proceeding.

### Configuration & Entry Point

```python
@dataclass
class SensitivityConfig:
    method: str = "morris"           # "morris" or "sobol"
    n_trajectories: int = 10         # Morris only
    n_levels: int = 4                # Morris: grid levels (delta = n_levels/(2*(n_levels-1)))
    n_samples: int = 1024            # Sobol only
    seed: int | None = None
    n_jobs: int = 1
    output_variable: str = "Biomass"
    output_group_idx: int | None = None  # specific group, or None → all
    ecopath_only: bool = False
    ecosim_years: range | None = None

def run_sensitivity(
    params: RpathParams,
    config: SensitivityConfig | None = None,
    *,
    pedigree_config: PedigreeConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> MorrisResult | SobolResult:
```

---

## Section 4: I/O & EwE Integration

### Schema Additions (`_ewe_schema.py`)

Based on actual EwE 6.6+ database structure (verified against LT2022 model):

```python
# Pedigree metadata: defines level names and CVs per variable type
"Pedigree": {
    "LevelID": "INTEGER",
    "LevelName": "TEXT",
    "VarName": "TEXT",          # PBInput, QBInput, BiomassAreaInput, DietComp, TCatchInput
    "Sequence": "INTEGER",
    "IndexValue": "DOUBLE",     # this IS the CV value (e.g. 0.1, 0.2, 0.5)
    "Confidence": "DOUBLE",     # confidence percentage (e.g. 70.0, 60.0)
    "LevelColor": "INTEGER",
    "Description": "TEXT",
},
# Per-group pedigree assignment: normalized (one row per group-variable pair)
"EcopathGroupPedigree": {
    "GroupID": "INTEGER",
    "VarName": "TEXT",          # matches Pedigree.VarName
    "LevelID": "INTEGER",      # FK to Pedigree.LevelID
},
# Monte Carlo sample metadata
"EcopathSample": {
    "SampleID": "INTEGER",
    "Hash": "TEXT",
    "Source": "TEXT",
    "Generated": "TEXT",
    "Rating": "DOUBLE",
    "SS": "DOUBLE",
},
# Sampled group parameters per MC iteration
"EcopathGroupSample": {
    "SampleID": "INTEGER",
    "GroupID": "INTEGER",
    "Biomass": "DOUBLE",
    "ProdBiom": "DOUBLE",
    "ConsBiom": "DOUBLE",
    "EcoEfficiency": "DOUBLE",
    "BiomAcc": "DOUBLE",
    "ImpVar": "DOUBLE",
    "BiomAccRate": "DOUBLE",
},
# Sampled diet composition per MC iteration
"EcopathDietCompSample": {
    "SampleID": "INTEGER",
    "PredID": "INTEGER",
    "PreyID": "INTEGER",
    "Diet": "DOUBLE",
},
# Sampled catch per MC iteration
"EcopathGroupCatchSample": {
    "SampleID": "INTEGER",
    "GroupID": "INTEGER",
    "FleetID": "INTEGER",
    "Landing": "DOUBLE",
    "Discards": "DOUBLE",
},
```

### Reader (`ewemdb.py`)

```python
def read_pedigree(db_path: str) -> tuple[PedigreeConfig, pd.DataFrame]:
    """Read pedigree tables from EwE database.

    Returns (config, group_pedigree) where:
    - config: PedigreeConfig with level_to_cv mapping from Pedigree table
    - group_pedigree: DataFrame with columns [GroupID, VarName, CV]
      where CV is the IndexValue from the Pedigree table for the assigned LevelID
    """
```

- Reads `Pedigree` table → builds `config.level_to_cv = {VarName: {LevelID: IndexValue}}` mapping
- Reads `EcopathGroupPedigree` → joins with Pedigree on (VarName, LevelID) to get CV
- Maps VarName to params.pedigree column names: `BiomassAreaInput→Biomass`, `PBInput→PB`, `QBInput→QB`, `DietComp→Diet`, `TCatchInput→Catch`
- Updates `params.pedigree` DataFrame with CVs from database
- Missing tables → returns default config + empty DataFrame

### Writer

- `write_pedigree()` on CsvBundleWriter and AccessWriter
- `write_ewemdb()` gains optional `pedigree_config` parameter
- Writes Pedigree and EcopathGroupPedigree tables

### MC Sample Writer

- `write_mc_samples(result: MCResult)` writes the 4 sample tables:
  - `EcopathSample` — one row per feasible run (SampleID, Hash, Source="PyPath MC")
  - `EcopathGroupSample` — sampled B, PB, QB per group per run
  - `EcopathDietCompSample` — sampled diet per pred-prey per run
  - `EcopathGroupCatchSample` — sampled landings/discards per group-fleet per run
- Only available when `store_runs=True` (raw samples required)

### MC Results Export (PyPath-native)

- `MCResult.to_dataframe()` → summary stats as pandas DataFrame
- `MCResult.to_dict()` → JSON-serializable dict

---

## Section 5: Testing Strategy

### Unit Tests

**`test_pedigree.py`** (~15 tests):
- TestParameterDistribution: construction, CV mapping, bounds
- TestBuildDistributions: correct dist_type per param (lognormal for B/PB/QB, dirichlet for diet), zero CV skipped
- TestSampleParameters: LHS returns correct shape, random returns correct shape, seed reproducibility
- TestApplySample: values correctly applied, original params unchanged (deep copy)
- TestPedigreeConfig: default EwE index→CV table, custom table override

**`test_montecarlo.py`** (~12 tests):
- TestMCConfig: defaults, validation
- TestMCResult: streaming stats computation, feasibility rate, store_runs toggle
- TestRunMontecarlo: small 3-group model, n_samples=10, ecopath_only=True
- TestRunMontecarloEcosim: n_samples=5, ecopath_only=False, verify ecosim_stats populated
- TestParallelFallback: n_jobs=2 without joblib falls back gracefully

**`test_sensitivity.py`** (~10 tests):
- TestMorrisDesign: trajectory generation produces correct number of points
- TestMorrisResult: mu_star/sigma on known analytic function
- TestRunSensitivityMorris: 3-group model, verify result structure
- TestSobolMissing: raises ImportError with hint when SALib absent
- TestSobolResult: structure validation (pytest.importorskip("SALib"))

### Integration Tests

**`test_mc_integration.py`** (~5 tests, `@pytest.mark.slow`):
- Full MC pipeline: pedigree → MC(n=50, ecopath_only) → feasibility_rate > 0
- Full MC pipeline: pedigree → MC(n=10, ecosim) → ecosim_stats shape correct
- Morris on 3-group model → all params ranked
- Regression: zero-CV pedigree → all samples identical to base
- store_runs=True → raw outputs accessible

### I/O Tests

**`test_pedigree_io.py`** (~6 tests):
- Schema table definitions present (Pedigree, EcopathGroupPedigree)
- read_pedigree with mocked tables → correct config + group pedigree
- Index→CV conversion matches EwE defaults
- Missing tables → graceful empty return

**Total: ~48 new tests**

---

## Section 6: Files Changed/Created

### New Files

| File | Purpose |
|------|---------|
| `core/pedigree.py` | ParameterDistribution, PedigreeConfig, build_distributions, sample_parameters, apply_sample |
| `core/montecarlo.py` | MCConfig, MCResult, run_montecarlo |
| `core/sensitivity.py` | MorrisResult, SobolResult, SensitivityConfig, run_sensitivity |
| `tests/test_pedigree.py` | ~15 unit tests |
| `tests/test_montecarlo.py` | ~12 unit tests |
| `tests/test_sensitivity.py` | ~10 unit tests |
| `tests/test_mc_integration.py` | ~5 integration tests |
| `tests/test_pedigree_io.py` | ~6 I/O tests |

### Modified Files

| File | Change |
|------|--------|
| `io/_ewe_schema.py` | Add 6 tables: Pedigree, EcopathGroupPedigree, EcopathSample, EcopathGroupSample, EcopathDietCompSample, EcopathGroupCatchSample |
| `io/ewemdb.py` | Add read_pedigree() function |
| `io/ewe_writer.py` | Add pedigree_config param to write_ewemdb() |
| `io/_csv_bundle_writer.py` | Add write_pedigree() method |
| `io/_access_writer.py` | Add write_pedigree() delegation |
| `core/__init__.py` | Export new classes/functions, HAS_JOBLIB + HAS_SALIB flags |
| `io/__init__.py` | Export read_pedigree |

### Optional Dependencies

| Package | Purpose | Fallback |
|---------|---------|----------|
| `joblib` | Parallel MC/sensitivity runs | `concurrent.futures.ProcessPoolExecutor` → sequential |
| `SALib` | Sobol sensitivity analysis | `ImportError` with install hint |
