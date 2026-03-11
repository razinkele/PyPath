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

```python
@dataclass
class ParameterDistribution:
    """A single parameter's sampling distribution."""
    param_name: str          # e.g. "Biomass", "PB", "QB", "Diet"
    group_idx: int           # 0-based group index
    base_value: float        # current Ecopath value
    cv: float                # coefficient of variation from pedigree
    dist_type: str           # "lognormal", "normal", "dirichlet"
    bounds: tuple[float, float] | None = None  # optional hard bounds
```

### PedigreeConfig

```python
@dataclass
class PedigreeConfig:
    """Configuration for pedigree-to-CV mapping."""
    index_to_cv: dict[int, float] = field(default_factory=lambda: {
        1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.4, 6: 0.5, 7: 0.6, 8: 0.8
    })
```

Used when importing from EwE databases where pedigree stores index values (1-8) rather than CVs directly. In the Python API, `params.pedigree` values are treated as CVs directly.

### Key Functions

- `build_distributions(params: RpathParams, config: PedigreeConfig | None = None) -> list[ParameterDistribution]`
  - Reads `params.pedigree` DataFrame columns (Biomass, PB, QB, Diet, Fleet1..N)
  - Maps each cell to a distribution:
    - Biomass, PB, QB, Catch → **log-normal** (strictly positive)
    - Diet → **Dirichlet** (per-consumer column, CV controls concentration parameter)
  - Skips parameters with CV = 0 (known exactly)
  - If `config` provided, converts index values to CVs first

- `sample_parameters(distributions: list[ParameterDistribution], n_samples: int, method: str = "lhs", rng: np.random.Generator | None = None) -> list[dict]`
  - Returns N parameter sets, each a dict mapping `(param_name, group_idx) → sampled_value`
  - `method="lhs"`: Uses `scipy.stats.qmc.LatinHypercube` for stratified coverage
  - `method="random"`: Direct sampling from distributions via `rng`
  - Diet columns sampled via Dirichlet, then renormalized to sum to 1.0

- `apply_sample(params: RpathParams, sample: dict) -> RpathParams`
  - Returns a deep copy of params with sampled values applied to `params.model` and `params.diet`
  - Original params object is never modified

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
    ecosim_years: range | None = None  # years for Ecosim runs
    store_runs: bool = False        # keep individual run outputs
    n_jobs: int = 1                 # parallelism (joblib → futures → sequential)
    mediation: "MediationCollection | None" = None  # pass through to rsim_run
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
    ecosim_stats: dict[str, np.ndarray] | None  # key=output, array shape: (timesteps, groups, 7_stats)

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
   b. `rpath(sampled_params)` → keep if mass balance succeeds (no exception)
   c. Collect Ecopath outputs (Biomass, TL, etc.)
4. If not `ecopath_only`, for each feasible result:
   a. `rsim_scenario(rpath_result, sampled_params, years=config.ecosim_years)`
   b. `rsim_run(scenario, mediation=config.mediation)`
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

**Total runs:** `n_samples * (2k + 2)`. For 30 params, 1024 base → 63,488 runs. User is warned if this exceeds a threshold (e.g. 10,000).

### Configuration & Entry Point

```python
@dataclass
class SensitivityConfig:
    method: str = "morris"           # "morris" or "sobol"
    n_trajectories: int = 10         # Morris only
    n_levels: int = 4                # Morris only
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
) -> MorrisResult | SobolResult:
```

---

## Section 4: I/O & EwE Integration

### Schema Additions (`_ewe_schema.py`)

```python
"Pedigree": {
    "PedigreeID": "INTEGER",
    "LevelName": "TEXT",
    "LevelDescription": "TEXT",
    "IndexValue": "INTEGER",
    "ConfidenceInterval": "DOUBLE",
},
"EcopathGroupPedigree": {
    "GroupID": "INTEGER",
    "ScenarioID": "INTEGER",
    "BiomassCI": "INTEGER",
    "PBCI": "INTEGER",
    "QBCI": "INTEGER",
    "DietCI": "INTEGER",
    "CatchCI": "INTEGER",
},
```

### Reader (`ewemdb.py`)

```python
def read_pedigree(db_path: str) -> tuple[PedigreeConfig, dict[int, dict[str, float]]]:
    """Read pedigree tables from EwE database.

    Returns (config, group_pedigree) where:
    - config: PedigreeConfig with index_to_cv mapping from Pedigree table
    - group_pedigree: {group_id: {"Biomass": cv, "PB": cv, ...}} with CVs already converted
    """
```

- Reads `Pedigree` table → builds `index_to_cv` mapping
- Reads `EcopathGroupPedigree` → converts indices to CVs using the mapping
- Missing tables → returns default config + empty dict

### Writer

- `write_pedigree()` on CsvBundleWriter and AccessWriter
- `write_ewemdb()` gains optional `pedigree_config` parameter
- Writes both Pedigree and EcopathGroupPedigree tables

### MC Results Export

No EwE schema for MC results — these are PyPath-native:
- `MCResult.to_dataframe()` → summary stats as pandas DataFrame
- `MCResult.to_dict()` → JSON-serializable dict
- Users save to CSV/JSON as preferred

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
| `io/_ewe_schema.py` | Add Pedigree + EcopathGroupPedigree table definitions |
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
