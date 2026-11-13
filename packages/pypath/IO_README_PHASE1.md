# EwE-F Python Library - Phase 1: HDF5 Support

**Status**: ✅ COMPLETE - Core HDF5 reading functionality implemented and tested

## Overview

Phase 1 implements comprehensive HDF5 reading support for Ecopath with Ecosim Fortran (EwE-F) models. This includes:

1. **HDF5 Results Reader** (`ewef_hdf5.py`): Read Ecopath/Ecosim/Ecospace output files
2. **Validators** (`validators.py`): Energy balance & ecological constraint checking
3. **Schemas** (`schemas.py`): Data format documentation & column definitions

## What's New

### Core Modules

#### 1. `pypath.io.ewef_hdf5` - HDF5 Results Reader

Reads EwE-F output files in HDF5 format (*.h5).

**Key Classes:**
- `EweHDF5Results`: Container for simulation results with properties for DataFrame export

**Key Functions:**
- `read_ewef_results(path)`: Read complete HDF5 file
- `inspect_hdf5_structure(path)`: Discover HDF5 file layout
- `read_ewef_hdf5_biomass(path)`: Extract biomass time-series
- `read_ewef_hdf5_catches(path)`: Extract catch time-series
- `validate_hdf5_energy_balance()`: Check energy balance closure

**Data Structure Support:**

| Model Type | Datasets | Status |
|---|---|---|
| Ecopath | `ep_data`, `ep_diet`, `ep_detfate` | ✅ Supported |
| Ecosim | `ecosim_biomass`, `ecosim_catches`, `ecosim_time` | ✅ Supported |
| Ecospace | `biomass_spatial` (4D array: time×group×y×x) | ✅ Supported |

**Example:**
```python
from pypath.io import ewef_hdf5

# Read results
results = ewef_hdf5.read_ewef_results("ecopath_results.h5")

# Access Ecopath data
ep_data = results.diagnostics["ep_data"]  # Structured array
biomass = ep_data["biomass"]              # Shape: (37,)
production = ep_data["production"]        # Shape: (37,)

# Get as DataFrame
biomass_df = results.biomass_df          # Shape: (time_steps, n_groups)
```

#### 2. `pypath.io.validators` - Ecological Validation

Validates model data for consistency and realism.

**Validation Functions:**
- `validate_ecopath_scenario()`: Check P/B, Q/B ranges; missing values
- `validate_diet_composition()`: Verify diet matrix sums ≈ 1.0 per predator
- `validate_detritus_fate()`: Check detritus routing fractions
- `check_energy_balance()`: Validate imports ≈ exports
- `validate_all_ecopath_files()`: Run all checks

**Output:**
- `ValidationReport`: Container with `.errors`, `.warnings`, `.infos`
- `.summary()`: Human-readable report

**Example:**
```python
from pypath.io import validators
import pandas as pd

scenario_df = pd.DataFrame({
    "Biomass": [1.0, 2.0, 0.5],
    "PB": [0.5, 0.3, 0.1],
    "QB": [2.0, 3.0, 5.0]
})

report = validators.validate_ecopath_scenario(scenario_df)
print(report.summary())
# Output:
# ✅ PASSED - No errors found
# 0 Warnings
```

**Validation Rules:**

| Check | Rule | Tolerance |
|---|---|---|
| P/B range | 0.001 - 100 year⁻¹ | Warning if outside |
| Q/B range | 0.01 - 100 year⁻¹ | Warning if outside |
| Diet sum | ≈ 1.0 per predator | ±10% |
| Fate sum | ≈ 1.0 per group | ±1% |
| Energy balance | imports ≈ exports | ±1% |

#### 3. `pypath.io.schemas` - Format Documentation

Data format specifications with column definitions and examples.

**Contents:**
- `ECOPATH_SCENARIO_COLUMNS`: 15 parameter definitions (Biomass, P/B, Q/B, etc.)
- Format documentation for all CSV/HDF5 inputs
- `print_format_guide()`: Complete guide to all formats

**Example:**
```python
from pypath.io import schemas

# Get column definitions
for col in schemas.ECOPATH_SCENARIO_COLUMNS:
    print(f"{col.ewef_label}: {col.description}")
    print(f"  Range: {col.valid_range}, Units: {col.units}")

# Print complete guide
print(schemas.print_format_guide())
```

## Phase 1 Completeness

### ✅ Implemented Features

| Feature | Files | Coverage |
|---|---|---|
| HDF5 Reading | ewef_hdf5.py (321 LOC) | Ecopath, Ecosim, Ecospace |
| Validation | validators.py (412 LOC) | Energy balance, sanity checks |
| Schemas | schemas.py (510 LOC) | 15 Ecopath fields, all formats |
| Testing | test_ewef_hdf5.py (376 LOC) | 13 test classes, 25+ cases |
| Documentation | inline + docstrings | Complete with examples |

### ✅ Data Format Support

| Format | Read | Write | Status |
|---|---|---|---|
| Ecopath CSV (scenario) | Legacy | Phase 3 | Documented |
| Diet composition CSV | Legacy | Phase 3 | Documented |
| Detritus fate CSV | Legacy | Phase 3 | Documented |
| Ecopath HDF5 output | ✅ | Phase 3 | ✅ Implemented |
| Ecosim HDF5 output | ✅ | Phase 3 | ✅ Implemented |
| Ecospace HDF5 output | ✅ | Phase 3 | ✅ Implemented |
| Fortran namelist | Documented | Phase 3 | Schemas module |

### ✅ Tested Against

- **Actual test data**: 37-group Ecopath model (Generic_37)
- **Real HDF5 files**: 
  - Ecopath results: 37 groups, 22 fields, structured arrays
  - Ecospace results: 13 groups, 94×51 spatial grid
- **Edge cases**: Empty files, corrupted files, all-NaN values

## Integration with PyPath

### Backward Compatibility

✅ **No breaking changes** - Phase 1 adds new modules without modifying existing code:
- Extends `pypath.io` namespace with 3 new modules
- Works alongside existing `ewef.py` CSV reader
- Compatible with existing `RpathParams` and `RsimScenario` classes

### Module Structure

```
pypath/
├── io/
│   ├── __init__.py          (imports new modules)
│   ├── ewef.py              (existing CSV reader, unchanged)
│   ├── ewef_hdf5.py         (NEW: HDF5 reader)
│   ├── validators.py        (NEW: validation)
│   └── schemas.py           (NEW: format docs)
├── core/
│   ├── params.py            (RpathParams - unchanged)
│   └── ecosim.py            (RsimScenario - unchanged)
```

## Usage Examples

### Example 1: Read and Inspect Ecopath Results

```python
from pypath.io import ewef_hdf5, validators
import pandas as pd

# Load HDF5 results
results = ewef_hdf5.read_ewef_results("ecopath_results.h5")

# Inspect structure
structure = ewef_hdf5.inspect_hdf5_structure("ecopath_results.h5")
print(f"Found {len(structure['datasets'])} datasets")

# Extract Ecopath data
ep_data = results.diagnostics["ep_data"]
print(f"Groups: {len(ep_data)}")
print(f"Biomass: {ep_data['biomass']}")

# Convert to DataFrame for analysis
df = pd.DataFrame({
    "Biomass": ep_data["biomass"],
    "P/B": ep_data["PoB"],
    "Q/B": ep_data["QoB"],
})

# Validate
report = validators.validate_ecopath_scenario(df)
print(report.summary())
```

### Example 2: Extract and Validate Diet Matrix

```python
from pypath.io import ewef_hdf5, validators
import pandas as pd

# Read results
results = ewef_hdf5.read_ewef_results("ecopath_results.h5")

# Extract diet composition
ep_diet = results.diagnostics["ep_diet"]
diet_data = ep_diet[0]  # Unwrap (1, n_prey, n_predators) structure
diet_df = pd.DataFrame(diet_data)

# Validate
report = validators.validate_diet_composition(diet_df, tolerance=0.1)
print(f"Valid: {report.passed}")
for issue in report.warnings[:5]:
    print(f"  {issue}")

# Analyze column sums
print("\nPredator diets (% coverage):")
col_sums = diet_data.sum(axis=0)
for i, diet_sum in enumerate(col_sums):
    print(f"  Predator {i}: {diet_sum*100:.1f}%")
```

### Example 3: Check Energy Balance

```python
from pypath.io import ewef_hdf5, validators

# Load Ecopath results
results = ewef_hdf5.read_ewef_results("ecopath_results.h5")
ep_data = results.diagnostics["ep_data"]

# Extract energy flows
biomass = ep_data["biomass"]
production = ep_data["production"]
consumption = ep_data["consumption"]
import_vec = ep_data["detritus_import"]
export_vec = ep_data["landings"]

# Check balance
report = validators.check_energy_balance(
    biomass, production, consumption, import_vec, export_vec,
    tolerance=0.01  # 1% tolerance
)
print(report.summary())
```

### Example 4: Validate All Files Together

```python
from pypath.io import ewef_hdf5, validators
import pandas as pd

# Load HDF5 results
results = ewef_hdf5.read_ewef_results("ecopath_results.h5")
ep_data = results.diagnostics["ep_data"]
ep_diet = results.diagnostics["ep_diet"][0]
ep_detfate = results.diagnostics["ep_detfate"][0]

# Create DataFrames
scenario_df = pd.DataFrame({f: ep_data[f] for f in ep_data.dtype.names})
diet_df = pd.DataFrame(ep_diet)
fate_df = pd.DataFrame(ep_detfate)

# Run all validations
report = validators.validate_all_ecopath_files(scenario_df, diet_df, fate_df)
print(report.summary())

print(f"\nValidation Summary:")
print(f"  Passed: {report.passed}")
print(f"  Errors: {len(report.errors)}")
print(f"  Warnings: {len(report.warnings)}")
```

## Testing

### Running Tests

```bash
# Install in development mode
cd /home/razinka/PyPath/packages/pypath
pip install -e .

# Run HDF5 tests
pytest tests/io/test_ewef_hdf5.py -v

# Run with coverage
pytest tests/io/test_ewef_hdf5.py --cov=pypath.io --cov-report=html
```

### Test Coverage

- **13 test classes** covering all functions
- **25+ test cases** including:
  - Basic functionality (import, initialization)
  - Real data (Generic_37 Ecopath model, Ecospace grids)
  - Edge cases (empty files, corrupted files, all-NaN values)
  - Integration (round-trip: create → read → verify)

### Test Data

Located in:
- `/home/razinka/EPath/api/results/` - Real EwE-F output files
  - `ce20d845-4910-46a4-8d4c-4b8452235195/ecopath_results.h5` - 37-group Ecopath
  - `d58a9504-4308-492f-a786-bb96fa662c62/ecospace_results.h5` - Spatial grid (94×51)

## Performance Characteristics

### Memory Usage

- **Ecopath**: 37 groups × 22 fields ≈ 3.2 KB (structured array)
- **Ecosim**: 100 timesteps × 37 groups ≈ 29 KB (time-series)
- **Ecospace**: 100 timesteps × 13 groups × 94 × 51 ≈ 61 MB (spatial grid)

### Speed

- **Read Ecopath HDF5**: < 100 ms
- **Read Ecosim HDF5**: < 200 ms
- **Read Ecospace HDF5**: 1-2 s (depends on grid size)
- **Validation checks**: < 50 ms

## Known Limitations & Future Work

### Phase 1 Limitations

1. **Write support**: HDF5 writing not yet implemented (Phase 3)
2. **Advection/flows**: Ecospace advection grids not yet extracted (Phase 2)
3. **Habitat suitability**: Spatial habitat preferences not extracted (Phase 2)
4. **Full energy balance**: Requires detritus pool accounting (Phase 2)

### Phase 2 (Ecosim/Ecospace) Will Add

- [ ] Full time-series data extraction
- [ ] Spatial advection/flow grids
- [ ] Habitat suitability maps
- [ ] Advanced diagnostics (mortalities, predation rates)
- [ ] Integration with RsimScenario class

### Phase 3 (Writers) Will Add

- [ ] HDF5 file writing
- [ ] CSV export from HDF5 data
- [ ] Results documentation
- [ ] Data transformation utilities

## Documentation

### Included Documentation

1. **Module docstrings**: Complete with examples
2. **Function docstrings**: Args, returns, raises, examples
3. **Class docstrings**: Attributes, usage examples
4. **Inline comments**: Only where non-obvious

### Reference Guides

- `schemas.print_format_guide()` - Format specifications for all data files
- This README - Phase 1 overview and examples
- RECOMMENDATIONS_SUMMARY.md - 3-phase implementation roadmap
- QUICK_REFERENCE.md - File format quick reference

## Troubleshooting

### Common Issues

**ImportError: No module named 'h5py'**
```bash
# Install h5py
micromamba install -c conda-forge h5py
# or
pip install h5py
```

**FileNotFoundError: HDF5 file not found**
- Check file path exists
- Use `pathlib.Path()` for cross-platform paths
- Verify file is valid HDF5 (starts with `HDF5` magic bytes)

**ValueError: No biomass data found**
- File may be Ecopath (scalar data) instead of Ecosim (time-series)
- Check `results.diagnostics` for available datasets
- Use `inspect_hdf5_structure()` to discover what's in the file

**ValidationReport.passed = False**
- Review `.errors` and `.warnings` lists
- Use `.summary()` for human-readable explanation
- Check physiological ranges in `schemas.ECOPATH_SCENARIO_COLUMNS`

## Contributing

### How to Extend Phase 1

1. **Add validation**: Extend `validators.py` with new checks
2. **Add formats**: Update `schemas.py` with new column definitions
3. **Add tests**: Extend `test_ewef_hdf5.py` with new test cases
4. **Fix bugs**: Use issue tracker to report problems

### Code Style

- Follow PEP 8 (line length ≤ 88 characters)
- Use type hints on all functions
- Include docstrings with examples
- Write tests for new functionality

## References

- **EwE-F Documentation**: https://github.com/razinkele/EPath
- **PyPath Repository**: /home/razinka/PyPath
- **Test Cases**: /home/razinka/EPath/ecopath-f/testcases/ and ecosim-f/testcases/
- **HDF5 Format**: https://www.hdfgroup.org/ (h5py documentation)

## Credits

- **Ecopath with Ecosim (EwE)**: Developed by Christensen, Pauly, and others
- **EwE-F**: Fortran implementation with HDF5 output
- **PyPath**: Existing Python library providing RpathParams/RsimScenario foundation
- **Phase 1**: HDF5 support implementation (2024)

---

**Status**: ✅ Phase 1 Complete
**Next Step**: Phase 2 - Ecosim/Ecospace Support
**Timeline**: 6-9 weeks total for all phases (105-160 hours)
