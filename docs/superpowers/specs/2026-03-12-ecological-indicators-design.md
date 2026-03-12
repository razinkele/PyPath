# Ecological Indicators Design Spec

**Goal:** Add flow analysis (Ulanowicz ascendency framework) and ecosystem summary indicators to PyPath, providing standard ecological metrics for ecosystem-based management reporting. Both static (from Ecopath) and dynamic (from Ecosim time series) versions.

**Approach:** New `core/indicators.py` module with pure functions returning dataclass results. Fix existing placeholders in `NetworkIndices`. No new dependencies.

---

## 1. Module Structure

### New file: `core/indicators.py`

**Flow analysis functions** (all take a balanced `Rpath` object):
- `flow_analysis(rpath) → FlowAnalysis` — TST, ascendency, capacity, overhead, Finn cycling, transfer efficiency
- `finn_cycling_index(rpath) → float` — standalone, also called by `calculate_network_indices()`
- `transfer_efficiency(rpath) → np.ndarray` — per-trophic-level efficiency, also called by `calculate_network_indices()`

**Ecosystem summary functions:**
- `ecosystem_indicators(rpath) → EcosystemIndicators` — static snapshot from balanced model
- `ecosystem_indicators_timeseries(output, scenario, rpath) → pd.DataFrame` — same metrics per year from Ecosim output

### Modified file: `core/analysis.py`

- `calculate_network_indices()` calls `finn_cycling_index()` and `transfer_efficiency()` to replace hardcoded placeholders
- `NetworkIndices.transfer_efficiency` type changes from `float` to `float` (mean of per-level array, was hardcoded 0.1)

### Exports: `core/__init__.py`

Add: `FlowAnalysis`, `EcosystemIndicators`, `flow_analysis`, `finn_cycling_index`, `transfer_efficiency`, `ecosystem_indicators`, `ecosystem_indicators_timeseries`

---

## 2. Data Types

```python
@dataclass
class FlowAnalysis:
    total_system_throughput: float  # TST: sum of all flows
    ascendency: float              # System organization (bits × flow)
    capacity: float                # Development capacity (upper bound)
    overhead: float                # Capacity - Ascendency (resilience)
    relative_ascendency: float     # Ascendency / Capacity [0-1]
    finn_cycling_index: float      # Fraction of TST recycled [0-1]
    transfer_efficiency: np.ndarray  # Per-TL efficiency array

@dataclass
class EcosystemIndicators:
    mtl_catch: float              # Mean trophic level of catch
    marine_trophic_index: float   # MTL catch excluding TL < 3.25
    catch_biomass_ratio: float    # Total catch / total biomass
    gross_efficiency: float       # Total catch / net primary production
    shannon_diversity: float      # Shannon H' of biomass (living groups)
    kempton_q: float              # Biomass evenness in TL 3-4 range
```

---

## 3. Flow Analysis Algorithms

### Flow matrix construction

From a balanced Ecopath model, build an extended flow matrix that includes internal compartments (living groups + detritus) plus external sinks (respiration, export). This ensures ascendency is computed over the complete system.

**Internal flows (between compartments):**
- **Consumption:** `T[pred, prey] = rpath.DC[prey, pred] × rpath.QB[pred] × rpath.Biomass[pred]`
- **Flow to detritus:** `FD[i] = Unassim[i] × QB[i] × B[i] + (1 - EE[i]) × PB[i] × B[i]`

**External flows (to sinks outside the system):**
- **Respiration per group:** `R[i] = (1 - Unassim[i]) × QB[i] × B[i] - PB[i] × B[i]`
- **Export (catch):** `E[i] = Catch[i]` (sum of landings + discards across fleets)

For ascendency/capacity: the flow matrix `T` is extended with additional sink rows for respiration and export. For Finn cycling: only internal flows (consumption + detritus) are used since cycling cannot involve external sinks.

All values come from `rpath` numpy arrays (Biomass, PB, QB, EE, Unassim, type) and `rpath.DC` (diet composition). All arrays are 1-based (index 0 unused).

### Total System Throughput (TST)

`TST = Σ all consumption flows + Σ respiration + Σ flow to detritus + Σ exports`

### Ascendency

```
A = Σᵢⱼ T[i,j] × log₂(T[i,j] × TST / (T_in[i] × T_out[j]))
```

Where `T_in[i] = Σⱼ T[i,j]` (total inflow to i) and `T_out[j] = Σᵢ T[i,j]` (total outflow from j). Only summed over terms where `T[i,j] > 0` AND `T_in[i] > 0` AND `T_out[j] > 0`. Uses log base 2.

### Development Capacity

```
C = -Σᵢⱼ T[i,j] × log₂(T[i,j] / TST)
```

### Overhead

`Overhead = C - A`

### Relative Ascendency

`A/C` — ratio in [0, 1]. Higher values indicate more organized, less redundant systems.

### Finn Cycling Index

Following Finn (1976) / Ulanowicz (1986):

1. Compute throughflow for each compartment: `throughflow[i] = Σⱼ T[i,j]` (total flow through compartment i; at steady state, inflow = outflow)
2. Build the output coefficient matrix: `G[i,j] = T[i,j] / throughflow[j]` (fraction of j's throughflow going to i)
3. Compute Leontief inverse: `N = (I - G)⁻¹` using `np.linalg.inv()`
4. Straight-through flow: `straight[i] = throughflow[i] / N[i,i]`
5. Cycled flow: `cycled[i] = throughflow[i] - straight[i]`
6. `FCI = Σ cycled[i] / TST`

If `(I - G)` is singular (degenerate model), return 0.0 with a warning. Guard against zero throughflow by skipping compartments with throughflow = 0.

### Transfer Efficiency

Simplified integer-bin approach (not full Lindeman spine decomposition). Sufficient for summary reporting; full fractional decomposition can be added later.

1. Compute trophic levels for all groups (from `rpath.TL`)
2. Assign integer TL bins: `bin[i] = floor(TL[i])`
3. For each trophic level L (from 2 upward):
   - Input = total consumption by groups in bin L
   - Output = total consumption of groups in bin L by groups in bin L+1
   - `TE[L] = Output / Input` (0.0 if Input = 0)
4. Return array indexed by TL bin (starting from TL 2)
5. Mean TE (excluding zeros) goes into `NetworkIndices.transfer_efficiency`

---

## 4. Ecosystem Summary Indicators

### Static (from `Rpath`)

**MTL catch:**
```
mtl_catch = Σ(TL[i] × Catch[i]) / Σ(Catch[i])
```
Over all groups with Catch > 0. Returns `np.nan` if total catch is 0.

**Marine Trophic Index:**
Same as MTL catch but only groups with TL ≥ 3.25 (Pauly & Watson 2005 cutoff).

**Catch/Biomass ratio:**
```
Σ(Catch[i]) / Σ(B[i])  (living groups only)
```

**Gross Efficiency:**
```
Σ(Catch[i]) / NPP
```
Where NPP = Σ(PB[i] × B[i]) for primary producers (Type = 1). Returns `np.nan` if NPP = 0.

**Shannon diversity:**
```
H' = -Σ(p[i] × ln(p[i]))  where p[i] = B[i] / Σ(B)
```
Over living groups (Type 0 and 1) with B > 0. Uses natural log.

**Kempton's Q:**
```
Q = 0.5 × S / (ln(B_75) - ln(B_25))
```
Where S = number of living groups with TL in [3, 4), B_75 and B_25 are the 75th and 25th percentile biomasses of those groups. Returns `np.nan` if fewer than 4 groups in range or if B_25 = B_75.

### Extracting values from `Rpath`

All `Rpath` arrays are 1-based (index 0 is unused). Groups are indexed 1..NUM_GROUPS.

- `B[i]` = `rpath.Biomass[i]`
- `PB[i]` = `rpath.PB[i]`
- `QB[i]` = `rpath.QB[i]`
- `EE[i]` = `rpath.EE[i]`
- `TL[i]` = `rpath.TL[i]` (trophic level, computed during balance)
- `Catch[i]` = `np.sum(rpath.Landings[i, 1:]) + np.sum(rpath.Discards[i, 1:])` (Landings/Discards are 2D: [groups+1, gears+1], 1-based)
- `Type[i]` = `rpath.type[i]` (0=consumer, 1=producer, 2=detritus, 3=fleet)
- `Unassim[i]` = `rpath.Unassim[i]`
- `DC[pred, prey]` = `rpath.DC[prey, pred]` (diet composition matrix)

### Dynamic (from `RsimOutput`)

`ecosystem_indicators_timeseries(output, scenario, rpath) → pd.DataFrame`

Parameters:
- `output`: `RsimOutput` with `annual_Biomass[years, groups+1]` and `annual_Catch[years, groups+1]` (pre-computed annual means, 1-based group indexing)
- `scenario`: `RsimScenario` for group count and time range
- `rpath`: balanced model for trophic levels and group types

Computes per-year using the pre-aggregated annual arrays:
- MTL catch, Marine Trophic Index, catch/biomass ratio, gross efficiency, Shannon diversity

Returns DataFrame with columns: `year`, `mtl_catch`, `marine_trophic_index`, `catch_biomass_ratio`, `gross_efficiency`, `shannon_diversity`. One row per year.

Trophic levels are static (from `rpath.TL`). Kempton Q is excluded from timeseries (rarely meaningful dynamically).

---

## 5. Integration with Existing Code

### `core/analysis.py` changes

In `calculate_network_indices()`, replace:
```python
transfer_efficiency = 0.1  # Default placeholder
finn_cycling_index = 0.0   # Placeholder
```

With:
```python
from pypath.core.indicators import finn_cycling_index as _finn_cycling_index
from pypath.core.indicators import transfer_efficiency as _transfer_efficiency

_te_array = _transfer_efficiency(rpath)
transfer_efficiency = float(np.mean(_te_array[_te_array > 0])) if np.any(_te_array > 0) else 0.0
finn_cycling_index = _finn_cycling_index(rpath)
```

No changes to the `NetworkIndices` dataclass fields or their types.

---

## 6. Testing Strategy

### Unit tests (`test_indicators.py`)

**FlowAnalysis tests (~10):**
- 2-group model (producer + consumer): TST matches manual calculation
- Ascendency > 0 and Ascendency < Capacity
- Overhead = Capacity - Ascendency (within tolerance)
- Relative ascendency in [0, 1]
- Finn cycling index = 0 for linear chain (no recycling)
- Finn cycling index > 0 when detritus feeds back to consumer
- Transfer efficiency returns per-level array with values in [0, 1]
- Transfer efficiency mean replaces placeholder in NetworkIndices
- Single-group model doesn't crash
- All-zero biomass model returns sensible defaults

**EcosystemIndicators tests (~8):**
- MTL catch weighted correctly (manual 2-group check)
- MTI excludes groups with TL < 3.25
- MTI = NaN when no groups above 3.25
- Catch/biomass ratio matches manual calculation
- Gross efficiency = total catch / primary production
- Shannon diversity of equal biomasses ≈ ln(n)
- Kempton Q returns NaN when fewer than 4 groups in TL 3-4
- Zero catch: MTL catch = NaN, no divide-by-zero error

**Timeseries tests (~3):**
- Returns DataFrame with correct columns and row count
- Values change over time when biomass/catch vary
- Consistent with static indicators at t=0

**Integration (~2):**
- `calculate_network_indices()` returns non-placeholder values
- Existing `test_analysis.py` tests still pass

**Total: ~23 new tests**

---

## 7. File Structure

### New files
| File | Purpose |
|------|---------|
| `core/indicators.py` | FlowAnalysis, EcosystemIndicators, all indicator functions |
| `tests/test_indicators.py` | ~23 unit tests |

### Modified files
| File | Change |
|------|--------|
| `core/analysis.py` | Replace 2 placeholders with calls to `indicators.py` |
| `core/__init__.py` | Export new types and functions |
