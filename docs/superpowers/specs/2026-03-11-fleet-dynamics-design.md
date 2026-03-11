# Fleet Dynamics & Quota Management Design Spec

**Goal:** Add profit-responsive fleet effort dynamics and TAC quota enforcement to Ecosim, faithfully replicating the EwE 6 capacity-investment model.

**Approach:** Lightweight coupling via keyword argument to `rsim_run()`, following the ecotracer/mediation pattern. Fleet dynamics code lives in a separate module (`core/fleet_dynamics.py`). Effort dynamics update capacity each month based on profitability; quota enforcement applies hard TAC caps per fleet-group pair.

---

## 1. Data Structures

### FleetEconParams

Dataclass holding per-fleet economic and effort dynamics parameters. Fleet arrays are 0-based, length `n_fleets`. Price array is per fishing link, length `n_links`.

```python
@dataclass
class FleetEconParams:
    # Economic parameters (from EcopathFleet)
    fixed_cost: np.ndarray       # (n_fleets,) annual fixed cost
    variable_cost: np.ndarray    # (n_fleets,) cost per unit effort
    sailing_cost: np.ndarray     # (n_fleets,) cost per unit effort (distance)
    price: np.ndarray            # (n_links,) price per unit catch, per fishing link

    # Effort dynamics (from EcosimScenarioFleet)
    cap_depreciate: np.ndarray   # (n_fleets,) capacity depreciation rate (fraction/yr)
    cap_base_growth: np.ndarray  # (n_fleets,) base capacity growth rate
    eff_power: np.ndarray        # (n_fleets,) effort power parameter

    # Quota management
    tac: np.ndarray | None       # (n_fleets, n_groups) TAC allocation, None = no quotas
```

### FleetDynamicsResult

Dataclass holding output time series.

```python
@dataclass
class FleetDynamicsResult:
    out_Effort: np.ndarray       # (n_months+1, n_fleets) monthly effort multipliers
    out_Revenue: np.ndarray      # (n_months+1, n_fleets) monthly revenue
    out_Cost: np.ndarray         # (n_months+1, n_fleets) monthly cost
    out_Profit: np.ndarray       # (n_months+1, n_fleets) monthly profit
    annual_Effort: np.ndarray    # (n_years, n_fleets) annual average effort
    annual_Profit: np.ndarray    # (n_years, n_fleets) annual total profit
    fleet_names: list[str]
```

### Factory

```python
def create_fleet_econ_params(n_fleets: int, n_links: int) -> FleetEconParams:
    """Create with defaults: zero costs, zero prices, no dynamics, no quotas.

    Defaults: fixed_cost=0, variable_cost=0, sailing_cost=0, price=0,
    cap_depreciate=0, cap_base_growth=0, eff_power=1.0, tac=None.
    """
```

---

## 2. Effort Dynamics Model

### Monthly update for each fleet g

```
revenue_g = sum over links i where FishThrough[i]==g: catch[i] * price[i]
cost_g = fixed_cost[g]/12 + (variable_cost[g] + sailing_cost[g]) * effort[g]
profit_g = revenue_g - cost_g

# Profit signal: normalized to [-1, 1] range
profit_signal_g = profit_g / max(revenue_g, cost_g, epsilon)

# Capacity update (investment/disinvestment)
capacity_g += (cap_base_growth[g] * max(profit_signal_g, 0) - cap_depreciate[g]) * capacity_g * dt

# Capacity cannot go below a floor (prevent extinction)
capacity_g = max(capacity_g, 0.01)

# Effort from capacity
effort_g = capacity_g ^ eff_power[g]
```

Key behaviors:
- **Profitable fleet**: capacity grows at `cap_base_growth * profit_signal` rate
- **Unprofitable fleet**: capacity only depreciates (base_growth term is zero since max(profit_signal, 0) = 0)
- **Depreciation always acts**: ensures fleets shrink when idle
- **eff_power**: typically 1.0 (linear), can be <1 for diminishing returns
- **Capacity floor**: 0.01 prevents fleet extinction

### Quota enforcement (separate from effort dynamics)

```
for each fishing link i:
    fleet_g = FishThrough[i] mapped to fleet index
    group_j = FishFrom[i]
    if tac is not None and tac[fleet_g, group_j] > 0:
        if cumulative_catch[fleet_g, group_j] >= tac[fleet_g, group_j]:
            effective_FishQ[i] = 0  # stop fishing this link for rest of year

# Reset cumulative catch at year boundary (month % 12 == 1)
```

### Functions

```python
_CAPACITY_FLOOR = 0.01
_EPSILON = 1e-10

def fleet_dynamics_step(
    capacity: np.ndarray,          # (n_fleets,) current fleet capacities
    monthly_catch: np.ndarray,     # (n_links,) catch from this month's fishing links
    cumulative_catch: np.ndarray,  # (n_fleets, n_groups) cumulative annual catch
    params: FleetEconParams,
    fish_through: np.ndarray,      # (n_links,) fleet index per fishing link (1-based)
    fish_from: np.ndarray,         # (n_links,) group index per fishing link (1-based)
    fleet_lookup: dict,            # maps gear_0based -> fleet array index (1-based)
    dt: float = 1.0 / 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Update fleet capacity and compute effort multipliers.

    Returns (new_capacity, effort_multipliers) where effort_multipliers
    has shape (n_fleets,) and replaces ForcedEffort for the next month.
    """

def apply_quota_caps(
    fish_q: np.ndarray,            # (n_links,) current FishQ values
    cumulative_catch: np.ndarray,  # (n_fleets, n_groups) cumulative annual catch
    tac: np.ndarray,               # (n_fleets, n_groups) TAC allocation
    fish_through: np.ndarray,      # (n_links,) fleet index per fishing link (1-based)
    fish_from: np.ndarray,         # (n_links,) group index per fishing link (1-based)
    fleet_lookup: dict,            # maps gear_0based -> fleet array index (1-based)
) -> np.ndarray:
    """Zero out FishQ for links that have reached their TAC.

    Returns modified copy of fish_q array.
    """
```

---

## 3. Integration with rsim_run

`rsim_run(scenario, fleet_dynamics=fleet_econ_params)` — keyword-only argument, default None.

When `fleet_dynamics` is provided:

1. **Before loop**: Initialize `capacity = ForcedEffort[0, 1:]` (initial effort = initial capacity for each gear). Allocate output arrays. Initialize `cumulative_catch = zeros(n_fleets, n_groups)`. Store initial effort at t=0.

2. **Each monthly step** (after catch computation at lines 2255-2287):
   - Accumulate catch into `cumulative_catch` by fleet and group
   - Call `fleet_dynamics_step(capacity, monthly_catch_links, cumulative_catch, fleet_dynamics, ...)` to get updated capacity and effort multipliers
   - Write new effort multipliers into `forcing_dict["ForcedEffort"]` for next month's derivative computation
   - If `tac` is set, call `apply_quota_caps()` and modify effective FishQ for next step
   - Store results in output arrays

3. **Year boundary** (when `month % 12 == 0`): Reset `cumulative_catch` to zeros for new quota year.

4. **After loop**: Compute annual averages/totals, attach `FleetDynamicsResult` to `RsimOutput.fleet_dynamics`.

**Return type**: `rsim_run` still returns `RsimOutput`. The `FleetDynamicsResult` is accessed via `output.fleet_dynamics` (None if not used). Adding `fleet_dynamics: FleetDynamicsResult | None = None` as a field of `RsimOutput` (with default `None`) is backward-compatible.

**Interaction with ForcedEffort**: When fleet dynamics is active, the dynamic effort *replaces* the forced effort for each gear each month. The initial ForcedEffort[0] serves as the starting capacity. Subsequent months use dynamically computed effort.

**Interaction with mediation**: Fleet mediation multipliers (from Phase 2) are applied *after* effort dynamics. The dynamic effort is the base, mediation modifies it further.

---

## 4. I/O Layer

### Schema tables (from real EwE 6 database)

```python
"EcosimScenarioFleet": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("EcopathFleetID", "INTEGER"),
    ("CapDepreciate", "DOUBLE"),
    ("CapBaseGrowth", "DOUBLE"),
    ("EffPower", "DOUBLE"),
    ("QmaxQbase", "DOUBLE"),
    ("QchangeRate", "DOUBLE"),
    ("CostOfEffort", "DOUBLE"),
])

"EcosimScenarioQuota": OrderedDict([
    ("ScenarioID", "INTEGER"),
    ("GroupID", "INTEGER"),
    ("FleetID", "INTEGER"),
    ("QuotaShare", "DOUBLE"),
    ("TAC", "DOUBLE"),
])
```

### read_fleet_dynamics

```python
def read_fleet_dynamics(
    db_path: str,
    n_fleets: int,
    n_links: int,
    n_groups: int,
    fleet_ids: list[int],
    fishing_links: dict,
) -> FleetEconParams:
    """Read fleet dynamics parameters from EwE database.

    Reads EcopathFleet for costs, EcopathCatch for prices,
    EcosimScenarioFleet for effort dynamics parameters,
    and EcosimScenarioQuota for TAC allocations.

    Maps EcopathFleetID (1-based) to 0-based fleet arrays.
    Maps fishing link prices via GroupID+FleetID matching.
    Returns default params if tables are missing/empty.
    """
```

---

## 5. File Structure

### New files
| File | Purpose |
|------|---------|
| `core/fleet_dynamics.py` | FleetEconParams, FleetDynamicsResult, create_fleet_econ_params, fleet_dynamics_step, apply_quota_caps |
| `tests/test_fleet_dynamics.py` | Unit tests for dataclasses, step function, quota caps |
| `tests/test_fleet_dynamics_io.py` | Schema + read_fleet_dynamics mock tests |
| `tests/test_fleet_dynamics_integration.py` | End-to-end with Ecosim model |

### Modified files
| File | Change |
|------|--------|
| `core/ecosim.py` | rsim_run() gains `fleet_dynamics=None` kwarg; monthly loop calls fleet_dynamics_step when present |
| `core/ecosim.py` | RsimOutput gains `fleet_dynamics: FleetDynamicsResult \| None = None` field |
| `core/__init__.py` | Export FleetEconParams, FleetDynamicsResult, create_fleet_econ_params |
| `io/_ewe_schema.py` | Add EcosimScenarioFleet, EcosimScenarioQuota tables |
| `io/ewemdb.py` | Add read_fleet_dynamics() |
| `io/__init__.py` | Export read_fleet_dynamics |

---

## 6. Testing Strategy

### Unit tests (`test_fleet_dynamics.py`)
- FleetEconParams construction and defaults
- create_fleet_econ_params() shapes and default values
- fleet_dynamics_step() with known inputs:
  - Profitable fleet → capacity grows, effort increases
  - Unprofitable fleet → capacity depreciates, effort decreases
  - Zero profit → only depreciation acts
  - Capacity floor enforced (never below 0.01)
  - eff_power != 1 produces correct effort (effort = capacity^power)
- apply_quota_caps():
  - Cumulative catch below TAC → FishQ unchanged
  - Cumulative catch at/above TAC → FishQ zeroed for that link
  - Links without TAC → unaffected

### I/O tests (`test_fleet_dynamics_io.py`)
- Schema tables exist with correct columns
- read_fleet_dynamics() with mocked database
- Missing tables return default params
- Price mapping from EcopathCatch to fishing links
- Quota reading from EcosimScenarioQuota

### Integration tests (`test_fleet_dynamics_integration.py`, @pytest.mark.slow)
- 3-group model + 1 fleet with fleet dynamics → effort changes over time
- High-price catch → fleet effort increases over months
- Zero-price (no revenue) → fleet effort decays toward floor
- Quota cap → catch stops at TAC, resets next year
- Without fleet_dynamics kwarg → output.fleet_dynamics is None
- Result shapes: (n_months+1, n_fleets) and (n_years, n_fleets)
