# Autofix: Diagnosing and Repairing Ecosim Crashes

When Ecosim simulations produce crashes (groups going to zero) or
explosions (biomass growing unrealistically), the `autofix` module helps
identify root causes and automatically apply parameter corrections.

## When to Use Autofix

- Groups crashing to zero during simulation
- Biomass exploding beyond realistic bounds
- Models imported from EwE databases that behave differently than expected
- New models that are balanced but dynamically unstable

## Diagnosing Crash Causes

`diagnose_crash_causes` inspects the balanced Ecopath model and Ecosim
parameters to identify potential sources of instability:

```python
from pypath.core.autofix import diagnose_crash_causes

report = diagnose_crash_causes(model, scenario.params)
```

The report contains two lists:

### Critical Issues

Problems very likely to cause crashes:

| Type | Meaning |
|------|---------|
| `ee_too_high` | Ecotrophic efficiency > 1.0 (group over-exploited) |
| `incomplete_diet` | Consumer diet columns don't sum to 1.0 |
| `unrealistic_qb_pb` | QB/PB ratio outside expected range |

### Warnings

Problems that may cause instability under certain conditions:

| Type | Meaning |
|------|---------|
| `low_biomass` | Baseline biomass very close to zero |
| `high_vulnerability` | VV values > 10 (strong top-down control) |
| `high_qq` | QQ density dependence values unusually high |

## Automatic Parameter Repair

`autofix_parameters` applies targeted corrections to improve stability:

```python
from pypath.core.autofix import autofix_parameters

fixed_params, result = autofix_parameters(model, scenario.params)

# Check what was changed
print(f"Success: {result.success}")
print(f"Fixes applied: {len(result.fixes_applied)}")
for fix in result.fixes_applied:
    print(f"  {fix}")
```

### What Gets Fixed

| Fix | Default Mode | Aggressive Mode |
|-----|-------------|-----------------|
| Cap high vulnerability (VV) | VV capped at 5.0 | VV capped at 3.0 |
| Enforce minimum biomass | B_BaseRef >= 1e-6 | B_BaseRef >= 1e-4 |
| Reduce extreme QQ | QQ capped at 10.0 | QQ capped at 5.0 |
| Adjust prey switching | DD capped at 5.0 | DD capped at 3.0 |

### Aggressive Mode

For models with many crashes, use aggressive mode for stronger corrections:

```python
fixed_params, result = autofix_parameters(model, scenario.params, aggressive=True)
```

## Complete Workflow

```python
from pypath import read_ewemdb, rpath, rsim_scenario, rsim_run
from pypath.core.autofix import validate_and_fix_scenario

# Load and balance
params = read_ewemdb("model.eweaccdb")
model = rpath(params)
scenario = rsim_scenario(model, params, years=range(1, 51))

# Validate and auto-fix in one step
fixed_scenario, report = validate_and_fix_scenario(scenario, model)

# Run with fixed parameters
output = rsim_run(fixed_scenario, method="AB")
print(f"Crashed groups: {output.crashed_groups}")
```

## API Reference

See [Core API: Autofix](../api/core.md#autofix-stability-diagnostics) for
complete function signatures and parameter details.
