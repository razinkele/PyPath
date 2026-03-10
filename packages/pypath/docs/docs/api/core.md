# Core API Reference

The core module implements the Ecopath mass-balance solver, Ecosim dynamic
simulation, and supporting subsystems (stanzas, forcing, optimization).

## Ecopath (Mass-Balance)

::: pypath.core.ecopath
    options:
      show_root_heading: true
      members_order: source

## Parameters

::: pypath.core.params
    options:
      show_root_heading: true
      members_order: source

## Ecosim (Dynamic Simulation)

The Ecosim engine runs time-dynamic food web simulations using foraging
arena theory. Key entry points:

- `rsim_scenario()` — create a scenario from a balanced Ecopath model
- `rsim_run()` — run the simulation with RK4 or Adams-Bashforth integration

### Integration methods

| Method | Description |
|--------|-------------|
| `"RK4"` | 4th-order Runge-Kutta. Stable, self-starting. Good for short runs. |
| `"AB"` | Adams-Bashforth 2-step (matches Rpath). Uses 1 month RK4 warmup then AB2: `B_{n+1} = B_n + dt/2 * (3*f_n - f_{n-1})`. Includes dynamic fast equilibrium for NoIntegrate groups and Rpath-style biomass bounds. Recommended for calibration and comparison with EwE/Rpath. |

### Consumption formula

The consumption kernel uses the full Rpath foraging arena functional response:

```
Q = QQbase * PDY * PYY^(HandleSwitch*COUPLED) *
    DD / (DD - 1 + ((1-Hself)*PYY + Hself*HandleSuite)^(HandleSwitch*COUPLED)) *
    VV / (VV - 1 + (1-Sself)*PDY + Sself*PredSuite)
```

Where `PredSuite` and `HandleSuite` are pooled competitor biomass sums
across all predator-prey links sharing the same prey or predator.

### Foraging time

Predators dynamically adjust foraging time using the Rpath formula:

```
Ftime_new = 0.1 + 0.9 * Ftime_old * ((1 - FtimeAdj) + FtimeAdj * QBopt / (FoodGain / B))
```

Capped at 2.0. Groups with `FtimeAdj = 0` do not adjust foraging time.

### NoIntegrate groups (fast equilibrium)

Groups flagged as NoIntegrate (typically detritus, meiobenthos, and other
fast-turnover groups) use instantaneous equilibrium instead of ODE
integration when running with the AB method:

```
biomeq = TotGain / (TotLoss / B)
B_new = 0.5 * biomeq + 0.5 * B_old
```

This matches Rpath's SORWT=0.5 smoothing.

::: pypath.core.ecosim
    options:
      show_root_heading: true
      members_order: source

## ODE Derivatives

The consumption kernel and derivative vector used by the Ecosim ODE solver.

Key functions:

- `deriv_vector()` — compute dB/dt for all groups at a given state
- `_compute_consumption_python()` — foraging arena consumption matrix (Rpath-compatible)
- `compute_biomeq()` — fast equilibrium for NoIntegrate groups
- `integrate_rk4()` — single RK4 step
- `integrate_ab()` — single AB2 step with Rpath-style biomass bounds

::: pypath.core.ecosim_deriv
    options:
      show_root_heading: true
      members_order: source

## Advanced Ecosim (State Forcing & Diet Rewiring)

State-variable forcing and dynamic diet rewiring extensions for Ecosim.

::: pypath.core.ecosim_advanced
    options:
      show_root_heading: true
      members_order: source

## Stanzas (Multi-Stanza Groups)

::: pypath.core.stanzas
    options:
      show_root_heading: true

## Adjustments

::: pypath.core.adjustments
    options:
      show_root_heading: true

## Forcing

::: pypath.core.forcing
    options:
      show_root_heading: true

## Optimization

::: pypath.core.optimization
    options:
      show_root_heading: true

## Plotting

::: pypath.core.plotting
    options:
      show_root_heading: true

## Analysis & Network Indices

Mixed Trophic Impacts, network indices (connectance, omnivory), and
Ecosim output summary statistics.

::: pypath.core.analysis
    options:
      show_root_heading: true
      members_order: source

## Autofix (Stability Diagnostics)

Automatic parameter calibration and diagnostic routines to prevent
simulation crashes and improve model stability.

::: pypath.core.autofix
    options:
      show_root_heading: true
      members_order: source

## Sparse Link Array

Compressed prey-predator link lists for efficient consumption kernel
computation in sparse food webs.

::: pypath.core.link_array
    options:
      show_root_heading: true

## Constants

Physical and biological constants used throughout PyPath.

::: pypath.core.constants
    options:
      show_root_heading: true
