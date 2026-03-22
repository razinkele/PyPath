# Smelt IBM Early Life Stages — Design Specification

**Date:** 2026-03-22
**Scope:** Enhance the PyPath SmeltIBM with mechanistic early life stages (egg, yolk-sac, larva), full lifecycle oxygen physiology, ontogenetic bioenergetics interpolation, and a zonal spatial model for the Curonian Lagoon.
**Primary use case:** Baltic-applied tool — parameterized for Lithuanian coastal / Curonian Lagoon smelt populations, incorporating Drewes et al. (2025) and Keller et al. (2020) early life stage science adapted for Baltic conditions.

## Scientific Context

Drewes et al. (2025) published the first IBM for European smelt (*O. eperlanus*), covering egg through ~20mm feeding larvae in the Elbe estuary. Their model uses temperature-dependent degree-day development, split Rs/Ra metabolism, and concentration-dependent consumption for larvae. Keller et al. (2020) provide the experimental foundation: DD_hatch = 149 °C·day, T₀ = 1.8°C, DD_mortality = 272.4 °C·day.

The current PyPath SmeltIBM covers the complementary lifecycle gap: juvenile through adult (Wisconsin bioenergetics, adaptive foraging, size-structured predation, spatial movement, Cushing match/mismatch recruitment). However, early life stages are collapsed into an instant eggs→recruits transition with no development, no yolk-sac stage, and no oxygen effects.

This design bridges the two, creating a complete lifecycle IBM for Baltic smelt with smooth ontogenetic transitions rather than hard stage boundaries.

## Key References

- Drewes, D., Schrum, C., Pein, J., Benkort, D. & Daewel, U. (2025). Environmental controls on the development of early life stages of European smelt in the Elbe estuary. *Ecological Modelling*, 510, 111313.
- Keller et al. (2020). Temperature effects on egg and larval development rate in European smelt. *Journal of Fish Biology*, 97(2), 368–381.
- Rose et al. (2013). Individual-based modeling of Delta Smelt population dynamics. *Transactions of the American Fisheries Society*, 142(5), 1238–1259.

## Design Decisions

1. **Baltic-applied tool** — Keller/Drewes parameter values used as configurable defaults, tunable to Curonian Lagoon field observations.
2. **Degree-day egg development** — Keller's values adopted with configurable overrides (DD_hatch, T₀, DD_mortality as parameters in `EggParams`).
3. **Full lifecycle oxygen effects** — metabolic scope reduction via Pcrit at all stages, plus lethal thresholds for early stages.
4. **Ontogenetic interpolation for bioenergetics** — body size as the master variable driving smooth sigmoid transitions between larval and adult physiology (no hard stage boundaries in the bioenergetics code).
5. **Zonal spatial model** — 3 functional zones (river spawning, lagoon nursery, coastal feeding) designed from the start, with ontogenetic habitat shifts.
6. **Validation data** — catch monitoring/CPUE for adults, recruitment indices for early life stages.

---

## Architecture Foundation

### SuperIndividual Extensions

Three new fields added to `SuperIndividual` dataclass:

```python
life_stage: int = 4         # 0=egg, 1=yolk_sac, 2=larva, 3=juvenile, 4=adult
degree_days: float = 0.0    # accumulated thermal development (°C·day)
starvation_days: float = 0.0  # consecutive days without sufficient feeding
```

All existing code continues unchanged — current individuals default to `life_stage=4` (adult).

### New Parameter Dataclasses

In a new `development.py` module:

- **`EggParams`** — DD_hatch, DD_mortality, T₀, oxygen thresholds, egg weight/length, per-zone hatch success
- **`YolkSacParams`** — yolk energy content, absorption rate vs temperature, starvation threshold for first-feeding transition, point of no return (days)
- **`LarvalParams`** — size-dependent bioenergetics coefficients (Rs/Ra split), Cmax allometric coefficients, ontogenetic interpolation sigmoid breakpoints (w_mid, w_scale for each transition)
- **`OxygenParams`** — Pcrit by life stage (or as size-dependent sigmoid), lethal thresholds, hypoxia mortality rate, behavioral avoidance weight
- **`ZoneParams`** — zone definitions (spawning/nursery/coastal), connectivity matrix, zone-specific temperature/O2/prey offsets, drift parameters

These are added to `SmeltParams` alongside existing `BioenergParams`, `PredationParams`, etc.

### compute_step() Restructure

The 5-phase cycle stays, but Phase 1 and Phase 2 get life-stage routing:

```
For each timestep:
  1a. Eggs: accumulate degree-days, check hatch threshold, apply egg mortality
  1b. Yolk-sac: deplete yolk reserves, check first-feeding transition, apply PNR starvation
  1c. Larvae + Juveniles + Adults: bioenergetics (ontogenetic interpolation selects params by body size)
  2.  Adults only: spawn → create eggs (not recruits)
  3.  All stages: predation mortality (size-dependent, includes larval vulnerability)
  4.  Bookkeeping: advance stages, advance degree-days, remove dead, add hatched/transitioned
  5.  Spatial: zone-based movement (ontogenetic habitat shifts, passive drift for early stages)
```

---

## Package 1: Egg Stage with Degree-Day Development

### Spawning Change

`spawn()` produces egg super-individuals (`life_stage=0`) instead of calling `create_recruits()`. Egg weight ~0.001g, length ~1.5mm. Eggs inherit `patch_idx` from spawning female (zone-aware deposition).

### Degree-Day Accumulation

Each timestep:

```
if temperature > T₀:
    degree_days += (temperature - T₀) * dt_days
```

- T₀ = 1.8°C (Keller et al. 2020 default, configurable)
- DD_hatch = 149.0 °C·day (configurable)
- DD_mortality = 272.4 °C·day (configurable)

### Hatching

When `degree_days >= DD_hatch`, egg transitions to `life_stage=1` (yolk-sac). Degree-days reset to 0 for the new stage.

### Egg Mortality — Three Sources

1. **Thermal mortality:** `degree_days >= DD_mortality` before hatching → lethal
2. **Oxygen mortality:** dissolved O2 < `egg_O2_lethal` → proportional mortality rate per day below threshold
3. **Background mortality:** constant daily rate (invertebrate predation on sessile eggs)

### Zone Interaction

Eggs are sessile — no movement in Phase 5 for `life_stage=0`.

### Review Checkpoint

Plot degree-day accumulation curves at 5.7°C, 9.1°C, 12.1°C. Hatching at ~40, ~23, ~12.5 days respectively (Keller et al. Table 4).

---

## Package 2: Yolk-Sac Stage and First Feeding Transition

### Yolk-Sac Model

Hatched individuals (`life_stage=1`) carry energy reserve as yolk. Depleted by basal metabolism only (no feeding, no active movement):

```
yolk_depletion_rate = basal_metabolism(weight, temperature)
energy_reserve -= yolk_depletion_rate * dt_days
```

Uses Q10 formulation with larval-specific `ra` (higher weight-specific metabolic rate). Duration emerges from the model (not hardcoded): ~25 days at 5.7°C, ~15 days at 9.1°C, ~14 days at 12.1°C.

### First Feeding Transition

When `energy_reserve <= first_feeding_threshold`:

```
if zooplankton_available >= minimum_prey_density:
    life_stage = 2 (larva, exogenous feeding)
    starvation_days = 0
else:
    starvation_days += dt_days
    if starvation_days > point_of_no_return:
        individual dies (starvation)
```

**Point of no return (PNR):** ~3–5 days (configurable). After PNR, larvae are too weak to feed.

### Cushing Match/Mismatch

Moved from the current instant-recruitment to here — where it biologically belongs. The match/mismatch between yolk exhaustion timing and zooplankton availability drives first-feeding success.

### Oxygen Effects

Yolk-sac larvae are immobile and sensitive. O2 below threshold accelerates yolk depletion (stress metabolism) and increases mortality.

### Passive Drift

Yolk-sac larvae begin drifting from spawning zone (0) toward nursery zone (1). Probability-based, modulated by seasonal flow.

### Review Checkpoint

Yolk-sac duration vs temperature matches literature. PNR starvation kills unfed larvae. Passive drift moves fraction from spawning to nursery zone.

---

## Package 3: Larval Bioenergetics with Ontogenetic Interpolation

### Metabolism Split — Rs + Ra

```
Rs(w, T) = rs_a * w^rs_b * Q10^((T - T_ref) / 10)     # basal
Ra(w, T) = activity_multiplier(w) * Rs(w, T)             # active
total_metabolism = Rs + Ra
```

Activity multiplier is size-dependent via sigmoid:

```
activity_multiplier(w) = am_min + (am_max - am_min) * sigmoid((w - w_mid) / w_scale)
```

- `am_min` ~ 0.3 (larvae, passive)
- `am_max` ~ 1.5 (adults, active foraging)

### Consumption — Concentration to Adaptive Foraging Blend

```
C_larval(w, zoo) = cmax(w) * (zoo / (zoo + K_half))    # Type II functional response
C_total = (1 - alpha(w)) * C_larval + alpha(w) * C_adaptive_forage
alpha(w) = sigmoid((w - w_forage_mid) / w_forage_scale)
```

5mm larva: pure concentration-dependent. 50mm juvenile: pure adaptive foraging. 15mm fish: blend.

### Cmax Scaling

```
cmax(w) = c_a * w^c_b * f(T)
```

Temperature dome function `f(T)` peaks at optimal temperature.

### Assimilation Efficiency — Size-Dependent

```
assimilation_efficiency(w) = ae_min + (ae_max - ae_min) * sigmoid((w - w_ae_mid) / w_ae_scale)
```

`ae_min` ~ 0.55 (larvae), `ae_max` ~ 0.73 (adults).

### Implementation

All sigmoid interpolations are vectorizable in `growth_step_batch()`. No branching by `life_stage` — body size drives everything.

### Juvenile Transition

At configurable size threshold (~20mm), `life_stage` advances to 3 (juvenile). Bookkeeping only — bioenergetics are already smoothly adult-like.

### Review Checkpoint

- Growth 5mm→20mm: ~0.3–0.5 mm/day at 15°C
- Metabolic scope dome-shaped vs temperature
- At adult sizes: identical to current Wisconsin model output (backward compatibility)

---

## Package 4: Oxygen Physiology (Full Lifecycle)

### Pcrit and Metabolic Scope Reduction

```
if O2 >= Pcrit:
    oxygen_scalar = 1.0
else:
    oxygen_scalar = max(0, O2 / Pcrit)
```

`oxygen_scalar` multiplies Cmax (not metabolism). Under hypoxia, fish can't eat as much — metabolic scope shrinks.

### Life Stage-Specific Pcrit (Defaults)

| Life stage | Pcrit (mg/L) | O2_lethal (mg/L) |
|-----------|-------------|-----------------|
| Egg | 4.0 | 2.0 |
| Yolk-sac | 3.5 | 1.5 |
| Larva | 3.0 | 1.0 |
| Juvenile | 2.5 | — |
| Adult | 2.0 | — |

Pcrit can also be expressed as a continuous size-dependent sigmoid (matching Package 3's ontogenetic interpolation philosophy):

```
Pcrit(w) = pcrit_larval + (pcrit_adult - pcrit_larval) * sigmoid((w - w_mid) / w_scale)
```

### Lethal Thresholds (Early Stages Only)

Below `O2_lethal`, mortality increases sharply:

```
if O2 < O2_lethal[life_stage]:
    mortality_rate = base_mortality + hypoxia_mortality * (1 - O2 / O2_lethal)
```

Juveniles and adults: sublethal effects only (reduced Cmax) — they can swim away.

### Behavioral Avoidance

In Phase 5 (movement), oxygen enters the movement score for mobile stages:

```
oxygen_score = min(O2_patch / Pcrit, 1.0)
movement_score += oxygen_weight * oxygen_score
```

### Backward Compatibility

When `dissolved_oxygen` is absent from `env_forcing`, `oxygen_scalar = 1.0` (no effect).

### Review Checkpoint

- O2 = 8 mg/L: no effect on any stage
- O2 = 3 mg/L: eggs affected, adults not
- O2 = 1.5 mg/L: egg mortality spikes, larvae reduced growth, adults avoid the zone
- Growth rate vs O2: flat above Pcrit, linear decline below

---

## Package 5: Curonian Lagoon Zonal Spatial Model

### Three Functional Zones

| Zone | Habitat | Smelt function | Character |
|------|---------|---------------|-----------|
| 0 | Nemunas delta, river mouths | Spawning grounds | Cooler spring, variable O2, low zooplankton |
| 1 | Open Curonian Lagoon | Nursery | Warmer, seasonal hypoxia risk, high zooplankton |
| 2 | Lithuanian coast, Klaipeda strait | Adult feeding | Stable O2, cooler, marine prey |

### Connectivity Matrix (Base Probabilities)

```
         To:  River  Lagoon  Coastal
From:
  River    0.7    0.3     0.0
  Lagoon   0.1    0.7     0.2
  Coastal  0.1    0.2     0.7
```

Configurable in `ZoneParams`. Actual movement modulated by behavioral scoring.

### Ontogenetic Habitat Shifts

| Life stage | Allowed zones | Movement type |
|-----------|--------------|--------------|
| Egg | Zone 0 only | None (sessile) |
| Yolk-sac | 0 → 1 | Passive drift |
| Larva | 0, 1 | Passive drift + weak active |
| Juvenile | 1, 2 | Active (full behavioral scoring) |
| Adult | All | Active + spawning migration |

### Spawning Migration

Adults in coastal zone migrate to river zone when spawning conditions approach. Reuses `should_migrate()` with zone-aware override:

```
if life_stage == 4 and is_mature and should_migrate(temperature, month, params):
    target_zone = 0
```

### Zone-Specific Forcing

Each zone has own temperature, oxygen, prey density — driven by seasonal profiles or external coupling:

```python
zone_forcing = {
    0: {'temperature': 8.0, 'dissolved_oxygen': 7.5, 'zoo_density': 0.3},
    1: {'temperature': 12.0, 'dissolved_oxygen': 4.5, 'zoo_density': 1.2},
    2: {'temperature': 9.0, 'dissolved_oxygen': 8.0, 'zoo_density': 0.8},
}
```

### Larval Drift

Passive transport probability depends on flow and larval age:

```
drift_probability = base_drift_rate * flow_multiplier(month)
```

Drift probability decreases as larvae grow and develop swimming capacity — smooth ontogenetic transition.

### Backward Compatibility

Without spatial context, model collapses to 0D (single well-mixed zone).

### Review Checkpoint

Eggs stay in zone 0. Larvae drift to zone 1. Juveniles concentrate in zone 1 with coastal migration. Adults cycle coastal → river (spawn) → coastal.

---

## Package 6: Calibration, Validation, and Sensitivity Analysis

### Two-Stage Calibration

**Stage A:** Ecosim-level — fit vulnerability parameters (VV) to adult biomass time series using `fit_to_timeseries()`. Anchors food web context.

**Stage B:** IBM-specific — fit early life stage parameters to recruitment indices:
- `larval_base_survival`
- `zooplankton_match_window`
- `point_of_no_return`
- `egg_O2_lethal`
- `DD_hatch` (if Curonian Lagoon data suggests deviation from Keller defaults)

Uses `differential_evolution` optimizer with log-ratio SS against recruitment index time series.

### Validation

| Target | Data source | Tests |
|--------|-----------|-------|
| Length-at-age | Survey length-frequency | VBGF + ontogenetic bioenergetics |
| Spawning timing | First egg/larval observations | Degree-day model + temperature forcing |
| Recruitment vs spring temperature | Multi-year correlation | Early life stage temperature sensitivity |
| Spatial distribution by age | Survey catch composition by zone | Ontogenetic habitat shift model |

### Sensitivity Analysis — Latin Hypercube Sampling

N = 500–1000 runs. Key parameters:

| Parameter | Range |
|-----------|-------|
| DD_hatch | 120–180 °C·day |
| T₀ | 1.0–3.0 °C |
| larval_base_survival | 0.001–0.05 |
| point_of_no_return | 2–7 days |
| Pcrit (larval) | 2.0–4.0 mg/L |
| activity_multiplier range | 0.2–0.5, 1.0–2.0 |
| w_mid (bioenergetics transition) | 1–10 g |
| zooplankton_match_window | 10–25 days |

Analysis: partial rank correlation coefficients (PRCC).

### Deliverables

1. Calibrated parameter set for Curonian Lagoon smelt
2. Validation report with observed vs predicted plots
3. Sensitivity ranking of early life stage parameters
4. Data gap identification — parameters with wide posterior uncertainty

---

## Implementation Order and Dependencies

```
Package 1 (Egg stage)
    ↓
Package 2 (Yolk-sac + first feeding)     ← depends on Package 1
    ↓
Package 3 (Larval bioenergetics)          ← depends on Package 2
    ↓
Package 4 (Oxygen physiology)             ← depends on Packages 1-3
    ↓
Package 5 (Zonal spatial)                 ← depends on Packages 1-4
    ↓
Package 6 (Calibration + validation)      ← depends on all above
```

Each package produces a reviewable, testable artifact with a defined scientific review checkpoint. Scientific review occurs after each package, comparing model outputs against published data before proceeding.

## Backward Compatibility

All changes are non-breaking:
- `SuperIndividual` new fields have defaults matching current behavior
- `compute_step()` routes by `life_stage`; existing `life_stage=4` adults follow the current code path
- Oxygen effects disabled when forcing data absent
- Spatial model collapses to 0D without spatial context
- Existing 162 IBM tests must continue to pass unchanged
