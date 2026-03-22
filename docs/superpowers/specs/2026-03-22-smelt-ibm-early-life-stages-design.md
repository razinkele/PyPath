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

Three new fields appended **after all existing non-default fields** in the `SuperIndividual` dataclass (required by Python dataclass field ordering — new default-valued fields must follow existing non-default fields):

```python
life_stage: int = 4         # 0=egg, 1=yolk_sac, 2=larva, 3=juvenile, 4=adult
degree_days: float = 0.0    # accumulated thermal development (°C·day)
starvation_days: float = 0.0  # consecutive days without sufficient feeding
```

All existing code continues unchanged — current individuals default to `life_stage=4` (adult). Existing code already uses keyword arguments for `SuperIndividual` construction (verified in `reproduction.py`), so adding default-valued fields at the end is safe.

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

Plot degree-day accumulation curves at 5.7°C, 9.1°C, and 12.1°C. The simple linear DD model predicts hatching at:

- 5.7°C: DD rate = 3.9 °C·day/day → 149 / 3.9 = **38.2 days**
- 9.1°C: DD rate = 7.3 °C·day/day → 149 / 7.3 = **20.4 days**
- 12.1°C: DD rate = 10.3 °C·day/day → 149 / 10.3 = **14.5 days**

**Note:** Keller et al. (2020) report empirical observations of ~40, ~23, and ~12.5 days respectively. The discrepancy (especially at 9.1°C and 12.1°C) arises because development rate is not perfectly linear with temperature — Keller also provides an Arrhenius formulation (T_A = 11,229 K) that captures this nonlinearity. The linear DD model is a first-order approximation; if validation shows systematic bias, the Arrhenius formulation can be substituted (same `EggParams` interface, different `accumulate_development()` implementation).

---

## Package 2: Yolk-Sac Stage and First Feeding Transition

### Yolk-Sac Model

Hatched individuals (`life_stage=1`) carry yolk energy in the `energy_reserve` field. **For early life stages (life_stage 0–2), `energy_reserve` is in kilojoules (kJ)**. This differs from the current adult usage where `energy_reserve` is a dimensionless index — the reinterpretation is safe because early and adult stages never share energy reserve values (recruits are re-initialized when transitioning to juvenile stage).

**Initial yolk energy at hatch:** defined in `YolkSacParams.initial_yolk_kj`. Default: 0.15 kJ (based on ~0.001g egg weight × 5 kJ/g energy density × ~30× yolk-to-body ratio for smelt eggs). Set on the `SuperIndividual` when transitioning from egg to yolk-sac stage.

Yolk is depleted by basal metabolism only (no feeding, no active movement):

```
yolk_depletion_rate_kj = rs_a_larval * weight^rs_b * Q10^((T - T_ref) / 10) * energy_density * dt_days
energy_reserve -= yolk_depletion_rate_kj
```

Uses Q10 formulation with larval-specific `rs_a_larval` from `LarvalParams` (higher weight-specific metabolic rate than adults). Duration emerges from the model (not hardcoded): ~25 days at 5.7°C, ~15 days at 9.1°C, ~14 days at 12.1°C.

### First Feeding Transition

**`first_feeding_threshold`**: defined in `YolkSacParams.first_feeding_threshold_kj` (default: 0.02 kJ — ~13% of initial yolk). When `energy_reserve <= first_feeding_threshold_kj`:

**Zooplankton availability source:** `env_forcing['zoo_density']` (mg C/m³), representing local zooplankton concentration. In zonal mode, this is zone-specific via `env_forcing['zone_forcing'][patch_idx]['zoo_density']`. **`minimum_prey_density`**: defined in `YolkSacParams.minimum_prey_density` (default: 50.0 mg C/m³ — minimum copepod nauplii density for first feeding success).

```
zoo_density = env_forcing['zoo_density']  # or zone-specific variant

if zoo_density >= params.yolk_sac.minimum_prey_density:
    life_stage = 2 (larva, exogenous feeding)
    starvation_days = 0
    energy_reserve = initial_larval_reserve  # re-initialize for larval bioenergetics
else:
    starvation_days += dt_days
    if starvation_days > params.yolk_sac.point_of_no_return:
        individual dies (starvation)
```

**Point of no return (PNR):** `YolkSacParams.point_of_no_return` (default: 4.0 days). After PNR, larvae are irreversibly starved.

### Cushing Match/Mismatch

The match/mismatch effect is now emergent rather than explicit. In the current code, `create_recruits()` applies `larval_survival_probability()` (Gaussian match/mismatch) to compute instantaneous survival. With the new early life stages, the match/mismatch arises naturally: if yolk is exhausted when zooplankton density is low, larvae hit PNR and die. The explicit `larval_survival_probability()` function and the current `create_recruits()` are **deprecated** — spawning now produces eggs directly, and survival emerges from the mechanistic egg→yolk-sac→larva pipeline. The deprecated functions remain in `reproduction.py` for backward compatibility (existing tests that call them directly still pass), but `SmeltIBM.compute_step()` no longer calls them.

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

**Larval consumption** uses a Type II functional response on zooplankton density from `env_forcing['zoo_density']`:

```
C_larval_scalar = cmax(w) * (zoo / (zoo + K_half))    # scalar consumption (g/timestep)
```

This scalar is allocated entirely to the **zooplankton prey group index** (defined in `LarvalParams.zooplankton_prey_idx`, default: 1). The result is a per-prey consumption dict: `{zooplankton_prey_idx: C_larval_scalar}`.

**Adult consumption** comes from the existing `adaptive_forage()` which returns `Dict[int, float]` — consumption allocated across all prey groups by profitability.

**Blending** operates on the per-prey consumption vectors:

```
alpha(w) = sigmoid((w - w_forage_mid) / w_forage_scale)

# Build larval consumption vector: all consumption goes to zooplankton group
C_larval_vec = np.zeros(n_groups)
C_larval_vec[zooplankton_prey_idx] = C_larval_scalar

# Build adaptive forage vector from dict
C_adaptive_vec = np.zeros(n_groups)
for prey_idx, amount in adaptive_forage_result.items():
    C_adaptive_vec[prey_idx] = amount

# Blend
C_total_vec = (1 - alpha) * C_larval_vec + alpha * C_adaptive_vec
```

At 5mm (alpha ≈ 0): pure zooplankton concentration-dependent. At 50mm (alpha ≈ 1): pure adaptive foraging across all prey. At 15mm: blend of both.

**`K_half`**: half-saturation constant in `LarvalParams.k_half_zoo` (default: 100.0 mg C/m³).

### Cmax Scaling

```
cmax(w, T) = c_a * w^c_b * f(T)
```

**Temperature dome function `f(T)`** — Thornton-Lessem formulation (standard in fish bioenergetics):

```
f(T) = K_A * K_B
where:
  K_A = (CQ * L1) / (1 + CQ * (L1 - 1))    for T <= T_opt
  K_B = (CQ * L2) / (1 + CQ * (L2 - 1))    for T >= T_opt
  L1 = exp(G1 * (T - T_opt))
  L2 = exp(G2 * (T_max - T))
  G1 = (1 / (T_opt - T_min)) * ln(CQ * (1 - V1) / V1)
  G2 = (1 / (T_max - T_opt)) * ln(CQ * (1 - V2) / V2)
```

Parameters in `LarvalParams`: `cmax_t_opt` (default: 18°C), `cmax_t_min` (default: 2°C), `cmax_t_max` (default: 28°C), `cmax_CQ` (default: 2.4, controls curve steepness), `cmax_V1` (default: 0.02, proportion of Cmax at T_min), `cmax_V2` (default: 0.02, proportion of Cmax at T_max). The dome shape ensures consumption drops at both cold and warm extremes.

### Assimilation Efficiency — Size-Dependent

```
assimilation_efficiency(w) = ae_min + (ae_max - ae_min) * sigmoid((w - w_ae_mid) / w_ae_scale)
```

`ae_min` ~ 0.55 (larvae), `ae_max` ~ 0.73 (adults).

**Relationship to existing `BioenergParams.unassimilated_fraction`:** The ontogenetic sigmoid AE **replaces** `unassimilated_fraction` when Package 3 is active. The old parameter is retained in `BioenergParams` for backward compatibility (used when `LarvalParams` is None / early life stages disabled), but the new `growth_step_batch_ontogenetic()` function uses `assimilation_efficiency(w)` instead of `1 - unassimilated_fraction`.

### Implementation

All sigmoid interpolations are vectorizable in `growth_step_batch_ontogenetic()` — a new function alongside the existing `growth_step_batch()`. No branching by `life_stage` — body size drives everything.

### Juvenile Transition

At configurable size threshold (~20mm), `life_stage` advances to 3 (juvenile). Bookkeeping only — bioenergetics are already smoothly adult-like.

### Review Checkpoint

- Growth 5mm→20mm: ~0.3–0.5 mm/day at 15°C
- Metabolic scope dome-shaped vs temperature
- **Backward compatibility via re-parameterization:** The Rs + Ra formulation with `activity_multiplier` changes the total metabolism calculation. At adult sizes, total metabolism = Rs × (1 + am_max). To match the current Wisconsin model, the new `rs_a` parameter must satisfy: `rs_a * (1 + am_max) = ra` (the existing metabolic intercept). With `am_max = 1.5`, this means `rs_a = ra / 2.5 = 0.0033 / 2.5 = 0.00132`. The `baltic_defaults()` method will set `rs_a` from this relationship automatically. Existing `growth_step_batch()` (used when `LarvalParams` is None) remains unchanged — only `growth_step_batch_ontogenetic()` uses the new formulation.

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
    mortality_rate = stage_background_mortality + OxygenParams.hypoxia_mortality_rate * (1 - O2 / O2_lethal)
```

Here `stage_background_mortality` is the life stage-specific baseline mortality (from `EggParams.background_mortality_rate` for eggs, zero for other stages that have predation mortality in Phase 3). `OxygenParams.hypoxia_mortality_rate` (default: 0.5 /day) is the maximum additional mortality at zero oxygen.

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

Configurable in `ZoneParams`. Actual movement for active stages (juvenile/adult) combines the base connectivity with behavioral scoring:

```
# For active movement (life_stage >= 3):
behavioral_score[j] = habitat_weight * quality[j] + food_weight * food[j]
                     + predator_weight * (1/(1+pred[j])) + oxygen_weight * O2_score[j]
final_prob[i, j] = normalize(base_connectivity[i, j] * behavioral_score[j])
```

For passive stages (yolk-sac, larvae), only the base connectivity matrix applies — no behavioral modulation.

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

Each zone has own temperature, oxygen, prey density. The `env_forcing` dict gains a `'zone_forcing'` key alongside existing top-level keys:

```python
env_forcing = {
    'temperature': 12.0,          # global default (used in non-zonal mode)
    'month': 6,
    'zoo_peak_day': 150,
    'dissolved_oxygen': 6.5,      # global default
    'zoo_density': 80.0,          # global default (mg C/m³)
    'zone_forcing': {             # NEW — overrides per zone when present
        0: {'temperature': 8.0, 'dissolved_oxygen': 7.5, 'zoo_density': 30.0},   # mg C/m³ — low in river
        1: {'temperature': 12.0, 'dissolved_oxygen': 4.5, 'zoo_density': 120.0},  # mg C/m³ — high in lagoon nursery
        2: {'temperature': 9.0, 'dissolved_oxygen': 8.0, 'zoo_density': 80.0},    # mg C/m³ — moderate coastal
    }
}
```

When `'zone_forcing'` is present, `compute_step()` resolves each individual's environmental conditions from their `patch_idx`. When absent, all individuals experience the top-level global values (backward compatible, non-zonal mode).

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
