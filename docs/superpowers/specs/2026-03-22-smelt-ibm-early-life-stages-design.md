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

## Conventions

- **`dt`** is in **years** (fraction of a year, typically 1/12 for monthly timesteps), matching the existing Ecosim convention.
- **`dt_days = dt * 365.0`** is the conversion used throughout (matching `bioenergetics.py` line 253). All daily-rate formulas use `dt_days`.
- **`n_groups`** refers to the total number of Ecosim groups (living + dead), consistent with `RsimParams.NUM_GROUPS`.

### Array indexing convention (critical for integration)

The Ecosim QQ matrix is `(n_groups+1, n_groups+1)` with 1-based group indices. However, the IBM integration layer (`integration.py`) translates to a different convention:

- **`prey_available`** passed to `compute_step()`: shape `(n_groups,)`, indexed with 1-based Ecosim indices stored in a 0-based array. Index 0 is unused (always 0.0). Indices 1 through n_groups-1 hold prey rates. This is the existing behavior of `extract_prey_availability()` → `prey_array`.
- **`consumption_by_prey`** returned in `IBMStepResult`: shape `(n_groups,)`, same convention. `apply_ibm_to_derivative()` iterates `consumption_by_prey[prey_idx]` and subtracts from `deriv[prey_idx]` directly.
- **`zooplankton_prey_idx`** and all prey group indices in this spec are **1-based Ecosim indices** stored in arrays indexed 0 through n_groups-1. Validated to be `>= 1` and `< n_groups` in `SmeltParams`.

**The blending pseudocode in Package 3 uses the same `(n_groups,)` convention** as the existing codebase. The consumption vectors in the blending code are NOT `(n_groups+1,)` — they match the existing `prey_array` shape.

---

## Architecture Foundation

### SuperIndividual Extensions

Four new fields appended **after all existing non-default fields** in the `SuperIndividual` dataclass (required by Python dataclass field ordering — new default-valued fields must follow existing non-default fields):

```python
life_stage: int = 4          # 0=egg, 1=yolk_sac, 2=larva, 3=juvenile, 4=adult
degree_days: float = 0.0     # accumulated thermal development (°C·day), egg stage only
starvation_days: float = 0.0 # consecutive days without sufficient feeding
yolk_energy_kj: float = 0.0  # yolk energy in kJ (only meaningful for life_stage 0-1)
```

All existing code continues unchanged — current individuals default to `life_stage=4` (adult). Existing code uses keyword arguments for `SuperIndividual` construction (verified in `reproduction.py` and `smelt.py`). `smelt.py:initialize_from_ecosim()` (line 282) also uses keyword construction. **Pre-implementation check:** verify no test constructs `SuperIndividual` via positional `*args` unpacking from a 9-element list — such code would silently lose the new fields.

### New Parameter Dataclasses

In a new `development.py` module:

- **`EggParams`** — DD_hatch, DD_mortality, T₀, oxygen thresholds, egg weight/length, per-zone hatch success
- **`YolkSacParams`** — yolk energy content, absorption rate vs temperature, starvation threshold for first-feeding transition, point of no return (days)
- **`LarvalParams`** — size-dependent bioenergetics coefficients (Rs/Ra split), Cmax allometric coefficients, ontogenetic interpolation sigmoid breakpoints (w_mid, w_scale for each transition)
- **`OxygenParams`** — Pcrit by life stage (or as size-dependent sigmoid), lethal thresholds, hypoxia mortality rate, behavioral avoidance weight
- **`ZoneParams`** — zone definitions (spawning/nursery/coastal), connectivity matrix, zone-specific temperature/O2/prey offsets, drift parameters

These are added to `SmeltParams` as **`Optional[...] = None`** fields, placed after the existing defaulted fields (`vbgf_k_mean`, `vbgf_k_sd`, `vbgf_linf_mean`, `vbgf_linf_sd`, `max_age`). When `None`, the corresponding feature is disabled (no early life stages, no oxygen effects, no zonal model) and the existing code path runs unchanged.

**Factory methods:**
- `baltic_defaults()` — **unchanged**, leaves all new params as `None`. Existing 162 tests continue to pass with identical behavior.
- `baltic_defaults_els()` — **new** factory method that populates all early life stage params (EggParams, YolkSacParams, LarvalParams, OxygenParams) with literature-derived defaults. `ZoneParams` remains `None` (zonal model is opted-in separately via `baltic_defaults_zonal()`).

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

`spawn()` produces egg super-individuals (`life_stage=0`) instead of calling `create_recruits()`. Egg weight ~0.001g, length = 0.10 cm (1.0 mm, matching Keller et al. 2020 egg diameter — note: `SuperIndividual.length` is always in **cm** throughout the codebase). Eggs inherit `patch_idx` from spawning female (zone-aware deposition).

**Larval allometry:** The adult allometric relationship (`a_length=0.55, b_length=0.333`) is not valid for sub-milligram larvae. `LarvalParams` includes larval-specific allometric parameters: `a_length_larval` (default: 5.0 cm/g^b), `b_length_larval` (default: 0.35). These produce realistic larval lengths: `5.0 * 0.001^0.35 = 0.45 cm` (4.5 mm) for a first-feeding larva, matching published TL of 4-5 mm (Keller et al. 2020). Length for life_stage 0-2 is computed from larval allometry; life_stage >= 3 uses adult allometry. The transition happens at the juvenile threshold (2.0 cm).

**Population management:** Each spawning event creates at most `EggParams.max_egg_cohorts` super-individuals (default: 3). Total eggs (from all females spawning in this timestep) are distributed across these cohorts via `n_represented`. For example, if 50 females produce 10 million eggs total, 3 egg super-individuals are created with `n_represented ≈ 3.33 million` each.

**Population cap:** `self.individuals` is capped at `SmeltParams.max_super_individuals` (default: 2000). When the cap is reached, same-stage, same-zone, same-sex cohorts are consolidated: the two smallest cohorts (by `n_represented`) are merged into one with:
- `n_represented` = sum
- `weight` = weighted average (by `n_represented`)
- `age` = weighted average
- `length` = recomputed from merged weight via allometry (`a_length * weight ^ b_length`)
- `degree_days` = weighted average (only meaningful for eggs)
- `yolk_energy_kj` = weighted average (only meaningful for yolk-sac)
- `starvation_days` = weighted average
- `energy_reserve` = weighted average
- `is_mature` = either (OR — if one is mature, merged cohort is mature)
- `id` = new id from `_next_id`

This keeps the list bounded while preserving total biomass and population structure.

**Performance guarantee:** Eggs (life_stage=0) and yolk-sac larvae (life_stage=1) skip Phase 1 foraging entirely — they are excluded from the `adaptive_forage()` loop and `growth_step_batch()` call. Only their degree-day/yolk depletion is computed (O(1) per cohort).

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

When `degree_days >= DD_hatch`, egg transitions to `life_stage=1` (yolk-sac) and `yolk_energy_kj` is initialized from `YolkSacParams.initial_yolk_kj`. The `degree_days` field is **only meaningful for the egg stage** (life_stage=0) — it is not used after hatching and is left at its final value (no reset needed).

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

Hatched individuals (`life_stage=1`) carry yolk energy in a **new field** `yolk_energy_kj` added to `SuperIndividual`:

```python
yolk_energy_kj: float = 0.0  # yolk energy in kJ (only meaningful for life_stage 0-1)
```

This avoids dual semantics on `energy_reserve`, which retains its existing dimensionless interpretation for juveniles and adults. Larvae (life_stage=2) use `energy_reserve` in its standard meaning once they enter the bioenergetics growth model. The `yolk_energy_kj` field is ignored (remains 0.0) for life_stage >= 2.

**Initial yolk energy at hatch:** defined in `YolkSacParams.initial_yolk_kj`. Default: 0.15 kJ (based on ~0.001g egg weight × 5 kJ/g energy density × ~30× yolk-to-body ratio for smelt eggs). Set on `SuperIndividual.yolk_energy_kj` when transitioning from egg to yolk-sac stage.

Yolk is depleted by basal metabolism only (no feeding, no active movement):

```
# rs_a_larval is mass-specific rate (g O2 / g fish / day)
# Multiply by weight to get total rate, convert g O2 to kJ via oxycalorific coefficient
total_metabolism_kj = rs_a_larval * weight^(1 + rs_b) * Q10^((T - T_ref) / 10) * oxycal * dt_days
yolk_energy_kj -= total_metabolism_kj
```

Where:
- `rs_a_larval`: larval basal metabolic rate intercept from `LarvalParams` (default: 0.12 g O2/g/day — approximately 91× adult `rs_a` of 0.00132). This high value reflects the extremely high mass-specific metabolic rate of sub-milligram yolk-sac larvae. The allometric relationship `weight^(1+rs_b)` with `rs_b = -0.227` produces very small values at 0.001g (0.001^0.773 ≈ 0.0048), so `rs_a_larval` must be correspondingly large to produce realistic depletion rates
- `rs_b`: metabolic weight exponent, shared with `BioenergParams.rb` (default: -0.227)
- Q10 and T_ref: shared with `BioenergParams.q10` (2.1) and `BioenergParams.t_ref` (10.0°C) — the spec uses the same temperature sensitivity for all life stages
- `oxycal`: oxycalorific coefficient from `YolkSacParams.oxycal_kj_per_g_o2` (default: 13.56 kJ/g O2 — standard conversion for protein-dominated larval metabolism)
- The `weight^(1 + rs_b)` term is the total metabolic rate: `weight^rs_b` (per-gram rate) × `weight` (total body mass)

Duration emerges from the model (not hardcoded). With `rs_a_larval = 0.12` and a 0.001g larva at 10°C: total_metabolism_kj per day = 0.12 × 0.001^0.773 × 1.0 × 13.56 ≈ 0.0079 kJ/day. At 0.15 kJ initial yolk and 0.02 kJ threshold, net depletion = 0.13 kJ, taking ~16.5 days at 10°C. With Q10 scaling at other temperatures: ~25 days at 5.7°C (Q10 factor 0.63), ~14 days at 12.1°C (Q10 factor 1.15).

### First Feeding Transition

**`first_feeding_threshold`**: defined in `YolkSacParams.first_feeding_threshold_kj` (default: 0.02 kJ — ~13% of initial yolk). When `yolk_energy_kj <= first_feeding_threshold_kj`:

**Zooplankton availability source:** `env_forcing['zoo_density']` (mg C/m³), representing local zooplankton concentration. In zonal mode, this is zone-specific via `env_forcing['zone_forcing'][patch_idx]['zoo_density']`.

**Deriving `zoo_density` in Ecosim-coupled runs:** When `zoo_density` is not explicitly provided in `env_forcing`, `compute_step()` derives it from the Ecosim biomass state: `zoo_density = BB[zooplankton_prey_idx] * zoo_conversion_factor`, where `zoo_conversion_factor` (defined in `LarvalParams`, default: 1000.0) converts Ecosim units (tonnes/km²) to mg C/m³. This derivation uses `prey_available` (already passed to `compute_step()`). If `zoo_density` IS explicitly provided, it overrides the derived value.

**Note on `zoo_conversion_factor`:** The default of 1000.0 assumes: 1 tonne/km² = 1 g/m² → ÷ 1m depth → 1 g/m³ → × 1000 → 1000 mg/m³, with carbon fraction ≈ 1.0. This is a rough approximation suitable for shallow lagoon environments (~1m mixed layer). Adjust for local bathymetry: deeper mixed layers need smaller conversion factors.

**`minimum_prey_density`**: defined in `YolkSacParams.minimum_prey_density` (default: 50.0 mg C/m³ — minimum copepod nauplii density for first feeding success).

```
zoo_density = env_forcing['zoo_density']  # or zone-specific variant

if zoo_density >= params.yolk_sac.minimum_prey_density:
    life_stage = 2 (larva, exogenous feeding)
    starvation_days = 0
    energy_reserve = weight * 0.1  # re-initialize using same convention as adult (weight * 0.1)
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

- `am_min` = 0.3 (larvae, passive)
- `am_max` = 1.5 (adults, active foraging)
- `w_activity_mid` = 5.0 g (sigmoid midpoint — a ~5g fish has intermediate activity costs)
- `w_activity_scale` = 3.0 g (sigmoid width — transition spans roughly 2-8g)

### Consumption — Concentration to Adaptive Foraging Blend

**Larval consumption** uses a Type II functional response on zooplankton density from `env_forcing['zoo_density']`:

```
C_larval_scalar = cmax(w) * (zoo / (zoo + K_half))    # scalar consumption (g/timestep)
```

This scalar is allocated entirely to the **zooplankton prey group index** (defined in `LarvalParams.zooplankton_prey_idx`, default: 1 — a **1-based Ecosim index**; validated in `SmeltParams` to be >= 1). The result is a per-prey consumption dict: `{zooplankton_prey_idx: C_larval_scalar}`.

**Adult consumption** comes from the existing `adaptive_forage()` which returns `Dict[int, float]` — consumption allocated across prey groups by profitability. Keys follow the `prey_available` indexing convention (1-based Ecosim indices stored in the dict from `extract_prey_availability()`).

**Blending** operates on the per-prey consumption vectors:

```
alpha(w) = sigmoid((w - w_forage_mid) / w_forage_scale)
# w_forage_mid = 2.0 g (default), w_forage_scale = 1.5 g (default)
# A 0.5g larva: alpha ≈ 0.12 (mostly concentration-dependent)
# A 5g juvenile: alpha ≈ 0.98 (mostly adaptive foraging)

# All vectors have shape (n_groups,) matching existing prey_array/consumption_by_prey convention.
# Index 0 is unused; indices 1..n_groups-1 hold 1-based Ecosim group values.

# Build larval consumption vector: all consumption goes to zooplankton group
C_larval_vec = np.zeros(n_groups)
C_larval_vec[zooplankton_prey_idx] = C_larval_scalar   # 1-based Ecosim index, >= 1

# Build adaptive forage vector from dict (keys are 1-based from prey_available)
C_adaptive_vec = np.zeros(n_groups)
for prey_idx, amount in adaptive_forage_result.items():
    if prey_idx < n_groups:
        C_adaptive_vec[prey_idx] = amount

# Blend
C_total_vec = (1 - alpha) * C_larval_vec + alpha * C_adaptive_vec
# This becomes IBMStepResult.consumption_by_prey (shape n_groups,)
```

At 5mm (alpha ≈ 0): pure zooplankton concentration-dependent. At 50mm (alpha ≈ 1): pure adaptive foraging across all prey. At 15mm: blend of both.

**`K_half`**: half-saturation constant in `LarvalParams.k_half_zoo` (default: 100.0 mg C/m³).

### Cmax Scaling

```
cmax(w, T) = c_a * w^c_b * f(T) * dt_days    # g/timestep (c_a is a daily rate, scaled by dt_days)
```

**Temperature dome function `f(T)`** — Thornton-Lessem formulation (Fish Bioenergetics 3.0/4.0 standard, Thornton & Lessem 1978):

```
f(T) = K_A * K_B
where:
  K_A = (CK1 * L1) / (1 + CK1 * (L1 - 1))    # ascending sigmoid (CK1 in numerator)
  K_B = (CK4 * L2) / (1 + CK4 * (L2 - 1))    # descending sigmoid (CK4 in numerator)
  # BOTH K_A and K_B are always computed for all T; they are NOT conditional branches
  L1 = exp(G1 * (T - CQ))
  L2 = exp(G2 * (CTL - T))
  G1 = (1 / (CTO - CQ)) * ln(0.98 * (1 - CK1) / (CK1 * 0.02))
  G2 = (1 / (CTL - CTM)) * ln(0.98 * (1 - CK4) / (CK4 * 0.02))
```

**FB3 parameter naming convention:**
- **CQ** = lower temperature where rate = CK1 fraction of max (≈ T_min). Default: 2°C
- **CTO** = temperature where ascending limb reaches 0.98 of max (≈ T_opt). Default: 18°C
- **CTM** = temperature where descending limb is still 0.98 of max (≈ T_opt + a few °C). Default: 20°C
- **CTL** = upper temperature where rate = CK4 fraction of max (≈ T_max). Default: 28°C
- **CK1** = small fraction of max at CQ (typically 0.01–0.05). Default: 0.01
- **CK4** = small fraction of max at CTL (typically 0.01–0.05). Default: 0.01

Note: G1 and G2 use hardcoded 0.98 and 0.02 — these define the logistic steepness and are NOT free parameters. The dome gives f(CTO) ≈ 0.98, f(CQ) ≈ CK1 ≈ 0.01, f(CTL) ≈ CK4 ≈ 0.01.

Parameters in `LarvalParams`: `cmax_CQ` (default: 2.0°C), `cmax_CTO` (default: 18.0°C), `cmax_CTM` (default: 20.0°C), `cmax_CTL` (default: 28.0°C), `cmax_CK1` (default: 0.01), `cmax_CK4` (default: 0.01).

### Assimilation Efficiency — Size-Dependent

```
assimilation_efficiency(w) = ae_min + (ae_max - ae_min) * sigmoid((w - w_ae_mid) / w_ae_scale)
```

`ae_min` = 0.55 (larvae), `ae_max` = 0.73 (adults), `w_ae_mid` = 5.0 g, `w_ae_scale` = 3.0 g.

**Relationship to existing `BioenergParams.unassimilated_fraction`:** The ontogenetic sigmoid AE **replaces** `unassimilated_fraction` when Package 3 is active. The old parameter is retained in `BioenergParams` for backward compatibility (used when `LarvalParams` is None / early life stages disabled), but the new `growth_step_batch_ontogenetic()` function uses `assimilation_efficiency(w)` instead of `1 - unassimilated_fraction`.

### Implementation

All sigmoid interpolations are vectorizable in `growth_step_batch_ontogenetic()` — a new function alongside the existing `growth_step_batch()`. No branching by `life_stage` — body size drives everything.

### Juvenile Transition

At configurable size threshold of `LarvalParams.juvenile_length_cm` (default: 2.0 cm = 20 mm, matching Drewes et al.'s model endpoint), `life_stage` advances to 3 (juvenile). Bookkeeping only — bioenergetics are already smoothly adult-like at this size due to the interpolation.

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

Here `stage_background_mortality` is the life stage-specific baseline mortality rate (/day):
- **Eggs:** `EggParams.background_mortality_rate` (default: 0.05 /day — invertebrate predation on sessile eggs)
- **Yolk-sac:** `YolkSacParams.background_mortality_rate` (default: 0.02 /day — some predation, but mainly PNR starvation drives mortality)
- **Larvae:** `LarvalParams.background_mortality_rate` (default: 0.01 /day — predation in Phase 3 is the main mortality source, this covers additional diffuse losses)
- **Juveniles/Adults:** 0.0 (predation in Phase 3 is the sole non-oxygen mortality source)

`OxygenParams.hypoxia_mortality_rate` (default: 0.5 /day) is the maximum additional mortality at zero oxygen.

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

### Ecospace Coexistence

The 3-zone model and full Ecospace grids are **mutually exclusive spatial modes**, resolved by priority:

1. **`SpatialContext` provided (Ecospace mode):** The IBM uses the full Ecospace grid. `ZoneParams` provides a `zone_of_patch: np.ndarray` of shape `(n_patches,)` mapping each Ecospace patch to a zone (0, 1, or 2). Zone-specific forcing is resolved from the patch's zone. Juveniles/adults use `SpatialContext.adjacency` for movement. Passive drift for early stages uses `ZoneParams.connectivity_matrix` applied at the zone level (a larva drifts from one zone to another, then is assigned a random patch within the target zone).

2. **`ZoneParams` only (standalone zonal mode):** The IBM uses its own 3-zone connectivity matrix. `patch_idx` values are 0, 1, 2 directly. No Ecospace adjacency matrix.

3. **Neither (0D fallback):** All individuals in a single well-mixed environment. Backward compatible with current behavior.

Note: Ecospace's `calculate_spatial_flux()` already skips IBM groups (`if group_idx in ibm_groups: continue`), so there is no conflict with Ecospace's diffusion/advection. The IBM always handles its own movement in Phase 5.

### Passive Drift and Ecospace

Passive drift for yolk-sac larvae and early larvae uses `ZoneParams.connectivity_matrix` (the 3×3 base probability matrix), **not** the Ecospace adjacency. This is because drift is a zone-level process (larvae move from spawning grounds to nursery), not a cell-level diffusion. Only juveniles and adults (life_stage >= 3) use the full `SpatialContext.adjacency` for within-zone and between-zone movement.

### Backward Compatibility

Without spatial context or ZoneParams, model collapses to 0D (single well-mixed zone).

### Review Checkpoint

Eggs stay in zone 0. Larvae drift to zone 1. Juveniles concentrate in zone 1 with coastal migration. Adults cycle coastal → river (spawn) → coastal.

---

## Package 6: Calibration, Validation, and Sensitivity Analysis

### Two-Stage Calibration

**Stage A:** Ecosim-level — fit vulnerability parameters (VV) to adult biomass time series using `fit_to_timeseries()`. Anchors food web context.

**Stage B:** IBM-specific — fit early life stage parameters to recruitment indices:
- `egg_background_mortality_rate` — controls egg-stage survival
- `minimum_prey_density` — first-feeding threshold
- `point_of_no_return` — starvation window severity
- `egg_O2_lethal` — oxygen mortality threshold
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
| egg_background_mortality_rate | 0.01–0.10 /day |
| minimum_prey_density | 20–100 mg C/m³ |
| point_of_no_return | 2–7 days |
| Pcrit (larval) | 2.0–4.0 mg/L |
| activity_multiplier range | 0.2–0.5, 1.0–2.0 |
| w_mid (bioenergetics transition) | 1–10 g |
| K_half (zooplankton half-saturation) | 50–200 mg C/m³ |

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
