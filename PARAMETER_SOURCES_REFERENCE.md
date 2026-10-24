# Curonian Lagoon Model (33-Group) — Parameter Sources Reference

**Model Version:** Phase 2, Checkpoint 3  
**Date Generated:** 2026-10-23  
**Groups Documented:** 33

## 1. BIOMASS PARAMETERS (B)

### Source Origin
- **Primary:** Phase 1 landings data (2001-2011 average)
- **Method:** Annual catch aggregated, normalized by fishing effort
- **Quality Assessment:** Commercial fish 75-80, Zooplankton 60-65, Producers 70

### Uncertainty Ranges
- Commercial fish: ±30-40% (CPUE uncertainty)
- Non-commercial fish: ±40-50%
- Zooplankton: ±40-50%
- Phytoplankton: ±35-45%

## 2. PRODUCTION/BIOMASS (P/B) PARAMETERS

### Source Origin
- **Primary:** FishBase, species-specific temperature adjustments
- **Method:** Q10=2.0 temperature correction for 5°C difference

| Species Group | Quality Score | Source |
|---|---|---|
| Fish | 78-85 | FishBase + literature |
| Zooplankton | 70-75 | Life table analysis |
| Phytoplankton | 80 | JGOFS/WOCE data |

### Uncertainty Ranges
- Fish species: ±15-20%
- Zooplankton: ±20-30%
- Bacteria: ±30-50%

## 3. CONSUMPTION/BIOMASS (Q/B) PARAMETERS

### Source Origin
- **Primary:** Expert bioenergetics estimates
- **Method:** Inferred from metabolic rate, activity, assimilation efficiency

| Species Group | Quality Score | Confidence |
|---|---|---|
| Fish | 60-70 | Medium |
| Zooplankton | 55-65 | Low-Medium |
| Benthos | 55 | Low |

### Known Limitations
1. Stomach content data minimal
2. Seasonal Q/B variation not captured
3. Zooplankton Q/B highly uncertain
4. No direct validation available

## 4. DIET COMPOSITION MATRIX (DC)

### Source Origin
- **Primary:** Expert opinion + historical stomach content
- **Method:** 33×33 matrix via Ecopath mass balance solver
- **Quality:** High-confidence diets 50%, low-confidence 35%

## 5. ECOTROPHIC EFFICIENCY (EE)

### Source Origin
- **Method:** EE = (Landings + Predation + M2) / Total production
- **Constraint:** 0 < EE ≤ 1.0
- **PEST Target:** Primary calibration variable

## 6. ENVIRONMENTAL FORCING

### River Discharge (Nemunas)
- **Quality Score:** 95 (measured, 123 years)
- **Uncertainty:** ±5%

### Temperature
- **Quality Score:** 90
- **Period:** 1910-2023

## PARAMETER QUALITY SUMMARY

| Category | Avg Quality | Priority | Next Action |
|---|---|---|---|
| Biomass (B) | 73 | Medium | Use in Ecopath |
| Production (P/B) | 76 | Medium | Review species-specific |
| Consumption (Q/B) | 63 | High | Needs validation |
| Diet Matrix (DC) | 50 | High | PEST fitting |
| EE | 62 | High | PEST primary target |
| Fishing | 80 | Medium | Use as constraint |
| Forcing | 90 | Medium | Use as-is |

## RECOMMENDATIONS FOR CHECKPOINT 4

### High-Priority Calibration Targets
1. Ecotrophic Efficiency (EE) for all consumers
2. Vulnerability (VV) parameters
3. Q/B for zooplankton

### Medium-Priority Parameters
1. P/B for non-commercial fish
2. Diet matrix entries via vulnerability
3. Natural mortality (M2)

### Low-Priority (assume fixed)
1. Commercial fish P/B
2. Phytoplankton P/B
3. Fishing selectivity

**Status:** Ready for Checkpoint 4 PEST Calibration Setup
