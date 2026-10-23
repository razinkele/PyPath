# Extended Multistanza Configuration for Curonian Lagoon

## Overview
This guide describes extensions to the multistanza model including additional fish
and bird species with age-structured populations.

## New Stanza Groups

### Planktivorous Fish (2 stanzas)
**Characteristics:**
- Fast-growing species (K=0.35, vs 0.12-0.15 for other fish)
- Early maturity (reproduce after 0.8 years)
- Short-lived (Z=1.8 juveniles, 0.8 adults)
- Zooplankton feeders

**Life History:**
- Juvenile (0-9 months): 0.5-15 g, weight gain ~20 g/month
- Adult (9+ months): 15-200 g, slower growth

**Ecological Role:**
- Link zooplankton to higher trophic levels
- Important forage for predatory fish
- Sensitive to productivity changes (environment forcing candidate)

**Rationale for Stanzas:**
- High juvenile mortality justifies separate juvenile class
- Rapid early growth phase distinct from asymptotic adult phase

### Cormorants (2 stanzas)
**Characteristics:**
- Piscivorous waterbird (eats fish, especially demersal and planktivorous)
- Top predator in model
- Long-lived (adults, Z=0.3 yr^-1)

**Life History:**
- Juvenile (0-1 year): 0.5-1.8 kg, fledgling to first year
- Adult (1+ years): 1.8-2.2 kg, breeding adults

**Ecological Role:**
- Control of fish populations
- Sensitive to fish abundance
- Candidate for bottom-up/top-down analysis

**Rationale for Stanzas:**
- Juvenile birds less competent foragers (higher Z)
- Adults occupy breeding habitat (distinct niche)

## Integration with Ecosystem

### Biomass Allocation
- Demersal fish: 1.0 t/km^2 (2 stanzas)
- Predatory fish: 0.5 t/km^2 (3 stanzas)
- Planktivorous fish: 0.2 t/km^2 (2 stanzas)
- Cormorants: 0.02 t/km^2 (2 stanzas)

### Diet Interactions
- Planktivorous -> Predatory, Cormorant
- Demersal -> Predatory, Cormorant
- Zooplankton -> Planktivorous

### Recruitment Coupling
All juvenile stanzas respond to recruitment forcing:
- Temperature response
- Productivity response
- Available habitat

## Analysis Opportunities

### 1. Sensitivity to Fishing
- Which stanzas drive system productivity?
- Juvenile harvest effects?
- Size-selective fishing strategies?

### 2. Predator-Prey Dynamics
- Cormorant growth limited by planktivorous fish?
- Fish recruitment driven by food availability?
- Cascading effects of bird population changes?

### 3. Ontogenetic Shifts
- Do planktivorous move to deeper water as they grow?
- Diet changes across stanzas?
- Spatial niche partitioning?

## Model Complexity Trade-offs

### Benefits of Extended Stanzas
+ More realistic population dynamics
+ Capacity to model age-specific harvests
+ Better recruitment dynamics
+ Habitat specialization by age class

### Costs
- More parameters to estimate/calibrate
- Increased computational cost
- More data needed for validation
- Potential overfitting with limited data

## Implementation Checklist

- [ ] Define Von Bertalanffy parameters (K, Linf, t0)
- [ ] Set natural mortality by stanza (juvenile > adult)
- [ ] Allocate biomass across stanzas
- [ ] Define inter-stanza diet overlaps
- [ ] Set up recruitment forcing linkages
- [ ] Validate mass balance for each species
- [ ] Run sensitivity analysis on K and Z
- [ ] Compare single vs. multi-stanza predictions
- [ ] Document data sources and assumptions
