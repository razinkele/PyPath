# Spatial Ecospace for Multistanza Models

## Overview
This guide demonstrates how to implement spatially-explicit simulations of 
age-structured fish populations with habitat preferences and ontogenetic migration.

## Spatial Grid

### Habitat Zones
1. **Littoral Shallow**: 0-1 m (nursery habitat for juveniles)
2. **Littoral Deep**: 1-3 m (shallow foraging habitat)
3. **Sublittoral**: 3-6 m (transition zone)
4. **Profundal Shallow**: 6-8 m (adult habitat)
5. **Profundal Deep**: 8-10 m (deep foraging)

### Connectivity
- Adjacent patches connected by dispersal
- Dispersal rates: 1% per month to neighboring zones
- Variable connectivity based on habitat proximity

## Habitat Preferences

### Demersal Fish
- **Juvenile**: Prefers littoral (0.8), avoids profundal (0.0)
- **Adult**: Prefers sublittoral-profundal (0.8), less littoral (0.3)

### Predatory Fish
- **Juvenile**: Prefers littoral (0.9), strong nursery habitat preference
- **Sub-adult**: Transitions to sublittoral (0.9), leaving littoral (0.5)
- **Adult**: Prefers profundal (0.8), avoids shallow (0.2)

## Ontogenetic Migration

### Process
1. Juveniles spawn in shallow littoral habitat
2. As juveniles grow, movement to deeper water increases
3. Adults occupy deeper profundal zones
4. Migration rates vary by stanza (5-20% per year)

### Drivers
- Size-dependent habitat preference
- Predation avoidance (deeper water reduces predation)
- Forage size selectivity (larger fish need larger prey)

## Simulation Process

### Initialization
- Distribute each stanza group across patches based on habitat preference
- Initial condition: all groups in preferred habitat

### Dynamics
- **Migration**: Stanza-specific movement from source to destination patch
- **Dispersal**: Local diffusion (1% monthly) between connected patches
- **Growth**: Implicit in stanza transitions (modeled in Ecopath/Ecosim)

### Timestep
- Monthly resolution (12 timesteps per year)
- 20-year simulation = 241 timesteps

## Ecosystem Effects

### Spatial Heterogeneity
- Juveniles concentrated in high-productivity littoral zones
- Adults distributed across diverse habitats
- Reduced overlap -> reduced inter-stanza predation

### Recovery Potential
- Juvenile nurseries act as population bottlenecks
- Protection of littoral habitat = protection of recruitment
- Loss of littoral habitat cascades to adult populations

## File Outputs

- `ecospace_spatial_dynamics.png`: Stacked area plots by patch
- `ecospace_summary.csv`: Initial, final, peak biomass fractions

## Future Enhancements

1. **Fishing Pressure**: Spatial targeting of specific zones
2. **Habitat Loss**: Remove littoral patches and observe recruitment collapse
3. **Marine Protected Areas**: Restrict fishing in nursery zones
4. **Temperature Coupling**: Link spatial distribution to thermal preferences
5. **Predator-Prey Overlap**: Calculate predation risk by patch
