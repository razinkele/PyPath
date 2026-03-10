# Analysis API Reference

Tools for model diagnostics, pre-balance checks, and post-simulation analysis.

## Pre-Balance Diagnostics

Pre-balance diagnostics help identify parameter issues before running
the Ecopath mass-balance solver. Checks include biomass ratios,
ecotrophic efficiency bounds, and diet composition consistency.

::: pypath.analysis.prebalance
    options:
      show_root_heading: true
      members_order: source

## Network Indices

Network analysis functions for food web structure: connectance, omnivory
index, mean trophic level, and system throughput.

See also [Core API: Analysis & Network Indices](core.md#analysis-network-indices)
for `mixed_trophic_impact()` and `summarize_ecosim_output()`.
