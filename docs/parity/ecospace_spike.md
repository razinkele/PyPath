# Ecospace oracle spike (2026-11-12)

**Question** (roadmap P6): can the headless EwE 6.7.144 oracle run Ecospace? If so, PyPath's Ecospace can be checked against EwE's own spatial trajectories, as Ecosim now is.

**Verdict: yes.** The unmodified EwE core runs an Ecospace scenario headless and returns its results.

## Harness

The private oracle repo `razinkele/ewe-oracle` has a fourth, optional stage in `EweOracle`. It is turned on with `--ecospace DBID`, or the workflow input `ecospace`.

What it does:

- loads the scenario after Ecosim (`cCore.LoadEcospaceScenario`);
- runs it unthreaded (`cCore.RunEcospace(handler, False)`);
- exits with 50 on failure. No EwE source is changed.

It writes three files:

| File | Content |
|---|---|
| `ecospace_steps.csv` | Area-mean biomass by group at every time step |
| `ecospace_maps.csv` | Biomass of every water cell at months 1, 12, 60 and 120 |
| `ecospace_grid.csv` | Depth and EwE's habitat capacity for every cell |

The comparison script is `tools/ecospace_compare.py` in the same repo.

## Case

The Black Sea EwE-F test case (`Data/BlackSea_EwEF_testcase.eweaccdb`):

- Ecospace scenario 1, on Ecosim scenario 1;
- 31 × 42 cells of 36.7 km, of which 304 are water;
- 10 years in monthly steps;
- no effort prediction and no spin-up.

Oracle run 43288513485 completed all 120 steps.

## First numbers

The PyPath Ecospace runs used:

- EwE's 304 water cells, with 4-neighbour links;
- EwE's habitat capacity in every cell;
- PyPath's dispersal of 300 km/yr for every group.

Deviations are of area-mean biomass, over every group and month.

| Comparison | Median | Max |
|---|---|---|
| PyPath Ecosim vs EwE Ecosim | 1e-7 | 3e-6 |
| EwE Ecospace vs EwE Ecosim | 13 % | 104 % |
| PyPath Ecospace vs EwE Ecospace | 15 % | 190 % |
| PyPath Ecospace vs EwE Ecosim | 1.1e-7 | 3.4e-6 |
| PyPath Ecospace with habitat capacity on vs EwE Ecospace | 22 % | 208 % |

With capacity off, PyPath's 304 patches stay identical: (max − min) / mean across patches is exactly 0 for every group and month. With capacity on, they spread to up to 11 times the mean, and the run is 21 % off EwE's Ecosim at the median.

## Findings

1. **The gap is spatial only.** PyPath's Ecosim matches EwE's on this model to 3e-6.
2. **EwE's Ecospace is far from its Ecosim.** On this model, Phytoplankton settles at about twice its Ecosim level, and the top predators end at 0.6 to 0.9 of theirs.
3. **PyPath's Ecospace reproduces Ecosim.** PyPath reads a habitat capacity map only when environmental drivers are given (`spatial/integration.py`, `_capacity_matrix`). An MPA capacity bonus alone multiplies a capacity of 1. Without drivers every patch evolves alike, whatever the map. This is open item B4, "capacity always on".
4. **Switching capacity on does not close the gap.** I passed an empty `EnvironmentalDrivers()`, which gives PyPath's #58 capacity rule: start biomass in proportion to capacity, and link flows scaled by c_prey·c_pred / MeanBP. The gap moves:
   - Phytoplankton improves, from −50 % to −31 %;
   - Bluefish (+208 %), Whiting (+178 %) and Shad (−74 %) get worse by year 10.
5. **What else could explain the gap: hypotheses, not yet tested.** First, what the scenario does not hold. It has:
   - no capacity drivers, data connections, driver layers, migration maps, MPAs, Ecospace fleets or relative-PP map;
   - Mvel 300 for every group (the database's value, not a reader default);
   - RelMoveBad and RelVulBad of 1.

   So none of these inputs explains the gap. EwE terms that PyPath lacks, read from the source:
   - EatEff = HabCap in poor cells (`cSpaceSolver.vb:549`);
   - producer shading PbSpace = pbbiomass / BRatio (`cEcoSpace.vb:6121`);
   - Ecospace's own detritus and emigration terms (deviations A1-space and A2-space);
   - Ecospace's nutrient pool, which sums biomass per cell (`cEcoSpace.vb:3323`). It is a candidate for Phytoplankton's flat doubling, on a model with nutrient forcing;
   - EwE's dispersal and solver (B1–B3).
6. **Reader gap.** `read_ecospace` does not read the depth map, so a database's land cells are modelled as water: here all 1302 cells, against EwE's 304. The map is stored in `EcospaceScenario.DepthMap` as rows separated by ';' and columns by ','. The spike took the water mask from the oracle's grid.
7. **Oracle outputs.** The area mean in `ecospace_steps.csv` is the mean over the 304 water cells; checked against the maps. Run 43288513485's maps also hold one padding cell, (32, 43), outside the 31 × 42 grid. The harness now writes only the model's cells.

## Next steps (P6)

This case is the yardstick for each step. Step 3 and the nutrient pool are hypotheses until a step is measured against it:

1. Read the depth map (the land mask) and EwE's per-group Ecospace inputs (Mvel and habitat) in `read_ecospace`.
2. Apply habitat capacity always (B4).
3. Add EatEff and RelVulBad, and producer PbSpace (B5).
4. Compare the cell maps in `ecospace_maps.csv`, not only area means.
5. Bring dispersal to EwE's rules (B1–B3).

Each step can then be pinned against the oracle, as the Ecosim fixes were. The oracle numbers (area means, cell maps, grid) can be committed as test data when the first of these steps lands.
