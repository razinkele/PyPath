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
- PyPath's dispersal of 300 km/yr for every group: on that grid an effective 2.70 per year across a cell side (see the update below the table).

Deviations are of area-mean biomass, over every group and month.

| Comparison | Median | Max |
|---|---|---|
| PyPath Ecosim vs EwE Ecosim | 1e-7 | 3e-6 |
| EwE Ecospace vs EwE Ecosim | 13 % | 104 % |
| PyPath Ecospace vs EwE Ecospace | 15 % | 190 % |
| PyPath Ecospace vs EwE Ecosim | 1.1e-7 | 3.4e-6 |
| PyPath Ecospace with habitat capacity on vs EwE Ecospace | 22 % | 208 % |

**Update (2026-11-13).** These runs placed the grid's centroids in km on a degree scale, so diffusion saw neighbours 111 times too far apart (4,074 km for 36.7 km cells), and the rate of 300 came to 2.70 per year across a cell side. That is within 4 % of EwE's movement rate, Mrate = Mvel / (3.14159 x CellLength) = 2.60 per year. `read_ecospace` now places the grid as EwE's basemap and converts Mvel to Mrate. Rerun with both: capacity off is unchanged (15 %, 190 %); capacity on gives 21.4 % and 208 %, and the cell maps are 0.286 from EwE's at year 10 (0.287 before). With the grid fixed but Mvel taken as the rate, the capacity-on run is not finite: 300 per year is far beyond what one explicit RK4 step a month keeps stable (B3). B3 has since replaced the explicit step with an implicit one: capacity on then gives 21.1 % and 416 % (Anchovy 0-0+, whose explicit run was unstable: P/B 23 per year plus diffusion), and the cell maps are 0.279 from EwE's at year 10.

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
6. **Reader gap.** `read_ecospace` does not read the depth map, so a database's land cells are modelled as water: here all 1302 cells, against EwE's 304. The map is stored in `EcospaceScenario.DepthMap` as rows separated by ';' and columns by ','. The spike took the water mask from the oracle's grid. **Closed** (step 1 below): `read_ecospace` now reads DepthMap and ExclusionMap as EwE does, and its grid is the oracle's 304 water cells. The capacity-on run on the reader's grid matches the run on the oracle's mask to 2e-15 (21.5 % median, 208 % max from EwE's Ecospace).
7. **Oracle outputs.** The area mean in `ecospace_steps.csv` is the mean over the 304 water cells; checked against the maps. Run 43288513485's maps also hold one padding cell, (32, 43), outside the 31 × 42 grid. The harness now writes only the model's cells.

## Next steps (P6)

This case is the yardstick for each step. Step 3 and the nutrient pool are hypotheses until a step is measured against it:

1. Read the depth map (the land mask) and EwE's per-group Ecospace inputs (Mvel and habitat) in `read_ecospace`. **Depth map done:** DepthMap and ExclusionMap give the grid's water cells, pinned to the oracle's 304 cells (`test_ecospace_depth_map.py`). Mvel was already read. **Habitat capacity done:** on EwE's grid `read_ecospace` sets each group's capacity as EwE's SetHabCap does (input capacity map, habitat preference x share, normalised by the maximum, the low-capacity gradient), pinned to the oracle's HabCap for all 22 groups and 304 cells to 1e-6 (`test_ecospace_habitat_capacity.py`); the capacity-on run on the reader's grid and capacity matches the run on the oracle's HabCap to 3e-8. Environmental responses and migration maps are not applied. **Step 2 done:** runs apply the habitat capacity always. Against the oracle's cell maps (months 1, 12, 60, 120), the distance between PyPath's and EwE's distribution of each group over the cells (total variation, median over groups) falls from 0.49 to 0.13, 0.27, 0.29 and 0.29; the area-mean deviation rises from 15.3 % to 21.5 % (max 190 % -> 208 %).
2. Apply habitat capacity always (B4). **Done** (see step 1).
3. Add EatEff and RelVulBad, and producer PbSpace (B5). **Done, with EwE's base calibration and nutrient rule** (`pypath.spatial.calibration`, `test_ecospace_ewe_calibration.py`).
   - **The calibration.** EwE's `AdjustSpaceParsNew` (BRatio, Vspace, the 5-pass search-rate iteration with MeanBP, PbSpace) is mapped onto the kernel with global bases B0 x BRatio. Each link gets X' = 1 + 2 Vspace / (A x Pb) and QQ' = Vspace x Bb / X'; producers' links from group 0 get Q0 x BRatio. Each patch's EatEff (its habitat capacity) divides the predator term of the vulnerable pool and scales producers' production. The nutrient pool keeps the map-wide base.
     - With a uniform habitat this is Ecosim exactly.
     - Against the oracle's own constants: Qlink and Vspace match, BRatio is within 0.4 %, and search rates are within 2.5 %. EwE computes TotHabCap before the low-capacity gradient, hence the BRatio gap. Of the 52 links, only the two into the anchovy stanzas could not be compared: EwE's stanza index is read after the run.
     - On X ≈ 1 links EwE's iteration does not converge, and map-wide base consumption is 2/3 of Ecopath's. The oracle's link flows confirm this (Zooplankton -> Sprat 0.670 at month 1).
   - **The nutrient rule.** EwE's Ecospace never applies Ecosim's nutrient forcing: a cell's free nutrient is NutTot x its relative productivity less its biomass (`cSpaceSolver.vb` 860). The Black Sea scenario forces nutrients between 0.7 and 1.15. Without that forcing EwE's Phytoplankton is twice its Ecosim level, which was the "flat doubling" of finding 2.
   - **How it was found.** The oracle harness now dumps map-wide link flows, group rates, the calibration constants and the cells' nutrient factor (oracle runs 43498801657, 43500051427 and 43503554514). PyPath's month-1 link flows then matched EwE's to a median of 0.8 %.
   - **Not followed:** EwE's Ecospace start runs a second derivative at the start biomass (SetBiomassesEcospace), advancing Hden again. It changes only the first months of handling groups (it brings the top predators' month-1 consumption from 1.2 to 1.1x EwE's) and nothing at 10 years. It is left out so that a uniform map stays equal to Ecosim.
   - **Result on this case:** area means median 3.9 %, max 62 % (before: 13.6 % / 225 % with B2, 21.1 % / 416 % before B2). Cell maps 0.008 / 0.020 / 0.035 at months 12 / 60 / 120. The largest remaining gaps are juvenile anchovy (a stanza group, 62 % max) and Aurelia (27 %).
   - **Earlier variants** that did not use EwE's calibration made things worse: EatEff on top of #58 (median 50 %), on producers only (54 %).
4. Compare the cell maps in `ecospace_maps.csv`, not only area means. **Done** at month 1, which showed where the runs part.
   - EwE's month-1 maps are B0 x BRatio x HabCap to within a few per cent (log correlation 1.000): EwE starts in proportion to capacity and keeps that pattern.
   - PyPath started the same way, but its plain diffusion levelled the pattern within the month: the per-cell log ratio to EwE spread 2.9 (p90 - p10).
   - With PyPath's dispersal switched off, that spread fell to 0.15.
5. Bring dispersal to EwE's rules (B1–B3). B1 and B3 done (#105, #108). **B2 done:** movement is EwE's capacity-weighted rule (`SetMovementParameters`, `cEcoSpace.vb` 3804-3815). The rate from cell x to y is Mrate x min(1, HabCap_y / HabCap_x): full rate into habitat at least as good, slowed by the habitat ratio into poorer habitat. Density in proportion to capacity is then at rest. On this case:

   | | area mean: median / max | cell maps (median distance), months 1 / 12 / 60 / 120 |
   |---|---|---|
   | plain diffusion | 21.1 % / 416 % | 0.111 / 0.266 / 0.273 / 0.279 |
   | B2 | 13.6 % / 225 % | 0.008 / 0.034 / 0.045 / 0.041 |

   Pinned in `test_ecospace_capacity_movement.py`: the medians at months 1 and 12 must stay below 0.03 and 0.06. The oracle's month 1 and 12 maps are in `ecospace_maps.csv` of the test data.

6. Distribute multi-stanza groups as EwE's NewMultiStanza model (B6). **Done** (`pypath.spatial.stanza_space`, `test_ecospace_new_multistanza.py`). The Black Sea scenario has ModelType 2, which EwE also gives every new scenario. EwE then leaves the stage groups out of the integration, keeps a population index per cell and stage, and spreads each stage's map-wide total by it every month.
   - **Two parts, both needed.** In a spike, the index alone brought the anchovy maps close to EwE's but made the totals worse (juvenile anchovy max 62 % -> 79 %). The age structure's update averaging the cells' rates weighted by the index is what brought the totals in line: in EwE a cell counts roughly by the square of its density, in a plain sum by its density.
   - **The start.** The index has to exist before the first update weights by it: setting it there moved the month-1 juvenile anchovy deviation from -16 % to -8 %.

   | | area mean: median / max | Anchovy 0-0+ max / median | anchovy maps at month 12 (0-0+ / 1-1+) | cell maps (median distance), months 12 / 60 / 120 |
   |---|---|---|---|---|
   | B5 | 3.9 % / 62 % | 62 % / 23 % | 0.138 / 0.044 | 0.008 / 0.020 / 0.035 |
   | B6 | 3.4 % / 35 % | 35 % / 8 % | 0.006 / 0.004 | 0.005 / 0.011 / 0.017 |

   The remaining juvenile anchovy maximum is late drift in year 10, where EwE has 0.004 t/km². Aurelia (27 %) and POM (22 %) are untouched.

7. Follow EwE's Ecospace detritus inflow (B7). **Done** (`EcospaceParams.detritus_pass_on`, `test_ecospace_detritus_pass_on.py`). After B6 the largest gaps were Aurelia (27 %) and POM (22 %).
   - **Aurelia was inherited.** Its month-1 link flows and Q/B match EwE's exactly and it has no predators: its level follows its food, half of it POM.
   - **POM and Detritus grew too fast.** Their losses matched EwE's (TotalLoss at month 1: POM 10.60 vs 10.58), and so did DetritusOut (an oracle dump: EwE's Ecospace 0.0462, Ecosim 0.0449, PyPath 0.0449) and the inflow parameters GS, mo, DF and DetPassedProp (a second dump). Yet EwE's POM gained about 2.9 less per unit area a month than its own formula gives from its own flows and biomass.
   - **The cause.** From month 3 that shortfall equals the flow the other pool passes on (POM: 0.0797 x B(Detritus) x 0.3 = 2.13 / 2.34 / 2.85 at months 12 / 24 / 60, against shortfalls of 2.17 / 2.34 / 2.83). EwE's `SimDetritusSpace` loops over group indices 1 to nDet, living groups, so no pool receives anything from another.
   - **Ruled out on the way:** detritus import and migration (none in the model), the flow a pool sends itself (both codes skip it), an open map edge (`SetBoundaryDepths`; no Black Sea water cell touches the grid edge) and the mortality multiplier (no response maps).

   | | area mean: median / max | Aurelia max / median | POM max / median | cell maps (median distance), months 12 / 60 / 120 |
   |---|---|---|---|---|
   | B6 | 3.4 % / 35 % | 27 % / 22 % | 22 % / 16 % | 0.005 / 0.011 / 0.017 |
   | B7 | 2.4 % / 26 % | 6 % / 3 % | at most 6 % max | 0.005 / 0.010 / 0.012 |

   Not followed: EwE's first Ecospace step applies 2/3 of the month's change (BDF2 starting from the start state), visible in the detritus pools' first month.

Each step can then be pinned against the oracle, as the Ecosim fixes were. The oracle numbers (area means, cell maps, grid) can be committed as test data when the first of these steps lands. The water cells are committed (`packages/pypath/tests/data/ewe_oracle/blacksea/`); the rest follows with the steps that use it.
