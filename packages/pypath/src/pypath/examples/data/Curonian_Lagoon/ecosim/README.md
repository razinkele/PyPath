# Curonian Lagoon test case (Ecosim)

Ecosim and Ecospace inputs of the Curonian Lagoon pelagic model (19 groups; see
`ecopath-f/testcases/Curonian_Lagoon/README.md` for the model and its source,
`curonian_pelagic.eweaccdb` here).

Files, all written by `api/scripts/make_curonian_testcase.py`:

| File | Content |
|---|---|
| `Curonian_Lagoon.h5` | Ecopath-F result of the Ecopath test case (the Ecosim input) |
| `Curonian_Lagoon_Scenario.csv` | Ecosim group settings at the EwE defaults, as in Generic_37 (max rel. P/B 2, feeding time adjust 0.5, Q/B max 1000, no switching); producers and detritus `-999` |
| `Curonian_Lagoon_vul.csv` | vulnerability 2 on every realised diet link (the EwE and PyPath default), `-999` elsewhere; 17 predator columns |
| `Curonian_Lagoon_F.csv` | fishing forcing type 5 (multiplier of the Ecopath fishing mortality) = 1.0 for 600 months, so Ecosim can run up to 50 years at constant fishing |
| `Curonian_Lagoon_Fleets.csv` | each group's Ecopath catch by fleet (one fleet, t/km²/yr); the API's fleet effort scenarios read it |
| `Curonian_Lagoon_filenames.nml` | namelist template; the API sets `tf` per run; no nutrient forcing (constant 1.0) |
| `Curonian_Lagoon_outline.geojson` | the lagoon outline (WGS84; from `curonian.gpkg`, layer `curonian_boundary__curonian_contour`, feature CODE 1, Marine Research Institute): with the strait corridor it defines the trimmed grid (lagoon + strait, the coastal Baltic dropped), and it is the only spatial input of the fallback grid |
| `Curonian_Lagoon_domain.geojson`, `Curonian_Lagoon_strait.geojson` | provenance: the dissolved 29-box model domain (`b29polys.gpkg` of the Curonian data repository) and the Klaipėda strait thalweg (SFINCS `channels.geojson`) extended to the mouth between the moles |
| `Curonian_Lagoon_grid.csv` | 94 × 51 land/sea mask, rows north to south, 0.009° × 0.0158° cells (~1 km): sea where at least half of the 200 m water mask (`resusp_grids/watermask.tif`, EPSG:3346) in the cell is water and the centre lies in the 29-box model domain, plus the cells the strait thalweg crosses (the 400 m channel is below the mask fraction); then trimmed to the lagoon (`api/scripts/curonian_lagoon_grid.py`): the cells inside the lagoon outline and the Klaipėda strait corridor up to the mouth between the moles (east of 21.093° E, 55.60–55.716° N), the largest 4-connected body, cropped with a one-cell land margin. 1576 sea cells: 1563 lagoon, 13 strait. The coastal Baltic of the domain (758 cells) is dropped (2026-11-08); the strait ends at the mouth |
| `Curonian_Lagoon_depth.csv` | cell-mean water depth (m) from the repository's filled depth raster (`resusp_grids/depth_filled.tif`; `gdalwarp -r average`); forced strait cells without a depth take 12 m (the dredged channel), other gaps their neighbours' mean; land -999 |
| `Curonian_Lagoon_habitat_depth.csv` | habitat preference of each group as a piecewise-linear curve of depth (nodes 0-6 m) with the ecological basis; edit and regenerate to change the maps |
| `Curonian_Lagoon_SpatialDistribution/Group_1..19.csv` | habitat maps: the group's depth curve evaluated on the cell depth (0 on land). Ecospace rescales each map to a mean of 1 over sea cells, so only the relative pattern matters |
| `Curonian_Lagoon_Advection.csv` | EwE-F direction code per cell (0 = none) from the residual circulation, see *Advection* below; the API's `advection=false` replaces it by zeros |
| `Curonian_Lagoon_residual_flow.json` | the residual circulation the field is made from: mean velocity vector of each of the 29 boxes and the net flow across every box interface, 1993-2014 means of the SHYFEM 29-box run (`--derive-residual-flow` rebuilds it from the Curonian and AQUABC repositories) |
| `Curonian_Lagoon_sediment.csv`, `Curonian_Lagoon_sediment_accumulation.csv`, `Curonian_Lagoon_sediment_types.csv` | bottom sediments on the grid (`api/scripts/curonian_sediment_grid.py` from `sedimenta.shp` of the 2005 zonation work, whole lagoon): grain-size class of the type covering most of each cell (1 = medium sand ... 7 = silty clayey mud, 0 = not mapped), accumulation zones, and per type organic carbon, N, P and sediment fluxes; Model inputs layers; chironomids and oligochaetes follow the sediment's organic carbon (×0.3 on sand to ×1 on mud, `Curonian_Lagoon_habitat_sediment.csv`, `api/scripts/curonian_sediment_habitat.py`) |
| `Curonian_Lagoon_georef.json` | exact georeference of the grid (`approximate: false`) for the maps |

The EwE 6 database holds no Ecosim scenario, time series or fitted vulnerabilities,
so this is a near-equilibrium baseline: only Demersal fish and Predatory fish carry
landings. A 50-year run with these settings stays finite and positive; the largest
excursions are chironomids rising to 1.39x and Predatory fish falling to 0.29x
their Ecopath biomass (birds settle at 0.35x; `python -m scripts.make_curonian_testcase
... --ecosim-years 50` prints the check). PyPath's own Ecosim demo on this
model reports a zooplankton crash after year 12; EwE-F with these settings does
not show it.

## Ecospace

Habitat maps follow water depth. The lagoon is 0-5.3 m deep by cell mean (3.3 m on
average), so the curves use the 0-6 m range (the dredged strait reaches 12 m
and takes each curve's 6 m value, so the lagoon model's groups treat it as deep water): water-column groups (phytoplankton, bacteria,
zooplankton, mysids, planktivorous fish, detritus) scale with depth because biomass per
area integrates the water column; soft-bottom benthos (chironomids, oligochaetes, deposit
feeders, meiobenthos) and the benthivorous demersal fish favour the deeper muddy basin;
Dreissena beds favour firmer shallower bottoms; grey heron wades the shores, the diving
birds and predatory fish prefer open deeper water, gulls have no preference. These are
stated assumptions, not fitted; `Curonian_Lagoon_habitat_depth.csv` holds them for editing.
`Advected?` = 1 in `Curonian_Lagoon_Scenario.csv` for Bacteria, both zooplankton groups,
phytoplankton and detritus (the Black Sea case advects plankton and POM); they follow the
advection field below instead of the predation-risk rule.

## Advection

EwE-F's advection is a direction per cell, not a current: once per month the spatial sweep moves
10 % of an advected group's cell biomass into the neighbour the code points at (about 1 km per
10 months), whatever the real speed. The field is therefore a bias towards the residual
circulation, not a representation of the exchange flows. It comes from the 29-box SHYFEM run
`run9314_minijafix` (1993-01-01 to 2014-04-30): for each box the volume-integrated velocity of a
divergence-free flow equals the sum over its interfaces of outward flux times interface position,
so the 21-year mean net flow across every interface (AQUABC link flows `FLOW_TS.txt`, river
boundaries included) gives a mean residual velocity per box (`Curonian_Lagoon_residual_flow.json`).
The Nemunas delta box runs west at 19 cm/s, the central and northern lagoon north at 2-10 cm/s,
the strait at 7-10 cm/s; the southern basin stays below 0.4 cm/s and carries no direction (the field was
built on the full domain, coastal Baltic included, then trimmed with the grid). Per cell: the box vector, blended
over a 5 x 5 window, then the best-aligned of the eight moves whose receiving cell is sea (within
60 degrees), else 0; the strait cells point along the thalweg to the mouth, where the outflow now
stops (the receiving Baltic cells are gone; EwE-F never advects onto land). 613 of the 1576 sea
cells carry a direction. Codes are defined by array displacement: the file's rows run north to
south, so EwE-F's code -90 (`lat-1`) is the move north here (`make_curonian_testcase.DISPLACEMENT`).

Advected groups also diffuse down their habitat-density gradient (`ecosim-f/src/ecospace.F90`,
2026-11-07), so a convergence of the direction field is not a permanent sink. Over ten years the
2nd-98th percentile spread of B / habitat stays below 2.5 for every advected group, the sea means
stay within 3 % of the non-spatial run, and the share of phytoplankton north of the lagoon rises
slightly (0.208 to 0.209): the effect is small, as the fixed 10 %/month rate implies.

Habitat maps are a relative carrying capacity (`docs/ecospace/HABITAT_CAPACITY.md`):
each cell integrates habitat-normalised densities and biomass diffuses towards
B ∝ habitat, so every cell holds its habitat value times the non-spatial run and the
depth pattern persists for the whole run (phytoplankton 6.9x across cells at month 24,
as at month 0).

`--ecospace-check 2` of the generator runs the spatial binary twice (about 8 s per two years). With the
advection field zeroed (the API's `advection=false`) it checks: no negative
values, land at -999, no mass-conservation warnings, the sea-mean biomass of every group
equal to the non-spatial run at the start and within 30 % of it over two years, B / habitat
equal to the non-spatial run in every cell (within 2e-3), and the phytoplankton contrast at
the end at least 5 with the deepest quartile over 1.5x the shallowest. With
`--uniform-habitat` every cell reproduces the non-spatial run exactly. With the shipped field it checks
no negative values, mass conservation, the advected groups' B / habitat spread below 20 and a net
movement of phytoplankton into the strait.

## Environmental driver maps (optional, 2026-11-10)

`Curonian_Lagoon_EnvMaps.h5` (+ `.json` provenance): monthly surface temperature and salinity, 1993-01 to 2024-12,
on this grid, from the E.U. Copernicus Marine Service Baltic Sea physics reanalysis BALTICSEA_MULTIYEAR_PHY_003_011
(CC-BY 4.0, https://doi.org/10.48670/moi-00013; built by `api/scripts/make_curonian_env_maps.py`). Drivers only:
it carries no response curves, so pointing a namelist at it (`EnvMaps_fname`) changes nothing until a run-specific
file adds a scenario's curves (`api/app/envmaps.py`). Map 1 is 1993-01 with `driver_first_month = 1`, i.e. run
month 1: a run that starts in January of year Y must start from map `12 (Y - 1993) + 1` (the cohort case,
starting 2003: map 121; the API slices the maps accordingly, `api/app/envmaps.py`), otherwise it sees 1993 conditions in
Y. The same file serves Curonian_Fish and
Curonian_Cohorts (same grid).
