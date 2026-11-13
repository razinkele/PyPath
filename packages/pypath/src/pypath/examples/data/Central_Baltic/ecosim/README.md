# Central_Baltic test case (Ecosim)

Ecosim inputs of the Central_Baltic model (22 groups, three stanza sets; modified from Tomczak et al. 2012
for a stationary baseline, see `ecopath-f/testcases/Central_Baltic/README.md` for the model, its source and the
changes).

| File | Content |
|---|---|
| `Central_Baltic.h5` | Ecopath-F result of the Ecopath test case (the Ecosim input, with the stanza data) |
| `Central_Baltic_Scenario.csv` | group settings: EwE's defaults for every group (the file stores no Ecosim scenario) |
| `Central_Baltic_vul.csv` | vulnerabilities: 2 (EwE's default) on every diet link, -999 elsewhere |
| `Central_Baltic_F.csv` | fishing forcing type 5 (multiplier of the Ecopath F) = 1.0 for 600 months (50 years) |
| `Central_Baltic_Fleets.csv` | each group's Ecopath catch by fleet (t/km²/yr); the three trawl fleets land 6 groups (Sprat 0 / Sprat 1, Ad. Sprat, Herring 0 / Herring 1 / Herring 2, Ad. Herring, Cod 2 / Cod 3, Ad. Cod); the API's fleet effort scenarios use these shares |
| `Central_Baltic_filenames.nml` | namelist template; the API sets `tf` per run; no nutrient forcing |

The nutrient settings (base free proportion 0.9999, maximum relative P/B 1.5) and the foraging-time floor (0.1)
are EwE-F's defaults. The file has no time series, no forcing shapes and no Ecospace scenario, so the case is a
baseline at constant fishing, without spatial inputs. Measured at constant fishing (the generator runs them):
10 years: every group within 0.86 % of its start at the end (largest Ad. Herring, 0.9914), within 1.2 % throughout; 30 years: every group within 0.94 % of its start at the end (largest Seals, 0.9906), within 1.2 % throughout; 50 years: every group within 1.1 % of its start at the end (largest Seals, 0.9894), within 1.2 % throughout. With the file's biomass accumulation kept instead, adult cod grows 1.6-fold and adult sprat falls to 0.86
in 30 years.
