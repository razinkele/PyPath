# Lithuanian_Coast test case (Ecosim)

Ecosim inputs of the Lithuanian_Coast model (44 groups, one stanza set; see
`ecopath-f/testcases/Lithuanian_Coast/README.md` for the model, its sources and the changes).

| File | Content |
|---|---|
| `Lithuanian_Coast.h5` | Ecopath-F result of the Ecopath test case (the Ecosim input, with the stanza data) |
| `Lithuanian_Coast_Scenario.csv` | the group settings of the file's Ecosim scenario "New Ecosim scenario": EwE's defaults for every group |
| `Lithuanian_Coast_vul.csv` | vulnerabilities: the scenario stores none (0 given), so 2 (EwE's default) on every diet link, -999 elsewhere |
| `Lithuanian_Coast_F.csv` | fishing forcing type 5 (multiplier of the Ecopath F) = 1.0 for 600 months (50 years) |
| `Lithuanian_Coast_Fleets.csv` | each group's Ecopath catch by fleet (t/km²/yr); the fleet "Fishery" lands 13 groups and discards wintering piscivorous birds and wintering diving ducks (by-catch); the API's fleet effort scenarios use these shares |
| `Lithuanian_Coast_filenames.nml` | namelist template; the API sets `tf` per run; no nutrient forcing |

The scenario's nutrient settings (base free proportion 0.9999, maximum
relative P/B 1.5) and foraging-time floor (0.1) are EwE-F's
defaults too. The file has no time series, no applied forcing shapes and no Ecospace scenario, so the case is a
baseline at constant fishing, without spatial inputs. At constant fishing the model stays where Ecopath put it: in a
30-year run every group ends within 0.1 % of its start
(`python -m scripts.make_lithuanian_coast_testcase --ecosim-years 30`).
