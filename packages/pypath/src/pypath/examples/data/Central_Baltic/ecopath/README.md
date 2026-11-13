# Central_Baltic test case (Ecopath)

Food-web model of the Baltic Proper (central Baltic Sea, 240 000 km²) in 1974, 22 groups, wet weight in
t/km², **modified for a stationary baseline** from the EwE 6 database *Central Baltic Sea* (EcoBase model 608,
imported by EwE in 2019; not in this repository), the model of

- Tomczak, M. T., Niiranen, S., Hjerne, O., & Blenckner, T. (2012). Ecosystem flow dynamics in the Baltic
  Proper—Using a multi-trophic dataset as a basis for food–web modelling. *Ecological Modelling*, 230, 123–147.
  https://doi.org/10.1016/j.ecolmodel.2011.12.014

Three stanza sets (Sprat: Sprat 0 / Sprat 1, Ad. Sprat (K 0.51, Wmat/Winf 0.26); Herring: Herring 0 / Herring 1 / Herring 2, Ad. Herring (K 0.38, Wmat/Winf 0.5); Cod: Cod larvae, Cod 0 / Cod 1, Cod 2 / Cod 3, Ad. Cod (K 0.23, Wmat/Winf 0.22)) and three trawl fleets (COD-Trawl, HER-Trawl, SPR-Trawl). Its values are the file's, except the
changes below; results of this case are not those of the published model.

Files, all written by `api/scripts/make_central_baltic_testcase.py` (the API's EwE 6 importer plus the changes
below):

| File | Content |
|---|---|
| `Central_Baltic_Scenario.csv` | basic input incl. the stanza columns; each set's leading stage carries B and Q/B, Ecopath-F derives the other stages |
| `Central_Baltic_GrowthParameters.csv` | K, recruitment power, BA/B (all 0, see below), Wmat/Winf of the three sets |
| `Central_Baltic_DC.csv`, `Central_Baltic_DetFate.csv` | diet (Import row last) and detritus fate (water column and sediment) |
| `Central_Baltic_filenames.nml` | namelist (paths relative to `ecopath-f/`) |

## Changes to the file

**Biomass accumulation moved into P/B** (asked for on 13 November 2026: biomass accumulation set to zero, for a
stationary Ecosim baseline). The file gives each stanza set a biomass accumulation rate BA/B, the stocks' growth
from 1974 on. Ecopath-F, like EwE, computes a set's survivorship from each stage's P/B plus the set's BA/B, and
each stage's BA as BA/B times its biomass. Setting BA/B to 0 alone therefore changes the age structure: the young
cod stages lose biomass and their predators eat more than they produce (cod larvae EE 5.0, cod 0-1 1.7), and no
cod BA/B up to 0.3 balances. So each set's BA/B is set to 0 and added to the P/B of each of its stages. The
survivorship, and with it every stage's biomass and Q/B, stays the file's; the production that accumulated becomes
other mortality, which lowers the EE and raises P/Q (adult cod's P/Q 0.37 is high for cod).

| Stage | Set BA/B | P/B in the file | P/B here | B, file | B, here | EE, file | EE, here | P/Q, file | P/Q, here |
|---|---|---|---|---|---|---|---|---|---|
| Sprat 0 / Sprat 1 | 0.017 | 1.98 | 1.997 | 0.9224 | 0.9224 | 0.607 | 0.602 | 0.180 | 0.182 |
| Ad. Sprat | 0.017 | 0.6 | 0.617 | 4.213 | 4.213 | 0.764 | 0.743 | 0.126 | 0.130 |
| Herring 0 / Herring 1 / Herring 2 | -0.009 | 1.9455 | 1.9365 | 3.974 | 3.974 | 0.297 | 0.298 | 0.401 | 0.399 |
| Ad. Herring | -0.009 | 0.416 | 0.407 | 5.619 | 5.619 | 0.666 | 0.681 | 0.208 | 0.203 |
| Cod larvae | 0.363282 | 13 | 13.3633 | 0.001606 | 0.001606 | 0.959 | 0.933 | 0.174 | 0.179 |
| Cod 0 / Cod 1 | 0.363282 | 0.9 | 1.26328 | 0.2701 | 0.2701 | 0.857 | 0.611 | 0.080 | 0.113 |
| Cod 2 / Cod 3 | 0.363282 | 0.62 | 0.983282 | 1.019 | 1.019 | 0.437 | 0.276 | 0.116 | 0.184 |
| Ad. Cod | 0.363282 | 0.92 | 1.28328 | 0.49 | 0.49 | 0.829 | 0.594 | 0.263 | 0.367 |

(B, EE and P/Q "file" are Ecopath-F's solution with the file's biomass accumulation.)

**One name completed.** The herring juvenile stage's name is cut off in the file ("Herring 0 / Herring 1 /
Herring"); we completed it to "Herring 0 / Herring 1 / Herring 2", as the stage covers ages 0–29 months.

**Two diet columns rescaled to 1.** In the file they sum slightly above 1:
Sprat 0 / Sprat 1 1.00040, Cod 0 / Cod 1 1.00005.

Result: Ecopath-F balances the model with its checks on. Every EE is at or below 1 (highest:
other phytoplankton 0.992, Cod larvae 0.933, Detritus-sediment 0.877). No respiration is negative.

Importer notes, on the file as stored: detritus fates from EcopathDietComp.DetritusFate; groups reordered for EwE-F: consumers, then producers, then detritus, stanza stages together in age order; 3 multi-stanza set(s) exported as EwE-F split groups; each set's B comes from the stage its LeadingLifeStage names and its Q/B from the one its LeadingCB names (EwE's leading stages, the oldest when none is named), Ecopath-F derives the other stages.
