# Curonian Lagoon test case (Ecopath)

Ecopath mass balance of the Curonian Lagoon pelagic food web: 19 functional groups
(17 consumers, phytoplankton, detritus), one fleet, EwE 6 model *curonian pelagic*
(2005–2007), Marine Research Institute, Klaipėda University. It is the model of the
PyPath Curonian Lagoon case study (https://github.com/razinkele/PyPath); the EwE 6
database it was generated from is kept beside the Ecosim inputs,
`ecosim-f/testcases/Curonian_Lagoon/curonian_pelagic.eweaccdb`.

`Curonian_Lagoon_Scenario.csv`, `_DC.csv` and `_DetFate.csv` were written by
`api/scripts/make_curonian_testcase.py` with the web API's EwE 6 importer
(`api/app/ewe_model.py`, PyPath's `read_ewemdb` + mass balance), so they are what
the API produces for any uploaded EwE database:

- group names exactly as the database spells them ("Fitoplankton", "oligochets",
  "larus");
- groups ordered consumers, then producers, then detritus, because Ecopath-F pairs
  diet-matrix column *j* with group *j* (the database lists phytoplankton first);
- EE left for Ecopath to estimate wherever the database does; Meiobenthos has B
  and EE both given, one parameter too many, so its B is passed as unknown and
  Ecopath-F estimates it from the EE of 0.95 (0.536 instead of the entered 0.1685;
  with the entered B the group's EE exceeds 1). This is exactly PyPath's balance of
  the database with that B left blank (`api/tests/test_ewe_model_import.py`);
- detritus fates from the database, no multi-stanza groups.

Regenerate with

    cd api && python -m scripts.make_curonian_testcase ../ecosim-f/testcases/Curonian_Lagoon/curonian_pelagic.eweaccdb --ecosim-years 50

No peer-reviewed reference is attached to the database; add it to
`api/ewef_citations.py` (`CURONIAN`) when known.
