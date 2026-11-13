# Lithuanian_Coast test case (Ecopath)

Food-web model of the exposed sandy coast of Lithuania (south-eastern Baltic Sea, the 0-20 m zone from Nida to
Būtingė), 44 groups, wet weight in g/m² (= t/km²): the EwE 6 model *coast* of 2017 (file
`Goby.ewemdb`, not in this repository), the model of

- Razinkovas-Baziukas, A., Morkūnė, R., Bacevičius, E., & Gasiūnaitė, Z. R. (2017). Trophic network model of
  exposed sandy coast: Linking continental and marine water ecosystems. *Estuarine, Coastal and Shelf Science*,
  195, 110–123. https://doi.org/10.1016/j.ecss.2017.04.030

with round goby (*Neogobius melanostomus*) added as a three-stage stanza set (Round goby juvenile (under 60 mm), Round goby sub-adult (60-120 mm), Round goby adult (over 120 mm);
K 0.3, Wmat/Winf 0.09) on the way to the model of

- Skabeikis, A. (2019). *Role of the round goby (Neogobius melanostomus) in the food web of the south-eastern
  Baltic Sea coastal waters*. Doctoral dissertation, Klaipėda University.
  https://gs.elaba.lt/object/elaba:39072875/39072875.pdf

The file is a working version: it lacks the thesis's cod and shorthorn sculpin groups, and several of its biomasses
differ from the thesis's (Annex 2). Its values are kept, except the changes below. Review and approval:
`docs/project/COAST_MODEL_REVIEW_2026-11-12.md` (12 November 2026).

Files, all written by `api/scripts/make_lithuanian_coast_testcase.py` (the API's EwE 6 importer plus the changes
below):

| File | Content |
|---|---|
| `Lithuanian_Coast_Scenario.csv` | basic input incl. the stanza columns and immigration; the adult round goby stage carries B and Q/B, Ecopath-F derives the other two stages |
| `Lithuanian_Coast_GrowthParameters.csv` | K, recruitment power, BA/B, Wmat/Winf of the round goby set |
| `Lithuanian_Coast_DC.csv`, `Lithuanian_Coast_DetFate.csv` | diet (Import row last) and detritus fate |
| `Lithuanian_Coast_filenames.nml` | namelist (paths relative to `ecopath-f/`) |

## Changes to the file

The round goby columns of the file were incomplete (no adult Q/B, placeholder P/B 2.0, empty diets), so Ecopath
could not solve it. Everything below except the detritus import comes from Skabeikis (2019):

- Round goby adult (over 120 mm), Q/B: -999 -> 2.44. Skabeikis 2019, Annex 2 row 23; Annex 8 gives 17.547 / 7.127 = 2.46
- Round goby juvenile (under 60 mm), P/B: 2 -> 1. Skabeikis 2019, Annex 2 row 22; the file's 2.0 is a placeholder
- Round goby sub-adult (60-120 mm), P/B: 2 -> 0.55. Skabeikis 2019, Annex 2 row 23; the file's 2.0 is a placeholder
- Round goby adult (over 120 mm), P/B: 2 -> 0.55. Skabeikis 2019, Annex 2 row 23; the file's 2.0 is a placeholder
- Three-spined stickleback, immigration: 0.017407 -> 0.093. Skabeikis 2019, Annex 2 row 15
- European smelt (juvenile), immigration: 1.5419 -> 4.32. Skabeikis 2019, Annex 2 row 13
- Detritus, detritus import: 204 -> 210. Ours, not from the sources: the smallest round value that keeps the detritus EE below 1 (0.999) once round goby eats benthos
- Round goby diets, all three stages: from none to the shares below. Skabeikis 2019, Annex 8

Round goby diet shares (Annex 8, the thesis model's consumption of each prey): juvenile stage mesozooplankton
0.616, zoobenthos filtrators 0.259, omnivores under 10 mm 0.125; sub-adult and adult stages zoobenthos filtrators
0.454, polychaetes 0.177, omnivores over 20 mm 0.158, omnivores under 10 mm 0.105, juvenile smelt 0.089, omnivores
10-20 mm 0.008, round goby juveniles 0.007, stickleback 0.002, common and sand gobies 0.0002, nectobenthos 0.00001.

Result: Ecopath-F balances the model with its checks on. Every EE is at or below 1: greater sandeel (adult) and
common and sand gobies take their biomass from EE 1, as the file asks, and 13 more groups and
detritus end between 0.998 and 1. No respiration is negative. Round goby comes out at
1.506, 0.653, 1.793 g/m² (juvenile, sub-adult, adult); the thesis
estimates 3.25 for juveniles and 7.13 for sub-adults and adults together.

## Names

Ecosim-F writes group names in 35-character fields and the binaries' comma-separated result files carry the names
unquoted, so the long names were shortened and the comma removed (the importer had already dropped the `<` and `>` of the size classes):

| In the file | In this case |
|---|---|
| Lesser sand-eel and Greater sand-eel (juvenile) | Sandeels (juvenile) |
| Greater sand-eel (adult) | Greater sandeel (adult) |
| Common and Sand gobies (juvenile/adult) | Common and sand gobies |
| European flounder (juvenile/sub-adult) | Flounder (juvenile/sub-adult) |
| European flounder (adult) | Flounder (adult) |
| Three-spined stickleback (juvenile/adult) | Three-spined stickleback |
| Round goby juvenile (<60 mm) | Round goby juvenile (under 60 mm) |
| Roud goby sub-adult (60-120 mm) | Round goby sub-adult (60-120 mm) |
| Round goby adult (>120 mm) | Round goby adult (over 120 mm) |
| Macrozoobenthos filtrators | Zoobenthos filtrators |
| Macrozoobenthos omnivoroes (<9,9mm) | Zoobenthos omnivores (under 10 mm) |
| Macrozoobenthos omnivoroes (10-20mm) | Zoobenthos omnivores (10-20 mm) |
| Macrozoobenthos omnivoroes (>20mm) | Zoobenthos omnivores (over 20 mm) |
| European perch (juvenile) | Perch (juvenile) |
| European perch (sub/adult) | Perch (sub-adult/adult) |
| Pike-perch (adult) | Pikeperch (adult) |
| Pike-perch (juvenile/sub-adult) | Pikeperch (juvenile/sub-adult) |

Importer notes, on the file as stored (before the changes above): detritus fates from EcopathDietComp.DetritusFate; groups reordered for EwE-F: consumers, then producers, then detritus, stanza stages together in age order; 1 multi-stanza set(s) exported as EwE-F split groups; each set's B comes from the stage its LeadingLifeStage names and its Q/B from the one its LeadingCB names (EwE's leading stages, the oldest when none is named), Ecopath-F derives the other stages; migration as EwE 6.7 (Ecopath balances it, Ecosim keeps immigration as an inflow and emigration as a loss): Pikeperch (juvenile/sub-adult) (immigration 0.005655), Baltic herring (juvenile) (immigration 0.22605), Baltic herring (adult) (immigration 0.62223), Baltic sprat (juvenile/adult) (immigration 0.236), Eelpout (juvenile) (immigration 0.006522), Eelpout (adult) (immigration 0.003002), Flounder (juvenile/sub-adult) (immigration 0.00209), Vimba (juvenile) (immigration 0.048426), Vimba (adult) (immigration 0.0510842), Three-spined stickleback (immigration 0.017407), Ruffe (adult) (immigration 0.0015809), Perch (juvenile) (immigration 0.0070173), Turbot (sub-adult/adult) (immigration 0.04156), European smelt (juvenile) (immigration 1.5419), European smelt (adult) (immigration 0.413791), Mesozooplankton (immigration 0.21), Phytoplankton (immigration 2.77).
