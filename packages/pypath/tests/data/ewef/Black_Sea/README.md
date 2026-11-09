# EwE-F Black Sea test case

Input files of the Black Sea test case distributed with EwE-F (the Fortran
implementation of Ecopath with Ecosim, IMS-METU and OGS), copied unchanged
from `ecopath-f/testcases/Black_Sea` and `ecosim-f/testcases/Black_Sea`:

| Folder | Files | Read by |
|---|---|---|
| `ecopath/` | `Black_Sea_filenames.nml`, `_Scenario.csv`, `_DC.csv`, `_DetFate.csv`, `_GrowthParameters.csv` | `pypath.io.ewef.read_ewef` |
| `ecosim/` | `Black_Sea_filenames.nml`, `_Scenario.csv`, `_vul.csv`, `_F.csv`, `_NutForce.csv` | `pypath.io.ewef.read_ewef_ecosim` |
| `ecosim/` | `Black_Sea.h5`: EwE-F's Ecopath result for these inputs | the reference values in `tests/test_ewef_import.py` |

The same model as an EwE 6 database, built from these files, is
`Data/BlackSea_EwEF_testcase.eweaccdb` at the repository root. Its forcing is the
test case's synthetic forcing; it is not the published Black Sea model.

The model is the Black Sea food web of the EwE-F authors (22 groups, including
*Mnemiopsis*, *Beroe* and two anchovy stanzas). If you use it, cite:

- Akoğlu, E., Salihoğlu, B., Libralato, S., Oguz, T., & Solidoro, C. (2014). An
  indicator-based evaluation of Black Sea food web dynamics during 1960–2000.
  *Journal of Marine Systems*, 134, 113–125. https://doi.org/10.1016/j.jmarsys.2014.02.010
- Akoğlu, E., Libralato, S., Salihoğlu, B., Oguz, T., & Solidoro, C. (2015). EwE-F 1.0:
  an implementation of Ecopath with Ecosim in Fortran 95/2003 for coupling and
  integration with other models. *Geoscientific Model Development*, 8(8), 2687–2699.
  https://doi.org/10.5194/gmd-8-2687-2015

EwE-F, including these test-case files, is distributed under the GNU General
Public License version 2 (see the header of the `.nml` files). They are
included here unchanged as test data under that licence; the rest of PyPath is
MIT-licensed.
