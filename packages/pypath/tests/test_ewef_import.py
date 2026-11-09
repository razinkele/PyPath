"""Import EwE-F models (the Fortran Ecopath with Ecosim) from their input files.

The reference is EwE-F's own Ecopath result for the Black Sea test case
(``tests/data/ewef/Black_Sea/ecosim/Black_Sea.h5``, ``ep_data``), pinned
below so the tests need no HDF5 reader. EwE-F stores single-precision
values, hence the 1e-5 tolerances.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecopath import rpath

CASE = Path(__file__).parent / "data" / "ewef" / "Black_Sea"
ECOPATH_NML = CASE / "ecopath" / "Black_Sea_filenames.nml"
ECOSIM_NML = CASE / "ecosim" / "Black_Sea_filenames.nml"
RESULTS_H5 = CASE / "ecosim" / "Black_Sea.h5"

# EwE-F Ecopath result: (B, P/B, Q/B, EE); -999 where EwE-F has none
EWEF_ECOPATH = {
    "Dolphins": (
        0.011864500120282173,
        0.43612998723983765,
        5.7728800773620605,
        0.19325673580169678,
    ),
    "Bonito": (
        0.04289683699607849,
        0.5038865804672241,
        5.289608001708984,
        0.6950969099998474,
    ),
    "Bluefish": (
        0.007222434040158987,
        0.5049999952316284,
        4.359222888946533,
        0.9800642728805542,
    ),
    "A. Mackerel": (
        0.03540651127696037,
        0.5121948719024658,
        5.385334014892578,
        0.5064571499824524,
    ),
    "Whiting": (
        0.006057451944798231,
        0.7054960131645203,
        2.1307690143585205,
        0.9955958724021912,
    ),
    "Turbot": (
        0.003856424242258072,
        0.6137825846672058,
        1.6367119550704956,
        0.7551165819168091,
    ),
    "R. Mullet": (
        0.009806456044316292,
        0.6530594825744629,
        2.1110329627990723,
        0.8282186985015869,
    ),
    "Dogfish": (
        0.0005884383572265506,
        0.5515018105506897,
        1.9035799503326416,
        0.43139997124671936,
    ),
    "H. Mackerel": (
        0.06538602709770203,
        2.739331007003784,
        9.164259910583496,
        0.3818385601043701,
    ),
    "Shad": (
        0.00044562405673787,
        2.2836530208587646,
        9.601008415222168,
        0.9826554656028748,
    ),
    "Sprat": (
        0.2575877904891968,
        2.887845993041992,
        13.01056957244873,
        0.3873234689235687,
    ),
    "Anchovy 0-0+": (
        0.01286410167813301,
        23.03274917602539,
        69.63777160644531,
        0.0001111254605348222,
    ),
    "Anchovy 1-1+": (
        0.25958749651908875,
        2.977022886276245,
        9.780116081237793,
        0.44656920433044434,
    ),
    "Aurelia": (0.03712702915072441, 12.392550468444824, 34.51361083984375, 0.0),
    "Beroe": (1.040000029206567e-06, 9.635578155517578, 27.55529022216797, 0.0),
    "Mnemiopsis": (
        9.540000064589549e-06,
        8.63093090057373,
        34.51361083984375,
        0.3132382929325104,
    ),
    "Pleurobrachia": (0.023776929825544357, 7.300000190734863, 37.18273162841797, 0.0),
    "Noctiluca": (0.08479569107294083, 8.76994800567627, 38.39453125, 0.0),
    "Zooplankton": (
        0.390474408864975,
        43.810001373291016,
        268.70709228515625,
        0.5098464488983154,
    ),
    "Phytoplankton": (0.33497390151023865, 291.0, 0.0, 0.9887863993644714),
    "POM": (20.0, -999.0, -999.0, 0.53681480884552),
    "Detritus": (80.0, -999.0, -999.0, 0.000875478086527437),
}
GROUPS = list(EWEF_ECOPATH)

# PyPath routes detritus as Rpath does: a pool's own fate share counts as its
# input, and the surplus passed between pools comes from living inflows only.
# EwE 6 (cEcoPathModel.CalcFateOfDetritus) and EwE-F (calculateDetritusFate)
# skip a pool's own fate and pass surpluses in group order, each including
# what earlier pools passed on. The Black Sea pools have both (POM -> POM 0.5,
# Detritus -> Detritus 0.7, Detritus -> POM 0.3), so their EE differ.
DETRITUS_ROUTING = pytest.mark.xfail(
    strict=True, reason="Rpath vs EwE detritus-to-detritus routing"
)


@pytest.fixture(scope="module")
def params():
    from pypath.io.ewef import read_ewef

    return read_ewef(ECOPATH_NML)


@pytest.fixture(scope="module")
def balanced(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


def _row(params, group):
    model = params.model.set_index("Group")
    return model.loc[group]


# --- Ecopath inputs ---------------------------------------------------------


def test_groups_types_and_fleet(params):
    model = params.model
    assert list(model["Group"])[:22] == GROUPS
    # EwE-F organism type 2/1/0 (consumer/producer/detritus) -> 0/1/2
    assert list(model["Type"])[:22] == [0] * 19 + [1, 2, 2]
    # landings make one fleet
    assert list(model["Type"])[22:] == [3]
    fleet = model["Group"].iloc[22]
    assert _row(params, "Dolphins")[fleet] == pytest.approx(0.001)
    assert _row(params, "Anchovy 1-1+")[fleet] == pytest.approx(0.0151)


def test_basic_inputs_and_missing_values(params):
    dolphins = _row(params, "Dolphins")
    assert dolphins["Biomass"] == pytest.approx(0.0118645)
    assert np.isnan(dolphins["EE"])  # -999 is EwE-F's missing marker
    assert dolphins["Unassim"] == pytest.approx(0.2)
    bonito = _row(params, "Bonito")
    assert np.isnan(bonito["Biomass"])
    assert bonito["EE"] == pytest.approx(0.6950969)
    assert _row(params, "Phytoplankton")["QB"] == 0.0
    pom = _row(params, "POM")
    assert pom["Biomass"] == 20.0 and pom["DetInput"] == 0.0


def test_diet_with_import(params):
    diet = params.diet.set_index("Group")
    assert diet.loc["Bonito", "Dolphins"] == pytest.approx(0.05)
    assert diet.loc["Mnemiopsis", "Beroe"] == pytest.approx(0.9)
    assert diet.loc["POM", "Beroe"] == pytest.approx(0.1)
    assert diet.loc["Import", "Bonito"] == pytest.approx(0.2)
    assert diet["Phytoplankton"].fillna(0).sum() == 0  # producers eat nothing


def test_detritus_fate(params):
    assert _row(params, "Dolphins")["Detritus"] == 1.0
    assert _row(params, "Anchovy 0-0+")["POM"] == 0.5
    assert _row(params, "Detritus")["Detritus"] == pytest.approx(0.7)
    assert _row(params, "Detritus")["POM"] == pytest.approx(0.3)


def test_stanzas(params):
    st = params.stanzas
    assert st.n_stanza_groups == 1
    g = st.stgroups.iloc[0]
    assert g["VBGF_Ksp"] == pytest.approx(0.3)
    assert g["Wmat"] == pytest.approx(0.177)
    assert g["RecPower"] == 1.0 and g["BAB"] == 0.0
    ind = st.stindiv.sort_values("StanzaNum")
    assert list(ind["Group"]) == ["Anchovy 0-0+", "Anchovy 1-1+"]
    assert list(ind["First"]) == [0, 3]  # months, as in EwE-F
    assert list(ind["Leading"].astype(bool)) == [False, True]
    assert list(ind["Z"]) == pytest.approx([23.03275, 2.977023])


# --- Ecopath balance against EwE-F --------------------------------------------


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(g, marks=DETRITUS_ROUTING) if g in ("POM", "Detritus") else g
        for g in GROUPS
    ],
)
def test_biomass_and_ee_match_ewef(balanced, group):
    b, _, _, ee = EWEF_ECOPATH[group]
    i = list(balanced.Group).index(group)
    assert balanced.Biomass[i] == pytest.approx(b, rel=1e-5)
    assert balanced.EE[i] == pytest.approx(ee, rel=1e-5, abs=1e-7)


def test_juvenile_anchovy_comes_from_the_stanza_derivation(params, balanced):
    """EwE-F leaves B and Q/B of the non-leading stanza to Ecopath (-999)."""
    i = list(balanced.Group).index("Anchovy 0-0+")
    b, _, qb, _ = EWEF_ECOPATH["Anchovy 0-0+"]
    assert balanced.Biomass[i] == pytest.approx(b, rel=1e-5)
    # PyPath ends the plus group where 99.999 % of the cumulative biomass is
    # reached (Rpath, 59 months here); EwE-F runs it to 135 months
    assert balanced.QB[i] == pytest.approx(qb, rel=1e-4)


def test_stanzas_can_be_left_to_the_caller():
    from pypath.io.ewef import read_ewef

    raw = read_ewef(ECOPATH_NML, compute_stanzas=False)
    assert np.isnan(_row(raw, "Anchovy 0-0+")["Biomass"])


def test_directory_and_missing_file(tmp_path):
    from pypath.io.ewef import read_ewef

    assert list(read_ewef(CASE / "ecopath").model["Group"])[:22] == GROUPS
    with pytest.raises(FileNotFoundError):
        read_ewef(tmp_path)


# --- Ecosim -----------------------------------------------------------------


@pytest.fixture(scope="module")
def ecosim(params):
    from pypath.io.ewef import read_ewef_ecosim

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        overrides = read_ewef_ecosim(ECOSIM_NML, params)
    return overrides, [str(w.message) for w in caught]


def test_vulnerabilities_land_on_the_named_links(ecosim):
    overrides, _ = ecosim
    vv = overrides["vv_overrides"]
    idx = {g: i for i, g in enumerate(GROUPS)}
    assert vv[(idx["Bonito"], idx["Dolphins"])] == pytest.approx(2.541453)
    assert vv[(idx["Sprat"], idx["Whiting"])] == pytest.approx(14731.61)
    assert vv[(idx["Detritus"], idx["R. Mullet"])] == pytest.approx(5.960643)
    assert vv[(idx["POM"], idx["Aurelia"])] == pytest.approx(4.718817)


def test_group_settings(ecosim):
    overrides, _ = ecosim
    idx = {g: i for i, g in enumerate(GROUPS)}
    assert overrides["mo_pred"][idx["Dolphins"]] == 1.0
    assert overrides["ftime_max"][idx["Zooplankton"]] == 2.0
    assert overrides["ftime_adjust"][idx["Sprat"]] == 0.0
    assert overrides["pb_maxs"][idx["Phytoplankton"]] == 2.0
    assert overrides["switch_power"][idx["Bonito"]] == 0.0
    assert idx["POM"] not in overrides["mo_pred"]  # -999: not set


def test_what_pypath_cannot_apply_is_reported(ecosim):
    overrides, messages = ecosim
    text = " ".join(messages)
    # Beroe and Mnemiopsis eat POM but have no vulnerability (EwE-F drops the
    # link; PyPath, like EwE6, keeps it at the default)
    assert "Beroe" in text and "Mnemiopsis" in text and "POM" in text
    assert "QBmax/QBo" in text  # handling time is not applied
    assert "nutrient" in text.lower()
    idx = {g: i for i, g in enumerate(GROUPS)}
    assert (idx["POM"], idx["Beroe"]) not in overrides["vv_overrides"]


def test_scenario_runs(params):
    from pypath.core.ecosim import rsim_run
    from pypath.io.ewef import ecosim_scenario_from_ewef

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = ecosim_scenario_from_ewef(ECOPATH_NML, ECOSIM_NML, years=range(1, 11))
        out = rsim_run(sc, method="AB")
    names = [str(n) for n in sc.params.spname]
    assert sc.params.MoPred[names.index("Dolphins")] == 1.0
    assert out.out_Biomass.shape[0] == 121
    assert np.all(np.isfinite(out.out_Biomass))


def test_results_reader():
    pytest.importorskip("h5py")
    from pypath.io.ewef import read_ewef_results

    res = read_ewef_results(RESULTS_H5, groups=GROUPS)
    assert list(res.index) == GROUPS
    for g in ("Dolphins", "Anchovy 0-0+", "Detritus"):
        b, pb, qb, ee = EWEF_ECOPATH[g]
        assert res.loc[g, "Biomass"] == pytest.approx(b)
        assert res.loc[g, "EE"] == pytest.approx(ee)
    assert np.isnan(res.loc["POM", "PB"])
