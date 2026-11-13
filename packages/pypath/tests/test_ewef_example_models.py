"""EPath's EwE-F models shipped with PyPath as examples.

Three models from EPath (EwE-F) test cases: Central Baltic, Curonian Lagoon
and Lithuanian Coast. Each loads through ``read_ewef`` and balances as
EPath's Ecopath-F does; the reference is Ecopath-F's own solution of each
model (biomass and EE per group, ``data/ewef_example_models_reference.csv``).
"""

import csv
import shutil
import warnings
from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run
from pypath.examples import (
    example_ecosim_scenario,
    example_model,
    list_example_models,
    load_example_model,
)
from pypath.io.ewef import read_ewef

REFERENCE = Path(__file__).parent / "data" / "ewef_example_models_reference.csv"
MODELS = ["Central_Baltic", "Curonian_Lagoon", "Lithuanian_Coast"]


def _reference(model):
    with open(REFERENCE, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == model]
    biomass = np.array([float(r["biomass"]) for r in rows])
    ee = np.array([float(r["EE"]) for r in rows])
    return [r["group"] for r in rows], biomass, np.where(ee == -999, np.nan, ee)


def _balance(params):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


# --- EwE-F layouts these models use ---


def test_growth_rows_belong_to_stanza_groups_in_order():
    # EwE-F reads row i of the growth file for stanza group i and ignores the
    # row's name; these rows are named "Sprat", "Herring", "Cod", not after a
    # stage
    params = read_ewef(
        example_model("Central_Baltic").ecopath_path, compute_stanzas=False
    )
    stgroups = params.stanzas.stgroups.set_index("StanzaGroup")
    order = ["Sprat", "Herring", "Cod"]  # split groups 1, 2, 3
    assert stgroups.loc[order, "VBGF_Ksp"].tolist() == pytest.approx([0.51, 0.38, 0.23])
    assert stgroups.loc[order, "Wmat"].tolist() == pytest.approx([0.26, 0.5, 0.22])


def test_a_growth_row_without_a_name_keeps_its_place(tmp_path):
    # rows are read by position, so an unnamed row is still split group 1's
    folder = tmp_path / "ecopath"
    shutil.copytree(example_model("Central_Baltic").ecopath_path, folder)
    growth = folder / "Central_Baltic_GrowthParameters.csv"
    rows = list(csv.reader(open(growth, newline="")))
    header = next(i for i, r in enumerate(rows) if r and r[0] == "Groups")
    rows[header + 1][0] = ""  # the Sprat row
    with open(growth, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    stanzas = read_ewef(folder, compute_stanzas=False).stanzas
    set_of = dict(zip(stanzas.stindiv["Group"], stanzas.stindiv["StanzaGroup"]))
    k = stanzas.stgroups.set_index("StanzaGroup")["VBGF_Ksp"]
    assert k[set_of["Ad. Sprat"]] == pytest.approx(0.51)
    assert k[set_of["Ad. Herring"]] == pytest.approx(0.38)
    assert k[set_of["Ad. Cod"]] == pytest.approx(0.23)


def test_stanza_sets_take_their_names_from_the_growth_rows():
    # the stages ("Sprat 0 / Sprat 1", "Ad. Sprat") share no name; the
    # growth rows carry EwE's stanza set names
    params = read_ewef(
        example_model("Central_Baltic").ecopath_path, compute_stanzas=False
    )
    assert sorted(params.stanzas.stgroups["StanzaGroup"]) == ["Cod", "Herring", "Sprat"]


def test_migration_and_accumulation_rate_columns_are_read(tmp_path):
    source = example_model("Lithuanian_Coast").ecopath_path
    folder = tmp_path / "ecopath"
    shutil.copytree(source, folder)
    scenario = folder / "Lithuanian_Coast_Scenario.csv"
    rows = list(csv.reader(open(scenario, newline="")))
    header = next(i for i, r in enumerate(rows) if r and r[0] == "Groups")
    columns = rows[header]
    perch = next(i for i, r in enumerate(rows) if r and r[0] == "Pikeperch (adult)")
    rows[perch][columns.index("Emigration rate")] = "0.05"
    rows[perch][columns.index("Biomass Accumulation Rate")] = "0.1"
    with open(scenario, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    model = read_ewef(folder).model.set_index("Group")
    assert model.loc["European smelt (juvenile)", "Immigration"] == pytest.approx(4.32)
    assert model.loc["Phytoplankton", "Immigration"] == pytest.approx(2.77)
    assert model.loc["Pikeperch (adult)", "EmigRate"] == pytest.approx(0.05)
    assert model.loc["Pikeperch (adult)", "BioAccRate"] == pytest.approx(0.1)
    assert model.loc["Pikeperch (adult)", "Emigration"] == 0


@pytest.mark.parametrize("model", MODELS)
def test_the_balance_is_ecopath_fs(model):
    groups, biomass, ee = _reference(model)
    params = load_example_model(model)
    assert list(params.model["Group"][: len(groups)]) == groups
    balanced = _balance(params)
    n = len(groups)
    assert np.asarray(balanced.Biomass[:n]) == pytest.approx(biomass, rel=2e-3)
    known = np.isfinite(ee)
    assert np.asarray(balanced.EE[:n])[known] == pytest.approx(
        ee[known], rel=1e-4, abs=1e-3
    )


# --- the examples API ---


def test_the_examples_are_listed_with_their_sources():
    models = list_example_models()
    assert [m.name for m in models] == MODELS
    for m in models:
        assert m.title and m.description and m.source
        assert m.ecopath_path.is_dir() and m.ecosim_path.is_dir()
        assert (m.path / "README.md").is_file()


def test_names_are_matched_without_case():
    assert example_model("central_baltic").name == "Central_Baltic"


def test_an_unknown_example_is_refused_with_the_names():
    with pytest.raises(ValueError, match="Central_Baltic"):
        load_example_model("Black_Sea")


@pytest.mark.parametrize("model", MODELS)
def test_each_example_runs_with_its_ecosim_settings(model):
    years = range(1, 6)
    scenario = example_ecosim_scenario(model, years=years)
    out = rsim_run(scenario, method="RK4", years=years)
    biomass = np.asarray(out.out_Biomass)
    assert biomass.shape[0] == 12 * len(years) + 1  # with the starting state
    living = biomass[:, 1 : scenario.params.NUM_LIVING + 1]
    assert np.isfinite(living).all() and (living > 0).all()
