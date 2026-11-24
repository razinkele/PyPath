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


# --- the Curonian Lagoon grid for Ecospace ----------------------------------


def test_the_curonian_example_has_the_29_boxes_as_its_grid():
    from pypath.examples import example_ecospace_grid
    from pypath.spatial.zone_fields import patch_zones

    assert example_model("Curonian_Lagoon").grid_path.name == "curonian29.grd"
    grid = example_ecospace_grid("curonian_lagoon")
    assert grid.n_patches == 29 and grid.geometry is None
    assert sorted(patch_zones(grid).tolist()) == list(range(1, 30))
    # lagoon, strait and the sea boxes off the coast
    assert grid.patch_areas.sum() == pytest.approx(2296.0, rel=0.01)
    assert (grid.adjacency_matrix != grid.adjacency_matrix.T).nnz == 0
    # the depths the README gives, from the mesh itself
    depth = dict(zip(patch_zones(grid).tolist(), grid.cell_metadata["depth"]))
    assert sorted(z for z, d in depth.items() if d > 10) == [1, 4, 7, 10, 12, 13, 16]
    assert depth[11] == pytest.approx(7.1, abs=0.05)
    shallow = [d for z, d in depth.items() if d <= 10 and z != 11]
    assert len(shallow) == 21
    assert (min(shallow), max(shallow)) == pytest.approx((1.4, 4.5), abs=0.05)
    cells = example_ecospace_grid("Curonian_Lagoon", patches="element")
    assert cells.n_patches == 2027
    assert cells.patch_areas.sum() == pytest.approx(grid.patch_areas.sum())


def test_the_grid_file_holds_nodes_and_elements_only():
    """Nothing of the project it comes from: no comments, paths or names."""
    lines = example_model("Curonian_Lagoon").grid_path.read_text().split("\n")
    kinds = {line.split()[0] for line in lines if line.strip()}
    assert kinds == {"1", "2"}
    assert all(
        part.replace(".", "", 1).replace("-", "", 1).isdigit()
        for line in lines
        for part in line.split()
    )


def test_examples_without_a_grid_say_which_have_one():
    from pypath.examples import example_ecospace_grid

    for name in ("Central_Baltic", "Lithuanian_Coast"):
        assert example_model(name).grid_path is None
        with pytest.raises(ValueError, match="one are: Curonian_Lagoon"):
            example_ecospace_grid(name)


def test_the_curonian_example_runs_on_its_grid():
    from pypath.examples import example_ecospace_grid
    from pypath.spatial import EcospaceParams, rsim_run_spatial

    scenario = example_ecosim_scenario("Curonian_Lagoon", years=range(1, 3))
    grid = example_ecospace_grid("Curonian_Lagoon")
    n = scenario.params.NUM_GROUPS
    ecospace = EcospaceParams(
        grid=grid,
        habitat_preference=np.ones((n, 29)),
        habitat_capacity=np.ones((n, 29)),
        dispersal_rate=np.full(n, 1.0),
        advection_enabled=np.zeros(n, dtype=bool),
        gravity_strength=np.zeros(n),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 3))
    biomass = out.out_Biomass_spatial
    assert np.isfinite(biomass).all() and biomass.shape[2] == 29
    # on a uniform map every box holds the same biomass per km2
    n_bio = scenario.params.NUM_LIVING + scenario.params.NUM_DEAD
    density = biomass[:, 1 : n_bio + 1] / grid.patch_areas
    np.testing.assert_allclose(
        density, np.repeat(density[:, :, :1], 29, axis=2), rtol=1e-6
    )
