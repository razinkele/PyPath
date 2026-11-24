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


def test_the_curonian_diets_include_what_is_eaten_outside_the_lagoon():
    """EcopathGroup.ImpVar of the database, in the Import row (EPath, 2026-11-18)."""
    diet = load_example_model("Curonian_Lagoon").diet
    row = diet[diet.iloc[:, 0].astype(str).str.strip() == "Import"].iloc[0, 1:]
    imported = {k: float(v) for k, v in row.astype(float).items() if v > 0}
    assert imported == pytest.approx(
        {
            "Demersal fish": 0.345,
            "grey heron": 0.2,
            "larus": 0.2,
            "goosander": 0.2,
            "Great Crested Grebe": 0.2,
            "Cormorants": 0.2,
            "Predatory fish": 0.306,
        },
        abs=1e-6,
    )
    # every consumer's diet sums to 1 with it (Predatory fish to 0.9997, as
    # the database has it)
    sums = diet.iloc[:, 1:].astype(float).sum(axis=0)
    assert all(s == pytest.approx(1.0, abs=5e-4) for s in sums[list(imported)])


def test_the_curonian_baseline_stays_where_ecopath_put_it():
    """Without the diet import it drifted to 0.38-1.16 of the start in 50 years."""
    years = range(1, 51)
    scenario = example_ecosim_scenario("Curonian_Lagoon", years=years)
    biomass = np.asarray(rsim_run(scenario, method="RK4", years=years).out_Biomass)
    living = biomass[:, 1 : scenario.params.NUM_LIVING + 1]
    ratio = living[-1] / living[0]
    assert 0.999 < ratio.min() and ratio.max() < 1.001


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


# --- the Curonian Lagoon's own Ecospace set-up (EPath's grid and maps) ---------


def _curonian_ecospace(**kwargs):
    from pypath.examples import example_ecospace

    return example_ecospace("Curonian_Lagoon", **kwargs)


def test_the_curonian_example_has_its_ecospace_set_up():
    """The lagoon and the strait as 1,576 cells of 1 km, a row per group."""
    model = example_model("Curonian_Lagoon")
    assert model.ecospace_path.name == "ecospace"
    files = sorted(p.name for p in model.ecospace_path.iterdir())
    assert files == [
        "Curonian_Lagoon_SpatialDistribution",
        "Curonian_Lagoon_depth.csv",
        "Curonian_Lagoon_georef.json",
        "Curonian_Lagoon_grid.csv",
        "curonian29.grd",
    ]
    maps = model.ecospace_path / "Curonian_Lagoon_SpatialDistribution"
    assert sorted(p.name for p in maps.iterdir()) == sorted(
        f"Group_{k}.csv" for k in range(1, 20)
    )
    ecospace = _curonian_ecospace()
    grid = ecospace.grid
    assert grid.n_patches == 1576 and grid.geometry is None
    np.testing.assert_array_equal(grid.patch_areas, 1.0)  # km2
    # a row for each of the 19 groups and the fleet
    assert ecospace.habitat_capacity.shape == (20, 1576)
    assert np.all(ecospace.habitat_capacity[-1] == 1.0)
    assert ecospace.dispersal_rate[-1] == 0.0
    # EwE's speeds for a new Ecospace group over a 1 km cell: 300 km a year,
    # 10 for detritus
    np.testing.assert_allclose(ecospace.dispersal_rate[:18], 300 / np.pi, rtol=1e-6)
    assert ecospace.dispersal_rate[18] == pytest.approx(10 / np.pi, rel=1e-6)
    # the cell depths: the lagoon is shallow, the dredged strait 12 m
    depth = grid.cell_metadata["depth"].to_numpy()
    assert depth.min() > 0 and depth.max() == pytest.approx(12.0)
    assert depth.mean() == pytest.approx(3.29, abs=0.01)
    # the cells lie where the case's own georeference puts them, to 0.001
    # degrees (the builder spreads the columns by 1 / cos of the middle
    # latitude; the georeference rounds the steps to 0.009 and 0.0158)
    import json

    with open(model.ecospace_path / "Curonian_Lagoon_georef.json") as fh:
        georef = json.load(fh)
    rows = grid.cell_metadata["row"].to_numpy()
    cols = grid.cell_metadata["col"].to_numpy()
    lon = georef["lon_west"] + (cols + 0.5) * georef["dlon"]
    lat = georef["lat_north"] - (rows + 0.5) * georef["dlat"]
    assert np.abs(grid.patch_centroids[:, 0] - lon).max() < 1e-3
    assert np.abs(grid.patch_centroids[:, 1] - lat).max() < 1e-3


def test_the_habitat_maps_are_the_capacity_of_their_groups():
    """Group_k.csv is the k-th group's map; EwE divides it by its largest value."""
    import csv as _csv

    model = example_model("Curonian_Lagoon")
    ecospace = _curonian_ecospace()
    meta = ecospace.grid.cell_metadata
    rows, cols = meta["row"].to_numpy(), meta["col"].to_numpy()
    names = [str(g) for g in load_example_model("Curonian_Lagoon").model["Group"]]
    maps = model.ecospace_path / "Curonian_Lagoon_SpatialDistribution"
    for k in (1, 6, 10, 19):  # bacteria, chironomids, gulls, detritus
        with open(maps / f"Group_{k}.csv", newline="") as fh:
            table = [r for r in _csv.reader(fh)][2:]
        cells = np.array([[float(v) for v in r] for r in table])[rows, cols]
        np.testing.assert_allclose(
            ecospace.habitat_capacity[k - 1],
            cells / cells.max(),
            rtol=1e-6,
            err_msg=names[k - 1],
        )
    # gulls have no preference; the plankton follow the depth
    assert np.all(ecospace.habitat_capacity[names.index("larus")] == 1.0)
    plankton = ecospace.habitat_capacity[names.index("Fitoplankton")]
    assert np.corrcoef(plankton, np.minimum(meta["depth"], 6.0))[0, 1] > 0.98
    # the groups EwE-F advects are flagged, as EwE's reader keeps the flag
    assert [names[i] for i in np.flatnonzero(ecospace.advection_enabled)] == [
        "Bacteria",
        "Grazing zooplankton",
        "Carnivorous zooplankton",
        "Fitoplankton",
        "Detritus",
    ]
    assert not ecospace.gravity_strength.any()


def test_the_set_up_is_the_one_the_sqlite_copy_of_the_model_holds():
    """Two routes to one model: the EwE-F files, and the scenario written into
    the database from them. The database lists the groups in another order."""
    from pypath.io.ewemdb import read_ecospace, read_ewemdb

    db = Path(__file__).resolve().parents[3] / "Data" / "curonian_pelagic.sqlite"
    if not db.exists():
        pytest.skip("the Curonian pelagic database is not here")
    ecospace = _curonian_ecospace()
    stored = read_ecospace(str(db), n_groups=19, cell_placement="ground").ecospace
    case = [
        str(g).strip() for g in load_example_model("Curonian_Lagoon").model["Group"]
    ]
    database = [str(g).strip() for g in read_ewemdb(str(db)).model["Group"]]
    assert case[:19] != database[:19] and set(case[:19]) == set(database[:19])
    order = [database.index(name) for name in case[:19]]
    np.testing.assert_array_equal(
        ecospace.grid.patch_centroids, stored.grid.patch_centroids
    )
    assert (ecospace.grid.adjacency_matrix != stored.grid.adjacency_matrix).nnz == 0
    np.testing.assert_array_equal(
        ecospace.habitat_capacity[:19], stored.habitat_capacity[order]
    )
    np.testing.assert_array_equal(
        ecospace.dispersal_rate[:19], stored.dispersal_rate[order]
    )
    np.testing.assert_array_equal(
        ecospace.advection_enabled[:19], stored.advection_enabled[order]
    )


def test_the_curonian_example_runs_on_its_own_grid():
    from pypath.spatial import rsim_run_spatial

    scenario = example_ecosim_scenario("Curonian_Lagoon", years=range(1, 3))
    ecospace = _curonian_ecospace()
    assert ecospace.habitat_capacity.shape[0] == scenario.params.NUM_GROUPS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = rsim_run_spatial(scenario, ecospace=ecospace, years=range(1, 2))
    biomass = out.out_Biomass_spatial
    assert biomass.shape == (13, 21, 1576) and np.isfinite(biomass).all()
    # the start is each group's Ecopath biomass spread by its capacity
    start = biomass[0, 1:20].sum(axis=1)
    np.testing.assert_allclose(
        start, np.asarray(scenario.start_state.Biomass)[1:20], rtol=1e-6
    )


def test_the_placement_and_polygons_of_the_set_up():
    pytest.importorskip("geopandas")
    ground = _curonian_ecospace(geometry=True)
    ewe = _curonian_ecospace(cell_placement="ewe")
    assert len(ground.grid.geometry) == 1576
    west, south, east, north = ground.grid.geometry.total_bounds
    # the lagoon: its eastern shore is past 21.25 E
    assert 20.5 < west < 20.56 and 21.25 < east < 21.32
    assert 54.88 < south < 54.9 and 55.71 < north < 55.73
    assert ewe.grid.patch_centroids[:, 0].max() < 21.0
    np.testing.assert_array_equal(ground.habitat_capacity, ewe.habitat_capacity)


def test_examples_without_an_ecospace_set_up_say_which_has_one():
    from pypath.examples import example_ecospace

    for name in ("Central_Baltic", "Lithuanian_Coast"):
        assert example_model(name).ecospace_path is None
        with pytest.raises(ValueError, match="with one are: Curonian_Lagoon"):
            example_ecospace(name)


def test_the_ecospace_reader_refuses_files_that_do_not_fit(tmp_path):
    import shutil

    from pypath.io.ewef import read_ewef_ecospace

    model = example_model("Curonian_Lagoon")
    params = load_example_model("Curonian_Lagoon")
    with pytest.raises(FileNotFoundError, match="No EwE-F Ecospace folder"):
        read_ewef_ecospace(tmp_path / "missing", params)
    case = tmp_path / "ecospace"
    shutil.copytree(model.ecospace_path, case)
    maps = case / "Curonian_Lagoon_SpatialDistribution"
    # a group without its map
    (maps / "Group_19.csv").rename(tmp_path / "Group_19.csv")
    with pytest.raises(
        FileNotFoundError, match="no habitat map Group_19.csv.*Detritus"
    ):
        read_ewef_ecospace(case, params)
    (tmp_path / "Group_19.csv").rename(maps / "Group_19.csv")
    # a map of another shape
    text = (maps / "Group_1.csv").read_text().split("\n")
    (maps / "Group_1.csv").write_text("\n".join(text[:2] + text[3:]))
    with pytest.raises(ValueError, match="Group_1.csv: expected 94 rows"):
        read_ewef_ecospace(case, params)
    (maps / "Group_1.csv").write_text("\n".join(text))
    # a sea cell without a depth: the first sea cell of the grid file
    grid_rows = (case / "Curonian_Lagoon_grid.csv").read_text().split("\n")[2:]
    row = next(i for i, line in enumerate(grid_rows) if "1" in line.split(","))
    col = grid_rows[row].split(",").index("1")
    depth_lines = (case / "Curonian_Lagoon_depth.csv").read_text().split("\n")
    cells = depth_lines[2 + row].split(",")
    assert float(cells[col]) > 0
    cells[col] = "-999"
    depth_lines[2 + row] = ",".join(cells)
    (case / "Curonian_Lagoon_depth.csv").write_text("\n".join(depth_lines))
    with pytest.raises(ValueError, match="1 sea cells of the grid have no depth"):
        read_ewef_ecospace(case, params)
    # "nan" parses as a number: refused, in the depth map and in a habitat map
    shutil.rmtree(case)
    shutil.copytree(model.ecospace_path, case)
    for name in (
        "Curonian_Lagoon_depth.csv",
        "Curonian_Lagoon_SpatialDistribution/Group_3.csv",
    ):
        original = (case / name).read_text()
        lines = original.split("\n")
        cells = lines[2 + row].split(",")
        cells[col] = "nan"
        lines[2 + row] = ",".join(cells)
        (case / name).write_text("\n".join(lines))
        with pytest.raises(ValueError, match="1 cells are not finite numbers"):
            read_ewef_ecospace(case, params)
        (case / name).write_text(original)
    assert read_ewef_ecospace(case, params).grid.n_patches == 1576
    # the speeds can be given, one per living and detritus group
    with pytest.raises(ValueError, match="dispersal_kmyr needs 19 values"):
        read_ewef_ecospace(model.ecospace_path, params, dispersal_kmyr=[300.0] * 5)
    slow = read_ewef_ecospace(
        model.ecospace_path, params, dispersal_kmyr=[30.0] * 19, cell_length_km=2.0
    )
    np.testing.assert_allclose(slow.dispersal_rate, 30 / (np.pi * 2.0), rtol=1e-6)
    np.testing.assert_array_equal(slow.grid.patch_areas, 4.0)


@pytest.mark.slow
def test_the_phytoplankton_map_sets_the_level_of_the_spatial_run():
    """What the README says of the run: uniform maps keep the Ecopath totals,
    and the phytoplankton map alone brings its total to 0.59 of the start,
    the food web behind it. (That the map's mean capacity is 0.593 too is a
    coincidence: test_a_producer_cannot_sustain_itself_below_its_capacity_limit
    has the rule.)"""
    import dataclasses

    from pypath.spatial import rsim_run_spatial

    years = range(1, 3)
    scenario = example_ecosim_scenario("Curonian_Lagoon", years=years)
    shipped = _curonian_ecospace()
    names = [str(n).strip() for n in scenario.params.spname]
    phyto = names.index("Fitoplankton")
    assert shipped.habitat_capacity[phyto - 1].mean() == pytest.approx(0.593, abs=2e-3)

    def totals(capacity):
        ecospace = dataclasses.replace(shipped, habitat_capacity=capacity)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rsim_run_spatial(scenario, ecospace=ecospace, years=years)
        total = out.out_Biomass_spatial.sum(axis=2)
        return total[-1, 1:20] / total[0, 1:20]

    uniform = np.ones_like(shipped.habitat_capacity)
    np.testing.assert_allclose(totals(uniform), 1.0, atol=2e-3)
    only_phyto = uniform.copy()
    only_phyto[phyto - 1] = shipped.habitat_capacity[phyto - 1]
    ratio = totals(only_phyto)
    assert ratio[phyto - 1] == pytest.approx(0.593, abs=0.02)
    assert 0.5 < np.median(ratio) < 0.7


def test_a_producer_cannot_sustain_itself_below_its_capacity_limit():
    """EwE 6's Ecospace, which PyPath follows: a producer's production in a
    cell is its capacity there times what it would be without it
    (cSpaceSolver.vb 1008), and can rise to PBmaxs x P/B. At capacity c it
    settles at (c x PBmaxs - 1) / (PBmaxs - 1) of its level at capacity 1,
    and at nothing where c < 1 / PBmaxs. Three cells, no movement."""
    import copy

    from pypath.io.ewemdb import _build_fallback_grid
    from pypath.spatial import EcospaceParams, rsim_run_spatial

    years = range(1, 6)
    scenario = example_ecosim_scenario("Curonian_Lagoon", years=years)
    params = scenario.params
    names = [str(n).strip() for n in params.spname]
    phyto = names.index("Fitoplankton")
    links = [
        k
        for k in range(len(params.PreyFrom))
        if params.PreyTo[k] == phyto and params.PreyFrom[k] == 0
    ]
    assert [float(params.VV[k]) for k in links] == [2.0]  # PBmaxs 2, EwE's default
    grid = _build_fallback_grid(1, 3, 10.0)
    n = params.NUM_GROUPS
    capacity = np.ones((n, 3))
    capacity[phyto - 1] = (1.0, 0.6, 0.2)  # mean 0.6

    def phytoplankton(pb_max):
        run = copy.deepcopy(scenario)
        for k in links:  # Rpath's producer link: VV = PBmaxs / (PBmaxs - 1)
            run.params.VV[k] = pb_max / (pb_max - 1.0)
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n, 3)),
            habitat_capacity=capacity,
            dispersal_rate=np.zeros(n),
            advection_enabled=np.zeros(n, dtype=bool),
            gravity_strength=np.zeros(n),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rsim_run_spatial(run, ecospace=ecospace, years=years)
        cells = out.out_Biomass_spatial[:, phyto]
        return cells[-1].sum() / cells[0].sum(), cells[-1] / cells[-1].sum()

    def rule(pb_max):
        c = capacity[phyto - 1]
        return np.maximum(c * pb_max - 1.0, 0.0).mean() / (pb_max - 1.0) / c.mean()

    total, share = phytoplankton(2.0)
    assert rule(2.0) == pytest.approx(0.667, abs=1e-3)  # not the mean, 0.6
    assert total == pytest.approx(rule(2.0), abs=0.02)
    assert share[2] < 1e-6 and share[0] > 2 * share[1]  # gone below 0.5
    # a higher ceiling lets it live in the poorer cells
    total, share = phytoplankton(5.0)
    assert rule(5.0) == pytest.approx(0.833, abs=1e-3)
    assert total == pytest.approx(rule(5.0), abs=0.03)
    # 0.2 is the limit at 5: the last cell keeps a little, as its grazers
    # have thinned out and its losses are below the Ecopath rate
    assert share[1] > 0.25 and 0.0 < share[2] < 0.1
