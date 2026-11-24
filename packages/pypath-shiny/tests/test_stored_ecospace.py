"""Stored spatial configurations and the one-step example open
(services.stored_ecospace, pages.home.publish_example_model)."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest
from pypath.core.ecopath import rpath
from pypath.io.ewemdb import read_ewemdb
from shiny import reactive

from pypath_shiny.pages import ecopath, ecosim, ecospace
from pypath_shiny.pages.ecospace import (
    WIZARD_ORIGIN,
    _duration_text,
    page_ecospace_arrays,
    run_ecospace_params,
    wizard_params_on_grid,
)
from pypath_shiny.pages.ewe_scenario import (
    ewe_source_matches,
    ewe_source_record,
    example_source_record,
)
from pypath_shiny.pages.home import open_example_model, publish_example_model
from pypath_shiny.services import stored_ecospace
from pypath_shiny.services.stored_ecospace import (
    ewe_ecospace_choices,
    load_example_grid,
    load_stored_ecospace,
)

pytestmark = pytest.mark.filterwarnings("ignore")

DATA = Path(__file__).resolve().parents[3] / "Data"
PELAGIC = DATA / "curonian_pelagic.sqlite"  # SQLite: needs no database driver
LT2022 = DATA / "LT2022_0.5ST_final7.eweaccdb"


@pytest.fixture(scope="module")
def pelagic():
    """The Curonian pelagic database as the Data Import page records it."""
    if not PELAGIC.exists():
        pytest.skip("the Curonian pelagic database is not here")
    params = read_ewemdb(str(PELAGIC))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)
    source = ewe_source_record(str(PELAGIC), PELAGIC.name, 1, params)
    return source, params, model


# --- what a database offers ---------------------------------------------------


def test_the_ecospace_scenarios_of_a_database_are_listed_with_their_grid(pelagic):
    source, _, _ = pelagic
    assert ewe_ecospace_choices(str(PELAGIC)) == source["ecospace_scenarios"]
    assert list(source["ecospace_scenarios"]) == [1]
    label = source["ecospace_scenarios"][1]
    assert label.startswith("1: ") and "94 x 51 cells of 1 km" in label


def test_a_database_without_an_ecospace_scenario_offers_none(tmp_path):
    assert ewe_ecospace_choices(str(tmp_path / "missing.sqlite")) == {}
    if LT2022.exists():
        from pypath.io.ewemdb import check_ewemdb_support

        if any(check_ewemdb_support().values()):
            # the table is there, with no scenario in it
            assert ewe_ecospace_choices(str(LT2022)) == {}


# --- loading a scenario -------------------------------------------------------


def test_a_stored_scenario_is_read_ready_for_a_run(pelagic):
    source, _, model = pelagic
    stored = load_stored_ecospace(source, "1", model)
    n_all = model.NUM_LIVING + model.NUM_DEAD + model.NUM_GEARS
    params = stored.params
    assert (stored.n_patches, stored.n_groups) == (1576, n_all)
    assert params.habitat_capacity.shape == (n_all, 1576)
    # the fleet's row is added as a group that lives everywhere and stays put
    assert np.all(params.habitat_capacity[-1] == 1.0)
    assert params.dispersal_rate[-1] == 0.0 and params.dispersal_rate[0] > 0.0
    # the groups' own capacity maps vary
    varying = [not np.allclose(row, row[0]) for row in params.habitat_capacity[:-1]]
    assert sum(varying) >= 15
    # polygons for the maps, one square per water cell, in the lagoon
    grid = params.grid
    assert grid.geometry is not None and len(grid.geometry) == 1576
    west, south, east, north = grid.geometry.total_bounds
    assert 20.4 < west < east < 21.4 and 54.8 < south < north < 55.8
    assert stored.origin == (
        "the Ecospace scenario 'EwE-F Curonian_Lagoon' of curonian_pelagic.sqlite"
    )
    assert "1,576 water cells of 1 km" in stored.summary
    assert stored.not_applied == ["the fleets' habitat rules"]


def test_the_corner_moves_the_grid_and_nothing_else(pelagic):
    source, _, model = pelagic
    north = load_stored_ecospace(source, 1, model)
    south = load_stored_ecospace(source, 1, model, grid_corner="south_west")
    np.testing.assert_array_equal(
        north.params.habitat_capacity, south.params.habitat_capacity
    )
    shift = south.params.grid.patch_centroids - north.params.grid.patch_centroids
    np.testing.assert_allclose(shift[:, 0], 0.0, atol=1e-12)
    assert np.allclose(shift[:, 1], shift[0, 1]) and shift[0, 1] > 0.5
    with pytest.raises(ValueError, match="grid_corner"):
        load_stored_ecospace(source, 1, model, grid_corner="middle")


def test_a_scenario_is_not_loaded_without_a_balanced_model_or_when_too_large(
    pelagic, monkeypatch
):
    source, params, model = pelagic
    with pytest.raises(ValueError, match="Balance the model first"):
        load_stored_ecospace(source, 1, params)
    with pytest.raises(ValueError, match="could not be read.*no Ecospace scenario 99"):
        load_stored_ecospace(source, 99, model)
    monkeypatch.setattr(stored_ecospace, "MAX_STORED_PATCHES", 1000)
    with pytest.raises(ValueError, match="1,576 water cells; up to 1,000"):
        load_stored_ecospace(source, 1, model)


def test_the_grid_size_is_checked_before_the_scenario_is_read(pelagic, monkeypatch):
    """An uploaded database must not make the reader build a grid of any size."""
    source, _, model = pelagic
    assert stored_ecospace._scenario_cells(str(PELAGIC), 1) == 94 * 51

    def never(*args, **kwargs):
        raise AssertionError("the scenario was read")

    monkeypatch.setattr("pypath.io.ewemdb.read_ecospace", never)
    monkeypatch.setattr(stored_ecospace, "MAX_STORED_CELLS", 94 * 51 - 1)
    with pytest.raises(ValueError, match=r"4,794 cells \(rows x columns\)"):
        load_stored_ecospace(source, 1, model)
    # a scenario that is not there is refused the same way, before a read
    with pytest.raises(ValueError, match="no Ecospace scenario 99"):
        load_stored_ecospace(source, 99, model)


def test_a_loaded_scenario_runs_in_place_of_the_page_settings(pelagic):
    source, _, model = pelagic
    stored = load_stored_ecospace(source, 1, model)
    grid, n_all = stored.params.grid, stored.n_groups
    page = page_ecospace_arrays(np.ones(grid.n_patches), n_all, 5.0, True, 0.5)
    used, note = run_ecospace_params(
        grid, n_all, page, wizard_params=stored.params, origin=stored.origin
    )
    assert used is stored.params
    assert note == f"Running with {stored.origin}."
    # the row gate: a scenario with another number of groups falls back, and says so
    used, note = run_ecospace_params(
        grid, n_all + 1, page, wizard_params=stored.params, origin=stored.origin
    )
    assert used is not stored.params
    assert note.startswith("The Ecospace scenario 'EwE-F Curonian_Lagoon'")
    assert "so this page's settings are used. Load it again" in note
    # a new grid on the page ends it
    assert wizard_params_on_grid(stored.params, object(), n_all) is None
    # the wizard's wording is as it was
    _, note = run_ecospace_params(grid, n_all, page, wizard_params=stored.params)
    assert note == "Running with the Ecospace Wizard's configuration."
    assert WIZARD_ORIGIN == "the Ecospace Wizard's configuration"


def test_a_loaded_scenario_runs(pelagic):
    """A year on the 1,576 cells, with the database's own Ecosim scenario."""
    from pypath.io.ewemdb import ecosim_scenario_from_ewemdb
    from pypath.spatial import rsim_run_spatial

    source, _, model = pelagic
    stored = load_stored_ecospace(source, 1, model)
    scenario = ecosim_scenario_from_ewemdb(str(PELAGIC), scenario=1)
    assert scenario.params.NUM_GROUPS == stored.n_groups
    out = rsim_run_spatial(scenario, ecospace=stored.params, years=range(1, 2))
    assert np.isfinite(out.out_Biomass_spatial).all()
    assert out.out_Biomass_spatial.shape[2] == 1576


# --- the examples -------------------------------------------------------------


def test_only_the_curonian_example_offers_a_grid():
    records = {}
    for name in ("Central_Baltic", "Curonian_Lagoon", "Lithuanian_Coast"):
        params, record = open_example_model(name)
        records[name] = record
        assert record == example_source_record(name, params)
    assert [n for n, r in records.items() if r["ecospace_grid"]] == ["Curonian_Lagoon"]
    grid = load_example_grid(records["Curonian_Lagoon"])
    assert grid.n_patches == 29 and len(grid.geometry) == 29
    with pytest.raises(ValueError, match="no spatial grid"):
        load_example_grid(records["Lithuanian_Coast"])
    with pytest.raises(ValueError, match="no example model is open"):
        load_example_grid({"path": "x", "name": "a database"})
    with pytest.raises(ValueError, match="no example model is open"):
        load_example_grid(None)


def test_an_example_is_published_for_a_look_or_for_one_step():
    with reactive.isolate():
        # the Home page's route: the unbalanced model, for the Ecopath page
        model_data, source, results = (
            reactive.Value("old model"),
            reactive.Value(None),
            reactive.Value({"old": 1}),
        )
        record = publish_example_model(
            "Lithuanian_Coast",
            model_data=model_data,
            ewe_source=source,
            sim_results=results,
        )
        assert results.get() is None and source.get() is record
        published = model_data.get()
        assert len(published.model) == 45
        assert ewe_source_matches(record, published)

        # the Ecosim page's route: handed over to be balanced, in one step
        model_data, source, request = (
            reactive.Value("old model"),
            reactive.Value(None),
            reactive.Value(None),
        )
        record = publish_example_model(
            "Curonian_Lagoon",
            model_data=model_data,
            ewe_source=source,
            balance_request=request,
        )
        assert model_data.get() == "old model"  # the Ecopath page publishes it
        assert len(request.get().model) == 20 and source.get() is record
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert ewe_source_matches(record, rpath(request.get()))

        # nothing is published when the example cannot be opened
        with pytest.raises(ValueError, match="No example model"):
            publish_example_model("Atlantis", model_data=model_data, ewe_source=source)
        assert source.get() is record and model_data.get() == "old model"


def test_the_pages_are_wired_for_the_new_routes():
    assert "balance_request" in inspect.signature(ecopath.ecopath_server).parameters
    assert "balance_request" in inspect.signature(ecosim.ecosim_server).parameters
    assert "ewe_source" in inspect.signature(ecospace.ecospace_server).parameters
    import pypath_shiny.app as app_module

    assert inspect.getsource(app_module).count("balance_request=balance_request") == 2
    # the Ecosim page's picker has ids of its own (the Home page has one too)
    page = str(ecosim.ecosim_ui())
    assert 'id="ecosim_example_model"' in page
    assert 'id="btn_ecosim_open_example"' in page
    assert 'id="example_model"' not in page
    assert 'id="stored_ecospace_ui"' in str(ecospace.ecospace_ui())


def test_run_times_are_worded_by_size():
    assert _duration_text(0.2) == "1 seconds"
    assert _duration_text(90) == "90 seconds"
    assert _duration_text(600) == "10 minutes"
    assert _duration_text(3 * 3600) == "3.0 hours"
