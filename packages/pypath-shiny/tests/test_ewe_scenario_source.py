"""The Ecosim page can run an imported EwE database's own Ecosim scenario.

Before, the app read only the Ecopath part of an EwE database: the scenario's
vulnerabilities, feeding-time, switching and forcing settings and its
time-series datasets never reached Ecosim, and the fast-group rule could not
be chosen.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

DB_PATH = Path(__file__).parents[3] / "Data" / "LT2022_0.5ST_final7.eweaccdb"


def _source(path, groups, scenarios=None):
    return {
        "path": str(path),
        "name": "model.eweaccdb",
        "scenario": 1,
        "groups": list(groups),
        "scenarios": {1: "1: Scenario"} if scenarios is None else scenarios,
        "datasets": {},
    }


class TestSourceMatchesModel:
    """The database scenario is offered only for the model it belongs to."""

    def test_same_groups_in_any_order(
        self, tmp_path, rpath_params, balanced_rpath_model
    ):
        from pypath_shiny.pages.ewe_scenario import (
            ewe_source_matches,
            model_group_names,
        )

        db = tmp_path / "model.eweaccdb"
        db.write_bytes(b"x")
        names = model_group_names(rpath_params)
        source = _source(db, reversed(names))
        assert ewe_source_matches(source, rpath_params)
        assert ewe_source_matches(source, balanced_rpath_model)

    def test_other_model_missing_file_or_no_scenarios(self, tmp_path, rpath_params):
        from pypath_shiny.pages.ewe_scenario import (
            ewe_source_matches,
            model_group_names,
        )

        db = tmp_path / "model.eweaccdb"
        db.write_bytes(b"x")
        names = model_group_names(rpath_params)
        assert not ewe_source_matches(_source(db, names[:-1]), rpath_params)
        assert not ewe_source_matches(
            _source(tmp_path / "gone.eweaccdb", names), rpath_params
        )
        assert not ewe_source_matches(_source(db, names, scenarios={}), rpath_params)
        assert not ewe_source_matches(None, rpath_params)
        assert not ewe_source_matches(_source(db, names), None)


class TestPreyForcingKeepsScenarioShapes:
    """The Biomass Forcing card used to reset ForcedPrey to 1 on every run,
    wiping the forcing shapes an EwE scenario brings."""

    def test_resets_to_the_scenario_own_forcing(
        self, rpath_params, balanced_rpath_model
    ):
        from pypath.core.ecosim import rsim_scenario

        from pypath_shiny.pages.ecosim import apply_prey_forcing

        sc = rsim_scenario(balanced_rpath_model, rpath_params, years=range(1, 4))
        names = [str(n) for n in sc.params.spname]
        shape = np.linspace(0.5, 1.5, sc.forcing.ForcedPrey.shape[0])
        sc.forcing.ForcedPrey[:, 2] = shape  # a forcing shape from the database

        assert apply_prey_forcing(sc, names[1], 2.0) is True
        np.testing.assert_allclose(sc.forcing.ForcedPrey[:, 2], shape)
        assert (sc.forcing.ForcedPrey[:, 1] == 2.0).all()

        # forcing a group that has its own shape scales that shape
        assert apply_prey_forcing(sc, names[2], 2.0) is True
        np.testing.assert_allclose(sc.forcing.ForcedPrey[:, 2], 2.0 * shape)
        assert (sc.forcing.ForcedPrey[:, 1] == 1.0).all()

        assert apply_prey_forcing(sc, names[2], 1.0) is False
        np.testing.assert_allclose(sc.forcing.ForcedPrey[:, 2], shape)


class TestEcosimControls:
    def test_sidebar_has_the_fast_group_rule_and_scenario_source(self):
        from pypath_shiny.pages.ecosim import ecosim_ui

        html = str(ecosim_ui())
        assert 'id="fast_group_rule"' in html
        assert 'value="ewe"' in html and 'value="rpath"' in html
        assert 'id="ewe_scenario_source_ui"' in html

    def test_servers_take_the_imported_database(self):
        from pypath_shiny.pages import data_import, ecosim

        assert "ewe_source" in inspect.signature(data_import.import_server).parameters
        assert "ewe_source" in inspect.signature(ecosim.ecosim_server).parameters

    def test_app_shares_one_imported_database(self):
        import pypath_shiny.app as app_module

        src = inspect.getsource(app_module)
        # Data Import and Home set it, Ecosim reads it
        assert src.count("ewe_source=ewe_source") == 3

    @pytest.mark.parametrize("rule, steps", [("ewe", 1), ("rpath", 4)])
    def test_model_scenario_uses_the_chosen_rule(
        self, rpath_params, balanced_rpath_model, rule, steps
    ):
        from pypath_shiny.pages.ecosim import scenario_from_balanced_model

        model = balanced_rpath_model
        sc = scenario_from_balanced_model(
            model,
            list(model.Group),
            list(model.type),
            years=range(1, 4),
            vulnerability=2.0,
            fast_group_rule=rule,
        )
        assert sc.params.RK4_STEPS == steps
        assert sc.params.RK4_FAST_EQUILIBRIUM == (rule == "ewe")
        assert sc.fishing.ForcedEffort.shape[0] == 36

    def test_database_scenarios_are_built_by_the_importer(self):
        from pypath_shiny.pages import ecosim

        assert "build_ewe_scenario(" in inspect.getsource(ecosim.ecosim_server)


@pytest.mark.integration
class TestLT2022Scenario:
    @pytest.fixture(scope="class")
    def db(self):
        if not DB_PATH.exists():
            pytest.skip("LT2022 database not found")
        from pypath.io.ewemdb import check_ewemdb_support

        if not any(check_ewemdb_support().values()):
            pytest.skip("No EwE database driver")
        return str(DB_PATH)

    def test_lists_scenarios_and_datasets(self, db):
        from pypath_shiny.pages.ewe_scenario import ewe_ecosim_choices

        choices = ewe_ecosim_choices(db)
        assert 16 in choices["scenarios"]
        assert 35 in choices["datasets"]

    def test_builds_the_scenario_with_the_rule_and_dataset(self, db):
        from pypath_shiny.pages.ewe_scenario import build_ewe_scenario

        sc = build_ewe_scenario({"path": db}, "16", "35", "rpath")
        assert sc.params.RK4_STEPS == 4
        assert not sc.params.RK4_FAST_EQUILIBRIUM
        names = [str(n).strip() for n in sc.params.spname]
        goby = names.index("Round goby")
        # dataset 35 forces Round goby from 0.03 up to 7.8
        assert np.nanmax(sc.forcing.ForcedBio[:, goby]) > 1.0


class TestExampleModelSource:
    """A shipped example model offers its own EwE-F Ecosim settings."""

    def test_the_record_matches_only_its_model(self, rpath_params):
        from pypath.examples import load_example_model

        from pypath_shiny.pages.ewe_scenario import (
            ewe_source_matches,
            example_source_record,
        )

        params = load_example_model("Central_Baltic")
        record = example_source_record("Central_Baltic", params)
        assert record["kind"] == "ewef_example"
        assert ewe_source_matches(record, params)
        assert not ewe_source_matches(record, rpath_params)

    def test_the_scenario_has_the_examples_settings(self):
        from pypath.examples import example_ecosim_scenario, load_example_model

        from pypath_shiny.pages.ewe_scenario import (
            build_ewe_scenario,
            example_source_record,
        )

        record = example_source_record(
            "Lithuanian_Coast", load_example_model("Lithuanian_Coast")
        )
        built = build_ewe_scenario(record, record["scenario"], "", "ewe")
        direct = example_ecosim_scenario("Lithuanian_Coast", fast_group_rule="ewe")
        assert built.fishing.ForcedEffort.shape[0] == 12 * 10  # EwE-F tf = 10
        assert np.array_equal(built.params.VV, direct.params.VV)
        assert built.params.RK4_FAST_EQUILIBRIUM

    def test_the_ecosim_page_starts_from_the_examples_settings(self, tmp_path):
        from pypath.examples import load_example_model

        from pypath_shiny.pages.ewe_scenario import (
            example_source_record,
            scenario_source_choices,
        )

        example = example_source_record(
            "Curonian_Lagoon", load_example_model("Curonian_Lagoon")
        )
        choices, selected, offers_datasets = scenario_source_choices(example)
        assert selected == "ewe" and not offers_datasets
        assert "Curonian Lagoon" in choices["ewe"]

        db = tmp_path / "model.eweaccdb"
        db.write_bytes(b"x")
        choices, selected, offers_datasets = scenario_source_choices(_source(db, ["A"]))
        assert selected == "model" and offers_datasets
        assert "model.eweaccdb" in choices["ewe"]
