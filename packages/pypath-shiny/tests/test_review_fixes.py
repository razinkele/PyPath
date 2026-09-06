"""Regression tests for the 2026-09-05 app review fixes.

Covers the pure pieces extracted while fixing the review findings: demo-page
code builders, matplotlib style/palette choices on the Results page, and the
Analysis page's use of the core analysis API.
"""

import inspect

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pypath_shiny.pages import analysis, results  # noqa: E402
from pypath_shiny.pages.diet_rewiring_demo import build_diet_rewiring_code  # noqa: E402
from pypath_shiny.pages.forcing_demo import build_forcing_code  # noqa: E402
from pypath_shiny.pages.optimization_demo import build_optimization_code  # noqa: E402


class TestDemoCodeBuilders:
    @pytest.mark.parametrize(
        "pattern", ["seasonal", "trend", "pulse", "step", "custom"]
    )
    def test_forcing_code_defines_values_for_every_pattern(self, pattern):
        code = build_forcing_code("biomass", "replace", 3, pattern)
        assert "values = " in code
        assert "from pypath.core.ecosim import rsim_run" in code
        assert "group_idx=3" in code
        compile(code, "<forcing_example>", "exec")  # must be valid Python

    def test_forcing_code_custom_variable(self):
        code = build_forcing_code(
            "fishing_mortality", "add", 1, "trend", {"start": 1, "end": 2}
        )
        assert "variable='fishing_mortality'" in code
        assert "np.linspace(1, 2, len(years))" in code

    def test_diet_code(self):
        code = build_diet_rewiring_code(2.5, 12, 0.01)
        assert "switching_power=2.5" in code
        assert "update_interval=12" in code
        compile(code, "<diet_example>", "exec")

    def test_optimization_code(self):
        code = build_optimization_code("PB", "rmse", 50, "ei")
        assert "'param': 'PB'" in code
        assert "n_iterations=50" in code
        assert "params.PB[group_idx]" in code
        compile(code, "<opt_example>", "exec")


class TestResultsPlotChoices:
    def _choices(self, html, input_id):
        """Extract option values for a select input from rendered HTML."""
        start = html.index(f'id="{input_id}"')
        end = html.index("</select>", start)
        chunk = html[start:end]
        return [seg.split('"')[0] for seg in chunk.split('<option value="')[1:]]

    def test_plot_styles_are_valid_matplotlib_styles(self):
        html = str(results.results_ui())
        styles = self._choices(html, "plot_style")
        assert styles, "plot_style select not found"
        for style in styles:
            if style != "default":
                assert style in plt.style.available, style

    def test_color_palettes_are_valid_colormaps(self):
        html = str(results.results_ui())
        palettes = self._choices(html, "color_palette")
        assert palettes, "color_palette select not found"
        for name in palettes:
            assert name in plt.colormaps, name

    def test_apply_plot_style_ignores_unknown_style(self):
        with plt.rc_context():
            results._apply_plot_style(plt, "no-such-style")  # must not raise
            results._apply_plot_style(plt, "default")

    def test_plot_style_does_not_leak_out_of_rc_context(self):
        before = dict(plt.rcParams)
        with plt.rc_context():
            results._apply_plot_style(plt, "dark_background")
            assert plt.rcParams["figure.facecolor"] != before["figure.facecolor"]
        assert plt.rcParams["figure.facecolor"] == before["figure.facecolor"]


class TestAnalysisPageCoreApiUsage:
    """The Analysis page must call the core API with its real signatures."""

    def test_source_uses_correct_keywords_and_keys(self):
        src = inspect.getsource(analysis)
        assert "plot_trophic_spectrum(model, by=" in src
        assert "metric=metric" not in src
        assert "plot_mti_heatmap(mti, model)" not in src
        assert 'check.get("is_balanced"' in src
        assert 'check.get("balanced"' not in src

    def test_network_index_attribute_names_exist(self):
        from pypath.core.analysis import NetworkIndices

        src = inspect.getsource(analysis)
        fields = set(NetworkIndices.__dataclass_fields__)
        for name in (
            "total_throughput",
            "total_biomass",
            "system_omnivory",
            "finn_cycling_index",
            "transfer_efficiency",
            "n_links",
        ):
            assert f'("{name}",' in src
            assert name in fields
        for stale in (
            "total_production",
            "system_omnivory_index",
            "num_links",
            "ascendency",
        ):
            assert f'("{stale}",' not in src


class TestBalancedModelAnalysis:
    """End-to-end: the core analysis functions the page relies on work on a real model."""

    def test_mti_keystoneness_align_with_groups(self, balanced_rpath_model):
        from pypath.core.analysis import keystoneness_index, mixed_trophic_impacts

        m = balanced_rpath_model
        n = m.NUM_LIVING + m.NUM_DEAD
        mti = mixed_trophic_impacts(m)
        ks = keystoneness_index(m, mti)
        assert mti.shape == (n, n)
        assert len(ks) == n
        assert len(list(m.Group[:n])) == n
        assert np.all(np.isfinite(ks))


class TestRecreateParamsKeepsFullDiet:
    """_recreate_params_from_model copies detritus, import and fate columns."""

    def test_detritus_and_import_rows_copied(self, balanced_rpath_model):
        from pypath_shiny.pages.ecopath import _recreate_params_from_model

        m = balanced_rpath_model
        p = _recreate_params_from_model(m)
        diet = p.diet.set_index("Group")
        n_prey = m.NUM_LIVING + m.NUM_DEAD
        for i in range(n_prey):
            prey = str(m.Group[i])
            for j in range(m.NUM_LIVING):
                pred = str(m.Group[j])
                assert diet.loc[prey, pred] == pytest.approx(m.DC[i, j])
        for j in range(m.NUM_LIVING):
            assert diet.loc["Import", str(m.Group[j])] == pytest.approx(m.DC[n_prey, j])

    def test_detritus_fate_copied(self, balanced_rpath_model):
        from pypath_shiny.pages.ecopath import _recreate_params_from_model

        m = balanced_rpath_model
        p = _recreate_params_from_model(m)
        det = str(np.asarray(m.Group)[np.asarray(m.type) == 2][0])
        non_fleet = np.asarray(m.type) < 3
        assert np.allclose(
            p.model.loc[p.model["Type"] < 3, det].to_numpy(dtype=float),
            m.DetFate[non_fleet, 0],
        )


class TestIbmGroupTable:
    def test_from_rpath_params(self, rpath_params):
        from pypath_shiny.pages.ibm import _model_group_table

        rows = _model_group_table(rpath_params)
        assert [r[1] for r in rows] == ["Fish", "Plankton", "Detritus"]
        assert rows[0][0] == 0 and rows[0][2] == 0

    def test_from_balanced_rpath(self, balanced_rpath_model):
        from pypath_shiny.pages.ibm import _model_group_table

        rows = _model_group_table(balanced_rpath_model)
        assert [r[1] for r in rows] == ["Fish", "Plankton", "Detritus"]
        assert [r[2] for r in rows] == [0, 1, 2]

    def test_none(self):
        from pypath_shiny.pages.ibm import _model_group_table

        assert _model_group_table(None) == []


class TestWizardSharedGroupNames:
    def _shared(self, value):
        from types import SimpleNamespace

        from shiny import reactive

        return SimpleNamespace(params=reactive.Value(value))

    def test_none_falls_back(self):
        from shiny import reactive

        from pypath_shiny.pages.ecospace_wizard import shared_group_names

        with reactive.isolate():
            assert shared_group_names(self._shared(None)) == [
                f"Group {i + 1}" for i in range(5)
            ]
            assert shared_group_names(None, default_n=2) == ["Group 1", "Group 2"]

    def test_rpath_params_and_rpath(self, rpath_params, balanced_rpath_model):
        from shiny import reactive

        from pypath_shiny.pages.ecospace_wizard import shared_group_names

        with reactive.isolate():
            assert shared_group_names(self._shared(rpath_params)) == [
                "Fish",
                "Plankton",
                "Detritus",
            ]
            assert shared_group_names(self._shared(balanced_rpath_model)) == [
                "Fish",
                "Plankton",
                "Detritus",
            ]


class TestForcingChoices:
    def test_forcing_type_keys_are_state_variables(self):
        from pypath.core.forcing import StateVariable

        from pypath_shiny.pages.forcing_demo import forcing_demo_ui

        html = str(forcing_demo_ui())
        start = html.index('id="forcing_type"')
        chunk = html[start : html.index("</select>", start)]
        keys = [seg.split('"')[0] for seg in chunk.split('<option value="')[1:]]
        valid = {v.value for v in StateVariable}
        assert keys and all(k in valid for k in keys), keys


class TestRecreateParamsCopiesLandings:
    def test_landings_and_discards_copied(self):
        from pypath.core.ecopath import rpath
        from pypath.core.params import create_rpath_params

        from pypath_shiny.pages.ecopath import _recreate_params_from_model

        p = create_rpath_params(["Fish", "Plankton", "Detritus", "Fleet"], [0, 1, 2, 3])
        p.model.loc[0, ["Biomass", "PB", "QB", "EE"]] = [10.0, 1.0, 5.0, 0.8]
        p.model.loc[1, ["Biomass", "PB", "EE"]] = [5.0, 50.0, 0.6]
        p.model.loc[2, "Biomass"] = 1.0
        p.diet.iloc[1, 1] = 1.0  # Plankton eaten by Fish
        p.model.loc[0, "Fleet"] = 0.5
        p.model.loc[0, "Fleet.disc"] = 0.1
        m = rpath(p)

        back = _recreate_params_from_model(m)
        non_fleet = back.model["Type"] < 3
        assert back.model.loc[non_fleet, "Fleet"].tolist() == pytest.approx(
            m.Landings[np.asarray(m.type) < 3, 0].tolist()
        )
        assert back.model.loc[non_fleet, "Fleet.disc"].tolist() == pytest.approx(
            m.Discards[np.asarray(m.type) < 3, 0].tolist()
        )
        assert back.model.loc[0, "Fleet"] == pytest.approx(0.5)
        assert back.model.loc[0, "Fleet.disc"] == pytest.approx(0.1)


class TestHabitatCsvParser:
    def _write(self, tmp_path, text):
        p = tmp_path / "habitat.csv"
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_with_header(self, tmp_path):
        from pypath_shiny.pages.ecospace import parse_habitat_csv

        path = self._write(tmp_path, "habitat\n0.1\n0.9\n0.5\n")
        assert parse_habitat_csv(path, 3) == pytest.approx([0.1, 0.9, 0.5])

    def test_headerless(self, tmp_path):
        from pypath_shiny.pages.ecospace import parse_habitat_csv

        path = self._write(tmp_path, "0.2\n0.4\n")
        assert parse_habitat_csv(path, 2) == pytest.approx([0.2, 0.4])

    def test_values_are_clipped(self, tmp_path):
        from pypath_shiny.pages.ecospace import parse_habitat_csv

        path = self._write(tmp_path, "h\n-1\n2.5\n")
        assert parse_habitat_csv(path, 2) == pytest.approx([0.0, 1.0])

    def test_wrong_length_raises(self, tmp_path):
        from pypath_shiny.pages.ecospace import parse_habitat_csv

        path = self._write(tmp_path, "h\n0.1\n0.2\n")
        with pytest.raises(ValueError, match="expected 5 rows"):
            parse_habitat_csv(path, 5)

    def test_non_numeric_raises(self, tmp_path):
        from pypath_shiny.pages.ecospace import parse_habitat_csv

        path = self._write(tmp_path, "a,b\nx,y\n")
        with pytest.raises(ValueError, match="no numeric column"):
            parse_habitat_csv(path, 2)


class TestIterPolygons:
    def test_polygon_multipolygon_and_none(self):
        from shapely.geometry import MultiPolygon, Polygon

        from pypath_shiny.pages.ecospace import _iter_polygons

        square = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        other = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        assert list(_iter_polygons(square)) == [square]
        assert len(list(_iter_polygons(MultiPolygon([square, other])))) == 2
        assert list(_iter_polygons(None)) == []


class TestDeadUiRemoved:
    """Controls the server never reads must not remain in the UI."""

    def test_habitat_view_group_gone(self):
        from pypath_shiny.pages import ecospace

        assert "habitat_view_group" not in str(ecospace.ecospace_ui())

    def test_custom_fishing_scenario_gone(self):
        from pypath_shiny.pages import ecosim

        html = str(ecosim.ecosim_ui())
        assert 'value="custom"' not in html

    def test_wired_controls_have_handlers(self):
        import inspect

        from pypath_shiny.pages import ecopath, ecosim, forcing_demo, multistanza

        assert "input.upload_params" in inspect.getsource(ecopath)
        assert "fisheries_table.set_patch_fn" in inspect.getsource(ecopath)
        assert "input.forcing_multiplier" in inspect.getsource(ecosim)
        assert "input.forcing_run_demo" in inspect.getsource(forcing_demo)
        assert "input.save_stanzas" in inspect.getsource(multistanza)


class TestLoggerConfiguration:
    def test_repeated_configure_does_not_duplicate_handlers(self, tmp_path):
        from pypath_shiny.logger import configure_logging, get_logger

        configure_logging(log_dir=tmp_path)
        first = len(get_logger().handlers)
        configure_logging(log_dir=tmp_path)
        assert len(get_logger().handlers) == first

    def test_file_handler_attached_when_dir_exists(self, tmp_path):
        import logging

        from pypath_shiny.logger import configure_logging

        tmp_path.mkdir(exist_ok=True)
        configure_logging(log_dir=tmp_path)  # directory already exists
        logger = configure_logging(log_dir=tmp_path)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)


class TestAllDownloadHandlersYield:
    """A @render.download that returns a str makes Shiny treat it as a path."""

    def test_no_download_handler_returns_a_value(self):
        import pathlib
        import re

        pages = pathlib.Path(inspect.getfile(analysis)).parent
        offenders = []
        for path in sorted(pages.glob("*.py")):
            src = path.read_text(encoding="utf-8")
            for m in re.finditer(
                r"@render\.download\([^)]*\)\s*\n\s*def (\w+)\(\):\n((?:.+\n)+?)(?=\n\s*@|\n\s*def |\Z)",
                src,
            ):
                body = m.group(2)
                returns = [
                    r.strip()
                    for r in re.findall(r"^\s+return (.+)$", body, re.M)
                    if r.strip() not in ("", "None")
                ]
                if "yield" not in body or returns:
                    offenders.append(f"{path.name}:{m.group(1)}")
        assert offenders == []


class TestApplyPreyForcing:
    """The Biomass Forcing card writes into the scenario's ForcedPrey array."""

    @pytest.fixture
    def scenario(self, rpath_params, balanced_rpath_model):
        from pypath.core.ecosim import rsim_scenario

        return rsim_scenario(balanced_rpath_model, rpath_params, years=range(1, 4))

    def test_sets_only_the_selected_group(self, scenario):
        from pypath_shiny.pages.ecosim import apply_prey_forcing

        names = [str(n) for n in scenario.params.spname]
        group = names[1]
        assert apply_prey_forcing(scenario, group, 2.0) is True
        col = names.index(group)
        assert (scenario.forcing.ForcedPrey[:, col] == 2.0).all()
        assert (np.delete(scenario.forcing.ForcedPrey, col, axis=1) == 1.0).all()

    def test_resets_previous_forcing(self, scenario):
        from pypath_shiny.pages.ecosim import apply_prey_forcing

        names = [str(n) for n in scenario.params.spname]
        apply_prey_forcing(scenario, names[1], 3.0)
        assert apply_prey_forcing(scenario, names[1], 1.0) is False
        assert (scenario.forcing.ForcedPrey == 1.0).all()

    def test_unknown_group_and_none_multiplier(self, scenario):
        from pypath_shiny.pages.ecosim import apply_prey_forcing

        assert apply_prey_forcing(scenario, "NoSuchGroup", 2.0) is False
        assert apply_prey_forcing(scenario, "Fish", None) is False
        assert (scenario.forcing.ForcedPrey == 1.0).all()


class TestSpatialFishingIsWired:
    """The Spatial Fishing controls must reach the spatial simulation."""

    def test_run_passes_spatial_fishing(self):
        from pypath_shiny.pages import ecospace

        src = inspect.getsource(ecospace)
        assert "spatial_fishing=spatial_fishing" in src
        assert "def _build_spatial_fishing" in src

    def test_preview_only_caption_removed(self):
        from pypath_shiny.pages import ecospace

        assert "not yet passed to the" not in str(ecospace.ecospace_ui())

    def test_every_allocation_choice_is_handled(self):
        from pypath_shiny.pages import ecospace

        html = str(ecospace.ecospace_ui())
        start = html.index('id="fishing_allocation"')
        chunk = html[start : html.index("</select>", start)]
        choices = [seg.split('"')[0] for seg in chunk.split('<option value="')[1:]]
        src = inspect.getsource(ecospace)
        assert choices
        for choice in choices:
            # each choice is either built into a SpatialFishing or handled
            # explicitly in _build_spatial_fishing
            assert f'"{choice}"' in src, choice


class TestReactiveDecoratorsBindTheRightFunction:
    """Inserting a helper between a decorator and its function silently
    unbinds the handler: the decorators wrap the helper, and the real
    handler becomes an orphan that nothing ever calls. Import succeeds and
    every unit test still passes, so only a structural check catches it.
    """

    @staticmethod
    def _page_files():
        import pathlib

        return sorted(pathlib.Path(inspect.getfile(analysis)).parent.glob("*.py"))

    def test_event_handlers_are_not_plain_helpers(self):
        """@reactive.event must sit on a handler, never on a value-returning helper."""
        import ast

        offenders = []
        for path in self._page_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                decorated = any(
                    "reactive.event" in ast.unparse(d) for d in node.decorator_list
                )
                if not decorated:
                    continue
                # An effect's return value is discarded, so a handler that
                # returns a value is really a helper that got captured.
                returns_value = any(
                    isinstance(n, ast.Return) and n.value is not None
                    for n in ast.walk(node)
                    if not isinstance(n, (ast.FunctionDef, ast.Lambda))
                )
                if returns_value and node.name.startswith(
                    ("_build", "_parse", "_make")
                ):
                    offenders.append(f"{path.name}:{node.name}")
        assert offenders == []

    def test_ecospace_run_button_has_a_handler(self):
        """The Run Spatial Simulation button must drive _run_spatial_simulation."""
        import ast

        path = next(p for p in self._page_files() if p.name == "ecospace.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any("run_spatial_sim" in ast.unparse(d) for d in node.decorator_list)
        }
        assert bound == {"_run_spatial_simulation"}, bound

    def test_spatial_fishing_helpers_are_called_by_the_handler(self):
        """_build_spatial_fishing must actually be invoked, not just defined."""
        src = next(p for p in self._page_files() if p.name == "ecospace.py").read_text(
            encoding="utf-8"
        )
        assert "spatial_fishing = _build_spatial_fishing()" in src
        assert "spatial_fishing=spatial_fishing" in src


class TestGravityTargetGroups:
    """The Ecospace page exposes SpatialFishing.target_groups."""

    @staticmethod
    def _ecospace_src():
        import pathlib

        return (
            pathlib.Path(inspect.getfile(analysis)).parent / "ecospace.py"
        ).read_text(encoding="utf-8")

    def test_control_exists_and_is_multi_select(self):
        src = self._ecospace_src()
        assert '"fishing_target_groups"' in src
        assert "input_selectize" in src
        # It only makes sense for the gravity model
        gravity_panel = src.split("input.fishing_allocation === 'gravity'")[1]
        assert "fishing_target_groups" in gravity_panel.split("panel_conditional")[0]

    def test_selection_reaches_both_the_run_and_the_preview(self):
        src = self._ecospace_src()
        assert "target_groups=_selected_target_groups()" in src
        # Once in SpatialFishing for the run, once in the sidebar preview
        assert src.count("target_groups=_selected_target_groups()") == 2

    def test_empty_selection_means_all_groups(self):
        """None is what allocate_gravity treats as 'every group'."""
        src = self._ecospace_src()
        body = src.split("def _selected_target_groups()")[1].split("\n    def ")[0]
        assert "if not selected:" in body
        assert "return None" in body

    def test_choices_use_ecosim_indices(self):
        """Group g of the Ecopath model is index g + 1 in the biomass array."""
        src = self._ecospace_src()
        body = src.split("def _populate_target_groups()")[1].split("\n    def ")[0]
        assert "if i > 0" in body or "i + 1" in body
        assert "update_selectize" in body
