"""Interface details: busy run buttons, full-screen plot cards, dark maps."""

import ast
from pathlib import Path

import pytest

from pypath_shiny.pages import ecospace as ecospace_page
from pypath_shiny.pages import ibm as ibm_page

PAGES = Path(ecospace_page.__file__).parent

# The Ecosim page keeps a plain Run button: it has its own progress bar and
# Cancel, and a click during a run must reach the handler (see
# test_ecosim_continuous_ui.test_run_simulation_invokes_the_task).
RUN_BUTTONS = [
    (ecospace_page, "ecospace_ui", "run_spatial_sim", "_spatial_task"),
    (ibm_page, "ibm_ui", "ibm_run", "_ibm_task"),
]


def _functions(page):
    tree = ast.parse(Path(page.__file__).read_text(encoding="utf-8"))
    return {
        f.name: f
        for f in ast.walk(tree)
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


# --- Run buttons show a busy state while their task runs ----------------


@pytest.mark.parametrize("page,ui_fn,button,task", RUN_BUTTONS)
def test_run_button_is_a_task_button(page, ui_fn, button, task):
    html = str(getattr(page, ui_fn)())
    tag = next(part for part in html.split("<button") if f'id="{button}"' in part[:400])
    opening = tag.split(">")[0]
    assert "bslib-task-button" in opening
    # It starts enabled: the handler explains what is missing.
    assert "disabled" not in opening


@pytest.mark.parametrize("page,ui_fn,button,task", RUN_BUTTONS)
def test_the_task_drives_its_button(page, ui_fn, button, task):
    decorators = [ast.unparse(d) for d in _functions(page)[task].decorator_list]
    assert f"ui.bind_task_button(button_id='{button}')" in decorators
    # Shiny wants bind_task_button above extended_task.
    assert decorators.index(
        f"ui.bind_task_button(button_id='{button}')"
    ) < decorators.index("reactive.extended_task")


def test_no_page_enables_a_task_button_as_an_action_button():
    # update_action_button does not reach a task button.
    for page, _, button, _ in RUN_BUTTONS:
        source = Path(page.__file__).read_text(encoding="utf-8")
        assert f'update_action_button("{button}"' not in source


# --- Plots and maps can be opened full screen ---------------------------

_CONTAINERS = (
    "ui.card",
    "ui.navset_card_tab",
    "ui.navset_card_pill",
    "ui.navset_card_underline",
)


def _plot_containers():
    found = []
    for path in sorted(PAGES.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call) and ast.unparse(node.func) in _CONTAINERS
            ):
                continue
            segment = ast.get_source_segment(source, node)
            if not any(
                marker in segment
                for marker in ("output_widget(", "output_plot(", "_MAP.ui(")
            ):
                continue
            full_screen = any(
                k.arg == "full_screen" and ast.unparse(k.value) == "True"
                for k in node.keywords
            )
            found.append((f"{path.name}:{node.lineno}", full_screen))
    return found


def test_every_plot_or_map_container_offers_full_screen():
    containers = _plot_containers()
    assert len(containers) >= 12
    assert [where for where, full_screen in containers if not full_screen] == []


# --- The Ecospace maps follow dark mode ---------------------------------


def test_ecospace_maps_have_a_dark_basemap():
    from shiny_deckgl import CARTO_DARK

    maps = [
        ecospace_page.GRID_MAP,
        ecospace_page.HABITAT_MAP,
        ecospace_page.EFFORT_MAP,
        ecospace_page.BIOMASS_MAP,
    ]
    assert ecospace_page._MAPS == tuple(maps) or list(ecospace_page._MAPS) == maps
    for widget in maps:
        assert widget.dark_style == CARTO_DARK
        assert widget.follow_dark_mode is True
        assert widget.style != widget.dark_style


def test_a_full_screen_card_lets_its_map_fill_it():
    css = (PAGES.parent / "static" / "custom.css").read_text(encoding="utf-8")
    rule = css.split('.card[data-full-screen="true"] .deckgl-map')[1].split("}")[0]
    assert "height: calc(100vh" in rule and "!important" in rule
