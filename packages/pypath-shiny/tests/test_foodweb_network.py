"""The interactive food-web network (pyvis) on the Results page."""

import numpy as np
import pytest

from pypath_shiny.pages import results as results_page
from pypath_shiny.services import network_viz

# Five groups in block order: a producer, two consumers, detritus, a fleet.
GROUPS = ["Phyto", "Zoo", "Fish", "Detritus", "Fleet"]
TYPES = [1, 0, 0, 2, 3]
TL = [1.0, 2.0, 3.1, 1.0, 4.0]
BIOMASS = [100.0, 25.0, 4.0, 50.0, 0.0]
# DC: a row per group plus the import row, a column per living group.
DC = np.array(
    [
        # Phyto  Zoo   Fish
        [0.0, 0.8, 0.0],  # Phyto eaten
        [0.0, 0.0, 0.95],  # Zoo eaten
        [0.0, 0.0, 0.05],  # Fish eaten (cannibalism: a self link)
        [0.0, 0.2, 0.0],  # Detritus eaten
        [0.0, 0.0, 0.0],  # Fleet
        [0.0, 0.0, 0.0],  # Import
    ]
)


def _build(**kwargs):
    pytest.importorskip("pyvis.shiny")
    args = dict(groups=GROUPS, types=TYPES, trophic_levels=TL, biomass=BIOMASS)
    args.update(diet_matrix=DC)
    args.update(kwargs)
    return network_viz.build_foodweb_network(**args)


def _node(net, name):
    return next(n for n in net.nodes if n["id"] == name)


def test_nodes_are_the_living_and_detritus_groups():
    net = _build()
    assert [n["id"] for n in net.nodes] == ["Phyto", "Zoo", "Fish", "Detritus"]
    assert all(n["label"] == n["id"] for n in net.nodes)


def test_higher_trophic_levels_sit_higher():
    net = _build()
    # Screen y grows downwards: a higher trophic level has the smaller y.
    assert _node(net, "Fish")["y"] < _node(net, "Zoo")["y"] < _node(net, "Phyto")["y"]
    assert _node(net, "Phyto")["y"] == _node(net, "Detritus")["y"]
    # Groups of one level are spread, not stacked.
    assert _node(net, "Phyto")["x"] != _node(net, "Detritus")["x"]
    # Positions are a starting layout the user can drag from, not a simulation.
    assert all(n["physics"] is False for n in net.nodes)


def test_links_run_from_prey_to_predator_without_self_links():
    net = _build(min_flow=0.0)
    links = {(e["from"], e["to"]) for e in net.edges}
    assert links == {("Phyto", "Zoo"), ("Zoo", "Fish"), ("Detritus", "Zoo")}
    assert all(e["arrows"] == "to" for e in net.edges)


def test_min_flow_hides_weak_links():
    net = _build(min_flow=0.5)
    assert {(e["from"], e["to"]) for e in net.edges} == {
        ("Phyto", "Zoo"),
        ("Zoo", "Fish"),
    }


def test_node_size_follows_biomass_only_when_asked():
    sized = _build(size_by_biomass=True)
    assert (
        _node(sized, "Phyto")["size"]
        > _node(sized, "Zoo")["size"]
        > _node(sized, "Fish")["size"]
    )
    flat = _build(size_by_biomass=False)
    assert len({n["size"] for n in flat.nodes}) == 1


def test_link_width_follows_diet_share_only_when_asked():
    scaled = {(e["from"], e["to"]): e["width"] for e in _build().edges}
    assert (
        scaled[("Zoo", "Fish")] > scaled[("Phyto", "Zoo")] > scaled[("Detritus", "Zoo")]
    )
    flat = _build(width_by_flow=False)
    assert len({e["width"] for e in flat.edges}) == 1


def test_tooltips_name_the_numbers():
    net = _build()
    assert "TL 3.10" in _node(net, "Fish")["title"]
    assert "4" in _node(net, "Fish")["title"]
    edge = next(e for e in net.edges if (e["from"], e["to"]) == ("Zoo", "Fish"))
    assert "95" in edge["title"] and "Fish" in edge["title"]


def test_colours_tell_producers_consumers_and_detritus_apart():
    net = _build()
    colours = {n["id"]: n["color"] for n in net.nodes}
    assert colours["Zoo"] == colours["Fish"]
    assert len({colours["Phyto"], colours["Zoo"], colours["Detritus"]}) == 3


def test_groups_are_picked_by_type_not_by_position():
    # The same model with the fleet listed first and detritus before a consumer.
    order = [4, 0, 3, 1, 2]  # Fleet, Phyto, Detritus, Zoo, Fish
    living_new = [i for i in order if TYPES[i] < 2]  # Phyto, Zoo, Fish
    dc = np.zeros((6, 3))
    for new_row, old_row in enumerate(order):
        for new_col, old_col in enumerate(living_new):
            dc[new_row, new_col] = DC[old_row, [0, 1, 2].index(old_col)]
    net = _build(
        groups=[GROUPS[i] for i in order],
        types=[TYPES[i] for i in order],
        trophic_levels=[TL[i] for i in order],
        biomass=[BIOMASS[i] for i in order],
        diet_matrix=dc,
        min_flow=0.0,
    )
    assert {n["id"] for n in net.nodes} == {"Phyto", "Zoo", "Fish", "Detritus"}
    assert {(e["from"], e["to"]) for e in net.edges} == {
        ("Phyto", "Zoo"),
        ("Zoo", "Fish"),
        ("Detritus", "Zoo"),
    }


def test_an_unbalanced_model_gives_an_empty_network():
    net = _build(groups=[], types=[], trophic_levels=[], biomass=[], diet_matrix=[])
    assert net.nodes == [] and net.edges == []


# --- The page offers the network only when pyvis' Shiny binding exists ---


def test_the_page_offers_the_network_view_when_pyvis_is_there(monkeypatch):
    pytest.importorskip("pyvis.shiny")
    monkeypatch.setattr(network_viz, "HAS_PYVIS_SHINY", True)
    html = str(results_page.results_ui())
    assert 'id="foodweb_view"' in html
    assert 'id="foodweb_network"' in html
    # The diagram stays the default view: its tab is the active one.
    import re

    tabs = {
        value: "active" in tag
        for tag, value in re.findall(
            r'(<a [^>]*data-value="(diagram|network)"[^>]*>)', html
        )
    }
    assert tabs == {"diagram": True, "network": False}
    assert "Interactive network" in html


def test_the_page_is_unchanged_without_pyvis(monkeypatch):
    monkeypatch.setattr(network_viz, "HAS_PYVIS_SHINY", False)
    html = str(results_page.results_ui())
    assert "foodweb_view" not in html
    assert "foodweb_network" not in html
    assert 'id="foodweb_plot"' in html


def test_the_dependency_is_optional():
    import tomllib
    from pathlib import Path

    project = tomllib.loads(
        (Path(results_page.__file__).parents[3] / "pyproject.toml").read_text()
    )["project"]
    assert not any("pyvis" in dep for dep in project["dependencies"])
    assert any(
        dep.startswith("pyvis-optimized>=5")
        for dep in project["optional-dependencies"]["network"]
    )


@pytest.mark.parametrize(
    "version,supported",
    [
        ("5.0.0", True),
        ("5.1.2", True),
        ("6.0", True),
        ("10.2.0", True),
        ("4.5.0", False),  # has pyvis.shiny, but no theme="auto"
        ("0.3.2", False),  # the original pyvis
        ("5", False),
        ("", False),
        ("unknown", False),
    ],
)
def test_only_pyvis_optimized_5_or_newer_turns_the_view_on(version, supported):
    assert network_viz._version_supported(version) is supported


# --- Group order, on models Ecopath really balanced ----------------------

_KINDS = {"Detritus": 2, "Phyto": 1, "Zoo": 0, "Fish": 0, "Fleet": 3}
_ORDERS = {
    "block order": ["Phyto", "Zoo", "Fish", "Detritus", "Fleet"],
    "detritus first": ["Detritus", "Phyto", "Zoo", "Fish", "Fleet"],
    "fleet first": ["Fleet", "Phyto", "Zoo", "Fish", "Detritus"],
    "fleet in the middle": ["Phyto", "Fleet", "Zoo", "Detritus", "Fish"],
}


def _balanced(order):
    """The same five-group model with its groups in ``order``, balanced."""
    import warnings

    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    params = create_rpath_params(groups=order, types=[_KINDS[g] for g in order])
    m = params.model
    m.loc[m["Group"] == "Detritus", "Biomass"] = 100.0
    m.loc[m["Group"] == "Phyto", ["Biomass", "PB", "EE"]] = [10.0, 50.0, 0.8]
    m.loc[m["Group"] == "Zoo", ["Biomass", "PB", "QB", "EE"]] = [5.0, 10.0, 30.0, 0.9]
    m.loc[m["Group"] == "Fish", ["Biomass", "PB", "QB", "EE"]] = [2.0, 1.0, 5.0, 0.5]
    m["BioAcc"] = 0.0
    m["Unassim"] = 0.2
    m.loc[m["Group"].isin(["Detritus", "Phyto"]), "Unassim"] = 0.0
    m.loc[m["Group"] == "Fleet", ["BioAcc", "Unassim"]] = np.nan
    m["Detritus"] = 1.0
    m.loc[m["Group"] == "Detritus", "Detritus"] = 0.0
    m.loc[m["Group"] == "Fleet", "Detritus"] = np.nan
    prey = list(params.diet["Group"])
    params.diet["Zoo"] = [{"Detritus": 0.2, "Phyto": 0.8}.get(g, 0.0) for g in prey]
    params.diet["Fish"] = [{"Zoo": 1.0}.get(g, 0.0) for g in prey]
    m.loc[m["Group"] == "Fish", "Fleet"] = 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rpath(params)


def _view(model):
    net = _build(
        groups=model.Group,
        types=model.type,
        trophic_levels=model.TL,
        biomass=model.Biomass,
        diet_matrix=model.DC,
    )
    nodes = {
        n["id"]: (round(n["y"], 6), round(n["size"], 6), n["color"]) for n in net.nodes
    }
    links = {(e["from"], e["to"], round(e["width"], 9)) for e in net.edges}
    return nodes, links


@pytest.mark.parametrize("name", [n for n in _ORDERS if n != "block order"])
def test_a_balanced_model_gives_the_same_network_in_any_group_order(name):
    reference = _view(_balanced(_ORDERS["block order"]))
    assert {(a, b) for a, b, _ in reference[1]} == {
        ("Phyto", "Zoo"),
        ("Detritus", "Zoo"),
        ("Zoo", "Fish"),
    }
    assert _view(_balanced(_ORDERS[name])) == reference


def test_the_two_views_are_tabs_not_conditional_panels():
    # A Plotly widget redrawn after a conditional panel hid and showed it made
    # shinywidgets log "Unexpected error while closing model" in the browser;
    # the same sequence with tabs does not (checked in a browser).
    import ast
    from pathlib import Path

    tree = ast.parse(Path(results_page.__file__).read_text(encoding="utf-8"))
    views = next(
        ast.unparse(f)
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef) and f.name == "_foodweb_views"
    )
    assert "ui.navset_pill(" in views and "id='foodweb_view'" in views
    assert "panel_conditional" not in views


def test_the_hidden_view_is_left_alone():
    # Tabs alone are not enough: a slider moved on the network tab closed the
    # hidden Plotly diagram, with the same browser error. Each view reads the
    # slider and the switches through values routed to the selected view.
    import ast
    from pathlib import Path

    tree = ast.parse(Path(results_page.__file__).read_text(encoding="utf-8"))
    fns = {
        f.name: ast.unparse(f) for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
    }
    for name, view in (("foodweb_plot", "diagram"), ("foodweb_network", "network")):
        body = fns[name]
        assert f"_settings_for('{view}')" in body
        # Neither reads the inputs, or the selected tab, itself.
        assert "input.foodweb_" not in body and "input.show_" not in body
    route = fns["_route_foodweb_settings"]
    assert "_view_settings[input.foodweb_view()]" in route
    # Unchanged settings are not set again: that would redraw the view.
    assert "if not unchanged:" in route and "held.set(settings)" in route
    held = fns["_settings_for"]
    # Without the network view there are no tabs: the diagram reads directly.
    assert "if not network_viz.HAS_PYVIS_SHINY:" in held
    assert "req(held is not None)" in held
