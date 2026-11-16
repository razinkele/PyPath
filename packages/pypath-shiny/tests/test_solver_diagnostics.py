"""Solver diagnostics panel: badges, summary and month tables."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from pypath.core.ecosim_continuous import KERNEL_CHECK_TOL
from pypath.examples import example_ecosim_scenario

from pypath_shiny.services.ecosim_solver import build_solver_request, run_ecosim
from pypath_shiny.services.solver_diagnostics import (
    QS_SEPARATION_MIN,
    Badge,
    flagged_months,
    solver_badges,
    solver_month_table,
    solver_summary,
)

PAGE = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "pypath_shiny"
    / "pages"
    / "ecosim.py"
)


def _month(i=1, **kw):
    base = {
        "month": i,
        "nfev": 10,
        "njev": 0,
        "status": 0,
        "steps": 4,
        "degraded": False,
        "clips": 0,
        "min_state": 0.1,
        "hden_passes_max": 2,
        "kernel_check": 0.0,
        "jac_fresh": 0,
        "jac_reused": 0,
        "budget_residual": 1e-12,
        "pool_budget_residual": 1e-12,
        "qs_inner_max": None,
    }
    base.update(kw)
    return base


def _info(months=None, **top):
    info = {
        "method": "lsoda",
        "rtol": 1e-6,
        "positivity": "clip",
        "forcing_interpolation": "constant",
        "fast_groups": "states",
        "qs_rate": 365.0,
        "qs_groups": [],
        "qs_separation": None,
        "corrections": [],
        "ignored_switches": [],
        "ftime_groups": [],
        "rhs_evals": 100,
        "months": [_month()] if months is None else months,
    }
    info.update(top)
    return info


def _levels(info, level):
    return [b for b in solver_badges(info) if b.level == level]


def test_no_badges_for_parity_runs():
    assert solver_badges(None) == []
    assert solver_summary(None) == []
    assert solver_month_table(None).empty
    assert flagged_months(None).empty


def test_constants():
    assert QS_SEPARATION_MIN == 10.0
    assert KERNEL_CHECK_TOL == 1e-9


def test_badge_edges():
    first = solver_badges(_info())[0]
    assert first == Badge("secondary", "Continuous path (not EwE parity): LSODA")
    assert len(solver_badges(_info())) == 1

    for kc in (float("nan"), None, 1e-9):
        assert not _levels(_info([_month(kernel_check=kc)]), "warning")
    kernel = _levels(_info([_month(kernel_check=1.1e-9)]), "warning")
    assert len(kernel) == 1 and "1.10e-09" in kernel[0].text

    for sep in (None, math.inf, 10.0):
        assert not _levels(_info(qs_separation=sep), "warning")
    sep = _levels(_info(qs_separation=9.99), "warning")
    assert len(sep) == 1 and "< 10" in sep[0].text

    assert _levels(_info([_month(degraded=True)]), "warning")
    assert not _levels(_info([_month(degraded=True)]), "danger")
    danger = _levels(_info([_month(2, degraded=True, status=1)]), "danger")
    assert len(danger) == 1 and "2" in danger[0].text

    big = _info(months=[_month(budget_residual=3.3e-6)], rtol=1e-10)
    assert len(solver_badges(big)) == 1

    qs = _info(fast_groups="quasi-steady", qs_groups=[])
    info_badges = _levels(qs, "info")
    assert len(info_badges) == 1
    assert info_badges[0].text == (
        "No group qualified as quasi-steady at 365 per year; every group was integrated"
    )

    smooth = _levels(_info(forcing_interpolation="smooth"), "info")
    assert [b.text for b in smooth] == ["Changes the model: smooth forcing"]
    qs_on = _info(fast_groups="quasi-steady", qs_groups=[2])
    names = ["Outside", "Phyto", "Detritus"]
    texts = [b.text for b in solver_badges(qs_on, names)]
    assert "Changes the model: quasi-steady Detritus" in texts

    assert not _levels(_info(positivity="log"), "info")
    clipped = _levels(_info([_month(clips=3)], positivity="clip"), "info")
    assert "log states avoid them" in clipped[0].text


def test_degraded_list_is_capped():
    months = [_month(i, degraded=True) for i in range(1, 9)]
    text = _levels(_info(months), "warning")[0].text
    assert "1, 2, 3, 4, 5 +3 more" in text


def test_month_table_with_none_values():
    months = [_month(i) for i in range(1, 601)]
    months[3] = _month(4, budget_residual=None, pool_budget_residual=None)
    months[10] = _month(11, degraded=True)
    months[20] = _month(21, kernel_check=1e-3)
    months[30] = _month(31, clips=2)
    info = _info(months)
    table = solver_month_table(info)
    assert len(table) == 600
    assert "qs_inner_max" not in table.columns
    assert math.isnan(table.loc[3, "budget_residual"])
    assert math.isnan(table.loc[3, "pool_budget_residual"])
    assert list(flagged_months(info)["Month"]) == [11, 21, 31]

    qs = _info(
        [_month(1), _month(2, qs_inner_max=3)],
        fast_groups="quasi-steady",
        qs_groups=[1],
    )
    qs_table = solver_month_table(qs)
    assert math.isnan(qs_table.loc[0, "qs_inner_max"])
    assert qs_table.loc[1, "qs_inner_max"] == 3
    assert solver_month_table(_info([])).empty


def test_real_run_renders():
    scen = example_ecosim_scenario(
        "Central_Baltic", years=range(1, 3), fast_group_rule="ewe"
    )
    out = run_ecosim(scen, build_solver_request("lsoda"))
    info = out.solver_info
    badges = solver_badges(info, scen.params.spname)
    assert [b.text for b in badges] == ["Continuous path (not EwE parity): LSODA"]
    summary = dict(solver_summary(info, scen.params.spname))
    assert "17" in summary["Feeding-time states"]
    for name in ("R2", "DF once", "Hden0 recalibrated"):
        assert name in summary["Corrections applied"]
    for name in (
        "RK4_FAST_EQUILIBRIUM",
        "STANZA_STAGE_RATES",
        "FTIME_START_OF_MONTH",
        "RK4_STEPS",
    ):
        assert name in summary["EwE switches ignored"]
    assert len(solver_month_table(info)) == 24
    assert flagged_months(info).empty


def test_panel_placeholders():
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    ids = {
        (n.func.attr, n.args[0].value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.args
        and isinstance(n.args[0], ast.Constant)
    }
    assert ("output_ui", "solver_diagnostics") in ids
    assert ("output_data_frame", "solver_month_table") in ids
    renderers = {
        n.name: [ast.unparse(d) for d in n.decorator_list]
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    }
    assert "render.ui" in renderers["solver_diagnostics"]
    assert "render.data_frame" in renderers["solver_month_table"]
    src = ast.unparse(tree)
    assert "render.DataGrid(_month_table(info)" in src


def test_progress_tab_holds_the_placeholder():
    text = PAGE.read_text(encoding="utf-8")
    tab = text.split('"Simulation Progress",', 1)[1].split('"Time Series"', 1)[0]
    assert 'ui.output_ui("solver_diagnostics")' in tab
    assert tab.index("progress_display") < tab.index("solver_diagnostics")


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_flagged_ignores_unchecked_kernel(bad):
    assert flagged_months(_info([_month(kernel_check=bad)])).empty


def test_summary_shows_the_quasi_steady_rate():
    names = ["Outside", "Phyto"]
    qs = _info(fast_groups="quasi-steady", qs_groups=[1], qs_rate=50.0)
    options = dict(solver_summary(qs, names))["Model options"]
    assert "fast groups: quasi-steady; quasi-steady above 50 per year" in options
    # Unused without quasi-steady, so not shown.
    assert "quasi-steady above" not in dict(solver_summary(_info()))["Model options"]
    # The progress display's "Model options" row is this same summary row.
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    progress = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "progress_display"
    )
    src = ast.unparse(progress)
    assert "_solver_summary(info" in src
    assert "'Model options'" in src


def test_flagged_months_view_scrolls_instead_of_growing():
    from pypath_shiny.pages.ecosim import _flagged_months_view

    assert "No month flagged." in str(_flagged_months_view(flagged_months(_info())))
    months = [_month(i, degraded=True) for i in range(1, 601)]
    view = str(_flagged_months_view(flagged_months(_info(months))))
    assert "Flagged months" in view
    # Every flagged month stays in the table, inside a box of capped height.
    assert view.count("<tr") == 601  # header and 600 months
    box = view[view.index("Flagged months") :]
    style = box[box.index('style="') :].split('"', 2)[1]
    assert "max-height:" in style
    assert "overflow-y: auto" in style
    tree = ast.parse(PAGE.read_text(encoding="utf-8"))
    panel = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "solver_diagnostics"
    )
    assert "_flagged_months_view(" in ast.unparse(panel)


@pytest.mark.parametrize("rhs", [None, float("nan"), math.inf])
def test_summary_survives_unrecorded_rhs_evals(rhs):
    summary = dict(solver_summary(_info(rhs_evals=rhs)))
    assert summary["Right-hand-side evaluations"] == "not recorded"
    assert dict(solver_summary(_info(rhs_evals=240)))[
        "Right-hand-side evaluations"
    ] == ("240 in total, 240 per month")


@pytest.mark.parametrize(
    "groups",
    [
        np.array([1, 2]),
        (1, 2),
        [np.int64(1), np.int64(2)],
        np.array([], dtype=int),
        (),
    ],
    ids=["array", "tuple", "numpy-ints", "empty-array", "empty-tuple"],
)
def test_qs_groups_as_array_or_tuple(groups):
    names = ["Outside", "Phyto", "Detritus"]
    info = _info(fast_groups="quasi-steady", qs_groups=groups)
    options = dict(solver_summary(info, names))["Model options"]
    texts = [b.text for b in solver_badges(info, names)]
    if len(groups):
        assert "quasi-steady groups: Phyto, Detritus" in options
        assert "Changes the model: quasi-steady Phyto, Detritus" in texts
    else:
        assert "quasi-steady groups: none" in options
        assert any(t.startswith("No group qualified as quasi-steady") for t in texts)
