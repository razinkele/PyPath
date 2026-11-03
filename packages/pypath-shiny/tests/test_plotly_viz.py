"""Tests for Plotly visualization helpers."""

from pypath_shiny.services.plotly_viz import build_biomass_figure, build_catch_figure


class DummySim:
    """Minimal simulation object used for Plotly tests."""

    def __init__(self):
        self.out_Biomass = [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
        self.annual_Catch = [[0.0, 10.0, 11.0], [1.0, 12.0, 13.0], [2.0, 14.0, 15.0]]


def test_build_biomass_figure_returns_traces():
    """Biomass figures should render the selected groups with trace data."""
    sim = DummySim()
    fig = build_biomass_figure(sim, ["A", "B", "C"], ["A", "C"], log_scale=False)
    assert len(fig.data) == 2
    assert fig.data[0].name == "A"
    assert fig.layout.xaxis.title.text == "Year"


def test_build_catch_figure_returns_trace():
    """Total catch should return a Plotly trace with annual values."""
    sim = DummySim()
    fig = build_catch_figure(sim)
    assert len(fig.data) == 1
    assert fig.layout.yaxis.title.text == "Total Catch"
    assert len(fig.data[0].y) == 3


def test_build_biomass_figure_handles_missing_selection():
    """Empty selections should still produce a valid figure."""
    sim = DummySim()
    fig = build_biomass_figure(sim, ["A", "B"], [], log_scale=True)
    assert fig is not None
    assert fig.layout.annotations[0].text == "No simulation data available"


def test_relative_biomass_starts_at_one_with_reference_line():
    """relative=True divides each series by its first value, line at y=1."""
    import pytest

    sim = DummySim()  # column B: 1, 2, 3 -> 1, 2, 3 ; column C: 2, 3, 4
    fig = build_biomass_figure(sim, ["A", "B", "C"], ["B", "C"], relative=True)
    assert list(fig.data[0].y) == pytest.approx([1.0, 2.0, 3.0])
    assert list(fig.data[1].y) == pytest.approx([1.0, 1.5, 2.0])
    assert fig.layout.yaxis.title.text == "Relative Biomass"
    assert any(s.y0 == 1 and s.y1 == 1 for s in fig.layout.shapes)


def test_relative_biomass_zero_start_gives_zeros():
    sim = DummySim()  # column A starts at 0
    fig = build_biomass_figure(sim, ["A", "B", "C"], ["A"], relative=True)
    assert list(fig.data[0].y) == [0.0, 0.0, 0.0]


def test_empty_message_is_configurable():
    fig = build_biomass_figure(
        None, [], [], empty_message="Run simulation to see results"
    )
    assert fig.layout.annotations[0].text == "Run simulation to see results"


def test_ecosim_output_offset_skips_outside_column():
    """Ecosim out_Biomass has a leading 'Outside' column."""

    class EcosimLike:
        out_Biomass = [[9.0, 1.0, 5.0], [9.0, 2.0, 6.0]]  # Outside, A, B

    fig = build_biomass_figure(EcosimLike(), ["A", "B"], ["B"])
    assert list(fig.data[0].y) == [5.0, 6.0]


# --- Analysis page builders -------------------------------------------------


def test_mti_heatmap_is_centred_on_zero():
    import numpy as np

    from pypath_shiny.services.plotly_viz import build_mti_heatmap

    mti = np.array([[0.0, 0.5], [-0.25, 0.1]])
    fig = build_mti_heatmap(mti, ["Fish", "Zoo"])
    hm = fig.data[0]
    assert hm.type == "heatmap"
    assert list(hm.x) == ["Fish", "Zoo"] and list(hm.y) == ["Fish", "Zoo"]
    assert hm.zmid == 0
    assert fig.layout.yaxis.autorange == "reversed"  # impacting group on top


def test_keystoneness_scatter_skips_invalid_points():
    import numpy as np
    import pytest

    from pypath_shiny.services.plotly_viz import build_keystoneness_figure

    fig = build_keystoneness_figure(
        ["A", "B", "C", "D"],
        np.array([10.0, 0.0, 1.0, 5.0]),  # B has no biomass
        np.array([0.5, 0.9, np.nan, -0.2]),  # C has no keystoneness
    )
    pts = fig.data[0]
    assert list(pts.text) == ["A", "D"]
    assert list(pts.x) == pytest.approx([1.0, np.log10(5.0)], abs=1e-3)
    assert fig.layout.xaxis.title.text == "Log10(Biomass)"


def test_ee_bars_mark_values_above_one():
    import numpy as np

    from pypath_shiny.services.plotly_viz import build_ee_figure

    fig = build_ee_figure(["A", "B", "Det"], np.array([0.4, 1.3, np.nan]))
    bars = fig.data[0]
    assert list(bars.y) == ["A", "B"]  # NaN EE (detritus) left out
    assert list(bars.marker.color) == ["#2ca02c", "#d62728"]
    assert any(s.x0 == 1 and s.x1 == 1 for s in fig.layout.shapes)


def test_analysis_builders_handle_empty_input():
    import numpy as np

    from pypath_shiny.services.plotly_viz import (
        build_ee_figure,
        build_keystoneness_figure,
        build_mti_heatmap,
    )

    for fig in (
        build_mti_heatmap(None, []),
        build_keystoneness_figure([], np.array([]), np.array([])),
        build_ee_figure([], np.array([])),
    ):
        assert fig.layout.annotations, "empty figures should explain themselves"


# --- Ecosim Summary builders --------------------------------------------------


def test_initial_vs_final_biomass_grouped_bars():
    from pypath_shiny.services.plotly_viz import build_initial_final_figure

    fig = build_initial_final_figure(["A", "B"], [1.0, 2.0], [1.5, 1.0])
    assert [t.name for t in fig.data] == ["Initial", "Final"]
    assert list(fig.data[1].y) == [1.5, 1.0]
    assert fig.layout.barmode == "group"


def test_biomass_change_percent_and_colours():
    import warnings

    import pytest

    from pypath_shiny.services.plotly_viz import build_biomass_change_figure

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero warning
        fig = build_biomass_change_figure(
            ["A", "B", "Z"], [2.0, 4.0, 0.0], [3.0, 1.0, 5.0]
        )
    bars = fig.data[0]
    assert list(bars.x) == pytest.approx([50.0, -75.0, 0.0])
    assert list(bars.marker.color) == ["#2ca02c", "#d62728", "#2ca02c"]
    assert any(s.x0 == 0 and s.x1 == 0 for s in fig.layout.shapes)


def test_summary_builders_empty():
    from pypath_shiny.services.plotly_viz import (
        build_biomass_change_figure,
        build_initial_final_figure,
    )

    for fig in (
        build_initial_final_figure([], [], []),
        build_biomass_change_figure([], [], []),
    ):
        assert fig.layout.annotations[0].text == "Run simulation first"


# --- Ecosim effort preview ----------------------------------------------------


def test_effort_figure_one_line_per_fleet():
    import numpy as np
    import pytest

    from pypath_shiny.services.plotly_viz import build_effort_figure

    effort = np.ones((24, 3))  # column 0 = Outside, then two fleets
    effort[12:, 2] = 0.5
    fig = build_effort_figure(effort, ["Trawl", "Gillnet"])
    assert [t.name for t in fig.data] == ["Trawl", "Gillnet"]
    assert list(fig.data[1].y[-1:]) == [0.5]
    assert list(fig.data[0].x[:2]) == pytest.approx([0.0, 1 / 12])
    assert any(s.y0 == 1 and s.y1 == 1 for s in fig.layout.shapes)
    assert fig.layout.yaxis.range[0] == 0 and fig.layout.yaxis.range[1] >= 1.5


def test_effort_figure_without_fleets_or_scenario():
    import numpy as np

    from pypath_shiny.services.plotly_viz import build_effort_figure

    assert (
        build_effort_figure(None, []).layout.annotations[0].text
        == "Create scenario to preview effort"
    )
    assert (
        build_effort_figure(np.ones((12, 1)), []).layout.annotations[0].text
        == "No fishing fleets in model"
    )


# --- Results: trophic charts and food web -------------------------------------


def test_tl_bars_sorted_high_to_low_with_line_at_one():
    from pypath_shiny.services.plotly_viz import build_tl_bar_figure

    fig = build_tl_bar_figure(["Phyto", "Fish", "Zoo"], [1.0, 3.2, 2.1])
    assert list(fig.data[0].y) == ["Fish", "Zoo", "Phyto"]
    assert list(fig.data[0].x) == [3.2, 2.1, 1.0]
    assert any(s.x0 == 1 and s.x1 == 1 for s in fig.layout.shapes)


def test_trophic_pyramid_bins_include_tl_above_six_and_put_tl1_at_bottom():
    import pytest

    from pypath_shiny.services.plotly_viz import build_trophic_pyramid_figure

    # TL:          1.0   2.5   6.3 (apex predator: was dropped by the old 5-6 bin)
    fig = build_trophic_pyramid_figure([1.0, 2.5, 6.3], [100.0, 10.0, 0.5])
    bars = fig.data[0]
    labels = list(bars.y)
    assert labels == ["TL 1-2", "TL 2-3", "TL 3-4", "TL 4-5", "TL 5+"]
    assert list(bars.x) == pytest.approx([100.0, 10.0, 0.0, 0.0, 0.5])
    assert fig.layout.yaxis.autorange != "reversed"  # TL 1-2 drawn at the bottom


def test_foodweb_nodes_at_true_trophic_level_and_bounded_sizes():
    import numpy as np

    from pypath_shiny.services.plotly_viz import build_foodweb_figure

    groups = ["Phyto", "Zoo", "Fish", "Det"]
    tl = np.array([1.0, 2.0, 3.9, 1.0])
    biomass = np.array([10.0, 5.0, 2.0, 500.0])
    dc = np.zeros((4, 3))  # prey rows x living predator columns
    dc[0, 1] = 1.0  # Zoo eats Phyto
    dc[1, 2] = 0.95  # Fish eats Zoo
    dc[3, 2] = 0.05  # ... and a little detritus
    fig = build_foodweb_figure(groups, tl, biomass, dc, n_living=3, min_flow=0.01)
    nodes = next(t for t in fig.data if t.mode and "markers" in t.mode)
    assert list(nodes.text) == groups
    assert nodes.y[2] == 3.9  # not int(TL)
    assert all(12 <= s <= 45 for s in nodes.marker.size)
    edges = [t for t in fig.data if t is not nodes]
    assert len(edges) == 3

    # min_flow drops weak links
    fig2 = build_foodweb_figure(groups, tl, biomass, dc, n_living=3, min_flow=0.1)
    assert len([t for t in fig2.data if t.mode == "lines"]) == 2


def test_foodweb_without_links_explains():
    import numpy as np

    from pypath_shiny.services.plotly_viz import build_foodweb_figure

    fig = build_foodweb_figure(
        ["A"], np.array([1.0]), np.array([1.0]), np.zeros((1, 1)), 1
    )
    assert "No diet connections" in fig.layout.annotations[0].text


# --- Results plot settings (palette and style selectors) ----------------------


def test_matplotlib_palette_names_are_honoured():
    from pypath_shiny.services.plotly_viz import _safe_palette

    assert _safe_palette("tab10", 2) == ["#1f77b4", "#ff7f0e"]
    assert _safe_palette("Paired", 2) == ["#a6cee3", "#1f78b4"]
    viridis = _safe_palette("viridis", 4)
    assert len(set(viridis)) == 4 and viridis[0].lower() == "#440154"
    assert len(_safe_palette("Set2", 3)) == 3  # Plotly qualitative name


def test_plot_style_maps_to_plotly_template():
    from pypath_shiny.services.plotly_viz import plotly_template_for

    assert plotly_template_for("default") == "plotly_white"
    assert plotly_template_for("seaborn-v0_8") == "seaborn"
    assert plotly_template_for("ggplot") == "ggplot2"
    assert plotly_template_for("dark_background") == "plotly_dark"
    assert plotly_template_for("unknown") == "plotly_white"
    assert plotly_template_for(None) == "plotly_white"


# --- Ecopath page reuse --------------------------------------------------------


def test_ee_bars_mark_negative_values_too():
    import numpy as np

    from pypath_shiny.services.plotly_viz import build_ee_figure

    fig = build_ee_figure(["A", "B"], np.array([-0.2, 0.5]))
    assert list(fig.data[0].marker.color) == ["#d62728", "#2ca02c"]


def test_tl_bars_coloured_by_role():
    from pypath_shiny.services.plotly_viz import build_tl_bar_figure

    fig = build_tl_bar_figure(
        ["Phyto", "Zoo", "Shark"], [1.0, 2.2, 4.4], top_predator_tl=4.0
    )
    by_group = dict(zip(fig.data[0].y, fig.data[0].marker.color))
    assert by_group == {"Shark": "#d62728", "Zoo": "#1f77b4", "Phyto": "#2ca02c"}


# --- Feedback and IBM builders -------------------------------------------------


def test_feedback_trend_bars_and_empty_state():
    from pypath_shiny.services.plotly_viz import build_feedback_trend_figure

    fig = build_feedback_trend_figure([("2026-W43", 2), ("2026-W44", 5)])
    assert list(fig.data[0].y) == [2, 5]
    assert fig.layout.xaxis.type == "category"
    empty = build_feedback_trend_figure([])
    assert empty.layout.annotations[0].text == "No feedback yet"


def test_ibm_size_figure_puts_length_on_a_right_axis():
    from pypath_shiny.services.plotly_viz import build_ibm_size_figure

    fig = build_ibm_size_figure([10.0, 12.0], [100, 200])
    assert fig.data[0].type == "bar"
    assert fig.data[1].yaxis == "y2"
    assert fig.layout.yaxis2.overlaying == "y"
    assert fig.layout.yaxis2.side == "right"


def test_ibm_builders_explain_why_they_are_empty():
    from pypath_shiny.services import plotly_viz as pv

    for fig in (
        pv.build_ibm_size_figure([], []),
        pv.build_ibm_age_figure([], []),
        pv.build_ibm_biomass_comparison_figure(None, None, ""),
        pv.build_ibm_dynamics_figure(None),
    ):
        assert fig.layout.annotations, fig.layout.title.text
        assert "IBM" in fig.layout.annotations[0].text


def test_ibm_biomass_comparison_has_both_runs():
    from pypath_shiny.services.plotly_viz import build_ibm_biomass_comparison_figure

    fig = build_ibm_biomass_comparison_figure([1, 2, 3], [1, 2], "Smelt")
    assert [t.name for t in fig.data] == ["Standard Ecosim", "IBM-enhanced"]
    assert len(fig.data[1].y) == 2
    assert "Smelt" in fig.layout.title.text


def test_ibm_growth_curve_reaches_toward_linf():
    from pypath_shiny.services.plotly_viz import build_ibm_growth_figure

    fig = build_ibm_growth_figure(10, 30.0, 0.5, [1.0], [12.0])
    curve = fig.data[0]
    assert curve.y[0] == 0.0
    assert 29.0 < curve.y[-1] < 30.0
    assert fig.data[1].mode == "markers"


def test_ibm_energy_fractions_sum_to_one():
    from pypath_shiny.services.plotly_viz import (
        build_ibm_energy_figure,
        ibm_energy_fractions,
    )

    f = ibm_energy_fractions(unassimilated=0.2, sda=0.1, reproduction=0.25)
    assert abs(sum(f.values()) - 1.0) < 1e-12
    assert abs(f["SDA"] - 0.08) < 1e-12
    fig = build_ibm_energy_figure(f)
    assert fig.data[0].orientation == "h"
    assert list(fig.data[0].text) == ["54.0%", "8.0%", "20.0%", "18.0%"]


def test_ibm_dynamics_splits_change_into_gain_and_loss():
    from pypath_shiny.services.plotly_viz import build_ibm_dynamics_figure

    fig = build_ibm_dynamics_figure([1.0, 3.0, 2.0])
    line, gain, loss = fig.data
    assert list(gain.y) == [2.0, 0.0]
    assert list(loss.y) == [0.0, -1.0]
    # Bottom panel shares the time axis with the top one
    assert gain.xaxis == "x2" and fig.layout.xaxis.matches == "x2"
