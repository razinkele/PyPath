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
