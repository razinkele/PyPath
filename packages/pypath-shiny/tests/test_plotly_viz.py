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
