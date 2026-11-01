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
