"""
Unit tests for the plotting module.

Tests for food web visualization, time series plots,
and other plotting functions.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

# Skip plotting tests if matplotlib not available
pytest.importorskip("matplotlib")

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from pypath.core.plotting import (
    HAS_NETWORKX,
    HAS_PLOTLY,
    plot_biomass,
    plot_biomass_grid,
    plot_catch,
    plot_ecosim_summary,
    plot_foodweb,
    plot_mti_heatmap,
    plot_trophic_spectrum,
    save_plots,
)


class TestPlotBiomass:
    """Tests for plot_biomass function."""

    def test_returns_figure(self):
        """Should return matplotlib Figure."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 5)
        output.out_Biomass_annual[:, 0] = 0  # Index 0 unused

        fig = plot_biomass(output, groups=[1, 2])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_relative_biomass(self):
        """Relative biomass should normalize to initial."""
        output = MagicMock()
        output.out_Biomass_annual = np.ones((10, 4))
        output.out_Biomass_annual[:, 1] = np.linspace(1, 2, 10)
        output.out_Biomass_annual[:, 0] = 0

        fig = plot_biomass(output, groups=[1], relative=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self):
        """Should respect custom figure size."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 4)
        output.out_Biomass_annual[:, 0] = 0

        fig = plot_biomass(output, groups=[1], figsize=(8, 4))

        # Check approximate figure size
        size = fig.get_size_inches()
        assert np.isclose(size[0], 8)
        assert np.isclose(size[1], 4)
        plt.close(fig)

    def test_auto_select_groups(self):
        """Should auto-select groups with biomass."""
        output = MagicMock()
        output.out_Biomass_annual = np.zeros((10, 5))
        output.out_Biomass_annual[:, 1] = 1  # Only group 1 has biomass
        output.out_Biomass_annual[:, 2] = 2

        fig = plot_biomass(output)  # No groups specified

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCatch:
    """Tests for plot_catch function."""

    def test_returns_figure(self):
        """Should return matplotlib Figure."""
        output = MagicMock()
        output.out_Catch_annual = np.random.rand(10, 4)
        output.out_Catch_annual[:, 0] = 0

        fig = plot_catch(output, groups=[1, 2])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_stacked_plot(self):
        """Should create stacked area plot when stacked=True."""
        output = MagicMock()
        output.out_Catch_annual = np.random.rand(10, 4)
        output.out_Catch_annual[:, 0] = 0

        fig = plot_catch(output, groups=[1, 2], stacked=True)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_catch_data(self):
        """Should handle case with no catch."""
        output = MagicMock()
        output.out_Catch_annual = np.zeros((10, 4))

        fig = plot_catch(output)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotBiomassGrid:
    """Tests for plot_biomass_grid function."""

    def test_returns_figure(self):
        """Should return matplotlib Figure."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 6)
        output.out_Biomass_annual[:, 0] = 0

        fig = plot_biomass_grid(output, groups=[1, 2, 3, 4])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_grid_dimensions(self):
        """Should create correct grid dimensions."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 10)
        output.out_Biomass_annual[:, 0] = 0

        # 6 groups with 4 columns = 2 rows
        fig = plot_biomass_grid(output, groups=[1, 2, 3, 4, 5, 6], n_cols=4)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTrophicSpectrum:
    """Tests for plot_trophic_spectrum function."""

    def test_returns_figure(self):
        """Should return matplotlib Figure."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 4
        rpath.TL = np.array([0, 1.0, 2.0, 2.5, 3.5])
        rpath.Biomass = np.array([0, 100, 50, 20, 5])
        rpath.PB = np.array([0, 2.0, 1.0, 0.5, 0.2])
        rpath.QB = np.array([0, 0, 10, 5, 2])

        fig = plot_trophic_spectrum(rpath)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_by_production(self):
        """Should aggregate by production when specified."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.TL = np.array([0, 1.0, 2.0, 3.0])
        rpath.Biomass = np.array([0, 100, 50, 10])
        rpath.PB = np.array([0, 2.0, 1.0, 0.5])
        rpath.QB = np.array([0, 0, 10, 5])

        fig = plot_trophic_spectrum(rpath, by="production")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_by_consumption(self):
        """Should aggregate by consumption when specified."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.TL = np.array([0, 1.0, 2.0, 3.0])
        rpath.Biomass = np.array([0, 100, 50, 10])
        rpath.PB = np.array([0, 2.0, 1.0, 0.5])
        rpath.QB = np.array([0, 0, 10, 5])

        fig = plot_trophic_spectrum(rpath, by="consumption")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_invalid_by_raises(self):
        """Should raise ValueError for invalid 'by' parameter."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 2
        rpath.TL = np.array([0, 1.0, 2.0])
        rpath.Biomass = np.array([0, 100, 50])
        rpath.PB = np.array([0, 2.0, 1.0])
        rpath.QB = np.array([0, 0, 10])

        with pytest.raises(ValueError):
            plot_trophic_spectrum(rpath, by="invalid")


class TestPlotMTIHeatmap:
    """Tests for plot_mti_heatmap function."""

    def test_returns_figure(self):
        """Should return matplotlib Figure."""
        mti = np.random.randn(5, 5)

        fig = plot_mti_heatmap(mti)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_group_names(self):
        """Should use custom group names."""
        mti = np.random.randn(3, 3)
        names = ["Phyto", "Zoo", "Fish"]

        fig = plot_mti_heatmap(mti, group_names=names)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_symmetric_colormap(self):
        """Colormap should be symmetric around zero."""
        mti = np.array([[-1, 0.5], [0.5, -0.5]])

        fig = plot_mti_heatmap(mti)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


@pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
class TestPlotFoodweb:
    """Tests for plot_foodweb function."""

    def test_returns_figure(self):
        """Should return matplotlib Figure."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 1
        rpath.TL = np.array([0, 1.0, 2.0, 3.0, 1.0])
        rpath.Biomass = np.array([0, 100, 50, 10, 20])
        rpath.DC = np.zeros((5, 5))
        rpath.DC[1, 2] = 0.5
        rpath.DC[2, 3] = 0.5
        rpath.DC[4, 2] = 0.5
        rpath.QB = np.array([0, 0, 10, 5, 0])
        rpath.PB = np.array([0, 2.0, 1.0, 0.5, 0])

        fig = plot_foodweb(rpath)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_trophic_layout(self):
        """Should create trophic level layout."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 0
        rpath.TL = np.array([0, 1.0, 2.0, 3.0])
        rpath.Biomass = np.array([0, 100, 50, 10])
        rpath.DC = np.zeros((4, 4))
        rpath.DC[1, 2] = 0.8
        rpath.DC[2, 3] = 0.8
        rpath.QB = np.array([0, 0, 10, 5])
        rpath.PB = np.array([0, 2.0, 1.0, 0.5])

        fig = plot_foodweb(rpath, layout="trophic")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_spring_layout(self):
        """Should create spring layout."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 0
        rpath.TL = np.array([0, 1.0, 2.0, 3.0])
        rpath.Biomass = np.array([0, 100, 50, 10])
        rpath.DC = np.zeros((4, 4))
        rpath.DC[1, 2] = 0.8
        rpath.QB = np.array([0, 0, 10, 5])
        rpath.PB = np.array([0, 2.0, 1.0, 0.5])

        fig = plot_foodweb(rpath, layout="spring")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_node_size_by_production(self):
        """Should size nodes by production."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 2
        rpath.NUM_DEAD = 0
        rpath.TL = np.array([0, 1.0, 2.0])
        rpath.Biomass = np.array([0, 100, 50])
        rpath.DC = np.zeros((3, 3))
        rpath.DC[1, 2] = 0.8
        rpath.QB = np.array([0, 0, 10])
        rpath.PB = np.array([0, 2.0, 1.0])

        fig = plot_foodweb(rpath, node_size_by="production")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotEcosimSummary:
    """Tests for plot_ecosim_summary function."""

    def test_returns_figure(self):
        """Should return matplotlib Figure."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 5)
        output.out_Biomass_annual[:, 0] = 0
        output.out_Catch_annual = np.random.rand(10, 5)
        output.out_Catch_annual[:, 0] = 0

        fig = plot_ecosim_summary(output, groups=[1, 2])

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_creates_four_subplots(self):
        """Should create 2x2 subplot grid."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 4)
        output.out_Biomass_annual[:, 0] = 0
        output.out_Catch_annual = np.random.rand(10, 4)
        output.out_Catch_annual[:, 0] = 0

        fig = plot_ecosim_summary(output, groups=[1, 2])

        axes = fig.get_axes()
        # 4 subplots + 2 potential colorbars
        assert len(axes) >= 4
        plt.close(fig)


class TestSavePlots:
    """Tests for save_plots function."""

    def test_save_single_figure(self, tmp_path):
        """Should save single figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        filepath = str(tmp_path / "test_plot")
        save_plots(fig, filepath, format="png")

        assert (tmp_path / "test_plot.png").exists()
        plt.close(fig)

    def test_save_multiple_figures(self, tmp_path):
        """Should save multiple figures with indices."""
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2, 3])

        fig2, ax2 = plt.subplots()
        ax2.plot([3, 2, 1])

        filepath = str(tmp_path / "test_plots")
        save_plots([fig1, fig2], filepath, format="png")

        assert (tmp_path / "test_plots_1.png").exists()
        assert (tmp_path / "test_plots_2.png").exists()
        plt.close(fig1)
        plt.close(fig2)


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly not installed")
class TestInteractivePlots:
    """Tests for interactive Plotly plots."""

    def test_plot_biomass_interactive(self):
        """Should return Plotly Figure."""
        import plotly.graph_objects as go

        from pypath.core.plotting import plot_biomass_interactive

        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 5)
        output.out_Biomass_annual[:, 0] = 0

        fig = plot_biomass_interactive(output, groups=[1, 2])

        assert isinstance(fig, go.Figure)

    @pytest.mark.skipif(not HAS_NETWORKX, reason="networkx not installed")
    def test_plot_foodweb_interactive(self):
        """Should return Plotly Figure."""
        import plotly.graph_objects as go

        from pypath.core.plotting import plot_foodweb_interactive

        rpath = MagicMock()
        rpath.NUM_LIVING = 3
        rpath.NUM_DEAD = 0
        rpath.TL = np.array([0, 1.0, 2.0, 3.0])
        rpath.Biomass = np.array([0, 100, 50, 10])
        rpath.DC = np.zeros((4, 4))
        rpath.DC[1, 2] = 0.5
        rpath.DC[2, 3] = 0.5
        rpath.QB = np.array([0, 0, 10, 5])

        fig = plot_foodweb_interactive(rpath)

        assert isinstance(fig, go.Figure)
