"""Unit tests for ecological indicators module."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pypath.core.analysis import calculate_network_indices
from pypath.core.indicators import (
    FlowAnalysis,
    ecosystem_indicators,
    ecosystem_indicators_timeseries,
    finn_cycling_index,
    flow_analysis,
    transfer_efficiency,
)


def _make_rpath_3group():
    """Create a simple 3-group model: producer(1), consumer(2), detritus(3).

    Producer: B=10, PB=2, QB=0 (type=1, producer)
    Consumer: B=5, PB=0.5, QB=2 (type=0, consumer), eats 100% producer
    Detritus: B=3, PB=0, QB=0 (type=2)

    Diet: consumer eats 100% producer.
    Landings: consumer caught at 0.5 by one fleet.
    """
    rpath = MagicMock()
    rpath.NUM_LIVING = 2
    rpath.NUM_DEAD = 1
    rpath.NUM_GEARS = 1

    # All arrays are 1-based (index 0 unused)
    rpath.Biomass = np.array([0.0, 10.0, 5.0, 3.0])
    rpath.PB = np.array([0.0, 2.0, 0.5, 0.0])
    rpath.QB = np.array([0.0, 0.0, 2.0, 0.0])
    rpath.EE = np.array([0.0, 0.8, 0.7, 0.5])
    rpath.Unassim = np.array([0.0, 0.0, 0.2, 0.0])
    rpath.TL = np.array([0.0, 1.0, 2.0, 1.0])
    rpath.type = np.array([0, 1, 0, 2])  # producer, consumer, detritus

    # DC[prey, pred]: consumer(2) eats 100% producer(1)
    rpath.DC = np.zeros((4, 4))
    rpath.DC[1, 2] = 1.0  # prey=1 (producer), pred=2 (consumer)

    # Landings/Discards: [groups+1, gears+1], 1-based
    rpath.Landings = np.zeros((4, 2))
    rpath.Landings[2, 1] = 0.5  # consumer caught by fleet 1
    rpath.Discards = np.zeros((4, 2))

    return rpath


class TestFlowAnalysis:
    """Tests for flow_analysis() function."""

    def test_tst_positive(self):
        """TST should be positive for any model with flows."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert result.total_system_throughput > 0

    def test_tst_manual_calculation(self):
        """TST should equal sum of all flows.

        Consumer consumption = QB[2]*B[2] = 2*5 = 10
        Consumer respiration = (1-Unassim[2])*QB[2]*B[2] - PB[2]*B[2]
                             = 0.8*10 - 2.5 = 5.5
        Consumer flow to detritus:
            unassim part = Unassim[2]*QB[2]*B[2] = 0.2*10 = 2.0
            non-EE part = (1-EE[2])*PB[2]*B[2] = 0.3*2.5 = 0.75
            total FD = 2.75
        Producer flow to detritus:
            (no consumption, QB=0) => unassim part = 0
            non-EE part = (1-EE[1])*PB[1]*B[1] = 0.2*20 = 4.0
            total FD = 4.0
        Detritus flow to detritus:
            (1-EE[3])*PB[3]*B[3] = 0.5*0*3 = 0
        Export (catch) = 0.5

        TST = consumption(10) + respiration(5.5) + FD(2.75+4.0+0) + export(0.5)
            = 22.75
        """
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert abs(result.total_system_throughput - 22.75) < 0.01

    def test_ascendency_positive(self):
        """Ascendency should be > 0 for a model with flows."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert result.ascendency > 0

    def test_ascendency_less_than_capacity(self):
        """Ascendency must always be <= Capacity."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert result.ascendency <= result.capacity + 1e-10

    def test_overhead_equals_capacity_minus_ascendency(self):
        """Overhead = Capacity - Ascendency."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert abs(result.overhead - (result.capacity - result.ascendency)) < 1e-10

    def test_relative_ascendency_in_unit_interval(self):
        """Relative ascendency should be in [0, 1]."""
        rpath = _make_rpath_3group()
        result = flow_analysis(rpath)
        assert 0 <= result.relative_ascendency <= 1

    def test_single_group_no_crash(self):
        """Single producer group should not crash."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 1
        rpath.NUM_DEAD = 0
        rpath.NUM_GEARS = 0
        rpath.Biomass = np.array([0.0, 5.0])
        rpath.PB = np.array([0.0, 1.0])
        rpath.QB = np.array([0.0, 0.0])
        rpath.EE = np.array([0.0, 0.5])
        rpath.Unassim = np.array([0.0, 0.0])
        rpath.TL = np.array([0.0, 1.0])
        rpath.type = np.array([0, 1])
        rpath.DC = np.zeros((2, 2))
        rpath.Landings = np.zeros((2, 1))
        rpath.Discards = np.zeros((2, 1))
        result = flow_analysis(rpath)
        assert isinstance(result, FlowAnalysis)

    def test_zero_biomass_returns_defaults(self):
        """All-zero biomass model returns zero TST."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 2
        rpath.NUM_DEAD = 1
        rpath.NUM_GEARS = 0
        rpath.Biomass = np.zeros(4)
        rpath.PB = np.zeros(4)
        rpath.QB = np.zeros(4)
        rpath.EE = np.zeros(4)
        rpath.Unassim = np.zeros(4)
        rpath.TL = np.zeros(4)
        rpath.type = np.array([0, 0, 0, 2])
        rpath.DC = np.zeros((4, 4))
        rpath.Landings = np.zeros((4, 1))
        rpath.Discards = np.zeros((4, 1))
        result = flow_analysis(rpath)
        assert result.total_system_throughput == 0.0
        assert result.ascendency == 0.0
        assert result.capacity == 0.0


class TestFinnCyclingIndex:
    """Tests for finn_cycling_index() function."""

    def test_linear_chain_no_cycling(self):
        """Linear chain (no recycling) should have FCI = 0."""
        rpath = _make_rpath_3group()
        fci = finn_cycling_index(rpath)
        assert fci == pytest.approx(0.0, abs=1e-10)

    def test_detritus_feedback_positive_cycling(self):
        """Detritus feeding back to consumer should give FCI > 0."""
        rpath = _make_rpath_3group()
        rpath.DC[1, 2] = 0.8  # prey=producer, pred=consumer
        rpath.DC[3, 2] = 0.2  # prey=detritus, pred=consumer
        fci = finn_cycling_index(rpath)
        assert fci > 0.0

    def test_fci_in_unit_interval(self):
        """FCI should be in [0, 1]."""
        rpath = _make_rpath_3group()
        rpath.DC[1, 2] = 0.8
        rpath.DC[3, 2] = 0.2
        fci = finn_cycling_index(rpath)
        assert 0 <= fci <= 1

    def test_fci_matches_flow_analysis(self):
        """finn_cycling_index() should match flow_analysis().finn_cycling_index."""
        rpath = _make_rpath_3group()
        rpath.DC[1, 2] = 0.8
        rpath.DC[3, 2] = 0.2
        fci_standalone = finn_cycling_index(rpath)
        fa = flow_analysis(rpath)
        assert fci_standalone == pytest.approx(fa.finn_cycling_index, abs=1e-10)


class TestTransferEfficiency:
    """Tests for transfer_efficiency() function."""

    def test_returns_array(self):
        """Should return numpy array."""
        rpath = _make_rpath_3group()
        te = transfer_efficiency(rpath)
        assert isinstance(te, np.ndarray)

    def test_values_in_unit_interval(self):
        """All TE values should be in [0, 1]."""
        rpath = _make_rpath_3group()
        te = transfer_efficiency(rpath)
        for val in te:
            assert 0 <= val <= 1

    def test_matches_flow_analysis(self):
        """transfer_efficiency() should match flow_analysis().transfer_efficiency."""
        rpath = _make_rpath_3group()
        te_standalone = transfer_efficiency(rpath)
        fa = flow_analysis(rpath)
        np.testing.assert_array_almost_equal(te_standalone, fa.transfer_efficiency)

    def test_single_tl_returns_empty(self):
        """Model with only TL=1 groups has no transfer to compute."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 1
        rpath.NUM_DEAD = 0
        rpath.NUM_GEARS = 0
        rpath.Biomass = np.array([0.0, 5.0])
        rpath.PB = np.array([0.0, 1.0])
        rpath.QB = np.array([0.0, 0.0])
        rpath.EE = np.array([0.0, 0.5])
        rpath.Unassim = np.array([0.0, 0.0])
        rpath.TL = np.array([0.0, 1.0])
        rpath.type = np.array([0, 1])
        rpath.DC = np.zeros((2, 2))
        rpath.Landings = np.zeros((2, 1))
        rpath.Discards = np.zeros((2, 1))
        te = transfer_efficiency(rpath)
        assert len(te) == 0


def _make_rpath_5group():
    """Create a 5-group model for ecosystem indicator tests.

    Groups:
    1: Phytoplankton (producer, TL=1.0, B=20, PB=50, QB=0)
    2: Zooplankton (consumer, TL=2.0, B=10, PB=10, QB=30)
    3: Small fish (consumer, TL=3.0, B=5, PB=1, QB=5)
    4: Large fish (consumer, TL=4.0, B=2, PB=0.3, QB=1.5)
    5: Detritus (type=2, TL=1.0, B=5)
    """
    rpath = MagicMock()
    rpath.NUM_LIVING = 4
    rpath.NUM_DEAD = 1
    rpath.NUM_GEARS = 1

    rpath.Biomass = np.array([0.0, 20.0, 10.0, 5.0, 2.0, 5.0])
    rpath.PB = np.array([0.0, 50.0, 10.0, 1.0, 0.3, 0.0])
    rpath.QB = np.array([0.0, 0.0, 30.0, 5.0, 1.5, 0.0])
    rpath.EE = np.array([0.0, 0.8, 0.7, 0.6, 0.5, 0.5])
    rpath.Unassim = np.array([0.0, 0.0, 0.2, 0.2, 0.2, 0.0])
    rpath.TL = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 1.0])
    rpath.type = np.array([0, 1, 0, 0, 0, 2])

    rpath.DC = np.zeros((6, 6))
    rpath.DC[1, 2] = 1.0  # zoo eats phyto
    rpath.DC[2, 3] = 1.0  # small fish eats zoo
    rpath.DC[3, 4] = 1.0  # large fish eats small fish

    # Fleet catches small fish (0.5) and large fish (0.3)
    rpath.Landings = np.zeros((6, 2))
    rpath.Landings[3, 1] = 0.5  # small fish landings
    rpath.Landings[4, 1] = 0.3  # large fish landings
    rpath.Discards = np.zeros((6, 2))

    return rpath


class TestEcosystemIndicators:
    """Tests for ecosystem_indicators() function."""

    def test_mtl_catch_weighted(self):
        """MTL catch = Σ(TL*Catch) / Σ(Catch).

        Small fish: TL=3.0, Catch=0.5
        Large fish: TL=4.0, Catch=0.3
        MTL = (3.0*0.5 + 4.0*0.3) / (0.5 + 0.3) = 2.7/0.8 = 3.375
        """
        rpath = _make_rpath_5group()
        result = ecosystem_indicators(rpath)
        assert result.mtl_catch == pytest.approx(3.375, abs=1e-10)

    def test_mti_excludes_low_tl(self):
        """Marine Trophic Index excludes groups with TL < 3.25.

        Only large fish (TL=4.0, Catch=0.3) qualifies.
        MTI = 4.0
        """
        rpath = _make_rpath_5group()
        result = ecosystem_indicators(rpath)
        assert result.marine_trophic_index == pytest.approx(4.0, abs=1e-10)

    def test_mti_nan_when_no_groups_above_cutoff(self):
        """MTI should be NaN when no groups have TL >= 3.25."""
        rpath = _make_rpath_3group()  # max TL=2.0
        result = ecosystem_indicators(rpath)
        assert np.isnan(result.marine_trophic_index)

    def test_catch_biomass_ratio(self):
        """Catch/Biomass = total catch / total living biomass.

        Catch = 0.5 + 0.3 = 0.8
        Living biomass = 20 + 10 + 5 + 2 = 37
        Ratio = 0.8/37 ≈ 0.02162
        """
        rpath = _make_rpath_5group()
        result = ecosystem_indicators(rpath)
        assert result.catch_biomass_ratio == pytest.approx(0.8 / 37.0, abs=1e-10)

    def test_gross_efficiency(self):
        """Gross efficiency = total catch / NPP.

        NPP = PB[1]*B[1] = 50*20 = 1000 (only phytoplankton is producer)
        Catch = 0.8
        GE = 0.8/1000 = 0.0008
        """
        rpath = _make_rpath_5group()
        result = ecosystem_indicators(rpath)
        assert result.gross_efficiency == pytest.approx(0.0008, abs=1e-10)

    def test_shannon_diversity_equal_biomass(self):
        """Shannon diversity of n equal-biomass groups ≈ ln(n)."""
        rpath = MagicMock()
        rpath.NUM_LIVING = 4
        rpath.NUM_DEAD = 0
        rpath.NUM_GEARS = 0
        rpath.Biomass = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        rpath.PB = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        rpath.QB = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        rpath.TL = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        rpath.type = np.array([0, 1, 0, 0, 0])
        rpath.Landings = np.zeros((5, 1))
        rpath.Discards = np.zeros((5, 1))
        rpath.EE = np.zeros(5)
        rpath.Unassim = np.zeros(5)
        rpath.DC = np.zeros((5, 5))
        result = ecosystem_indicators(rpath)
        assert result.shannon_diversity == pytest.approx(np.log(4), abs=0.01)

    def test_kempton_q_few_groups(self):
        """Kempton Q returns NaN when fewer than 4 groups in TL 3-4."""
        rpath = _make_rpath_3group()  # only TL 1 and 2
        result = ecosystem_indicators(rpath)
        assert np.isnan(result.kempton_q)

    def test_zero_catch_mtl_nan(self):
        """MTL catch should be NaN when total catch is 0."""
        rpath = _make_rpath_3group()
        rpath.Landings = np.zeros((4, 2))
        rpath.Discards = np.zeros((4, 2))
        result = ecosystem_indicators(rpath)
        assert np.isnan(result.mtl_catch)


class TestEcosystemIndicatorsTimeseries:
    """Tests for ecosystem_indicators_timeseries() function."""

    def _make_ecosim_output(self, n_years=5, n_groups=5):
        """Create mock RsimOutput with annual arrays.

        n_groups must match rpath's NUM_LIVING + NUM_DEAD (5 for _make_rpath_5group).
        Arrays are 1-based: shape (n_years, n_groups+1).
        """
        output = MagicMock()
        output.annual_Biomass = np.ones((n_years, n_groups + 1)) * 5.0
        output.annual_Biomass[:, 0] = 0.0  # index 0 unused
        # Vary biomass over time for group 1
        for yr in range(n_years):
            output.annual_Biomass[yr, 1] = 20.0 - yr * 2  # declining

        output.annual_Catch = np.zeros((n_years, n_groups + 1))
        output.annual_Catch[:, 3] = 0.5  # small fish catch
        output.annual_Catch[:, 4] = 0.3  # large fish catch
        return output

    def _make_scenario(self, n_years=5, n_groups=5):
        """Create mock RsimScenario (n_groups matches rpath)."""
        scenario = MagicMock()
        scenario.params = MagicMock()
        scenario.params.NUM_GROUPS = n_groups
        scenario.params.NUM_LIVING = n_groups - 1
        scenario.params.NUM_DEAD = 1
        return scenario

    def test_returns_dataframe(self):
        """Should return a pandas DataFrame."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output()
        scenario = self._make_scenario()
        result = ecosystem_indicators_timeseries(output, scenario, rpath)
        assert isinstance(result, pd.DataFrame)

    def test_correct_columns(self):
        """DataFrame should have expected columns."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output()
        scenario = self._make_scenario()
        result = ecosystem_indicators_timeseries(output, scenario, rpath)
        expected_cols = {
            "year",
            "mtl_catch",
            "marine_trophic_index",
            "catch_biomass_ratio",
            "gross_efficiency",
            "shannon_diversity",
        }
        assert set(result.columns) == expected_cols

    def test_correct_row_count(self):
        """Should have one row per year."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output(n_years=10)
        scenario = self._make_scenario()
        result = ecosystem_indicators_timeseries(output, scenario, rpath)
        assert len(result) == 10

    def test_values_change_over_time(self):
        """Shannon diversity should change when biomass varies."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output()
        scenario = self._make_scenario()
        result = ecosystem_indicators_timeseries(output, scenario, rpath)
        # Biomass of group 1 declines, so diversity changes
        assert (
            result["shannon_diversity"].iloc[0] != result["shannon_diversity"].iloc[-1]
        )

    def test_consistent_with_static_at_t0(self):
        """Timeseries year 0 should match static indicators when biomass matches."""
        rpath = _make_rpath_5group()
        output = self._make_ecosim_output(n_years=1)
        scenario = self._make_scenario()
        # Set annual biomass to match rpath.Biomass exactly
        for i in range(1, rpath.NUM_LIVING + rpath.NUM_DEAD + 1):
            output.annual_Biomass[0, i] = rpath.Biomass[i]
        # Set annual catch to match rpath landings+discards
        for i in range(1, rpath.NUM_LIVING + rpath.NUM_DEAD + 1):
            output.annual_Catch[0, i] = np.sum(rpath.Landings[i, 1:]) + np.sum(
                rpath.Discards[i, 1:]
            )
        ts = ecosystem_indicators_timeseries(output, scenario, rpath)
        static = ecosystem_indicators(rpath)
        assert ts["mtl_catch"].iloc[0] == pytest.approx(static.mtl_catch, abs=1e-10)
        assert ts["shannon_diversity"].iloc[0] == pytest.approx(
            static.shannon_diversity, abs=1e-10
        )


class TestIntegration:
    """Tests for integration with analysis.py."""

    def test_network_indices_transfer_efficiency_not_placeholder(self):
        """calculate_network_indices() should return computed TE, not 0.1 placeholder."""
        rpath = _make_rpath_5group()
        indices = calculate_network_indices(rpath)
        # 5-group model has groups at TL 1,2,3,4 so TE should be meaningful
        assert indices.transfer_efficiency != 0.1
        assert indices.transfer_efficiency >= 0.0

    def test_network_indices_finn_cycling_not_placeholder(self):
        """calculate_network_indices() should compute FCI, not return 0.0 placeholder."""
        rpath = _make_rpath_3group()
        # Add detritus feedback to create cycling
        rpath.DC[1, 2] = 0.8
        rpath.DC[3, 2] = 0.2
        indices = calculate_network_indices(rpath)
        assert indices.finn_cycling_index > 0.0
