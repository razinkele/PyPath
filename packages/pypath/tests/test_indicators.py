"""Unit tests for ecological indicators module."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from pypath.core.indicators import FlowAnalysis, finn_cycling_index, flow_analysis


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
        rpath.DC[1, 2] = 0.8   # prey=producer, pred=consumer
        rpath.DC[3, 2] = 0.2   # prey=detritus, pred=consumer
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
