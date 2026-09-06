"""Tests for Ftime (foraging time) dynamic adjustment matching Rpath."""

import pytest

from pypath.core.ecosim import _ftime_update_rpath


class TestFtimeUpdateFormula:
    def test_ftime_rpath_formula_basic(self):
        """Ftime update should match Rpath: 0.1 + 0.9*Ft*((1-adj) + adj*QBopt/(FG/B))."""
        old_ftime = 1.0
        ftadj = 0.5
        qbopt = 5.0
        food_gain = 40.0
        biomass = 10.0

        actual_qb = food_gain / biomass  # = 4.0
        # 0.1 + 0.9 * 1.0 * (0.5 + 0.5 * 5.0 / 4.0) = 0.1 + 0.9 * 1.125 = 1.1125
        expected = 0.1 + 0.9 * 1.0 * (0.5 + 0.5 * qbopt / actual_qb)

        result = _ftime_update_rpath(old_ftime, ftadj, qbopt, food_gain, biomass)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_ftime_at_equilibrium(self):
        """At equilibrium (FoodGain/B == QBopt), Ftime should stay near 1.0."""
        result = _ftime_update_rpath(1.0, 0.5, 25.0, 250.0, 10.0)
        # actual_qb = 250/10 = 25 = qbopt, so ratio = 1.0
        # 0.1 + 0.9 * 1.0 * (0.5 + 0.5*1.0) = 0.1 + 0.9 = 1.0
        assert result == pytest.approx(1.0, rel=1e-6)

    def test_ftime_cap_at_2(self):
        """Rpath caps Ftime at 2.0."""
        # Very low consumption -> Ftime tries to increase a lot
        result = _ftime_update_rpath(1.5, 0.9, 50.0, 1.0, 10.0)
        assert result <= 2.0

    def test_ftime_floor_at_01(self):
        """Rpath has implicit floor at 0.1 (the constant term)."""
        # Very high consumption -> ratio is tiny
        result = _ftime_update_rpath(0.5, 0.5, 5.0, 10000.0, 1.0)
        # actual_qb = 10000, qbopt=5, ratio ≈ 0.0005
        # 0.1 + 0.9 * 0.5 * (0.5 + 0.5*0.0005) ≈ 0.1 + 0.225 ≈ 0.325
        assert result >= 0.1

    def test_ftime_no_update_when_no_food(self):
        """When food_gain=0, Ftime should not change."""
        result = _ftime_update_rpath(1.5, 0.5, 25.0, 0.0, 10.0)
        assert result == 1.5

    def test_ftime_no_update_when_ftadj_zero(self):
        """When ftadj=0, Ftime should stay constant (no adjustment)."""
        result = _ftime_update_rpath(1.0, 0.0, 25.0, 30.0, 10.0)
        # 0.1 + 0.9 * 1.0 * (1.0 + 0.0) = 0.1 + 0.9 = 1.0
        assert result == pytest.approx(1.0, rel=1e-6)
