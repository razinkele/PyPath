"""Tests for pypath.core.constants module."""

import pytest

from pypath.core import constants


class TestPhysicalConstants:
    def test_km_per_degree_lat_reasonable(self):
        assert 100 < constants.KM_PER_DEGREE_LAT < 120


class TestBiologicalConstants:
    def test_vulnerability_bounds(self):
        assert constants.MIN_VULNERABILITY < constants.DEFAULT_VULNERABILITY
        assert constants.DEFAULT_VULNERABILITY < constants.MAX_VULNERABILITY_SAFE

    def test_prey_switching_bounds(self):
        assert (
            constants.MIN_PREY_SWITCHING_POWER < constants.DEFAULT_PREY_SWITCHING_POWER
        )
        assert (
            constants.DEFAULT_PREY_SWITCHING_POWER < constants.MAX_PREY_SWITCHING_POWER
        )


class TestNumericalThresholds:
    def test_biomass_thresholds_ordered(self):
        assert (
            constants.MIN_BIOMASS_CRASH_THRESHOLD
            < constants.MIN_BIOMASS_VIABLE
            < constants.MIN_BIOMASS_RECOVERY_THRESHOLD
        )

    def test_ee_bounds(self):
        assert constants.MIN_EE == 0.0
        assert constants.MAX_EE == 1.0
        assert constants.MIN_EE < constants.MAX_EE_WARNING < constants.MAX_EE

    def test_ge_bounds_consistent_with_qb_pb(self):
        # GE = PB/QB, so MIN_GE = 1/MAX_QB_PB_RATIO, MAX_GE = 1/MIN_QB_PB_RATIO
        assert constants.MIN_GE_CONSUMER == pytest.approx(
            1.0 / constants.MAX_QB_PB_RATIO, rel=0.1
        )
        assert constants.MAX_GE_CONSUMER == pytest.approx(
            1.0 / constants.MIN_QB_PB_RATIO, rel=0.1
        )

    def test_epsilon_positive_and_small(self):
        assert 0 < constants.EPSILON < 1e-6
        assert 0 < constants.BALANCE_TOLERANCE < 1e-3


class TestSimulationParameters:
    def test_months_per_year(self):
        assert constants.MONTHS_PER_YEAR == 12
        assert constants.STEPS_PER_YEAR_MONTHLY == 12

    def test_default_simulation_durations(self):
        assert constants.DEFAULT_SIMULATION_MONTHS > 0
        assert constants.DEFAULT_SIMULATION_YEARS > 0


class TestParameterBounds:
    def test_pb_bounds(self):
        assert constants.MIN_PB < constants.MAX_PB_CONSUMER

    def test_qb_bounds(self):
        assert constants.MIN_QB < constants.MAX_QB

    def test_biomass_bounds(self):
        assert constants.MIN_BIOMASS < constants.MAX_BIOMASS


class TestFileConstants:
    def test_valid_db_extensions(self):
        assert ".ewemdb" in constants.VALID_DB_EXTENSIONS
        assert ".accdb" in constants.VALID_DB_EXTENSIONS
