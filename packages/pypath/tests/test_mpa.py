"""Tests for pypath.spatial.mpa module."""

import numpy as np
import pytest

from pypath.spatial.mpa import (
    MPAZone,
    MPAConfig,
    create_mpa_config,
)


class TestMPAZone:
    def test_construction_defaults(self):
        z = MPAZone(mpa_id=1, name="Reserve A", patches=[0, 1, 2])
        assert z.mpa_id == 1
        assert z.name == "Reserve A"
        assert z.patches == [0, 1, 2]
        assert z.start_month == 0
        assert z.end_month is None
        assert z.excluded_fleets is None
        assert z.capacity_bonus == 1.0

    def test_construction_full(self):
        z = MPAZone(
            mpa_id=2,
            name="Seasonal",
            patches=[3, 4],
            start_month=6,
            end_month=18,
            excluded_fleets=[0, 2],
            capacity_bonus=1.3,
        )
        assert z.start_month == 6
        assert z.end_month == 18
        assert z.excluded_fleets == [0, 2]
        assert z.capacity_bonus == 1.3


class TestCreateMPAConfig:
    def test_default_empty(self):
        cfg = create_mpa_config()
        assert cfg.zones == []

    def test_with_zones(self):
        zones = [MPAZone(mpa_id=1, name="A", patches=[0])]
        cfg = create_mpa_config(zones)
        assert len(cfg.zones) == 1


class TestMPAConfigGetActiveZones:
    def test_permanent_zone_always_active(self):
        z = MPAZone(mpa_id=1, name="Perm", patches=[0])
        cfg = MPAConfig(zones=[z])
        assert z in cfg.get_active_zones(0)
        assert z in cfg.get_active_zones(100)

    def test_temporal_zone_active_in_window(self):
        z = MPAZone(mpa_id=1, name="Temp", patches=[0], start_month=6, end_month=18)
        cfg = MPAConfig(zones=[z])
        assert z not in cfg.get_active_zones(5)
        assert z in cfg.get_active_zones(6)
        assert z in cfg.get_active_zones(17)
        assert z not in cfg.get_active_zones(18)

    def test_empty_config(self):
        cfg = MPAConfig(zones=[])
        assert cfg.get_active_zones(0) == []


class TestMPAConfigIsClosed:
    def test_no_take_zone(self):
        """excluded_fleets=None means all fleets excluded."""
        z = MPAZone(mpa_id=1, name="NoTake", patches=[0, 1])
        cfg = MPAConfig(zones=[z])
        assert cfg.is_closed(0, 0, 0) is True
        assert cfg.is_closed(0, 5, 0) is True
        assert cfg.is_closed(2, 0, 0) is False  # patch 2 not in MPA

    def test_fleet_selective(self):
        z = MPAZone(mpa_id=1, name="Sel", patches=[0], excluded_fleets=[1])
        cfg = MPAConfig(zones=[z])
        assert cfg.is_closed(0, 0, 0) is False  # fleet 0 not excluded
        assert cfg.is_closed(0, 1, 0) is True  # fleet 1 excluded

    def test_inactive_zone_not_closed(self):
        z = MPAZone(mpa_id=1, name="Future", patches=[0], start_month=12)
        cfg = MPAConfig(zones=[z])
        assert cfg.is_closed(0, 0, 5) is False  # before activation


class TestMPAConfigGetEffortMask:
    def test_no_mpa_all_open(self):
        cfg = MPAConfig(zones=[])
        mask = cfg.get_effort_mask(5, 2, 0)
        assert mask.shape == (5, 2)
        np.testing.assert_array_equal(mask, 1.0)

    def test_no_take_zeros_all_fleets(self):
        z = MPAZone(mpa_id=1, name="NoTake", patches=[1, 2])
        cfg = MPAConfig(zones=[z])
        mask = cfg.get_effort_mask(5, 3, 0)
        # Patches 1, 2 closed to all fleets
        np.testing.assert_array_equal(mask[0, :], 1.0)
        np.testing.assert_array_equal(mask[1, :], 0.0)
        np.testing.assert_array_equal(mask[2, :], 0.0)
        np.testing.assert_array_equal(mask[3, :], 1.0)
        np.testing.assert_array_equal(mask[4, :], 1.0)

    def test_fleet_selective_mask(self):
        z = MPAZone(mpa_id=1, name="Sel", patches=[0], excluded_fleets=[1])
        cfg = MPAConfig(zones=[z])
        mask = cfg.get_effort_mask(3, 3, 0)
        assert mask[0, 0] == 1.0  # fleet 0 open
        assert mask[0, 1] == 0.0  # fleet 1 closed
        assert mask[0, 2] == 1.0  # fleet 2 open
        np.testing.assert_array_equal(mask[1, :], 1.0)  # other patches open

    def test_overlapping_mpas(self):
        z1 = MPAZone(mpa_id=1, name="A", patches=[0], excluded_fleets=[0])
        z2 = MPAZone(mpa_id=2, name="B", patches=[0], excluded_fleets=[1])
        cfg = MPAConfig(zones=[z1, z2])
        mask = cfg.get_effort_mask(3, 3, 0)
        assert mask[0, 0] == 0.0  # fleet 0 closed by A
        assert mask[0, 1] == 0.0  # fleet 1 closed by B
        assert mask[0, 2] == 1.0  # fleet 2 open

    def test_out_of_range_patch_skipped(self):
        """Patches outside [0, n_patches) are silently skipped."""
        z = MPAZone(mpa_id=1, name="Bad", patches=[0, 99])
        cfg = MPAConfig(zones=[z])
        mask = cfg.get_effort_mask(5, 2, 0)
        assert mask[0, 0] == 0.0  # patch 0 closed
        np.testing.assert_array_equal(mask[4, :], 1.0)  # patch 4 open

    def test_returns_float_array(self):
        cfg = MPAConfig(zones=[])
        mask = cfg.get_effort_mask(3, 2, 0)
        assert mask.dtype == np.float64


class TestMPAConfigGetCapacityMultipliers:
    def test_no_mpa_all_ones(self):
        cfg = MPAConfig(zones=[])
        mult = cfg.get_capacity_multipliers(5, 0)
        assert mult.shape == (5,)
        np.testing.assert_array_equal(mult, 1.0)

    def test_single_zone_bonus(self):
        z = MPAZone(mpa_id=1, name="R", patches=[1, 2], capacity_bonus=1.3)
        cfg = MPAConfig(zones=[z])
        mult = cfg.get_capacity_multipliers(5, 0)
        assert mult[0] == 1.0
        assert mult[1] == pytest.approx(1.3)
        assert mult[2] == pytest.approx(1.3)
        assert mult[3] == 1.0

    def test_overlapping_zones_multiply(self):
        z1 = MPAZone(mpa_id=1, name="A", patches=[0], capacity_bonus=1.3)
        z2 = MPAZone(mpa_id=2, name="B", patches=[0], capacity_bonus=1.2)
        cfg = MPAConfig(zones=[z1, z2])
        mult = cfg.get_capacity_multipliers(3, 0)
        assert mult[0] == pytest.approx(1.3 * 1.2)

    def test_no_bonus_zone_returns_one(self):
        z = MPAZone(mpa_id=1, name="NB", patches=[0])  # default bonus=1.0
        cfg = MPAConfig(zones=[z])
        mult = cfg.get_capacity_multipliers(3, 0)
        np.testing.assert_array_equal(mult, 1.0)

    def test_inactive_zone_ignored(self):
        z = MPAZone(
            mpa_id=1, name="Future", patches=[0], start_month=12, capacity_bonus=1.5
        )
        cfg = MPAConfig(zones=[z])
        mult = cfg.get_capacity_multipliers(3, 5)  # before activation
        np.testing.assert_array_equal(mult, 1.0)
