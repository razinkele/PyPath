import numpy as np
import pytest
from pypath.ibm.development import EggParams, accumulate_degree_days, check_hatching, check_thermal_mortality, apply_egg_mortality


def test_egg_params_defaults():
    p = EggParams()
    assert p.dd_hatch == 149.0
    assert p.dd_mortality == 272.4
    assert p.t_zero == 1.8
    assert p.egg_weight == 0.001
    assert p.egg_length_cm == 0.10
    assert p.max_egg_cohorts == 3
    assert p.background_mortality_rate == 0.05
    assert p.o2_lethal == 2.0


def test_degree_day_accumulation_above_t_zero():
    dd = accumulate_degree_days(current_dd=0.0, temperature=9.1, t_zero=1.8, dt_days=1.0)
    assert dd == pytest.approx(7.3, abs=0.01)


def test_degree_day_no_accumulation_below_t_zero():
    dd = accumulate_degree_days(current_dd=50.0, temperature=1.5, t_zero=1.8, dt_days=30.0)
    assert dd == 50.0


def test_degree_day_no_accumulation_at_t_zero():
    dd = accumulate_degree_days(current_dd=50.0, temperature=1.8, t_zero=1.8, dt_days=30.0)
    assert dd == 50.0  # strict >


def test_check_hatching_triggers():
    assert check_hatching(degree_days=149.0, dd_hatch=149.0) is True
    assert check_hatching(degree_days=148.9, dd_hatch=149.0) is False
    assert check_hatching(degree_days=200.0, dd_hatch=149.0) is True


def test_hatching_at_different_temperatures():
    params = EggParams()
    for temp, expected_days in [(5.7, 38.2), (9.1, 20.4), (12.1, 14.5)]:
        dd = 0.0
        days = 0
        while dd < params.dd_hatch:
            dd = accumulate_degree_days(dd, temp, params.t_zero, dt_days=1.0)
            days += 1
        assert days == pytest.approx(expected_days, abs=1.0)


def test_thermal_mortality():
    assert check_thermal_mortality(degree_days=272.4, dd_mortality=272.4) is True
    assert check_thermal_mortality(degree_days=200.0, dd_mortality=272.4) is False


def test_egg_background_mortality():
    n = apply_egg_mortality(
        n_represented=1e6, background_rate=0.05, dt_days=30.0,
        o2=8.0, o2_lethal=2.0, degree_days=50.0, dd_mortality=272.4,
    )
    assert n == pytest.approx(1e6 * np.exp(-0.05 * 30), rel=0.01)


def test_egg_oxygen_mortality():
    n = apply_egg_mortality(
        n_represented=1e6, background_rate=0.0, dt_days=30.0,
        o2=1.0, o2_lethal=2.0, degree_days=50.0, dd_mortality=272.4,
    )
    assert n < 1e6


def test_egg_thermal_mortality_kills_all():
    n = apply_egg_mortality(
        n_represented=1e6, background_rate=0.0, dt_days=1.0,
        o2=8.0, o2_lethal=2.0, degree_days=272.4, dd_mortality=272.4,
    )
    assert n == 0.0


def test_egg_no_mortality_good_conditions():
    n = apply_egg_mortality(
        n_represented=1e6, background_rate=0.0, dt_days=1.0,
        o2=8.0, o2_lethal=2.0, degree_days=50.0, dd_mortality=272.4,
    )
    assert n == 1e6


def test_yolk_sac_params_defaults():
    from pypath.ibm.development import YolkSacParams
    p = YolkSacParams()
    assert p.initial_yolk_kj == 0.15
    assert p.first_feeding_threshold_kj == 0.02
    assert p.minimum_prey_density == 50.0
    assert p.point_of_no_return == 4.0
    assert p.oxycal_kj_per_g_o2 == 13.56
    assert p.background_mortality_rate == 0.02


def test_larval_params_defaults():
    from pypath.ibm.development import LarvalParams
    p = LarvalParams()
    assert p.rs_a_larval == 0.12
    assert p.zooplankton_prey_idx == 1
    assert p.k_half_zoo == 100.0
    assert p.juvenile_length_cm == 2.0
    assert p.w_forage_mid == 2.0
    assert p.w_activity_mid == 5.0
    assert p.ae_min == 0.55
    assert p.ae_max == 0.73
    assert p.cmax_CTO == 18.0
    assert p.zoo_conversion_factor == 1000.0
    assert p.background_mortality_rate == 0.01


def test_oxygen_params_defaults():
    from pypath.ibm.development import OxygenParams
    p = OxygenParams()
    assert p.pcrit_egg == 4.0
    assert p.pcrit_adult == 2.0
    assert p.hypoxia_mortality_rate == 0.5


def test_zone_params_defaults():
    from pypath.ibm.development import ZoneParams
    p = ZoneParams()
    assert p.connectivity.shape == (3, 3)
    assert p.connectivity[0].sum() == pytest.approx(1.0)
