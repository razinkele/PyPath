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
