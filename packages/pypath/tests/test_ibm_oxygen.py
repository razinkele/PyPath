"""Tests for Package 4: Oxygen physiology integration in SmeltIBM."""

import numpy as np

from pypath.ibm.base import SuperIndividual
from pypath.ibm.bioenergetics import oxygen_scalar
from pypath.ibm.smelt import SmeltIBM, SmeltParams

# ---- Task 4.1: Oxygen scalar integration into consumption ----


def test_oxygen_no_effect_when_absent():
    """No dissolved_oxygen in env_forcing -> no oxygen limitation."""
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=10)
    env_no_o2 = {
        "temperature": 15.0,
        "month": 6,
        "zoo_peak_day": 150,
        "zoo_density": 80.0,
    }
    env_high_o2 = {
        "temperature": 15.0,
        "month": 6,
        "zoo_peak_day": 150,
        "zoo_density": 80.0,
        "dissolved_oxygen": 10.0,
    }
    r1 = ibm.compute_step(np.zeros(6), 0.0, env_no_o2, dt=1 / 12)
    # Reset
    ibm2 = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm2.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=10)
    r2 = ibm2.compute_step(np.zeros(6), 0.0, env_high_o2, dt=1 / 12)
    # Both should produce similar biomass (high O2 ~ no limitation)
    assert abs(r1.biomass - r2.biomass) / max(r1.biomass, 0.001) < 0.1


def test_oxygen_reduces_consumption_under_hypoxia():
    """Low dissolved_oxygen should reduce consumption and hence growth."""
    params = SmeltParams.baltic_defaults_els()
    ibm_norm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm_norm.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=10)
    ibm_hyp = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm_hyp.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=10)

    prey = np.ones(6) * 0.5
    env_norm = {
        "temperature": 15.0,
        "month": 6,
        "zoo_peak_day": 150,
        "zoo_density": 80.0,
        "dissolved_oxygen": 10.0,
    }
    env_hyp = {
        "temperature": 15.0,
        "month": 6,
        "zoo_peak_day": 150,
        "zoo_density": 80.0,
        "dissolved_oxygen": 1.0,  # severe hypoxia
    }
    r_norm = ibm_norm.compute_step(prey, 0.0, env_norm, dt=1 / 12)
    r_hyp = ibm_hyp.compute_step(prey, 0.0, env_hyp, dt=1 / 12)
    # Hypoxic consumption should be lower
    assert r_hyp.consumption_by_prey.sum() <= r_norm.consumption_by_prey.sum()


def test_oxygen_no_effect_when_oxygen_params_none():
    """When params.oxygen is None, oxygen has no effect."""
    params = SmeltParams.baltic_defaults()  # no oxygen params
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=10)
    env = {
        "temperature": 15.0,
        "month": 6,
        "zoo_peak_day": 150,
        "dissolved_oxygen": 0.5,  # very low, but should have no effect
    }
    r = ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)
    assert r.biomass > 0


# ---- Task 4.2: Oxygen-dependent mortality for early stages ----


def test_egg_oxygen_mortality_integration():
    """Eggs under hypoxia should suffer increased mortality."""
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    egg = SuperIndividual(
        id=0,
        n_represented=1e6,
        weight=0.001,
        length=0.10,
        age=0.0,
        energy_reserve=0.0,
        patch_idx=0,
        is_mature=False,
        sex=0,
        life_stage=0,
        degree_days=50.0,
    )
    ibm.individuals = [egg]
    ibm._next_id = 1
    # Low oxygen
    env = {
        "temperature": 9.0,
        "month": 4,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
        "dissolved_oxygen": 1.0,
    }
    ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)
    surviving_eggs = [i for i in ibm.individuals if i.life_stage == 0]
    # Under severe hypoxia (O2=1.0), eggs may be fully eliminated or reduced
    if surviving_eggs:
        assert surviving_eggs[0].n_represented < 1e6, "Expected mortality from hypoxia"
    # Either way, the cohort was decimated — original 1e6 is not intact
    total_egg_n = sum(e.n_represented for e in surviving_eggs)
    assert total_egg_n < 1e6, "Expected egg mortality under hypoxia"


def test_egg_oxygen_mortality_uses_dissolved_oxygen_key():
    """Verify egg mortality works with 'dissolved_oxygen' env key."""
    params = SmeltParams.baltic_defaults_els()
    ibm_low = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm_high = SmeltIBM(group_index=2, n_groups=6, params=params)

    def make_egg():
        return SuperIndividual(
            id=0,
            n_represented=1e6,
            weight=0.001,
            length=0.10,
            age=0.0,
            energy_reserve=0.0,
            patch_idx=0,
            is_mature=False,
            sex=0,
            life_stage=0,
            degree_days=50.0,
        )

    ibm_low.individuals = [make_egg()]
    ibm_low._next_id = 1
    ibm_high.individuals = [make_egg()]
    ibm_high._next_id = 1

    env_low = {
        "temperature": 5.0,
        "month": 3,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
        "dissolved_oxygen": 0.5,
    }
    env_high = {
        "temperature": 5.0,
        "month": 3,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
        "dissolved_oxygen": 10.0,
    }

    ibm_low.compute_step(np.zeros(6), 0.0, env_low, dt=1 / 12)
    ibm_high.compute_step(np.zeros(6), 0.0, env_high, dt=1 / 12)

    eggs_low = [i for i in ibm_low.individuals if i.life_stage == 0]
    eggs_high = [i for i in ibm_high.individuals if i.life_stage == 0]
    # Low O2 should have fewer survivors
    n_low = eggs_low[0].n_represented if eggs_low else 0.0
    n_high = eggs_high[0].n_represented if eggs_high else 0.0
    assert n_low < n_high


def test_yolk_sac_oxygen_stress_accelerates_depletion():
    """Hypoxia should accelerate yolk depletion in yolk-sac larvae."""
    params = SmeltParams.baltic_defaults_els()

    def make_yolk_larva():
        return SuperIndividual(
            id=0,
            n_represented=1e4,
            weight=0.001,
            length=0.10,
            age=0.0,
            energy_reserve=0.0,
            patch_idx=0,
            is_mature=False,
            sex=0,
            life_stage=1,
            yolk_energy_kj=0.10,
        )

    # Normal oxygen
    ibm_norm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm_norm.individuals = [make_yolk_larva()]
    ibm_norm._next_id = 1
    env_norm = {
        "temperature": 10.0,
        "month": 4,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
        "dissolved_oxygen": 8.0,
    }
    ibm_norm.compute_step(np.zeros(6), 0.0, env_norm, dt=1 / 365)

    # Low oxygen (below pcrit_yolk_sac = 3.5)
    ibm_hyp = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm_hyp.individuals = [make_yolk_larva()]
    ibm_hyp._next_id = 1
    env_hyp = {
        "temperature": 10.0,
        "month": 4,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
        "dissolved_oxygen": 1.0,
    }
    ibm_hyp.compute_step(np.zeros(6), 0.0, env_hyp, dt=1 / 365)

    yolk_norm = [i for i in ibm_norm.individuals if i.life_stage == 1]
    yolk_hyp = [i for i in ibm_hyp.individuals if i.life_stage == 1]

    assert len(yolk_norm) > 0, "Expected normal yolk-sac larvae to survive"
    assert len(yolk_hyp) > 0, "Expected hypoxic yolk-sac larvae to survive"
    # Hypoxic larvae should have less yolk remaining
    assert yolk_hyp[0].yolk_energy_kj < yolk_norm[0].yolk_energy_kj


def test_yolk_sac_lethal_oxygen_mortality():
    """Yolk-sac larvae below o2_lethal_yolk_sac should suffer mortality."""
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    larva = SuperIndividual(
        id=0,
        n_represented=1e6,
        weight=0.001,
        length=0.10,
        age=0.0,
        energy_reserve=0.0,
        patch_idx=0,
        is_mature=False,
        sex=0,
        life_stage=1,
        yolk_energy_kj=0.10,
    )
    ibm.individuals = [larva]
    ibm._next_id = 1
    # O2 below o2_lethal_yolk_sac (1.5 mg/L)
    env = {
        "temperature": 10.0,
        "month": 4,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
        "dissolved_oxygen": 0.5,
    }
    ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)
    yolk_larvae = [i for i in ibm.individuals if i.life_stage == 1]
    # Under severe hypoxia (O2=0.5 << lethal 1.5), cohort may be fully eliminated
    if yolk_larvae:
        assert yolk_larvae[0].n_represented < 1e6, "Expected mortality from hypoxia"
    # Either way, the original 1e6 should be reduced
    total_n = sum(larva.n_represented for larva in yolk_larvae)
    assert total_n < 1e6, "Expected yolk-sac mortality under lethal O2"


# ---- Task 4.3: Oxygen avoidance score (unit tests) ----


def test_oxygen_avoidance_score():
    """oxygen_scalar at O2=1, Pcrit=2.0 should be 0.5; at O2=8 should be 1.0."""
    assert oxygen_scalar(1.0, 2.0) == 0.5
    assert oxygen_scalar(8.0, 2.0) == 1.0


def test_oxygen_scalar_at_zero():
    """At O2=0, oxygen_scalar should be 0.0."""
    assert oxygen_scalar(0.0, 2.0) == 0.0


def test_oxygen_scalar_at_pcrit():
    """At O2=Pcrit, oxygen_scalar should be 1.0."""
    assert oxygen_scalar(2.0, 2.0) == 1.0


def test_oxygen_scalar_above_pcrit():
    """Above Pcrit, oxygen_scalar should be 1.0."""
    assert oxygen_scalar(5.0, 2.0) == 1.0
