"""End-to-end early life stage integration tests for SmeltIBM."""

import numpy as np
import pytest

from pypath.ibm.base import SuperIndividual
from pypath.ibm.smelt import SmeltIBM, SmeltParams


def test_spawning_produces_eggs():
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=50)
    env = {
        "temperature": 8.0,
        "month": 4,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
    }
    result = ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)
    eggs = [i for i in ibm.individuals if i.life_stage == 0]
    assert len(eggs) <= params.egg.max_egg_cohorts
    if eggs:
        assert eggs[0].weight == params.egg.egg_weight
        assert eggs[0].length == params.egg.egg_length_cm


def test_existing_behavior_when_els_disabled():
    params = SmeltParams.baltic_defaults()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    ibm.initialize_from_ecosim(biomass=1.0, params={}, n_super_individuals=50)
    env = {"temperature": 8.0, "month": 4, "zoo_peak_day": 120}
    result = ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)
    eggs = [i for i in ibm.individuals if i.life_stage == 0]
    assert len(eggs) == 0


def test_egg_degree_day_accumulation():
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
        degree_days=0.0,
    )
    ibm.individuals = [egg]
    ibm._next_id = 1
    env = {
        "temperature": 9.1,
        "month": 4,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
    }
    ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)
    # After 1 month at 9.1C: DD = (9.1-1.8) * 30.4 ~ 221.9 > 149 -> should hatch
    hatched = [i for i in ibm.individuals if i.life_stage == 1]
    assert len(hatched) > 0


def test_egg_no_hatching_below_threshold():
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
        degree_days=0.0,
    )
    ibm.individuals = [egg]
    ibm._next_id = 1
    env = {
        "temperature": 1.5,
        "month": 3,
        "zoo_peak_day": 120,
        "zoo_density": 80.0,
    }
    ibm.compute_step(np.zeros(6), 0.0, env, dt=1 / 12)
    eggs = [i for i in ibm.individuals if i.life_stage == 0]
    assert len(eggs) == 1
    assert eggs[0].degree_days == 0.0


def test_population_cap_consolidation():
    params = SmeltParams.baltic_defaults_els()
    params.max_super_individuals = 10
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    for i in range(15):
        ibm.individuals.append(
            SuperIndividual(
                id=i,
                n_represented=1000.0,
                weight=0.001,
                length=0.10,
                age=0.0,
                energy_reserve=0.0,
                patch_idx=0,
                is_mature=False,
                sex=0,
                life_stage=0,
            )
        )
    ibm._next_id = 15
    biomass_before = sum(i.n_represented * i.weight for i in ibm.individuals)
    ibm._consolidate_population()
    assert len(ibm.individuals) <= params.max_super_individuals
    biomass_after = sum(i.n_represented * i.weight for i in ibm.individuals)
    assert biomass_after == pytest.approx(biomass_before, rel=1e-10)
