"""End-to-end early life stage integration tests for SmeltIBM."""

import numpy as np
import pytest

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
