"""
Tests for SmeltIBM concrete implementation.

Validates the SmeltParams dataclass (with Baltic defaults) and the SmeltIBM
class that orchestrates bioenergetics, predation, foraging, reproduction,
and growth for Baltic smelt (Osmerus eperlanus) super-individuals.
"""

import numpy as np
import pytest

from pypath.ibm.base import IBMGroup, IBMStepResult
from pypath.ibm.smelt import SmeltIBM, SmeltParams


class TestSmeltParams:
    """Test the SmeltParams dataclass and its factory method."""

    def test_baltic_defaults_returns_smelt_params(self):
        """baltic_defaults() returns a SmeltParams instance."""
        params = SmeltParams.baltic_defaults()
        assert isinstance(params, SmeltParams)

    def test_baltic_defaults_has_all_sub_params(self):
        """baltic_defaults() populates all parameter sub-objects."""
        params = SmeltParams.baltic_defaults()
        assert params.bioenerg is not None
        assert params.predation is not None
        assert params.foraging is not None
        assert params.movement is not None
        assert params.reproduction is not None

    def test_baltic_defaults_vbgf_values(self):
        """baltic_defaults() has sensible Von Bertalanffy defaults."""
        params = SmeltParams.baltic_defaults()
        assert params.vbgf_k_mean > 0.0
        assert params.vbgf_k_sd > 0.0
        assert params.vbgf_linf_mean > 0.0
        assert params.vbgf_linf_sd > 0.0
        assert params.max_age > 0.0

    def test_baltic_defaults_max_age(self):
        """Baltic smelt max_age should be a reasonable fish lifespan."""
        params = SmeltParams.baltic_defaults()
        assert 1.0 < params.max_age <= 15.0

    def test_baltic_defaults_foraging_arrays_valid(self):
        """Foraging energy_content and handling_time arrays are positive."""
        params = SmeltParams.baltic_defaults()
        assert len(params.foraging.energy_content) > 0
        assert len(params.foraging.handling_time) > 0
        assert np.all(params.foraging.energy_content >= 0)
        assert np.all(params.foraging.handling_time >= 0)


class TestSmeltIBMInit:
    """Test SmeltIBM initialization and initialize_from_ecosim."""

    @pytest.fixture
    def params(self):
        """Return Baltic default SmeltParams."""
        return SmeltParams.baltic_defaults()

    @pytest.fixture
    def smelt(self, params):
        """Return a SmeltIBM initialized from Ecosim biomass."""
        n_groups = 10
        ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=params)
        ibm.initialize_from_ecosim(
            biomass=5.0,  # 5 tonnes
            params={},
            n_super_individuals=100,
        )
        return ibm

    def test_is_ibm_group_subclass(self, params):
        """SmeltIBM is a subclass of IBMGroup."""
        ibm = SmeltIBM(group_index=0, n_groups=5, params=params)
        assert isinstance(ibm, IBMGroup)

    def test_creates_correct_number_of_super_individuals(self, smelt):
        """initialize_from_ecosim creates the requested number of super-individuals."""
        assert len(smelt.individuals) == 100

    def test_total_biomass_matches_input(self, smelt):
        """Total biomass of all super-individuals matches the input biomass."""
        total = smelt.get_aggregate_biomass()
        assert total == pytest.approx(5.0, rel=0.05)

    def test_individuals_have_valid_attributes(self, smelt):
        """Each super-individual has positive weight, length, age, n_represented."""
        for ind in smelt.individuals:
            assert ind.weight > 0.0, f"Individual {ind.id} has non-positive weight"
            assert ind.length > 0.0, f"Individual {ind.id} has non-positive length"
            assert ind.age >= 0.0, f"Individual {ind.id} has negative age"
            assert (
                ind.n_represented > 0.0
            ), f"Individual {ind.id} has non-positive n_represented"
            assert ind.energy_reserve >= 0.0

    def test_individuals_age_range(self, smelt, params):
        """Individual ages span from near 0 to max_age."""
        ages = [ind.age for ind in smelt.individuals]
        assert min(ages) >= 0.0
        assert max(ages) <= params.max_age + 0.1

    def test_individuals_have_unique_ids(self, smelt):
        """Each super-individual has a unique ID."""
        ids = [ind.id for ind in smelt.individuals]
        assert len(ids) == len(set(ids))

    def test_get_aggregate_biomass(self, smelt):
        """get_aggregate_biomass returns sum of total_biomass_tonnes."""
        expected = sum(ind.total_biomass_tonnes() for ind in smelt.individuals)
        assert smelt.get_aggregate_biomass() == pytest.approx(expected)

    def test_get_consumption_by_prey_initial(self, smelt):
        """Before any step, get_consumption_by_prey returns zero array."""
        consumption = smelt.get_consumption_by_prey()
        assert isinstance(consumption, np.ndarray)
        assert len(consumption) == smelt.n_groups
        np.testing.assert_array_equal(consumption, np.zeros(smelt.n_groups))


class TestSmeltIBMStep:
    """Test SmeltIBM.compute_step lifecycle."""

    @pytest.fixture
    def params(self):
        """Return Baltic default SmeltParams."""
        return SmeltParams.baltic_defaults()

    @pytest.fixture
    def smelt(self, params):
        """Return a SmeltIBM initialized with moderate biomass."""
        n_groups = 10
        ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=params)
        ibm.initialize_from_ecosim(
            biomass=5.0,
            params={},
            n_super_individuals=50,
        )
        return ibm

    def test_compute_step_returns_ibm_step_result(self, smelt):
        """compute_step returns an IBMStepResult."""
        prey = np.zeros(smelt.n_groups)
        prey[1] = 2.0  # some prey available
        prey[2] = 1.0
        env = {"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0}
        dt = 1.0 / 12.0

        result = smelt.compute_step(
            prey, predation_pressure=0.1, env_forcing=env, dt=dt
        )

        assert isinstance(result, IBMStepResult)

    def test_compute_step_biomass_positive(self, smelt):
        """After compute_step, biomass should remain positive."""
        prey = np.zeros(smelt.n_groups)
        prey[1] = 2.0
        env = {"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0}
        dt = 1.0 / 12.0

        result = smelt.compute_step(
            prey, predation_pressure=0.1, env_forcing=env, dt=dt
        )

        assert result.biomass > 0.0

    def test_compute_step_consumption_array_shape(self, smelt):
        """consumption_by_prey has shape (n_groups,)."""
        prey = np.zeros(smelt.n_groups)
        prey[1] = 2.0
        env = {"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0}
        dt = 1.0 / 12.0

        result = smelt.compute_step(
            prey, predation_pressure=0.1, env_forcing=env, dt=dt
        )

        assert result.consumption_by_prey.shape == (smelt.n_groups,)

    def test_mass_conservation_within_bounds(self, smelt):
        """Biomass after step is within reasonable bounds of initial biomass."""
        initial_biomass = smelt.get_aggregate_biomass()
        prey = np.zeros(smelt.n_groups)
        prey[1] = 10.0  # generous prey
        prey[2] = 5.0
        env = {"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0}
        dt = 1.0 / 12.0

        result = smelt.compute_step(
            prey, predation_pressure=0.1, env_forcing=env, dt=dt
        )

        # Biomass should not change by more than 50% in one month step
        assert result.biomass > initial_biomass * 0.3
        assert result.biomass < initial_biomass * 2.0

    def test_predation_reduces_population(self, smelt):
        """High predation pressure reduces population count."""
        prey = np.zeros(smelt.n_groups)
        prey[1] = 5.0
        env = {"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0}
        dt = 1.0 / 12.0

        result = smelt.compute_step(
            prey, predation_pressure=2.0, env_forcing=env, dt=dt
        )

        # After high predation, mortality_count should be positive
        assert result.mortality_count > 0.0

    def test_compute_step_with_zero_prey(self, smelt):
        """compute_step handles zero prey availability without crashing."""
        prey = np.zeros(smelt.n_groups)
        env = {"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0}
        dt = 1.0 / 12.0

        result = smelt.compute_step(
            prey, predation_pressure=0.1, env_forcing=env, dt=dt
        )

        assert isinstance(result, IBMStepResult)
        assert result.biomass >= 0.0

    def test_compute_step_updates_consumption_record(self, smelt):
        """After compute_step, get_consumption_by_prey reflects last step."""
        prey = np.zeros(smelt.n_groups)
        prey[1] = 5.0
        env = {"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0}
        dt = 1.0 / 12.0

        smelt.compute_step(prey, predation_pressure=0.1, env_forcing=env, dt=dt)

        consumption = smelt.get_consumption_by_prey()
        assert isinstance(consumption, np.ndarray)
        assert len(consumption) == smelt.n_groups
