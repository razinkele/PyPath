"""
Tests for IBM (Individual-Based Model) base data structures.

Validates the foundational dataclasses and abstract base class
used by all IBM group implementations in PyPath.
"""

import numpy as np
import pytest

from pypath.ibm import IBMGroup, IBMStepResult, SuperIndividual


class TestSuperIndividual:
    """Test the SuperIndividual dataclass."""

    def test_creation_with_valid_attributes(self):
        """SuperIndividual can be created with all required fields."""
        si = SuperIndividual(
            id=1,
            n_represented=1000.0,
            weight=0.05,
            length=12.0,
            age=2.0,
            energy_reserve=0.8,
            patch_idx=3,
            is_mature=True,
            sex=1,
        )
        assert si.id == 1
        assert si.n_represented == 1000.0
        assert si.weight == 0.05
        assert si.length == 12.0
        assert si.age == 2.0
        assert si.energy_reserve == 0.8
        assert si.patch_idx == 3
        assert si.is_mature is True
        assert si.sex == 1

    def test_total_biomass_tonnes(self):
        """total_biomass_tonnes returns n_represented * weight / 1e6."""
        si = SuperIndividual(
            id=0,
            n_represented=2_000_000.0,
            weight=0.5,
            length=10.0,
            age=1.0,
            energy_reserve=1.0,
            patch_idx=0,
            is_mature=False,
            sex=0,
        )
        # 2_000_000 * 0.5 / 1e6 = 1.0
        assert si.total_biomass_tonnes() == pytest.approx(1.0)

    def test_total_biomass_tonnes_zero_weight(self):
        """total_biomass_tonnes returns 0 when weight is zero."""
        si = SuperIndividual(
            id=0,
            n_represented=1000.0,
            weight=0.0,
            length=0.0,
            age=0.0,
            energy_reserve=0.0,
            patch_idx=0,
            is_mature=False,
            sex=0,
        )
        assert si.total_biomass_tonnes() == pytest.approx(0.0)


class TestIBMStepResult:
    """Test the IBMStepResult dataclass."""

    def test_creation(self):
        """IBMStepResult can be created with all required fields."""
        consumption = np.array([0.1, 0.2, 0.3])
        result = IBMStepResult(
            biomass=10.0,
            production=1.5,
            consumption_by_prey=consumption,
            mortality_count=50.0,
            recruitment_count=200.0,
        )
        assert result.biomass == 10.0
        assert result.production == 1.5
        np.testing.assert_array_equal(result.consumption_by_prey, consumption)
        assert result.mortality_count == 50.0
        assert result.recruitment_count == 200.0

    def test_consumption_by_prey_is_ndarray(self):
        """consumption_by_prey field holds a numpy array."""
        result = IBMStepResult(
            biomass=5.0,
            production=0.5,
            consumption_by_prey=np.zeros(4),
            mortality_count=0.0,
            recruitment_count=0.0,
        )
        assert isinstance(result.consumption_by_prey, np.ndarray)
        assert result.consumption_by_prey.shape == (4,)


class TestIBMGroupInterface:
    """Test the IBMGroup abstract base class interface."""

    def test_cannot_instantiate_directly(self):
        """IBMGroup raises TypeError when instantiated directly."""
        with pytest.raises(TypeError):
            IBMGroup(group_index=0, n_groups=5)

    def test_concrete_subclass_must_implement_abstract_methods(self):
        """A subclass missing abstract methods also raises TypeError."""

        class IncompleteGroup(IBMGroup):
            pass

        with pytest.raises(TypeError):
            IncompleteGroup(group_index=0, n_groups=5)

    def test_concrete_subclass_can_be_instantiated(self):
        """A fully implemented subclass can be instantiated."""

        class MockGroup(IBMGroup):
            def compute_step(self, prey_available, predation_pressure,
                             env_forcing, dt):
                return IBMStepResult(
                    biomass=0.0,
                    production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0,
                    recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 0.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params,
                                       n_super_individuals=500):
                pass

        group = MockGroup(group_index=2, n_groups=10)
        assert group.group_index == 2
        assert group.n_groups == 10
        assert group.individuals == []

    def test_concrete_subclass_individuals_list(self):
        """A concrete IBMGroup starts with an empty individuals list."""

        class MockGroup(IBMGroup):
            def compute_step(self, prey_available, predation_pressure,
                             env_forcing, dt):
                return IBMStepResult(
                    biomass=0.0,
                    production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0,
                    recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 0.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params,
                                       n_super_individuals=500):
                pass

        group = MockGroup(group_index=0, n_groups=5)
        assert isinstance(group.individuals, list)
        assert len(group.individuals) == 0

        # Can append SuperIndividuals
        si = SuperIndividual(
            id=0, n_represented=100.0, weight=0.01, length=5.0,
            age=0.5, energy_reserve=0.9, patch_idx=0,
            is_mature=False, sex=0,
        )
        group.individuals.append(si)
        assert len(group.individuals) == 1
