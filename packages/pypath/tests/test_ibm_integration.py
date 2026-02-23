"""
Tests for IBM derivative override integration.

Validates the functions that bridge IBM groups with the Ecosim derivative
calculation loop: prey extraction, predation pressure, mass balance checks,
and the in-place derivative override.
"""

from typing import Any, Dict

import numpy as np
import pytest

from pypath.ibm.base import IBMGroup, IBMStepResult
from pypath.ibm.integration import (
    apply_ibm_to_derivative,
    check_ibm_mass_balance,
    extract_predation_pressure,
    extract_prey_availability,
)


class MockIBM(IBMGroup):
    """A minimal concrete IBMGroup for testing the integration layer."""

    def __init__(
        self,
        group_index: int,
        n_groups: int,
        step_biomass: float = 10.0,
        step_production: float = 1.0,
        step_consumption: np.ndarray | None = None,
    ):
        super().__init__(group_index, n_groups)
        self._step_biomass = step_biomass
        self._step_production = step_production
        self._step_consumption = (
            step_consumption if step_consumption is not None else np.zeros(n_groups)
        )

    def compute_step(
        self,
        prey_available: np.ndarray,
        predation_pressure: float,
        env_forcing: Dict[str, Any],
        dt: float,
        spatial_context=None,
    ) -> IBMStepResult:
        return IBMStepResult(
            biomass=self._step_biomass,
            production=self._step_production,
            consumption_by_prey=self._step_consumption,
            mortality_count=0.0,
            recruitment_count=0.0,
        )

    def get_aggregate_biomass(self) -> float:
        return self._step_biomass

    def get_consumption_by_prey(self) -> np.ndarray:
        return self._step_consumption

    def initialize_from_ecosim(
        self,
        biomass: float,
        params: Dict[str, Any],
        n_super_individuals: int = 500,
    ) -> None:
        pass


class TestExtractPreyAvailability:
    """Tests for extract_prey_availability."""

    def test_extracts_nonzero_prey(self):
        """Should return only non-zero consumption entries for the predator."""
        n = 5
        QQ = np.zeros((n + 1, n + 1))
        # Predator index 3 eats prey 1 and prey 4
        QQ[1, 3] = 2.5
        QQ[4, 3] = 1.2

        result = extract_prey_availability(QQ, predator_idx=3, n_groups=n)

        assert result == {1: 2.5, 4: 1.2}

    def test_empty_when_no_prey(self):
        """Should return empty dict when predator has no prey."""
        n = 5
        QQ = np.zeros((n + 1, n + 1))

        result = extract_prey_availability(QQ, predator_idx=3, n_groups=n)

        assert result == {}

    def test_ignores_zero_entries(self):
        """Exactly zero consumption should be excluded."""
        n = 4
        QQ = np.zeros((n + 1, n + 1))
        QQ[1, 2] = 0.0
        QQ[2, 2] = 3.0
        QQ[3, 2] = 0.0

        result = extract_prey_availability(QQ, predator_idx=2, n_groups=n)

        assert result == {2: 3.0}


class TestExtractPredationPressure:
    """Tests for extract_predation_pressure."""

    def test_sums_all_predators(self):
        """Should sum QQ[prey_idx, 1:n_living+1] for all predators."""
        n = 5
        n_living = 3
        QQ = np.zeros((n + 1, n + 1))
        # Prey index 2, consumed by predators 1, 2, 3
        QQ[2, 1] = 1.0
        QQ[2, 2] = 0.5  # self-predation
        QQ[2, 3] = 2.0

        result = extract_predation_pressure(QQ, prey_idx=2, n_living=n_living)

        assert result == pytest.approx(3.5)

    def test_zero_when_no_predation(self):
        """Should return 0.0 when no predators consume this prey."""
        n = 5
        n_living = 3
        QQ = np.zeros((n + 1, n + 1))

        result = extract_predation_pressure(QQ, prey_idx=2, n_living=n_living)

        assert result == 0.0


class TestMassBalance:
    """Tests for check_ibm_mass_balance."""

    def test_balanced_result(self):
        """Should return (True, small_error) for a valid result."""
        result = IBMStepResult(
            biomass=10.0,
            production=1.0,
            consumption_by_prey=np.array([0.5, 0.3, 0.2]),
            mortality_count=0.0,
            recruitment_count=0.0,
        )

        is_balanced, error = check_ibm_mass_balance(result)

        assert is_balanced is True
        assert error >= 0.0

    def test_negative_biomass_fails(self):
        """Negative biomass should fail the mass balance check."""
        result = IBMStepResult(
            biomass=-1.0,
            production=1.0,
            consumption_by_prey=np.array([0.5]),
            mortality_count=0.0,
            recruitment_count=0.0,
        )

        is_balanced, error = check_ibm_mass_balance(result)

        assert is_balanced is False

    def test_negative_consumption_fails(self):
        """Negative consumption values should fail the mass balance check."""
        result = IBMStepResult(
            biomass=10.0,
            production=1.0,
            consumption_by_prey=np.array([0.5, -0.3]),
            mortality_count=0.0,
            recruitment_count=0.0,
        )

        is_balanced, error = check_ibm_mass_balance(result)

        assert is_balanced is False

    def test_custom_tolerance(self):
        """check_ibm_mass_balance accepts a custom tolerance parameter."""
        result = IBMStepResult(
            biomass=10.0,
            production=1.0,
            consumption_by_prey=np.array([1.0]),
            mortality_count=0.0,
            recruitment_count=0.0,
        )

        is_balanced, error = check_ibm_mass_balance(result, tolerance=0.01)

        assert is_balanced is True


class TestApplyIBMToDerivative:
    """Tests for apply_ibm_to_derivative."""

    def test_overrides_derivative_for_ibm_group(self):
        """deriv[group_idx] should be (ibm_biomass - BB[group_idx]) / dt."""
        n_groups = 5
        dt = 1.0 / 12.0

        # IBM group at index 3, returns biomass=12.0
        ibm = MockIBM(group_index=3, n_groups=n_groups, step_biomass=12.0)

        # Current biomass
        BB = np.zeros(n_groups + 1)
        BB[3] = 10.0

        # QQ matrix (predator 3 eats prey 1)
        QQ = np.zeros((n_groups + 1, n_groups + 1))
        QQ[1, 3] = 0.5

        deriv = np.zeros(n_groups + 1)
        forcing = {}

        apply_ibm_to_derivative(
            deriv=deriv,
            QQ=QQ,
            BB=BB,
            ibm_group=ibm,
            forcing=forcing,
            dt=dt,
        )

        expected_deriv = (12.0 - 10.0) / dt
        assert deriv[3] == pytest.approx(expected_deriv)

    def test_subtracts_consumption_from_prey_derivatives(self):
        """Prey derivatives should be reduced by IBM consumption / dt."""
        n_groups = 5
        dt = 1.0 / 12.0

        # IBM consumes 0.6 from prey 1 and 0.4 from prey 2
        consumption = np.zeros(n_groups)
        consumption[1] = 0.6
        consumption[2] = 0.4

        ibm = MockIBM(
            group_index=3,
            n_groups=n_groups,
            step_biomass=10.0,
            step_consumption=consumption,
        )

        BB = np.zeros(n_groups + 1)
        BB[3] = 10.0

        QQ = np.zeros((n_groups + 1, n_groups + 1))
        QQ[1, 3] = 0.5  # predator 3 eats prey 1

        deriv = np.zeros(n_groups + 1)
        deriv[1] = 5.0  # some existing derivative for prey 1
        deriv[2] = 3.0  # some existing derivative for prey 2

        forcing = {}

        apply_ibm_to_derivative(
            deriv=deriv,
            QQ=QQ,
            BB=BB,
            ibm_group=ibm,
            forcing=forcing,
            dt=dt,
        )

        assert deriv[1] == pytest.approx(5.0 - 0.6 / dt)
        assert deriv[2] == pytest.approx(3.0 - 0.4 / dt)

    def test_no_consumption_leaves_prey_unchanged(self):
        """When IBM consumes nothing, prey derivatives should not change."""
        n_groups = 5
        dt = 1.0 / 12.0

        ibm = MockIBM(
            group_index=3,
            n_groups=n_groups,
            step_biomass=10.0,
            step_consumption=np.zeros(n_groups),
        )

        BB = np.zeros(n_groups + 1)
        BB[3] = 10.0

        QQ = np.zeros((n_groups + 1, n_groups + 1))
        deriv = np.ones(n_groups + 1) * 2.0
        forcing = {}

        apply_ibm_to_derivative(
            deriv=deriv,
            QQ=QQ,
            BB=BB,
            ibm_group=ibm,
            forcing=forcing,
            dt=dt,
        )

        # Only group 3 derivative should change; all others stay at 2.0
        for i in range(n_groups + 1):
            if i != 3:
                assert deriv[i] == pytest.approx(2.0)
