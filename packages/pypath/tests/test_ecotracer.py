"""Tests for pypath.core.ecotracer module."""

import numpy as np
import pytest

from pypath.core.ecotracer import (
    EcotracerParams,
    EcotracerResult,
    create_ecotracer_params,
)


class TestEcotracerParams:
    def test_construction(self):
        n = 3
        p = EcotracerParams(
            czero=np.zeros(n),
            cenv=np.zeros(n),
            cimmig=np.zeros(n),
            cdecay=np.zeros(n),
            cassim=np.ones(n),
            cmetab=np.zeros(n),
        )
        assert p.czero.shape == (3,)
        assert p.cassim[0] == 1.0

    def test_custom_values(self):
        p = EcotracerParams(
            czero=np.array([1.0, 0.0, 0.0]),
            cenv=np.array([0.1, 0.0, 0.0]),
            cimmig=np.zeros(3),
            cdecay=np.array([0.05, 0.05, 0.01]),
            cassim=np.array([1.0, 0.8, 0.0]),
            cmetab=np.array([0.02, 0.03, 0.0]),
        )
        assert p.czero[0] == 1.0
        assert p.cassim[1] == 0.8


class TestEcotracerResult:
    def test_construction(self):
        r = EcotracerResult(
            out_Conc=np.zeros((13, 3)),
            annual_Conc=np.zeros((1, 3)),
            group_names=["A", "B", "C"],
        )
        assert r.out_Conc.shape == (13, 3)
        assert r.annual_Conc.shape == (1, 3)
        assert len(r.group_names) == 3


class TestCreateEcotracerParams:
    def test_defaults(self):
        p = create_ecotracer_params(4)
        assert p.czero.shape == (4,)
        np.testing.assert_array_equal(p.czero, 0.0)
        np.testing.assert_array_equal(p.cenv, 0.0)
        np.testing.assert_array_equal(p.cimmig, 0.0)
        np.testing.assert_array_equal(p.cdecay, 0.0)
        np.testing.assert_array_equal(p.cassim, 1.0)
        np.testing.assert_array_equal(p.cmetab, 0.0)

    def test_shape(self):
        p = create_ecotracer_params(10)
        for arr in [p.czero, p.cenv, p.cimmig, p.cdecay, p.cassim, p.cmetab]:
            assert arr.shape == (10,)


from pypath.core.ecotracer import ecotracer_deriv, ecotracer_step


class TestEcotracerDeriv:
    def _make_params(self, n=3):
        return EcotracerParams(
            czero=np.zeros(n),
            cenv=np.array([0.1, 0.0, 0.0]),
            cimmig=np.array([0.0, 0.05, 0.0]),
            cdecay=np.array([0.01, 0.02, 0.005]),
            cassim=np.array([1.0, 0.8, 0.0]),
            cmetab=np.array([0.02, 0.03, 0.0]),
        )

    def test_zero_conc_only_inputs(self):
        """With zero concentration, only cenv and cimmig contribute."""
        params = self._make_params()
        conc = np.zeros(3)
        biomass = np.array([10.0, 5.0, 100.0])
        Q = np.zeros((3, 3))
        deriv = ecotracer_deriv(conc, biomass, Q, params, n_living=2)
        # dC/dt = cenv + cimmig - (cdecay + cmetab) * 0 = cenv + cimmig
        assert deriv[0] == pytest.approx(0.1)  # cenv only
        assert deriv[1] == pytest.approx(0.05)  # cimmig only
        assert deriv[2] == pytest.approx(0.0)  # detritus, no input, no fate

    def test_decay_losses(self):
        """Positive concentration with decay/metabolism loses mass."""
        params = self._make_params()
        conc = np.array([1.0, 2.0, 0.5])
        biomass = np.array([10.0, 5.0, 100.0])
        Q = np.zeros((3, 3))
        deriv = ecotracer_deriv(conc, biomass, Q, params, n_living=2)
        # Group 0: dC/dt = 0.1 + 0 - (0.01 + 0.02) * 1.0 = 0.07
        assert deriv[0] == pytest.approx(0.07)
        # Group 1: dC/dt = 0 + 0.05 - (0.02 + 0.03) * 2.0 = -0.05
        assert deriv[1] == pytest.approx(-0.05)

    def test_dietary_intake(self):
        """Known Q matrix produces expected dietary uptake."""
        params = EcotracerParams(
            czero=np.zeros(3),
            cenv=np.zeros(3),
            cimmig=np.zeros(3),
            cdecay=np.zeros(3),
            cassim=np.array([1.0, 1.0, 0.0]),
            cmetab=np.zeros(3),
        )
        conc = np.array([2.0, 0.0, 0.0])  # only prey 0 contaminated
        biomass = np.array([10.0, 5.0, 100.0])
        # Predator 1 eats prey 0: Q[0, 1] = 10.0 (consumption rate)
        Q = np.zeros((3, 3))
        Q[0, 1] = 10.0
        deriv = ecotracer_deriv(conc, biomass, Q, params, n_living=2)
        # dietary_intake_1 = cassim_1 * Q[0,1] * C[0] / B[1] = 1.0 * 10 * 2 / 5 = 4.0
        assert deriv[1] == pytest.approx(4.0)
        # Group 0 has no predator eating it that returns contaminant
        assert deriv[0] == pytest.approx(0.0)

    def test_zero_biomass_no_division_error(self):
        """B_i = 0 should not cause division by zero."""
        params = self._make_params()
        conc = np.array([1.0, 1.0, 0.5])
        biomass = np.array([0.0, 5.0, 100.0])  # group 0 crashed
        Q = np.zeros((3, 3))
        Q[0, 1] = 10.0  # pred 1 eats prey 0
        # Should not raise
        deriv = ecotracer_deriv(conc, biomass, Q, params, n_living=2)
        assert np.all(np.isfinite(deriv))


class TestEcotracerStep:
    def test_analytic_update_no_loss(self):
        """With zero decay/metab, step is simple Euler: C += input * dt."""
        params = EcotracerParams(
            czero=np.zeros(2),
            cenv=np.array([1.2, 0.0]),
            cimmig=np.zeros(2),
            cdecay=np.zeros(2),
            cassim=np.ones(2),
            cmetab=np.zeros(2),
        )
        conc = np.array([0.0, 0.0])
        biomass = np.array([10.0, 5.0])
        Q = np.zeros((2, 2))
        dt = 1.0 / 12
        new_conc = ecotracer_step(conc, biomass, Q, params, dt, n_living=2)
        assert new_conc[0] == pytest.approx(1.2 / 12, rel=1e-6)

    def test_analytic_update_with_loss(self):
        """Analytic solution matches exact for constant input."""
        import math

        params = EcotracerParams(
            czero=np.zeros(1),
            cenv=np.array([1.0]),
            cimmig=np.zeros(1),
            cdecay=np.array([0.5]),
            cassim=np.ones(1),
            cmetab=np.array([0.5]),
        )
        conc = np.array([0.0])
        biomass = np.array([10.0])
        Q = np.zeros((1, 1))
        dt = 1.0 / 12
        new_conc = ecotracer_step(conc, biomass, Q, params, dt, n_living=1)
        # Exact: input=1.0, loss_rate=1.0
        # C(dt) = input/loss + (C0 - input/loss) * exp(-loss*dt)
        #       = 1.0 + (0 - 1.0) * exp(-1/12) = 1 - exp(-1/12) ≈ 0.0800
        expected = 1.0 - math.exp(-1.0 / 12)
        assert new_conc[0] == pytest.approx(expected, rel=1e-6)

    def test_stable_high_decay(self):
        """High decay rate (cdecay*dt > 1) should not go negative."""
        params = EcotracerParams(
            czero=np.zeros(1),
            cenv=np.zeros(1),
            cimmig=np.zeros(1),
            cdecay=np.array([100.0]),  # very high
            cassim=np.ones(1),
            cmetab=np.zeros(1),
        )
        conc = np.array([5.0])
        biomass = np.array([10.0])
        Q = np.zeros((1, 1))
        dt = 1.0 / 12
        new_conc = ecotracer_step(conc, biomass, Q, params, dt, n_living=1)
        assert new_conc[0] >= 0.0
        assert new_conc[0] < 5.0  # should decay

    def test_clamps_to_zero(self):
        """Result is clamped to >= 0."""
        params = EcotracerParams(
            czero=np.zeros(1),
            cenv=np.zeros(1),
            cimmig=np.zeros(1),
            cdecay=np.array([10.0]),
            cassim=np.ones(1),
            cmetab=np.zeros(1),
        )
        conc = np.array([0.001])
        biomass = np.array([10.0])
        Q = np.zeros((1, 1))
        new_conc = ecotracer_step(conc, biomass, Q, params, dt=1.0, n_living=1)
        assert new_conc[0] >= 0.0
