"""Tests for fast equilibrium (NoIntegrate groups) matching Rpath."""

import pytest


class TestFastEquilibrium:
    def test_biomeq_computation(self):
        """biomeq = TotGain / (TotLoss / B) should be computed for NoIntegrate groups."""
        from pypath.core.ecosim_deriv import compute_biomeq

        total_gain = 100.0
        total_loss = 80.0
        biomass = 10.0

        # biomeq = TotGain / (TotLoss / B) = 100 / (80/10) = 100/8 = 12.5
        # smoothed: (1-0.5)*12.5 + 0.5*10.0 = 11.25
        result = compute_biomeq(total_gain, total_loss, biomass)
        assert result == pytest.approx(11.25, rel=1e-6)

    def test_biomeq_zero_loss_returns_biomass(self):
        """If total_loss is zero, return current biomass unchanged."""
        from pypath.core.ecosim_deriv import compute_biomeq

        result = compute_biomeq(100.0, 0.0, 10.0)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_biomeq_zero_biomass_returns_zero(self):
        """If biomass is zero, return zero."""
        from pypath.core.ecosim_deriv import compute_biomeq

        result = compute_biomeq(100.0, 80.0, 0.0)
        assert result == pytest.approx(0.0, rel=1e-6)

    def test_biomeq_equilibrium_no_change(self):
        """When gain equals loss, biomeq should equal current biomass."""
        from pypath.core.ecosim_deriv import compute_biomeq

        # If gain=loss, biomeq = gain/(loss/B) = B
        # smoothed: 0.5*B + 0.5*B = B
        result = compute_biomeq(80.0, 80.0, 10.0)
        assert result == pytest.approx(10.0, rel=1e-6)

    def test_biomeq_custom_sorwt(self):
        """SORWT parameter controls smoothing weight."""
        from pypath.core.ecosim_deriv import compute_biomeq

        # biomeq = 100/(80/10) = 12.5
        # with sorwt=0.8: 0.2*12.5 + 0.8*10.0 = 2.5 + 8.0 = 10.5
        result = compute_biomeq(100.0, 80.0, 10.0, sorwt=0.8)
        assert result == pytest.approx(10.5, rel=1e-6)
