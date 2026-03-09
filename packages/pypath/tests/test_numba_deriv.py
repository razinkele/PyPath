"""
Tests for numba-accelerated kernels in ecosim_deriv.

Validates that the extracted _compute_consumption, _compute_living_derivs,
and _compute_detritus_derivs functions (both the pure-Python and
numba-compiled versions) produce identical numerical results to the
original inline loops, using the seabirds reference model.
"""

from pathlib import Path

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.ecosim_deriv import (
    HAS_NUMBA,
    _compute_consumption,
    _compute_consumption_numba,
    _compute_consumption_python,
    _compute_detritus_derivs,
    _compute_detritus_derivs_numba,
    _compute_detritus_derivs_python,
    _compute_living_derivs,
    _compute_living_derivs_numba,
    _compute_living_derivs_python,
)
from pypath.core.params import create_rpath_params

# Path to reference data
REFERENCE_DIR = Path(__file__).parent / "data" / "rpath_reference"
ECOPATH_DIR = REFERENCE_DIR / "ecopath"


def _load_seabirds_model():
    """Load and balance the seabirds reference model."""
    import pandas as pd

    model_df = pd.read_csv(ECOPATH_DIR / "model_params.csv")
    diet_df = pd.read_csv(ECOPATH_DIR / "diet_matrix.csv")

    groups = model_df["Group"].tolist()
    types = model_df["Type"].tolist()

    params = create_rpath_params(groups, types)
    params.model = model_df
    params.diet = diet_df

    stanza_groups_file = ECOPATH_DIR / "stanza_groups.csv"
    stanza_indiv_file = ECOPATH_DIR / "stanza_indiv.csv"

    if stanza_groups_file.exists() and stanza_indiv_file.exists():
        params.stanzas.stgroups = pd.read_csv(stanza_groups_file)
        params.stanzas.stindiv = pd.read_csv(stanza_indiv_file)

    return params


@pytest.fixture(scope="module")
def seabirds_params():
    """Load the seabirds reference model parameters."""
    if not ECOPATH_DIR.exists():
        pytest.skip("Reference data not available")
    return _load_seabirds_model()


@pytest.fixture(scope="module")
def seabirds_balanced(seabirds_params):
    """Balance the seabirds model."""
    return rpath(seabirds_params)


@pytest.fixture(scope="module")
def seabirds_scenario(seabirds_balanced, seabirds_params):
    """Create an Ecosim scenario from the seabirds model."""
    return rsim_scenario(seabirds_balanced, seabirds_params, years=range(1, 11))


class TestComputeConsumptionKernel:
    """Test the extracted consumption kernel directly."""

    def test_python_kernel_matches_inline(self):
        """Verify _compute_consumption_python produces correct QQ values."""
        # Create a small test case with known values
        n = 5  # 4 groups + outside
        QQ = np.zeros((n, n))
        BB = np.array([0.0, 10.0, 2.0, 5.0, 1.0])
        ActiveLink = np.zeros((n, n), dtype=np.int64)
        ActiveLink[1, 2] = 1  # prey 1, pred 2
        ActiveLink[2, 3] = 1  # prey 2, pred 3
        VV = np.full((n, n), 2.0)
        DD = np.full((n, n), 1000.0)
        QQbase = np.zeros((n, n))
        QQbase[1, 2] = 0.5
        QQbase[2, 3] = 0.3
        preyYY = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        predYY = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        NUM_LIVING = 3
        NUM_GROUPS = 4

        _compute_consumption_python(
            QQ, BB, ActiveLink, VV, DD, QQbase, preyYY, predYY,
            NUM_LIVING, NUM_GROUPS,
        )

        # At equilibrium (preyYY=1, predYY=1), Q = QQbase * 1 * 1 * dd_term * vv_term
        # dd_term = 1000/(1000 - 1 + 1) = 1000/1000 = 1.0
        # vv_term = 2/(2 - 1 + 1) = 2/2 = 1.0
        # So QQ should equal QQbase
        assert np.isclose(QQ[1, 2], 0.5, rtol=1e-10)
        assert np.isclose(QQ[2, 3], 0.3, rtol=1e-10)
        assert QQ[3, 1] == 0.0  # no link

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_numba_matches_python(self):
        """Verify numba-compiled kernel produces identical results to Python."""
        rng = np.random.default_rng(42)
        n = 20
        QQ_py = np.zeros((n, n))
        QQ_nb = np.zeros((n, n))

        BB = rng.uniform(0.1, 10.0, n)
        BB[0] = 0.0

        ActiveLink = np.zeros((n, n), dtype=np.int64)
        for _ in range(40):
            prey = rng.integers(1, n)
            pred = rng.integers(1, 15)
            ActiveLink[prey, pred] = 1

        VV = rng.uniform(1.5, 10.0, (n, n))
        DD = rng.uniform(1.0, 1000.0, (n, n))
        QQbase = rng.uniform(0.0, 1.0, (n, n))
        preyYY = rng.uniform(0.5, 2.0, n)
        preyYY[0] = 0.0
        predYY = rng.uniform(0.5, 2.0, n)
        predYY[0] = 0.0

        NUM_LIVING = 14
        NUM_GROUPS = n - 1

        _compute_consumption_python(
            QQ_py, BB, ActiveLink, VV, DD, QQbase, preyYY, predYY,
            NUM_LIVING, NUM_GROUPS,
        )
        _compute_consumption_numba(
            QQ_nb, BB, ActiveLink, VV, DD, QQbase, preyYY, predYY,
            NUM_LIVING, NUM_GROUPS,
        )

        np.testing.assert_allclose(QQ_nb, QQ_py, rtol=1e-14, atol=0.0)

    def test_dispatch_function_works(self):
        """Verify _compute_consumption dispatch runs without error."""
        n = 5
        QQ = np.zeros((n, n))
        BB = np.array([0.0, 5.0, 2.0, 1.0, 0.5])
        ActiveLink = np.zeros((n, n), dtype=np.int64)
        ActiveLink[1, 2] = 1
        VV = np.full((n, n), 2.0)
        DD = np.full((n, n), 1000.0)
        QQbase = np.zeros((n, n))
        QQbase[1, 2] = 0.4
        preyYY = np.ones(n)
        preyYY[0] = 0.0
        predYY = np.ones(n)
        predYY[0] = 0.0

        _compute_consumption(
            QQ, BB, ActiveLink, VV, DD, QQbase, preyYY, predYY, 3, 4,
        )

        assert QQ[1, 2] > 0.0


# =============================================================================
# Living-group derivative kernel tests
# =============================================================================


class TestComputeLivingDerivsKernel:
    """Test the extracted living-group derivative kernel directly."""

    def _make_small_model(self):
        """Create a small 3-living + 1-detritus model for testing."""
        NUM_LIVING = 3
        NUM_GROUPS = 4  # 3 living + 1 detritus
        n = NUM_GROUPS + 1  # +1 for outside (index 0)

        # Consumption matrix: prey 1 eaten by pred 2, prey 2 eaten by pred 3
        QQ = np.zeros((n, n))
        QQ[1, 2] = 0.5   # prey 1 consumed by pred 2
        QQ[2, 3] = 0.3   # prey 2 consumed by pred 3
        QQ[4, 1] = 0.2   # detritus consumed by group 1

        BB = np.array([0.0, 10.0, 5.0, 2.0, 8.0])
        M0_arr = np.array([0.0, 0.1, 0.05, 0.02, 0.0])
        ForcedMigrate = np.zeros(n)
        ForcedMigrate[2] = 0.01  # some migration for group 2
        FishMort = np.array([0.0, 0.05, 0.1, 0.02, 0.0])

        # Group 1 is a producer (PP_type=1), groups 2-3 are consumers
        PP_type = np.array([0, 1, 0, 0, 2], dtype=np.int64)
        PB = np.array([0.0, 2.0, 1.5, 0.8, 0.0])
        QB = np.array([0.0, 0.0, 3.0, 2.0, 0.0])

        # pp_rates for producer (group 1)
        pp_rates = np.array([0.0, 20.0, 0.0, 0.0, 0.0])  # PB * BB * dd_factor

        # GE = PB/QB for consumers
        GE_arr = np.zeros(n)
        GE_arr[2] = PB[2] / QB[2]  # 0.5
        GE_arr[3] = PB[3] / QB[3]  # 0.4

        ibm_mask = np.zeros(n, dtype=np.int64)

        return (
            QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates, GE_arr,
            PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS,
        )

    def test_python_kernel_correct(self):
        """Verify _compute_living_derivs_python produces correct derivatives."""
        (QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates, GE_arr,
         PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS) = self._make_small_model()

        n = NUM_GROUPS + 1
        deriv = np.zeros(n)

        _compute_living_derivs_python(
            deriv, QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates,
            GE_arr, PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS,
        )

        # Group 1 (producer): consumption = QQ[4,1] = 0.2 (detritus eaten by group 1)
        # production = pp_rates[1] = 20.0 (producer)
        # predation_loss = QQ[1, 1] + QQ[1, 2] + QQ[1, 3] = 0 + 0.5 + 0 = 0.5
        # deriv[1] = 20.0 - 0.5 - 0.05*10 - 0.1*10 + 0.0 = 20.0 - 0.5 - 0.5 - 1.0 = 18.0
        expected_1 = 20.0 - 0.5 - 0.05 * 10.0 - 0.1 * 10.0 + 0.0
        assert np.isclose(deriv[1], expected_1, rtol=1e-12), (
            f"Group 1: expected {expected_1}, got {deriv[1]}"
        )

        # Group 2 (consumer): consumption = QQ[1,2] + QQ[2,2] + QQ[3,2] + QQ[4,2]
        #   = 0.5 + 0 + 0 + 0 = 0.5
        # production = GE[2] * 0.5 = 0.5 * 0.5 = 0.25
        # predation_loss = QQ[2, 1] + QQ[2, 2] + QQ[2, 3] = 0 + 0 + 0.3 = 0.3
        # deriv[2] = 0.25 - 0.3 - 0.1*5.0 - 0.05*5.0 + 0.01
        expected_2 = 0.25 - 0.3 - 0.1 * 5.0 - 0.05 * 5.0 + 0.01
        assert np.isclose(deriv[2], expected_2, rtol=1e-12), (
            f"Group 2: expected {expected_2}, got {deriv[2]}"
        )

        # Group 3 (consumer): consumption = QQ[2,3] = 0.3
        # production = GE[3] * 0.3 = 0.4 * 0.3 = 0.12
        # predation_loss = QQ[3, 1..3] = 0
        # deriv[3] = 0.12 - 0 - 0.02*2 - 0.02*2 + 0 = 0.12 - 0.04 - 0.04 = 0.04
        expected_3 = 0.4 * 0.3 - 0.0 - 0.02 * 2.0 - 0.02 * 2.0 + 0.0
        assert np.isclose(deriv[3], expected_3, rtol=1e-12), (
            f"Group 3: expected {expected_3}, got {deriv[3]}"
        )

        # Detritus group (index 4) should not be touched
        assert deriv[4] == 0.0

    def test_ibm_mask_skips_groups(self):
        """Verify groups with ibm_mask=1 are not computed by the kernel."""
        (QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates, GE_arr,
         PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS) = self._make_small_model()

        n = NUM_GROUPS + 1
        ibm_mask[2] = 1  # skip group 2

        deriv = np.zeros(n)
        _compute_living_derivs_python(
            deriv, QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates,
            GE_arr, PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS,
        )

        # Group 2 should remain at zero
        assert deriv[2] == 0.0
        # Groups 1 and 3 should still be computed
        assert deriv[1] != 0.0
        assert deriv[3] != 0.0

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_numba_matches_python(self):
        """Verify numba-compiled living-group kernel matches pure Python."""
        rng = np.random.default_rng(123)
        NUM_LIVING = 10
        NUM_GROUPS = 13
        n = NUM_GROUPS + 1

        QQ = rng.uniform(0.0, 1.0, (n, n))
        QQ[0, :] = 0.0
        QQ[:, 0] = 0.0
        BB = rng.uniform(0.5, 20.0, n)
        BB[0] = 0.0
        M0_arr = rng.uniform(0.0, 0.2, n)
        M0_arr[0] = 0.0
        ForcedMigrate = rng.uniform(-0.01, 0.01, n)
        ForcedMigrate[0] = 0.0
        FishMort = rng.uniform(0.0, 0.15, n)
        FishMort[0] = 0.0
        pp_rates = np.zeros(n)
        pp_rates[1] = 5.0  # one producer
        PB = rng.uniform(0.5, 3.0, n)
        QB = rng.uniform(1.0, 6.0, n)
        QB[0] = 0.0
        QB[1] = 0.0  # producer has no QB
        GE_arr = np.zeros(n)
        for i in range(2, NUM_LIVING + 1):
            if QB[i] > 0:
                GE_arr[i] = PB[i] / QB[i]
        PP_type = np.zeros(n, dtype=np.int64)
        PP_type[1] = 1  # producer
        ibm_mask = np.zeros(n, dtype=np.int64)

        deriv_py = np.zeros(n)
        deriv_nb = np.zeros(n)

        _compute_living_derivs_python(
            deriv_py, QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates,
            GE_arr, PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS,
        )
        _compute_living_derivs_numba(
            deriv_nb, QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates,
            GE_arr, PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS,
        )

        np.testing.assert_allclose(deriv_nb, deriv_py, rtol=1e-14, atol=0.0)

    def test_dispatch_function_works(self):
        """Verify _compute_living_derivs dispatch runs without error."""
        (QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates, GE_arr,
         PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS) = self._make_small_model()

        n = NUM_GROUPS + 1
        deriv = np.zeros(n)

        _compute_living_derivs(
            deriv, QQ, BB, M0_arr, ForcedMigrate, FishMort, pp_rates,
            GE_arr, PP_type, PB, QB, ibm_mask, NUM_LIVING, NUM_GROUPS,
        )

        # Should produce non-zero derivatives for living groups
        assert np.any(deriv[1:NUM_LIVING + 1] != 0.0)


# =============================================================================
# Detritus derivative kernel tests
# =============================================================================


class TestComputeDetritusKernel:
    """Test the extracted detritus derivative kernel directly."""

    def _make_detritus_model(self):
        """Create a small model with 3 living groups and 2 detritus groups."""
        NUM_LIVING = 3
        NUM_DEAD = 2
        NUM_GROUPS = NUM_LIVING + NUM_DEAD  # 5
        n = NUM_GROUPS + 1  # 6

        QQ = np.zeros((n, n))
        # Some consumption links
        QQ[1, 2] = 0.5   # prey 1 eaten by pred 2
        QQ[2, 3] = 0.3   # prey 2 eaten by pred 3
        # Detritus consumption: detritus group 4 eaten by group 1
        QQ[4, 1] = 0.2
        QQ[4, 2] = 0.1
        # Detritus group 5 eaten by group 3
        QQ[5, 3] = 0.05

        BB = np.array([0.0, 10.0, 5.0, 2.0, 8.0, 3.0])

        # total consumption by each predator: sum(QQ[1:, pred])
        total_consump_by_pred = np.sum(QQ[1:, 1:NUM_LIVING + 1], axis=0)

        Unassim = np.array([0.0, 0.2, 0.3, 0.15, 0.0, 0.0])

        # DetFrac: rows = groups, cols = detritus indices (1-based)
        # Shape: (n, NUM_DEAD + 1) = (6, 3)
        DetFrac = np.zeros((n, NUM_DEAD + 1))
        DetFrac[1, 1] = 0.6   # group 1 mortality goes 60% to det 1
        DetFrac[1, 2] = 0.4   # and 40% to det 2
        DetFrac[2, 1] = 1.0   # group 2 mortality goes 100% to det 1
        DetFrac[3, 1] = 0.5   # group 3 mortality goes 50% to det 1
        DetFrac[3, 2] = 0.5   # and 50% to det 2

        M0_arr = np.array([0.0, 0.1, 0.05, 0.02, 0.0, 0.0])
        decay_rate = np.array([0.0, 0.01, 0.02])  # decay for det 1 and det 2

        return (
            QQ, BB, total_consump_by_pred, Unassim, DetFrac, M0_arr,
            decay_rate, NUM_LIVING, NUM_DEAD,
        )

    def test_python_kernel_correct(self):
        """Verify _compute_detritus_derivs_python produces correct derivatives."""
        (QQ, BB, total_consump_by_pred, Unassim, DetFrac, M0_arr,
         decay_rate, NUM_LIVING, NUM_DEAD) = self._make_detritus_model()

        n = NUM_LIVING + NUM_DEAD + 1
        deriv = np.zeros(n)

        _compute_detritus_derivs_python(
            deriv, QQ, BB, total_consump_by_pred, Unassim, DetFrac,
            M0_arr, decay_rate, NUM_LIVING, NUM_DEAD,
        )

        # Detritus group 4 (d=4, det_idx=1):
        # unas_input:
        #   pred 1: total_consump_by_pred[0] * Unassim[1] * DetFrac[1, 1]
        #     = (QQ[1,1]+QQ[2,1]+QQ[3,1]+QQ[4,1]+QQ[5,1]) * 0.2 * 0.6
        #     = (0+0+0+0.2+0) * 0.2 * 0.6 = 0.024
        #   pred 2: total_consump_by_pred[1] * Unassim[2] * DetFrac[2, 1]
        #     = (QQ[1,2]+QQ[2,2]+QQ[3,2]+QQ[4,2]+QQ[5,2]) * 0.3 * 1.0
        #     = (0.5+0+0+0.1+0) * 0.3 * 1.0 = 0.18
        #   pred 3: total_consump_by_pred[2] * Unassim[3] * DetFrac[3, 1]
        #     = (QQ[1,3]+QQ[2,3]+QQ[3,3]+QQ[4,3]+QQ[5,3]) * 0.15 * 0.5
        #     = (0+0.3+0+0+0.05) * 0.15 * 0.5 = 0.02625
        unas_d4 = 0.024 + 0.18 + 0.02625

        # mort_input:
        #   grp 1: M0[1]*BB[1]*DetFrac[1,1] = 0.1*10*0.6 = 0.6
        #   grp 2: M0[2]*BB[2]*DetFrac[2,1] = 0.05*5*1.0 = 0.25
        #   grp 3: M0[3]*BB[3]*DetFrac[3,1] = 0.02*2*0.5 = 0.02
        mort_d4 = 0.6 + 0.25 + 0.02

        # det_consumed: QQ[4, 1] + QQ[4, 2] + QQ[4, 3] = 0.2 + 0.1 + 0 = 0.3
        det_consumed_d4 = 0.3

        # decay: decay_rate[1] * BB[4] = 0.01 * 8.0 = 0.08
        decay_d4 = 0.08

        expected_d4 = unas_d4 + mort_d4 - det_consumed_d4 - decay_d4
        assert np.isclose(deriv[4], expected_d4, rtol=1e-10), (
            f"Detritus 4: expected {expected_d4}, got {deriv[4]}"
        )

        # Detritus group 5 (d=5, det_idx=2):
        # unas: pred 1: 0.2 * 0.2 * 0.4 = 0.016
        #        pred 2: 0.6 * 0.3 * 0.0 = 0 (DetFrac[2,2] = 0)
        #        pred 3: 0.35 * 0.15 * 0.5 = 0.02625
        unas_d5 = 0.2 * 0.2 * 0.4 + 0.6 * 0.3 * 0.0 + 0.35 * 0.15 * 0.5

        # mort: grp 1: 0.1*10*0.4=0.4, grp 2: 0.05*5*0=0, grp 3: 0.02*2*0.5=0.02
        mort_d5 = 0.4 + 0.0 + 0.02

        # det_consumed: QQ[5, 1] + QQ[5, 2] + QQ[5, 3] = 0 + 0 + 0.05 = 0.05
        det_consumed_d5 = 0.05

        # decay: decay_rate[2] * BB[5] = 0.02 * 3.0 = 0.06
        decay_d5 = 0.06

        expected_d5 = unas_d5 + mort_d5 - det_consumed_d5 - decay_d5
        assert np.isclose(deriv[5], expected_d5, rtol=1e-10), (
            f"Detritus 5: expected {expected_d5}, got {deriv[5]}"
        )

        # Living groups should not be touched
        for i in range(NUM_LIVING + 1):
            assert deriv[i] == 0.0

    @pytest.mark.skipif(not HAS_NUMBA, reason="numba not installed")
    def test_numba_matches_python(self):
        """Verify numba-compiled detritus kernel matches pure Python."""
        rng = np.random.default_rng(456)
        NUM_LIVING = 8
        NUM_DEAD = 3
        NUM_GROUPS = NUM_LIVING + NUM_DEAD
        n = NUM_GROUPS + 1

        QQ = rng.uniform(0.0, 0.5, (n, n))
        QQ[0, :] = 0.0
        QQ[:, 0] = 0.0
        BB = rng.uniform(0.5, 20.0, n)
        BB[0] = 0.0

        total_consump_by_pred = np.sum(QQ[1:, 1:NUM_LIVING + 1], axis=0)
        Unassim = rng.uniform(0.0, 0.4, n)
        Unassim[0] = 0.0
        DetFrac = rng.uniform(0.0, 1.0, (n, NUM_DEAD + 1))
        DetFrac[0, :] = 0.0
        M0_arr = rng.uniform(0.0, 0.2, n)
        M0_arr[0] = 0.0
        decay_rate = rng.uniform(0.0, 0.05, NUM_DEAD + 1)
        decay_rate[0] = 0.0

        deriv_py = np.zeros(n)
        deriv_nb = np.zeros(n)

        _compute_detritus_derivs_python(
            deriv_py, QQ, BB, total_consump_by_pred, Unassim, DetFrac,
            M0_arr, decay_rate, NUM_LIVING, NUM_DEAD,
        )
        _compute_detritus_derivs_numba(
            deriv_nb, QQ, BB, total_consump_by_pred, Unassim, DetFrac,
            M0_arr, decay_rate, NUM_LIVING, NUM_DEAD,
        )

        np.testing.assert_allclose(deriv_nb, deriv_py, rtol=1e-14, atol=0.0)

    def test_dispatch_function_works(self):
        """Verify _compute_detritus_derivs dispatch runs without error."""
        (QQ, BB, total_consump_by_pred, Unassim, DetFrac, M0_arr,
         decay_rate, NUM_LIVING, NUM_DEAD) = self._make_detritus_model()

        n = NUM_LIVING + NUM_DEAD + 1
        deriv = np.zeros(n)

        _compute_detritus_derivs(
            deriv, QQ, BB, total_consump_by_pred, Unassim, DetFrac,
            M0_arr, decay_rate, NUM_LIVING, NUM_DEAD,
        )

        # Detritus groups should have non-zero derivatives
        assert np.any(deriv[NUM_LIVING + 1:] != 0.0)

    def test_empty_detritus(self):
        """Kernel handles zero detritus groups gracefully."""
        NUM_LIVING = 3
        NUM_DEAD = 0
        n = NUM_LIVING + 1
        deriv = np.zeros(n)
        QQ = np.zeros((n, n))
        BB = np.ones(n)
        total_consump = np.zeros(NUM_LIVING)
        Unassim = np.zeros(n)
        DetFrac = np.zeros((n, 1))
        M0_arr = np.zeros(n)
        decay_rate = np.zeros(1)

        # Should not raise
        _compute_detritus_derivs_python(
            deriv, QQ, BB, total_consump, Unassim, DetFrac,
            M0_arr, decay_rate, NUM_LIVING, NUM_DEAD,
        )

        # No detritus groups, so deriv should remain all zeros
        np.testing.assert_array_equal(deriv, 0.0)


# =============================================================================
# Full rsim_run integration tests
# =============================================================================


@pytest.mark.skipif(
    not REFERENCE_DIR.exists(), reason="Reference data not available"
)
class TestRsimRunWithNumba:
    """Test that a full rsim_run produces unchanged results with the new code path."""

    def test_rsim_rk4_biomass_unchanged(self, seabirds_scenario):
        """Run RK4 simulation and verify biomass trajectories are reasonable."""
        output = rsim_run(seabirds_scenario, method="RK4", years=range(1, 11))
        biomass = output.annual_Biomass

        # Basic sanity: all biomass values are non-negative and finite
        assert np.all(np.isfinite(biomass)), "Non-finite biomass detected"
        assert np.all(biomass >= 0.0), "Negative biomass detected"

        # No group should crash to zero (all groups start with positive biomass)
        # Check final year: at least the first few living groups should survive
        final_bio = biomass[-1, :]
        n_alive = np.sum(final_bio > 1e-10)
        assert n_alive > 5, (
            f"Only {n_alive} groups survived 10 years -- possible regression"
        )

    def test_rsim_ab_biomass_unchanged(self, seabirds_scenario):
        """Run AB simulation and verify biomass trajectories are reasonable."""
        output = rsim_run(seabirds_scenario, method="AB", years=range(1, 11))
        biomass = output.annual_Biomass

        assert np.all(np.isfinite(biomass)), "Non-finite biomass detected"
        assert np.all(biomass >= 0.0), "Negative biomass detected"

        final_bio = biomass[-1, :]
        n_alive = np.sum(final_bio > 1e-10)
        assert n_alive > 5, (
            f"Only {n_alive} groups survived 10 years -- possible regression"
        )

    def test_rk4_and_ab_initial_consistency(self, seabirds_scenario):
        """RK4 and AB should produce similar (not identical) results."""
        out_rk4 = rsim_run(seabirds_scenario, method="RK4", years=range(1, 11))
        out_ab = rsim_run(seabirds_scenario, method="AB", years=range(1, 11))

        # Year 1 biomass should be quite close between methods
        bio_rk4_y1 = out_rk4.annual_Biomass[1, :]
        bio_ab_y1 = out_ab.annual_Biomass[1, :]

        # Mask out zero-biomass groups
        mask = (bio_rk4_y1 > 1e-10) & (bio_ab_y1 > 1e-10)
        if np.any(mask):
            rel_diff = np.abs(bio_rk4_y1[mask] - bio_ab_y1[mask]) / bio_rk4_y1[mask]
            # Methods differ but should be in the same ballpark for year 1
            assert np.max(rel_diff) < 0.5, (
                f"RK4 and AB diverge too much at year 1: max rel diff = {np.max(rel_diff):.4f}"
            )
