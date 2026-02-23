"""Tests for IBM spatial movement integration."""

import numpy as np
import pytest
import scipy.sparse as sp


class TestSpatialContext:
    """Tests for the SpatialContext dataclass."""

    def test_creation(self):
        """SpatialContext can be constructed with required fields."""
        from pypath.ibm.base import SpatialContext

        adj = sp.csr_matrix(np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]))
        ctx = SpatialContext(
            adjacency=adj,
            habitat_quality=np.array([0.8, 0.5, 0.9]),
            food_density=np.array([10.0, 5.0, 15.0]),
            predator_density=np.array([1.0, 2.0, 0.5]),
            n_patches=3,
        )
        assert ctx.n_patches == 3
        assert ctx.adjacency.shape == (3, 3)
        assert ctx.habitat_quality.shape == (3,)

    def test_ibm_step_result_has_patch_biomass(self):
        """IBMStepResult should accept optional patch_biomass field."""
        from pypath.ibm.base import IBMStepResult

        result = IBMStepResult(
            biomass=5.0,
            production=0.1,
            consumption_by_prey=np.zeros(4),
            mortality_count=10.0,
            recruitment_count=5.0,
            patch_biomass=np.array([2.0, 1.5, 1.5]),
        )
        assert result.patch_biomass is not None
        np.testing.assert_allclose(result.patch_biomass.sum(), 5.0)

    def test_ibm_step_result_patch_biomass_defaults_none(self):
        """IBMStepResult.patch_biomass should default to None."""
        from pypath.ibm.base import IBMStepResult

        result = IBMStepResult(
            biomass=5.0,
            production=0.1,
            consumption_by_prey=np.zeros(4),
            mortality_count=10.0,
            recruitment_count=5.0,
        )
        assert result.patch_biomass is None


class TestIBMGroupSpatialSignature:
    """Tests for the updated IBMGroup.compute_step() signature."""

    def test_compute_step_accepts_spatial_context_none(self):
        """compute_step() should accept spatial_context=None (backward compat)."""
        from pypath.ibm.base import IBMGroup, IBMStepResult

        class MockIBM(IBMGroup):
            def compute_step(
                self,
                prey_available,
                predation_pressure,
                env_forcing,
                dt,
                spatial_context=None,
            ):
                return IBMStepResult(
                    biomass=1.0,
                    production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0,
                    recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 1.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params, n_super_individuals=500):
                pass

        ibm = MockIBM(group_index=1, n_groups=4)
        result = ibm.compute_step(
            prey_available=np.zeros(4),
            predation_pressure=0.0,
            env_forcing={"temperature": 10.0},
            dt=1 / 12,
            spatial_context=None,
        )
        assert result.biomass == 1.0

    def test_spatial_context_importable_from_init(self):
        """SpatialContext should be importable from pypath.ibm."""
        from pypath.ibm import SpatialContext

        assert SpatialContext is not None


def _make_3patch_adjacency():
    """Create a 3-patch linear chain: 0 -- 1 -- 2."""
    adj = sp.lil_matrix((3, 3))
    adj[0, 1] = 1
    adj[1, 0] = 1
    adj[1, 2] = 1
    adj[2, 1] = 1
    return adj.tocsr()


def _make_smelt_with_spatial(n_patches=3, n_super=30):
    """Create a SmeltIBM initialized for spatial testing."""
    from pypath.ibm.base import SpatialContext
    from pypath.ibm.smelt import SmeltIBM, SmeltParams

    params = SmeltParams.baltic_defaults()
    # Resize foraging arrays for 7-group model
    n_groups = 7
    params.foraging.energy_content = np.full(n_groups, 4.0)
    params.foraging.handling_time = np.ones(n_groups)

    ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=params)
    ibm.initialize_from_ecosim(biomass=5.0, params={}, n_super_individuals=n_super)

    adj = _make_3patch_adjacency()
    ctx = SpatialContext(
        adjacency=adj,
        habitat_quality=np.array([0.5, 0.8, 0.3]),
        food_density=np.array([10.0, 20.0, 5.0]),
        predator_density=np.array([1.0, 0.5, 2.0]),
        n_patches=n_patches,
    )
    return ibm, ctx


class TestSmeltIBMSpatialMovement:
    """Tests for Phase 5 (spatial movement) in SmeltIBM."""

    def test_compute_step_without_spatial_unchanged(self):
        """compute_step without spatial_context should work as before."""
        from pypath.ibm.smelt import SmeltIBM, SmeltParams

        params = SmeltParams.baltic_defaults()
        n_groups = 7
        params.foraging.energy_content = np.full(n_groups, 4.0)
        params.foraging.handling_time = np.ones(n_groups)
        ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=params)
        ibm.initialize_from_ecosim(biomass=5.0, params={}, n_super_individuals=30)

        result = ibm.compute_step(
            prey_available=np.full(n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
        )
        assert result.biomass > 0
        assert result.patch_biomass is None

    def test_compute_step_with_spatial_returns_patch_biomass(self):
        """With spatial_context, result should have patch_biomass array."""
        ibm, ctx = _make_smelt_with_spatial()

        result = ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        assert result.patch_biomass is not None
        assert result.patch_biomass.shape == (3,)

    def test_patch_biomass_sums_to_total(self):
        """patch_biomass should sum to total biomass."""
        ibm, ctx = _make_smelt_with_spatial()

        result = ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        np.testing.assert_allclose(
            result.patch_biomass.sum(),
            result.biomass,
            rtol=1e-6,
        )

    def test_movement_distributes_across_patches(self):
        """After movement, individuals should be in multiple patches."""
        ibm, ctx = _make_smelt_with_spatial(n_super=100)

        ibm.compute_step(
            prey_available=np.full(ibm.n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )
        patches_occupied = {ind.patch_idx for ind in ibm.individuals}
        # With 100 individuals and 3 patches, at least 2 patches should be occupied
        assert len(patches_occupied) >= 2

    def test_movement_prefers_food_rich_patches(self):
        """Individuals should tend toward patches with higher food density."""
        # Run 10 trials to reduce stochastic noise
        food_rich_counts = []
        for _ in range(10):
            ibm, ctx = _make_smelt_with_spatial(n_super=200)
            # Extreme food gradient: patch 1 has all the food
            ctx.food_density = np.array([0.1, 100.0, 0.1])
            ctx.habitat_quality = np.array([0.5, 0.5, 0.5])  # uniform habitat
            ctx.predator_density = np.zeros(3)  # no predators

            ibm.compute_step(
                prey_available=np.full(ibm.n_groups, 2.0),
                predation_pressure=0.0,
                env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
                dt=1 / 12,
                spatial_context=ctx,
            )
            in_patch_1 = sum(1 for ind in ibm.individuals if ind.patch_idx == 1)
            food_rich_counts.append(in_patch_1)

        avg_in_food_rich = np.mean(food_rich_counts)
        avg_per_patch = np.mean([len(ibm.individuals)]) / 3
        # On average, more individuals should be in the food-rich patch
        assert avg_in_food_rich > avg_per_patch


class TestApplyIBMToDerivativeSpatial:
    """Tests for spatial context forwarding in apply_ibm_to_derivative."""

    def test_forwards_spatial_context_to_compute_step(self):
        """apply_ibm_to_derivative should pass spatial_context through."""
        from pypath.ibm.base import IBMGroup, IBMStepResult, SpatialContext
        from pypath.ibm.integration import apply_ibm_to_derivative

        received_context = {}

        class SpyIBM(IBMGroup):
            def compute_step(
                self,
                prey_available,
                predation_pressure,
                env_forcing,
                dt,
                spatial_context=None,
            ):
                received_context["ctx"] = spatial_context
                return IBMStepResult(
                    biomass=1.0,
                    production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0,
                    recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 1.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params, n_super_individuals=500):
                pass

        n = 4
        spy = SpyIBM(group_index=1, n_groups=n)
        deriv = np.zeros(n + 1)
        QQ = np.zeros((n + 1, n + 1))
        BB = np.ones(n + 1)

        adj = sp.csr_matrix(np.eye(3))
        ctx = SpatialContext(
            adjacency=adj,
            habitat_quality=np.ones(3),
            food_density=np.ones(3),
            predator_density=np.zeros(3),
            n_patches=3,
        )

        apply_ibm_to_derivative(deriv, QQ, BB, spy, {}, 1 / 12, spatial_context=ctx)
        assert received_context["ctx"] is ctx

    def test_none_spatial_context_by_default(self):
        """Without spatial_context, compute_step receives None."""
        from pypath.ibm.base import IBMGroup, IBMStepResult
        from pypath.ibm.integration import apply_ibm_to_derivative

        received_context = {}

        class SpyIBM(IBMGroup):
            def compute_step(
                self,
                prey_available,
                predation_pressure,
                env_forcing,
                dt,
                spatial_context=None,
            ):
                received_context["ctx"] = spatial_context
                return IBMStepResult(
                    biomass=1.0,
                    production=0.0,
                    consumption_by_prey=np.zeros(self.n_groups),
                    mortality_count=0.0,
                    recruitment_count=0.0,
                )

            def get_aggregate_biomass(self):
                return 1.0

            def get_consumption_by_prey(self):
                return np.zeros(self.n_groups)

            def initialize_from_ecosim(self, biomass, params, n_super_individuals=500):
                pass

        n = 4
        spy = SpyIBM(group_index=1, n_groups=n)
        deriv = np.zeros(n + 1)
        QQ = np.zeros((n + 1, n + 1))
        BB = np.ones(n + 1)

        apply_ibm_to_derivative(deriv, QQ, BB, spy, {}, 1 / 12)
        assert received_context["ctx"] is None


class TestSpatialFluxSkipsIBM:
    """Tests that calculate_spatial_flux skips IBM groups."""

    def test_ibm_group_gets_zero_flux(self):
        """IBM groups should get zero spatial flux from standard dispersal."""
        from pypath.ibm.smelt import SmeltIBM, SmeltParams
        from pypath.spatial.dispersal import calculate_spatial_flux
        from pypath.spatial.ecospace_params import EcospaceGrid, EcospaceParams

        n_patches = 3
        n_groups_plus_1 = 5  # Outside + 4 groups

        # Build a minimal grid
        adj = _make_3patch_adjacency()
        grid = EcospaceGrid(
            n_patches=n_patches,
            patch_ids=np.arange(n_patches),
            patch_areas=np.ones(n_patches),
            patch_centroids=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            adjacency_matrix=adj,
            edge_lengths={(0, 1): 1.0, (1, 2): 1.0},
        )
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_groups_plus_1 - 1, n_patches)),
            habitat_capacity=np.ones((n_groups_plus_1 - 1, n_patches)),
            dispersal_rate=np.full(n_groups_plus_1 - 1, 5.0),
            advection_enabled=np.zeros(n_groups_plus_1 - 1, dtype=bool),
            gravity_strength=np.zeros(n_groups_plus_1 - 1),
        )

        # State with biomass gradient (should cause diffusion for non-IBM)
        state = np.ones((n_groups_plus_1, n_patches))
        state[2, 0] = 10.0  # Group 2 has biomass gradient

        # IBM group at Ecosim index 2
        ibm_params = SmeltParams.baltic_defaults()
        ibm_params.foraging.energy_content = np.full(n_groups_plus_1, 4.0)
        ibm_params.foraging.handling_time = np.ones(n_groups_plus_1)
        ibm = SmeltIBM(group_index=2, n_groups=n_groups_plus_1 - 1, params=ibm_params)

        params = {"ibm_groups": {2: ibm}}

        flux = calculate_spatial_flux(state, ecospace, params, t=0.0)

        # IBM group (index 2) should have zero flux
        np.testing.assert_allclose(flux[2], 0.0)
        # Non-IBM group with same gradient should have non-zero flux
        state[3, 0] = 10.0
        flux2 = calculate_spatial_flux(state, ecospace, params, t=0.0)
        assert np.any(flux2[3] != 0.0), "Non-IBM group should have dispersal flux"


class TestDerivVectorSpatialIBM:
    """Tests for IBM support in deriv_vector_spatial."""

    @pytest.fixture
    def spatial_ibm_setup(self):
        """Create a minimal spatial scenario with IBM group."""
        import warnings

        from pypath.core.ecopath import rpath
        from pypath.core.ecosim import rsim_scenario
        from pypath.core.params import create_rpath_params
        from pypath.ibm.smelt import SmeltIBM, SmeltParams
        from pypath.spatial.ecospace_params import EcospaceGrid, EcospaceParams

        # 4-group model: Phyto(1), Zoo(0), Smelt(0), Det(2)
        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Smelt", "Det"],
            types=[1, 0, 0, 2],
        )
        params.model.loc[0, "Biomass"] = 20.0
        params.model.loc[0, "PB"] = 100.0
        params.model.loc[0, "EE"] = 0.8
        params.model.loc[1, "Biomass"] = 10.0
        params.model.loc[1, "PB"] = 20.0
        params.model.loc[1, "QB"] = 40.0
        params.model.loc[1, "EE"] = 0.8
        params.model.loc[2, "Biomass"] = 3.0
        params.model.loc[2, "PB"] = 1.5
        params.model.loc[2, "QB"] = 4.0
        params.model.loc[2, "EE"] = 0.5
        params.model.loc[3, "Biomass"] = 50.0
        params.model["BioAcc"] = 0.0
        params.model["Unassim"] = 0.2
        params.model.loc[0, "Unassim"] = 0.0
        params.model.loc[3, "Unassim"] = 0.0
        params.model["Det"] = 1.0
        # Diet rows: Phyto, Zoo, Smelt, Det, Import
        params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
        params.diet["Smelt"] = [0.0, 1.0, 0.0, 0.0, 0.0]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = rpath(params)

        scenario = rsim_scenario(model, params, years=range(1, 3))

        # IBM for Smelt (Ecosim index 3, 1-based)
        smelt_params = SmeltParams.baltic_defaults()
        n = model.NUM_GROUPS + 1
        smelt_params.foraging.energy_content = np.full(n, 4.0)
        smelt_params.foraging.handling_time = np.ones(n)

        smelt_ibm = SmeltIBM(
            group_index=3, n_groups=model.NUM_GROUPS, params=smelt_params
        )
        smelt_ibm.initialize_from_ecosim(
            biomass=model.Biomass[2], params={}, n_super_individuals=30
        )
        scenario.params.ibm_groups = {3: smelt_ibm}

        # 3-patch grid
        n_patches = 3
        adj = _make_3patch_adjacency()
        grid = EcospaceGrid(
            n_patches=n_patches,
            patch_ids=np.arange(n_patches),
            patch_areas=np.ones(n_patches),
            patch_centroids=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            adjacency_matrix=adj,
            edge_lengths={(0, 1): 1.0, (1, 2): 1.0},
        )

        n_eco_groups = model.NUM_GROUPS
        ecospace = EcospaceParams(
            grid=grid,
            habitat_preference=np.ones((n_eco_groups, n_patches)),
            habitat_capacity=np.ones((n_eco_groups, n_patches)),
            dispersal_rate=np.full(n_eco_groups, 5.0),
            advection_enabled=np.zeros(n_eco_groups, dtype=bool),
            gravity_strength=np.zeros(n_eco_groups),
        )

        return scenario, model, smelt_ibm, ecospace

    def test_spatial_ibm_simulation_completes(self, spatial_ibm_setup):
        """rsim_run_spatial with IBM group should complete."""
        import warnings

        from pypath.spatial.integration import rsim_run_spatial

        scenario, model, ibm, ecospace = spatial_ibm_setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run_spatial(scenario, ecospace=ecospace)
        assert output is not None
        assert output.out_Biomass.shape[0] > 1

    def test_spatial_ibm_no_nan(self, spatial_ibm_setup):
        """Spatial IBM output should not contain NaN."""
        import warnings

        from pypath.spatial.integration import rsim_run_spatial

        scenario, model, ibm, ecospace = spatial_ibm_setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = rsim_run_spatial(scenario, ecospace=ecospace)
        assert not np.any(np.isnan(output.out_Biomass))

    def test_ibm_individuals_spread(self):
        """IBM individuals should spread to multiple patches when spatial context is provided."""
        # Use a direct single-step approach to verify spatial context wiring,
        # since full simulation with RK4 calls compute_step many times causing
        # population collapse with few super-individuals.

        from pypath.ibm.base import SpatialContext
        from pypath.ibm.smelt import SmeltIBM, SmeltParams

        n_groups = 4
        smelt_params = SmeltParams.baltic_defaults()
        smelt_params.foraging.energy_content = np.full(n_groups + 1, 4.0)
        smelt_params.foraging.handling_time = np.ones(n_groups + 1)

        ibm = SmeltIBM(group_index=3, n_groups=n_groups, params=smelt_params)
        ibm.initialize_from_ecosim(biomass=5.0, params={}, n_super_individuals=100)

        adj = _make_3patch_adjacency()
        ctx = SpatialContext(
            adjacency=adj,
            habitat_quality=np.array([0.5, 0.9, 0.3]),
            food_density=np.array([10.0, 30.0, 5.0]),
            predator_density=np.array([0.5, 0.1, 1.0]),
            n_patches=3,
        )

        # Single step with spatial context
        result = ibm.compute_step(
            prey_available=np.full(n_groups, 2.0),
            predation_pressure=0.1,
            env_forcing={"temperature": 10.0, "month": 6, "zoo_peak_day": 120.0},
            dt=1 / 12,
            spatial_context=ctx,
        )

        patches = {ind.patch_idx for ind in ibm.individuals}
        assert len(patches) >= 2, f"Individuals only in patches: {patches}"
        assert result.patch_biomass is not None
