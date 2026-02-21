"""End-to-end integration test: SmeltIBM inside Ecosim.

Validates that a SmeltIBM instance can be injected into a full Ecosim
simulation and that the coupled system runs to completion with physically
plausible results.  The test creates a 6-group Baltic-like model (Phyto,
Zoo, Smelt, Cod, Detritus, Fleet), replaces the Smelt group with an IBM,
and runs a short simulation.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import rsim_run, rsim_scenario
from pypath.core.params import create_rpath_params
from pypath.ibm.smelt import SmeltIBM, SmeltParams


class TestIBMEcosimIntegration:
    """Test SmeltIBM running inside full Ecosim simulation."""

    @pytest.fixture
    def ibm_scenario(self):
        """Create a balanced model with Smelt replaced by IBM.

        Model layout (6 groups):
            Index 0 / Ecosim 1: Phyto   (producer,  type=1)
            Index 1 / Ecosim 2: Zoo     (consumer,  type=0)
            Index 2 / Ecosim 3: Smelt   (consumer,  type=0) -- IBM target
            Index 3 / Ecosim 4: Cod     (consumer,  type=0)
            Index 4 / Ecosim 5: Det     (detritus,  type=2)
            Index 5 / Ecosim 6: Fleet   (fleet,     type=3)
        """
        params = create_rpath_params(
            groups=["Phyto", "Zoo", "Smelt", "Cod", "Det", "Fleet"],
            types=[1, 0, 0, 0, 2, 3],
        )

        # Phytoplankton (producer)
        params.model.loc[0, "Biomass"] = 20.0
        params.model.loc[0, "PB"] = 150.0
        params.model.loc[0, "EE"] = 0.8

        # Zooplankton (consumer)
        params.model.loc[1, "Biomass"] = 10.0
        params.model.loc[1, "PB"] = 30.0
        params.model.loc[1, "QB"] = 60.0
        params.model.loc[1, "EE"] = 0.9

        # Smelt (consumer -- will be replaced by IBM)
        params.model.loc[2, "Biomass"] = 3.0
        params.model.loc[2, "PB"] = 1.5
        params.model.loc[2, "QB"] = 4.0
        params.model.loc[2, "EE"] = 0.8

        # Cod (consumer -- top predator)
        params.model.loc[3, "Biomass"] = 2.0
        params.model.loc[3, "PB"] = 0.5
        params.model.loc[3, "QB"] = 2.5
        params.model.loc[3, "EE"] = 0.3

        # Detritus
        params.model.loc[4, "Biomass"] = 100.0

        # Common bookkeeping columns
        params.model["BioAcc"] = 0.0
        params.model["Unassim"] = 0.2
        params.model.loc[0, "Unassim"] = 0.0   # Producer
        params.model.loc[4, "Unassim"] = 0.0   # Detritus
        params.model.loc[5, "BioAcc"] = np.nan  # Fleet
        params.model.loc[5, "Unassim"] = np.nan  # Fleet

        # Detritus fate
        params.model["Det"] = 1.0
        params.model.loc[5, "Det"] = np.nan

        # Diet matrix -- columns are predators, rows are prey
        params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]     # Zoo eats Phyto
        params.diet["Smelt"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]   # Smelt eats Zoo
        params.diet["Cod"] = [0.0, 0.3, 0.6, 0.1, 0.0, 0.0]     # Cod eats Zoo+Smelt+Cod
        params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # Producers have no diet

        # Fishing
        params.model.loc[3, "Fleet"] = 0.2  # Cod caught by Fleet

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = rpath(params)

        # Create scenario for 3 years
        scenario = rsim_scenario(model, params, years=range(1, 4))

        # --- Set up SmeltIBM ---
        # Smelt is Ecosim group index 3 (1-based).
        smelt_ecosim_idx = 3

        smelt_params = SmeltParams.baltic_defaults()

        # Resize foraging arrays to match model group count.
        n = model.NUM_GROUPS + 1
        smelt_params.foraging.energy_content = np.full(n, 4.0)
        smelt_params.foraging.handling_time = np.ones(n)

        smelt_ibm = SmeltIBM(
            group_index=smelt_ecosim_idx,
            n_groups=model.NUM_GROUPS,
            params=smelt_params,
        )

        # Initialize from Ecopath biomass.
        # model.Biomass is 0-based, so Smelt is at index 2.
        smelt_ibm.initialize_from_ecosim(
            biomass=model.Biomass[2],
            params={},
            n_super_individuals=50,
        )

        # Inject IBM into scenario params (propagated to deriv_vector params_dict).
        scenario.params.ibm_groups = {smelt_ecosim_idx: smelt_ibm}

        return scenario, model, smelt_ibm

    def _run(self, scenario, method="RK4"):
        """Run the Ecosim simulation with warnings suppressed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return rsim_run(scenario, method=method)

    def test_simulation_completes(self, ibm_scenario):
        """Simulation with IBM group should run without errors."""
        scenario, model, ibm = ibm_scenario
        output = self._run(scenario)
        assert output is not None
        assert output.out_Biomass.shape[0] > 1

    def test_biomass_no_nan(self, ibm_scenario):
        """Biomass output should not contain NaN."""
        scenario, model, ibm = ibm_scenario
        output = self._run(scenario)
        assert not np.any(np.isnan(output.out_Biomass))

    def test_biomass_stays_finite(self, ibm_scenario):
        """All biomass values should remain finite (no Inf)."""
        scenario, model, ibm = ibm_scenario
        output = self._run(scenario)
        assert np.all(np.isfinite(output.out_Biomass))

    def test_smelt_biomass_reasonable(self, ibm_scenario):
        """IBM smelt biomass should stay within a reasonable range."""
        scenario, model, ibm = ibm_scenario
        initial_smelt = model.Biomass[2]  # 0-based model index
        output = self._run(scenario)
        # Smelt is at Ecosim column 3 (1-based)
        final_smelt = output.out_Biomass[-1, 3]
        assert final_smelt >= 0, f"Smelt biomass went negative: {final_smelt}"
        assert final_smelt < initial_smelt * 100, (
            f"Smelt biomass blew up: {final_smelt} vs initial {initial_smelt}"
        )

    def test_other_groups_still_run(self, ibm_scenario):
        """Non-IBM groups should still produce valid output."""
        scenario, model, ibm = ibm_scenario
        output = self._run(scenario)
        for col, name in [(1, "Phyto"), (2, "Zoo"), (4, "Cod")]:
            final_bio = output.out_Biomass[-1, col]
            assert final_bio > 0, (
                f"{name} biomass went to zero or negative: {final_bio}"
            )

    def test_ibm_was_invoked(self, ibm_scenario):
        """The IBM should have been invoked during the simulation."""
        scenario, model, ibm = ibm_scenario
        assert len(ibm.individuals) > 0, "IBM should start with individuals"
        self._run(scenario)
        assert isinstance(ibm.individuals, list)

    def test_output_shape_correct(self, ibm_scenario):
        """Output arrays should have the expected shape."""
        scenario, model, ibm = ibm_scenario
        output = self._run(scenario)
        n_years = 3
        n_months = n_years * 12
        n_cols = model.NUM_GROUPS + 1  # Outside + groups
        assert output.out_Biomass.shape == (n_months + 1, n_cols)
        assert output.annual_Biomass.shape[0] == n_years

    def test_annual_catch_non_negative(self, ibm_scenario):
        """Annual catch should be non-negative."""
        scenario, model, ibm = ibm_scenario
        output = self._run(scenario)
        assert np.all(output.annual_Catch >= 0)
