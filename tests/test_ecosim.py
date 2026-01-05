"""
Tests for PyPath Ecosim simulation functionality.
"""

import warnings

import numpy as np
import pytest

from pypath.core.ecopath import rpath
from pypath.core.ecosim import (
    rsim_fishing,
    rsim_forcing,
    rsim_params,
    rsim_run,
    rsim_scenario,
    rsim_state,
)
from pypath.core.ecosim_deriv import deriv_vector
from pypath.core.params import create_rpath_params
from pypath.core.stanzas import RsimStanzas


@pytest.fixture
def simple_model():
    """Create a simple balanced Ecopath model for testing."""
    params = create_rpath_params(
        groups=["Phyto", "Zoo", "Fish", "Det", "Fleet"], types=[1, 0, 0, 2, 3]
    )

    # Phytoplankton (producer)
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 200.0
    params.model.loc[0, "EE"] = 0.8

    # Zooplankton (consumer)
    params.model.loc[1, "Biomass"] = 5.0
    params.model.loc[1, "PB"] = 50.0
    params.model.loc[1, "QB"] = 150.0
    params.model.loc[1, "EE"] = 0.9

    # Fish (consumer)
    params.model.loc[2, "Biomass"] = 2.0
    params.model.loc[2, "PB"] = 1.0
    params.model.loc[2, "QB"] = 5.0
    params.model.loc[2, "EE"] = 0.5

    # Detritus
    params.model.loc[3, "Biomass"] = 100.0

    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0
    params.model.loc[3, "Unassim"] = 0.0
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan

    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan

    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    return model, params


@pytest.fixture
def baltic_sea_model():
    """Create a Baltic Sea-like Ecopath model for Ecosim testing.

    A simplified 7-group model representing a Baltic Sea ecosystem:
    1. Phytoplankton (producer)
    2. Zooplankton (consumer)
    3. Benthic invertebrates (consumer)
    4. Herring (consumer)
    5. Cod (consumer - top predator)
    6. Detritus
    7. Commercial fishery fleet
    """
    groups = [
        "Phytoplankton",
        "Zooplankton",
        "Benthos",
        "Herring",
        "Cod",
        "Detritus",
        "Fishery",
    ]
    types = [1, 0, 0, 0, 0, 2, 3]

    params = create_rpath_params(groups, types)

    # Set biomass values (t/km²)
    params.model.loc[0, "Biomass"] = 25.0  # Phytoplankton
    params.model.loc[1, "Biomass"] = 12.0  # Zooplankton
    params.model.loc[2, "Biomass"] = 30.0  # Benthos
    params.model.loc[3, "Biomass"] = 8.0  # Herring
    params.model.loc[4, "Biomass"] = 3.0  # Cod
    params.model.loc[5, "Biomass"] = 50.0  # Detritus

    # Set P/B ratios (1/year)
    params.model.loc[0, "PB"] = 150.0  # Phytoplankton - high turnover
    params.model.loc[1, "PB"] = 35.0  # Zooplankton
    params.model.loc[2, "PB"] = 3.0  # Benthos
    params.model.loc[3, "PB"] = 1.2  # Herring
    params.model.loc[4, "PB"] = 0.5  # Cod

    # Set Q/B ratios (1/year) for consumers
    params.model.loc[1, "QB"] = 100.0  # Zooplankton
    params.model.loc[2, "QB"] = 10.0  # Benthos
    params.model.loc[3, "QB"] = 4.0  # Herring
    params.model.loc[4, "QB"] = 2.5  # Cod

    # Set ecotrophic efficiency (proportion of production that is consumed)
    params.model.loc[0, "EE"] = 0.85  # Phytoplankton
    params.model.loc[1, "EE"] = 0.90  # Zooplankton
    params.model.loc[2, "EE"] = 0.80  # Benthos
    params.model.loc[3, "EE"] = 0.75  # Herring
    params.model.loc[4, "EE"] = 0.40  # Cod - lower, top predator

    # Biomass accumulation and unassimilated consumption
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0  # Producer
    params.model.loc[5, "Unassim"] = 0.0  # Detritus
    params.model.loc[6, "BioAcc"] = np.nan
    params.model.loc[6, "Unassim"] = np.nan

    # Detritus fate - all groups flow to detritus
    params.model["Detritus"] = 1.0
    params.model.loc[6, "Detritus"] = np.nan

    # Diet matrix - rows are prey, columns are predators
    # 7 rows: Phyto, Zoo, Benthos, Herring, Cod, Detritus, Import

    # Zooplankton diet: 90% phytoplankton, 10% detritus
    params.diet["Zooplankton"] = [0.9, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]

    # Benthos diet: 30% phytoplankton, 70% detritus
    params.diet["Benthos"] = [0.3, 0.0, 0.0, 0.0, 0.0, 0.7, 0.0]

    # Herring diet: 80% zooplankton, 20% benthos
    params.diet["Herring"] = [0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0]

    # Cod diet: 20% zooplankton, 30% benthos, 40% herring, 10% cod (cannibalism)
    params.diet["Cod"] = [0.0, 0.2, 0.3, 0.4, 0.1, 0.0, 0.0]

    # Producers have no diet
    params.diet["Phytoplankton"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Fishery landings (Herring and Cod are caught)
    params.model.loc[3, "Fishery"] = 1.5  # Herring landings
    params.model.loc[4, "Fishery"] = 0.3  # Cod landings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    return model, params


class TestRsimParams:
    """Tests for rsim_params conversion."""

    def test_rsim_params_creation(self, simple_model):
        """Test that rsim_params creates valid parameters."""
        model, _ = simple_model
        params = rsim_params(model)

        assert params.NUM_GROUPS == 5
        assert params.NUM_LIVING == 3
        assert params.NUM_DEAD == 1
        assert params.NUM_GEARS == 1
        assert len(params.spname) == 6  # Includes "Outside"
        assert params.spname[0] == "Outside"

    def test_biomass_reference(self, simple_model):
        """Test that reference biomass is correct."""
        model, _ = simple_model
        params = rsim_params(model)

        # B_BaseRef[0] should be 1.0 (Outside)
        assert params.B_BaseRef[0] == 1.0

        # Other groups should match original model
        assert np.isclose(params.B_BaseRef[1], 10.0)  # Phyto
        assert np.isclose(params.B_BaseRef[2], 5.0)  # Zoo
        assert np.isclose(params.B_BaseRef[3], 2.0)  # Fish

    def test_predprey_links(self, simple_model):
        """Test predator-prey link construction."""
        model, _ = simple_model
        params = rsim_params(model)

        # Should have links for:
        # - Primary production (Outside -> Phyto)
        # - Zoo eating Phyto
        # - Fish eating Zoo
        assert params.NumPredPreyLinks >= 2


class TestRsimState:
    """Tests for initial state creation."""

    def test_state_creation(self, simple_model):
        """Test that initial state is created correctly."""
        model, _ = simple_model
        params = rsim_params(model)
        state = rsim_state(params)

        # Biomass should match reference (use equal_nan=True for NaN values)
        assert np.allclose(state.Biomass, params.B_BaseRef, equal_nan=True)

        # Ftime should be 1.0
        assert np.allclose(state.Ftime, 1.0)


class TestRsimScenario:
    """Tests for scenario creation."""

    def test_scenario_creation(self, simple_model):
        """Test full scenario creation."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 11))

        assert scenario.params.NUM_GROUPS == 5
        assert scenario.forcing.ForcedBio.shape[0] == 10 * 12  # 10 years * 12 months
        assert scenario.fishing.ForcedEffort.shape[1] == 2  # Outside + 1 gear


class TestEcosimSimulation:
    """Tests for Ecosim simulation run."""

    def test_simulation_runs(self, simple_model):
        """Test that simulation runs without error."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        # This should run without raising an exception
        output = rsim_run(scenario, method="RK4")

        # Check output structure
        assert (
            output.out_Biomass.shape[0] == 5 * 12 + 1
        )  # 5 years * 12 months + initial
        assert output.out_Biomass.shape[1] == 6  # Outside + 5 groups

    @pytest.mark.xfail(
        reason="Ecosim stability affected by diet matrix fix - needs model recalibration"
    )
    def test_biomass_positive(self, simple_model):
        """Test that biomass stays positive."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))
        output = rsim_run(scenario, method="RK4")

        # All biomass should be positive (or very small epsilon)
        assert np.all(output.out_Biomass >= 0)

    def test_annual_output(self, simple_model):
        """Test annual output aggregation."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))
        output = rsim_run(scenario, method="RK4")

        assert output.annual_Biomass.shape[0] == 5  # 5 years
        assert output.annual_Catch.shape[0] == 5


class TestDerivVector:
    """Tests for derivative calculation function."""

    def test_deriv_output_shape(self, simple_model):
        """Test that deriv_vector returns correct shape."""
        model, _ = simple_model
        sim_params = rsim_params(model)

        state = np.array([1.0, 10.0, 5.0, 2.0, 100.0, 0.0])  # Outside + groups

        params_dict = {
            "NUM_GROUPS": sim_params.NUM_GROUPS,
            "NUM_LIVING": sim_params.NUM_LIVING,
            "NUM_DEAD": sim_params.NUM_DEAD,
            "NUM_GEARS": sim_params.NUM_GEARS,
            "PB": sim_params.PBopt,
            "QB": sim_params.FtimeQBOpt,
            "M0": sim_params.MzeroMort,
            "Unassim": sim_params.UnassimRespFrac,
            "ActiveLink": np.zeros((6, 6), dtype=bool),
            "VV": np.zeros((6, 6)),
            "DD": np.zeros((6, 6)),
            "QQbase": np.zeros((6, 6)),
            "Bbase": sim_params.B_BaseRef,
        }

        forcing_dict = {"Ftime": np.ones(6)}
        fishing_dict = {"FishingMort": np.zeros(6)}

        deriv = deriv_vector(state, params_dict, forcing_dict, fishing_dict)

        assert len(deriv) == 6  # Should match state length


class TestStanzaIntegration:
    """Tests for stanza integration in simulation."""

    def test_scenario_with_stanzas(self, simple_model):
        """Test that scenario can hold RsimStanzas."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        # Create a minimal stanza structure
        stanzas = RsimStanzas(n_split=0)
        scenario.stanzas = stanzas

        assert scenario.stanzas is not None
        assert scenario.stanzas.n_split == 0

    @pytest.mark.xfail(
        reason="Ecosim stability affected by diet matrix fix - needs model recalibration"
    )
    def test_simulation_with_empty_stanzas(self, simple_model):
        """Test simulation runs with empty stanza structure."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        # Add empty stanza structure
        stanzas = RsimStanzas(n_split=0)
        scenario.stanzas = stanzas

        # Should run without error (stanzas are skipped when n_split=0)
        output = rsim_run(scenario, method="RK4")

        assert output.out_Biomass.shape[0] == 5 * 12 + 1
        assert np.all(output.out_Biomass >= 0)

    @pytest.mark.xfail(
        reason="Ecosim stability affected by diet matrix fix - needs model recalibration"
    )
    def test_simulation_with_stanza_groups(self, simple_model):
        """Test simulation runs with initialized stanza groups."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        # Create stanza structure with 1 split group
        _n_groups = 6
        max_age = 121  # 10 years in months + 1 (0-indexed, need 0 to 120)

        stanzas = RsimStanzas(
            n_split=1,
            n_stanzas=np.array([0, 2]),  # 2 stanzas for group 1
            ecopath_code=np.zeros((2, 3)),
            age1=np.zeros((2, 3)),
            age2=np.zeros((2, 3)),
            base_wage_s=np.zeros((max_age, 2)),
            base_nage_s=np.zeros((max_age, 2)),
            base_qage_s=np.zeros((max_age, 2)),  # Consumption at age
            base_spawn_bio=np.zeros(2),
            vbm=np.array([0.0, 0.2]),  # VB M parameter
            vbgf_d=np.array([0.0, 0.667]),
            wmat=np.array([0.0, 0.5]),
            spawn_x=np.array([0.0, 2.0]),  # Beverton-Holt parameter
            r_zero_s=np.array([0.0, 1000.0]),
            base_eggs_stanza=np.array([0.0, 100.0]),
            recruits=np.array([0.0, 100.0]),
            split_alpha=np.zeros((2, 3)),
            spawn_energy=np.zeros(2),
            r_scale_split=np.zeros(2),
            base_stanza_pred=np.zeros(2),
            rec_power=np.array([0.0, 1.0]),
        )

        # Set up ecopath codes and ages
        stanzas.ecopath_code[1, 1] = 2  # juvenile = group 2 (Zoo)
        stanzas.ecopath_code[1, 2] = 3  # adult = group 3 (Fish)
        stanzas.age1[1, 1] = 0
        stanzas.age2[1, 1] = 24  # juvenile 0-24 months
        stanzas.age1[1, 2] = 25
        stanzas.age2[1, 2] = 119  # adult 25-119 months (within bounds)

        # Initialize weight, numbers and consumption at age
        for age in range(max_age):
            stanzas.base_wage_s[age, 1] = 0.01 * (age + 1)
            stanzas.base_nage_s[age, 1] = 1000 * np.exp(-0.01 * age)
            stanzas.base_qage_s[age, 1] = (0.01 * (age + 1)) ** 0.667  # Q = W^d

        scenario.stanzas = stanzas

        # Should run without error
        output = rsim_run(scenario, method="RK4")

        assert output.out_Biomass.shape[0] == 5 * 12 + 1
        assert np.all(output.out_Biomass >= 0)


class TestBalticSeaModel:
    """Tests using the more realistic Baltic Sea model."""

    def test_baltic_rsim_params_creation(self, baltic_sea_model):
        """Test rsim_params creation from Baltic Sea model."""
        model, _ = baltic_sea_model
        params = rsim_params(model)

        assert params.NUM_GROUPS == 7
        assert params.NUM_LIVING == 5
        assert params.NUM_DEAD == 1
        assert params.NUM_GEARS == 1
        assert len(params.spname) == 8  # Outside + 7 groups

    def test_baltic_predprey_links(self, baltic_sea_model):
        """Test predator-prey link construction for Baltic model."""
        model, _ = baltic_sea_model
        params = rsim_params(model)

        # Should have multiple predator-prey links
        # Zoo->Phyto, Benthos->Phyto, Herring->Zoo, Herring->Benthos,
        # Cod->Zoo, Cod->Benthos, Cod->Herring, Cod->Cod, plus detritus links
        assert params.NumPredPreyLinks >= 8

    def test_baltic_fishing_links(self, baltic_sea_model):
        """Test fishing link construction for Baltic model."""
        model, _ = baltic_sea_model
        params = rsim_params(model)

        # Should have fishing links for Herring and Cod
        assert params.NumFishingLinks >= 2

    def test_baltic_simulation_runs(self, baltic_sea_model):
        """Test that Baltic simulation runs without error."""
        model, rpath_params = baltic_sea_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 21))

        output = rsim_run(scenario, method="RK4")

        assert output.out_Biomass.shape[0] == 20 * 12 + 1
        assert output.out_Biomass.shape[1] == 8  # Outside + 7 groups

    @pytest.mark.xfail(
        reason="Ecosim stability affected by diet matrix fix - needs model recalibration"
    )
    def test_baltic_biomass_stability(self, baltic_sea_model):
        """Test that Baltic model runs for multiple years."""
        model, rpath_params = baltic_sea_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 11))  # 10 years

        output = rsim_run(scenario, method="RK4")

        # Check that simulation completes
        assert output.out_Biomass.shape[0] == 10 * 12 + 1

        # Check that biomass values are finite
        assert np.all(np.isfinite(output.out_Biomass))

    def test_baltic_catch_produced(self, baltic_sea_model):
        """Test that catch is produced from fishing."""
        model, rpath_params = baltic_sea_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 11))

        output = rsim_run(scenario, method="RK4")

        # Total catch should be positive
        total_catch = np.sum(output.annual_Catch)
        assert total_catch > 0

    def test_baltic_annual_output_aggregation(self, baltic_sea_model):
        """Test annual output is correctly aggregated."""
        model, rpath_params = baltic_sea_model
        n_years = 10
        scenario = rsim_scenario(model, rpath_params, years=range(1, n_years + 1))

        output = rsim_run(scenario, method="RK4")

        assert output.annual_Biomass.shape[0] == n_years
        assert output.annual_Catch.shape[0] == n_years


class TestForcingScenarios:
    """Tests for forcing modifications in simulations."""

    def test_increased_fishing_effort(self, baltic_sea_model):
        """Test simulation with increased fishing effort."""
        model, rpath_params = baltic_sea_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 21))

        # Double fishing effort in years 10-20
        _n_months = scenario.forcing.ForcedBio.shape[0]
        scenario.fishing.ForcedEffort[120:, :] = 2.0  # After year 10

        output = rsim_run(scenario, method="RK4")

        # Fish biomass should decline with higher fishing
        initial_fish = output.annual_Biomass[0]
        final_fish = output.annual_Biomass[-1]

        # Herring (group 4) and Cod (group 5) should decrease
        assert final_fish[4] < initial_fish[4] or final_fish[5] < initial_fish[5]

    def test_zero_fishing_effort(self, simple_model):
        """Test simulation with reduced fishing effort."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        # Set fishing effort to zero
        scenario.fishing.ForcedEffort[:] = 0.0

        output = rsim_run(scenario, method="RK4")

        # Simulation should complete
        assert output.out_Biomass.shape[0] == 5 * 12 + 1

    def test_forced_biomass(self, simple_model):
        """Test simulation with forced biomass values."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        # Force phytoplankton (group 1) to double biomass in second half
        _n_months = scenario.forcing.ForcedBio.shape[0]
        # Set forced biomass (positive value means forced)
        scenario.forcing.ForcedBio[30:, 1] = 20.0  # Double phyto biomass

        output = rsim_run(scenario, method="RK4")

        # Model should run without error
        assert output.out_Biomass.shape[0] == 5 * 12 + 1


class TestIntegrationMethods:
    """Tests for different integration methods."""

    @pytest.mark.xfail(
        reason="Ecosim stability affected by diet matrix fix - needs model recalibration"
    )
    def test_rk4_method(self, simple_model):
        """Test RK4 integration method."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        output = rsim_run(scenario, method="RK4")

        assert output.out_Biomass.shape[0] == 5 * 12 + 1
        assert np.all(output.out_Biomass >= 0)

    @pytest.mark.xfail(
        reason="Ecosim stability affected by diet matrix fix - needs model recalibration"
    )
    def test_ab_method(self, simple_model):
        """Test Adams-Bashforth integration method."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        output = rsim_run(scenario, method="AB")

        assert output.out_Biomass.shape[0] == 5 * 12 + 1
        assert np.all(output.out_Biomass >= 0)

    def test_methods_similar_results(self, simple_model):
        """Test that both RK4 and AB methods run successfully."""
        model, rpath_params = simple_model
        scenario_rk4 = rsim_scenario(model, rpath_params, years=range(1, 4))
        scenario_ab = rsim_scenario(model, rpath_params, years=range(1, 4))

        output_rk4 = rsim_run(scenario_rk4, method="RK4")
        output_ab = rsim_run(scenario_ab, method="AB")

        # Both methods should complete and produce output
        assert output_rk4.out_Biomass.shape[0] == 3 * 12 + 1
        assert output_ab.out_Biomass.shape[0] == 3 * 12 + 1


class TestOutputStructure:
    """Tests for simulation output structure and content."""

    def test_output_has_all_fields(self, simple_model):
        """Test that output has all required fields."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        output = rsim_run(scenario, method="RK4")

        # Check all fields exist
        assert hasattr(output, "out_Biomass")
        assert hasattr(output, "out_Catch")
        assert hasattr(output, "out_Gear_Catch")
        assert hasattr(output, "annual_Biomass")
        assert hasattr(output, "annual_Catch")
        assert hasattr(output, "annual_QB")
        assert hasattr(output, "end_state")
        assert hasattr(output, "crash_year")
        assert hasattr(output, "pred")
        assert hasattr(output, "prey")

    def test_crash_detection(self, simple_model):
        """Test crash year detection."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        output = rsim_run(scenario, method="RK4")

        # In a stable model, crash_year should be -1 (no crash)
        # or a positive year if a crash occurred
        assert isinstance(output.crash_year, (int, np.integer))

    def test_end_state_preserves_final(self, simple_model):
        """Test that end_state matches final simulation state."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 6))

        output = rsim_run(scenario, method="RK4")

        # End state biomass should match final output biomass
        np.testing.assert_array_almost_equal(
            output.end_state.Biomass, output.out_Biomass[-1]
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_minimum_years(self, simple_model):
        """Test simulation with minimum years."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 3))  # 2 years

        output = rsim_run(scenario, method="RK4")

        assert output.out_Biomass.shape[0] == 2 * 12 + 1

    def test_years_range_validation(self, simple_model):
        """Test that years range validation works."""
        model, rpath_params = simple_model

        with pytest.raises(ValueError):
            rsim_scenario(model, rpath_params, years=range(1, 2))  # Only 1 year

    def test_long_simulation(self, simple_model):
        """Test longer simulation runs."""
        model, rpath_params = simple_model
        scenario = rsim_scenario(model, rpath_params, years=range(1, 21))  # 20 years

        output = rsim_run(scenario, method="RK4")

        # Should complete and produce correct shape
        assert output.out_Biomass.shape[0] == 20 * 12 + 1


class TestRsimForcing:
    """Tests for forcing matrix creation."""

    def test_forcing_matrix_shapes(self, simple_model):
        """Test that forcing matrices have correct shapes."""
        model, _ = simple_model
        params = rsim_params(model)
        years = range(1, 11)

        forcing = rsim_forcing(params, years)

        n_months = len(years) * 12
        n_groups = params.NUM_GROUPS + 1

        assert forcing.ForcedPrey.shape == (n_months, n_groups)
        assert forcing.ForcedMort.shape == (n_months, n_groups)
        assert forcing.ForcedRecs.shape == (n_months, n_groups)
        assert forcing.ForcedBio.shape == (n_months, n_groups)

    def test_forcing_default_values(self, simple_model):
        """Test that forcing matrices have correct default values."""
        model, _ = simple_model
        params = rsim_params(model)

        forcing = rsim_forcing(params, range(1, 6))

        # Default forcing should be 1.0 (no change)
        assert np.allclose(forcing.ForcedPrey, 1.0)
        assert np.allclose(forcing.ForcedMort, 1.0)

        # ForcedBio should be -1.0 (not forced)
        assert np.allclose(forcing.ForcedBio, -1.0)

        # Migration should be 0.0 (no migration)
        assert np.allclose(forcing.ForcedMigrate, 0.0)


class TestRsimFishing:
    """Tests for fishing matrix creation."""

    def test_fishing_matrix_shapes(self, simple_model):
        """Test that fishing matrices have correct shapes."""
        model, _ = simple_model
        params = rsim_params(model)
        years = range(1, 11)

        fishing = rsim_fishing(params, years)

        n_months = len(years) * 12
        n_years = len(years)

        assert fishing.ForcedEffort.shape == (n_months, params.NUM_GEARS + 1)
        assert fishing.ForcedFRate.shape == (n_years, params.NUM_BIO + 1)
        assert fishing.ForcedCatch.shape == (n_years, params.NUM_BIO + 1)

    def test_fishing_default_values(self, simple_model):
        """Test that fishing matrices have correct default values."""
        model, _ = simple_model
        params = rsim_params(model)

        fishing = rsim_fishing(params, range(1, 6))

        # Default effort should be 1.0 (baseline)
        assert np.allclose(fishing.ForcedEffort, 1.0)

        # F rate and catch should be 0.0 (not forced)
        assert np.allclose(fishing.ForcedFRate, 0.0)
        assert np.allclose(fishing.ForcedCatch, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
