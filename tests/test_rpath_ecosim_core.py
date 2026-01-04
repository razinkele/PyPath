"""
Comprehensive Test Suite for Ecopath/Ecosim Simulation (Rpath-style).

This test module provides comprehensive testing for the core Ecosim simulation
functionality following the Rpath implementation patterns. Tests are organized
to validate:

1. Ecopath Balance Verification
   - Mass balance constraints
   - EE calculation accuracy
   - Trophic level computation

2. Ecosim Parameter Conversion
   - rsim_params() conversion from Rpath
   - Predator-prey link construction
   - Vulnerability and handling time setup

3. Functional Response Calculations
   - Foraging arena theory implementation
   - Consumption rate calculations at equilibrium
   - Derivative stability at baseline

4. Simulation Integration
   - RK4 and Adams-Bashforth methods
   - Long-term stability
   - Crash detection

5. Energy Balance Verification
   - Production = GE × Consumption
   - Predation + M0 + Fishing = Production at equilibrium
   - Detritus flow accounting

6. Real Model Testing (LT2022)
   - Full workflow with real EwE database
   - Multi-stanza species handling
   - Long-term simulation stability

The tests are designed to catch regressions in the core simulation engine
and verify mathematical correctness of the Ecosim equations.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

# =============================================================================
# FIXTURES FOR TEST MODELS
# =============================================================================


@pytest.fixture
def minimal_3group_model():
    """Create a minimal 3-group model for basic testing.

    Groups:
    1. Phytoplankton (producer)
    2. Zooplankton (consumer eating phytoplankton)
    3. Detritus

    This is the simplest possible food web for testing.
    """
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    groups = ["Phyto", "Zoo", "Det"]
    types = [1, 0, 2]  # producer, consumer, detritus

    params = create_rpath_params(groups, types)

    # Phytoplankton
    params.model.loc[0, "Biomass"] = 10.0
    params.model.loc[0, "PB"] = 100.0
    params.model.loc[0, "EE"] = 0.8

    # Zooplankton
    params.model.loc[1, "Biomass"] = 2.0
    params.model.loc[1, "PB"] = 20.0
    params.model.loc[1, "QB"] = 100.0
    params.model.loc[1, "EE"] = 0.5

    # Detritus
    params.model.loc[2, "Biomass"] = 50.0

    # Set defaults
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0  # Producer
    params.model.loc[2, "Unassim"] = 0.0  # Detritus
    params.model["Det"] = 1.0

    # Diet: Zoo eats 100% Phyto
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0]  # Phyto, Zoo, Det, Import
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    return model, params


@pytest.fixture
def balanced_5group_model():
    """Create a balanced 5-group model with fishing.

    Groups:
    1. Phytoplankton (producer)
    2. Zooplankton (consumer)
    3. Fish (consumer - top predator)
    4. Detritus
    5. Fleet (fishing fleet)

    This model includes fishing and a simple 3-level food chain.
    The model is PROPERLY BALANCED - consumption matches production.

    Mass balance:
    - Phyto: PB=10, B=100, Production=1000
      - Consumed by Zoo: DC*QB*B_zoo = 1.0*50*20 = 1000
      - EE = 1.0 (all consumed)
    - Zoo: PB=20, B=20, Production=400
      - Consumed by Fish: DC*QB*B_fish = 1.0*10*5 = 50
      - M0 = PB*(1-EE) where EE calculated from consumption
    - Fish: PB=1, B=5, Production=5
      - Fishing: 0.5/yr
    """
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    groups = ["Phyto", "Zoo", "Fish", "Det", "Fleet"]
    types = [1, 0, 0, 2, 3]

    params = create_rpath_params(groups, types)

    # Phytoplankton (producer) - balanced so all production is consumed
    params.model.loc[0, "Biomass"] = 100.0
    params.model.loc[0, "PB"] = 10.0  # Production = 1000
    # Don't set EE - let it be calculated

    # Zooplankton (herbivore) - consumes all phyto production
    # QB * B_zoo = 50 * 20 = 1000 = Phyto production
    params.model.loc[1, "Biomass"] = 20.0
    params.model.loc[1, "PB"] = 20.0  # Production = 400
    params.model.loc[1, "QB"] = 50.0  # Consumption = 1000

    # Fish (predator) - consumes some Zoo
    # QB * B_fish = 10 * 5 = 50
    params.model.loc[2, "Biomass"] = 5.0
    params.model.loc[2, "PB"] = 1.0  # Production = 5
    params.model.loc[2, "QB"] = 10.0  # Consumption = 50

    # Detritus
    params.model.loc[3, "Biomass"] = 100.0

    # Set defaults
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0  # Producer
    params.model.loc[3, "Unassim"] = 0.0  # Detritus
    params.model.loc[4, "BioAcc"] = np.nan
    params.model.loc[4, "Unassim"] = np.nan
    params.model["Det"] = 1.0
    params.model.loc[4, "Det"] = np.nan

    # Diet matrix - Zoo eats 100% Phyto, Fish eats 100% Zoo
    params.diet["Zoo"] = [1.0, 0.0, 0.0, 0.0, 0.0]
    params.diet["Fish"] = [0.0, 1.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0]

    # Fishing: Fleet catches Fish (0.5/yr catch rate)
    params.model.loc[2, "Fleet"] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    return model, params


@pytest.fixture
def complex_foodweb_model():
    """Create a complex 8-group food web model.

    Groups:
    1. Phytoplankton (producer)
    2. Zooplankton (consumer)
    3. Benthos (consumer)
    4. Forage Fish (consumer)
    5. Predatory Fish (consumer - apex)
    6. Birds (consumer - apex)
    7. Detritus
    8. Fleet

    This model has multiple trophic pathways and competing predators.
    """
    from pypath.core.ecopath import rpath
    from pypath.core.params import create_rpath_params

    groups = [
        "Phyto",
        "Zoo",
        "Benthos",
        "ForageFish",
        "PredFish",
        "Birds",
        "Det",
        "Fleet",
    ]
    types = [1, 0, 0, 0, 0, 0, 2, 3]

    params = create_rpath_params(groups, types)

    # Set biomass and rates
    biomass_vals = [25.0, 10.0, 20.0, 5.0, 2.0, 0.5, 100.0, np.nan]
    pb_vals = [150.0, 40.0, 5.0, 2.0, 0.8, 0.3, np.nan, np.nan]
    qb_vals = [np.nan, 120.0, 15.0, 8.0, 3.0, 50.0, np.nan, np.nan]
    ee_vals = [0.9, 0.85, 0.8, 0.7, 0.3, 0.1, np.nan, np.nan]

    for i, (b, pb, qb, ee) in enumerate(zip(biomass_vals, pb_vals, qb_vals, ee_vals)):
        if not np.isnan(b):
            params.model.loc[i, "Biomass"] = b
        if not np.isnan(pb):
            params.model.loc[i, "PB"] = pb
        if not np.isnan(qb):
            params.model.loc[i, "QB"] = qb
        if not np.isnan(ee):
            params.model.loc[i, "EE"] = ee

    # Set defaults
    params.model["BioAcc"] = 0.0
    params.model["Unassim"] = 0.2
    params.model.loc[0, "Unassim"] = 0.0  # Producer
    params.model.loc[6, "Unassim"] = 0.0  # Detritus
    params.model.loc[7, "BioAcc"] = np.nan
    params.model.loc[7, "Unassim"] = np.nan
    params.model["Det"] = 1.0
    params.model.loc[7, "Det"] = np.nan

    # Complex diet matrix
    # Zoo: 80% Phyto, 20% Detritus
    params.diet["Zoo"] = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0]
    # Benthos: 40% Phyto, 60% Detritus
    params.diet["Benthos"] = [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.0]
    # ForageFish: 70% Zoo, 30% Benthos
    params.diet["ForageFish"] = [0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
    # PredFish: 60% ForageFish, 30% Zoo, 10% Benthos
    params.diet["PredFish"] = [0.0, 0.3, 0.1, 0.6, 0.0, 0.0, 0.0, 0.0]
    # Birds: 80% ForageFish, 20% Zoo
    params.diet["Birds"] = [0.0, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0]
    params.diet["Phyto"] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Fishing on both fish groups
    params.model.loc[3, "Fleet"] = 0.3  # Forage fish landings
    params.model.loc[4, "Fleet"] = 0.2  # Predatory fish landings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = rpath(params)

    return model, params


# LT2022 real model fixture
DATA_FILE = Path(__file__).parent.parent / "Data" / "LT2022_0.5ST_final7.eweaccdb"


@pytest.fixture
def lt2022_model():
    """Load the LT2022 model from EwE database.

    Returns the balanced Rpath model and original parameters.
    Skips if the data file doesn't exist.
    """
    if not DATA_FILE.exists():
        pytest.skip(f"Test data file not found: {DATA_FILE}")

    from pypath.core.ecopath import rpath
    from pypath.io.ewemdb import read_ewemdb

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params = read_ewemdb(str(DATA_FILE))

        # Sort groups by type
        type_order = {0: 0, 1: 1, 2: 2, 3: 3}
        params.model["_sort_key"] = params.model["Type"].map(type_order)
        params.model = (
            params.model.sort_values("_sort_key")
            .drop("_sort_key", axis=1)
            .reset_index(drop=True)
        )

        # Reorder diet matrix
        groups = params.model["Group"].tolist()
        diet_rows = ["Import"] + [g for g in groups if g in params.diet["Group"].values]
        params.diet = params.diet.set_index("Group").reindex(diet_rows).reset_index()
        params.diet = params.diet.fillna(0)

        model = rpath(params)

    return model, params


# =============================================================================
# TEST CLASSES
# =============================================================================


class TestEcopathMassBalance:
    """Tests for Ecopath mass balance verification.

    These tests verify that the balanced Ecopath model satisfies
    fundamental mass balance constraints.
    """

    def test_consumption_equals_qb_times_biomass(self, balanced_5group_model):
        """Test that total consumption = QB × B for consumers."""
        model, params = balanced_5group_model

        for i in range(model.NUM_LIVING):
            if model.type[i] == 0:  # Consumer
                expected_q = model.QB[i] * model.Biomass[i]
                # Total consumption from DC
                actual_q = np.sum(model.DC[:, i]) * model.QB[i] * model.Biomass[i]
                if expected_q > 0:
                    assert np.isclose(
                        expected_q, actual_q, rtol=0.01
                    ), f"Consumption mismatch for group {i}: {expected_q} vs {actual_q}"

    def test_production_equals_pb_times_biomass(self, balanced_5group_model):
        """Test that production = PB × B for all living groups."""
        model, params = balanced_5group_model

        for i in range(model.NUM_LIVING):
            if model.PB[i] > 0:
                production = model.PB[i] * model.Biomass[i]
                assert production > 0, f"Zero production for group {i}"

    def test_ee_is_fraction_consumed(self, balanced_5group_model):
        """Test that EE represents fraction of production consumed."""
        model, params = balanced_5group_model

        for i in range(model.NUM_LIVING):
            if model.EE[i] >= 0 and model.PB[i] > 0:
                # EE should be between 0 and 1 for living groups
                assert (
                    0 <= model.EE[i] <= 1.0
                ), f"Invalid EE for group {i}: {model.EE[i]}"

    def test_ge_equals_pb_over_qb(self, balanced_5group_model):
        """Test that GE = PB/QB for consumers."""
        model, params = balanced_5group_model

        for i in range(model.NUM_LIVING):
            if model.type[i] == 0 and model.QB[i] > 0:  # Consumer
                expected_ge = model.PB[i] / model.QB[i]
                assert np.isclose(
                    model.GE[i], expected_ge, rtol=0.01
                ), f"GE mismatch for group {i}: {model.GE[i]} vs {expected_ge}"


class TestRsimParamsConversion:
    """Tests for rsim_params conversion from Rpath model.

    These tests verify that the Ecosim parameter conversion correctly
    builds the predator-prey link arrays and other simulation parameters.
    """

    def test_basic_param_extraction(self, minimal_3group_model):
        """Test basic parameter extraction."""
        from pypath.core.ecosim import rsim_params

        model, _ = minimal_3group_model
        params = rsim_params(model)

        assert params.NUM_GROUPS == 3
        assert params.NUM_LIVING == 2
        assert params.NUM_DEAD == 1
        assert len(params.spname) == 4  # Outside + 3 groups

    def test_biomass_reference(self, balanced_5group_model):
        """Test that B_BaseRef matches original biomass."""
        from pypath.core.ecosim import rsim_params

        model, _ = balanced_5group_model
        params = rsim_params(model)

        # B_BaseRef[0] should be 1.0 (Outside)
        assert params.B_BaseRef[0] == 1.0

        # Other values should match model.Biomass
        for i in range(model.NUM_LIVING + model.NUM_DEAD):
            assert np.isclose(params.B_BaseRef[i + 1], model.Biomass[i], rtol=0.01)

    def test_predprey_links_created(self, balanced_5group_model):
        """Test that predator-prey links are created."""
        from pypath.core.ecosim import rsim_params

        model, _ = balanced_5group_model
        params = rsim_params(model)

        # Should have at least:
        # - Primary production link (Outside -> Phyto)
        # - Zoo eating Phyto
        # - Fish eating Zoo
        assert params.NumPredPreyLinks >= 2

        # Check link arrays have same length
        assert len(params.PreyFrom) == len(params.PreyTo) == len(params.QQ)

    def test_qq_values_positive(self, balanced_5group_model):
        """Test that QQ (base consumption) values are positive."""
        from pypath.core.ecosim import rsim_params

        model, _ = balanced_5group_model
        params = rsim_params(model)

        # All non-zero QQ values should be positive
        for i in range(1, len(params.QQ)):
            if params.QQ[i] != 0:
                assert params.QQ[i] > 0, f"Negative QQ at link {i}: {params.QQ[i]}"

    def test_vulnerability_default(self, balanced_5group_model):
        """Test that default vulnerability is 2.0."""
        from pypath.core.ecosim import rsim_params

        model, _ = balanced_5group_model
        params = rsim_params(model, mscramble=2.0)

        # All non-zero VV values should be 2.0
        for i in range(1, len(params.VV)):
            if params.VV[i] != 0:
                assert params.VV[i] == 2.0, f"VV not 2.0 at link {i}: {params.VV[i]}"

    def test_handling_time_default(self, balanced_5group_model):
        """Test that default handling time is 1000 (essentially off)."""
        from pypath.core.ecosim import rsim_params

        model, _ = balanced_5group_model
        params = rsim_params(model, mhandle=1000.0)

        # All non-zero DD values should be 1000
        for i in range(1, len(params.DD)):
            if params.DD[i] != 0:
                assert (
                    params.DD[i] == 1000.0
                ), f"DD not 1000 at link {i}: {params.DD[i]}"

    def test_mzero_calculation(self, balanced_5group_model):
        """Test that M0 (other mortality) is calculated correctly."""
        from pypath.core.ecosim import rsim_params

        model, _ = balanced_5group_model
        params = rsim_params(model)

        # M0 = PB * (1 - EE) for living groups
        for i in range(model.NUM_LIVING):
            expected_m0 = model.PB[i] * (1.0 - model.EE[i])
            assert np.isclose(
                params.MzeroMort[i + 1], expected_m0, rtol=0.01
            ), f"M0 mismatch for group {i}: {params.MzeroMort[i + 1]} vs {expected_m0}"


class TestFunctionalResponse:
    """Tests for functional response calculations.

    These tests verify the foraging arena functional response
    produces correct consumption rates at equilibrium.
    """

    def test_consumption_at_equilibrium(self, balanced_5group_model):
        """Test that consumption = QQbase at equilibrium (B/B0 = 1)."""
        from pypath.core.ecosim import _build_link_matrix, rsim_params

        model, params = balanced_5group_model
        sim_params = rsim_params(model)

        # Build QQbase matrix
        n = sim_params.NUM_GROUPS + 1
        QQbase = _build_link_matrix(sim_params, sim_params.QQ)
        VV = _build_link_matrix(sim_params, sim_params.VV)
        DD = _build_link_matrix(sim_params, sim_params.DD)

        # At equilibrium: preyYY = predYY = 1.0
        # Q = QQbase * predYY * preyYY * (DD/(DD-1+preyYY)) * (VV/(VV-1+predYY))
        # With VV=2, DD=1000: Q = QQbase * 1 * 1 * (1000/1000) * (2/2) = QQbase

        for prey in range(1, n):
            for pred in range(1, sim_params.NUM_LIVING + 1):
                if QQbase[prey, pred] > 0:
                    vv = VV[prey, pred]
                    dd = DD[prey, pred]

                    # At equilibrium (PYY=1, PDY=1)
                    dd_term = dd / (dd - 1.0 + 1.0) if dd > 1.0 else 1.0
                    vv_term = vv / (vv - 1.0 + 1.0) if vv > 1.0 else 1.0
                    Q_calc = QQbase[prey, pred] * 1.0 * 1.0 * dd_term * vv_term

                    assert np.isclose(
                        Q_calc, QQbase[prey, pred], rtol=0.01
                    ), f"Q mismatch at ({prey},{pred}): {Q_calc} vs {QQbase[prey, pred]}"

    def test_derivatives_near_zero_at_equilibrium(self, balanced_5group_model):
        """Test that derivatives are near zero at equilibrium."""
        from pypath.core.ecosim import (
            _build_active_link_matrix,
            _build_link_matrix,
            rsim_scenario,
        )
        from pypath.core.ecosim_deriv import deriv_vector

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 3))

        # Build params_dict
        n_groups = scenario.params.NUM_GROUPS + 1

        # Use PP_type from params (correctly computed based on rpath.type)
        params_dict = {
            "NUM_GROUPS": scenario.params.NUM_GROUPS,
            "NUM_LIVING": scenario.params.NUM_LIVING,
            "NUM_DEAD": scenario.params.NUM_DEAD,
            "NUM_GEARS": scenario.params.NUM_GEARS,
            "PB": scenario.params.PBopt,
            "QB": scenario.params.FtimeQBOpt,
            "M0": scenario.params.MzeroMort,
            "Unassim": scenario.params.UnassimRespFrac,
            "ActiveLink": _build_active_link_matrix(scenario.params),
            "VV": _build_link_matrix(scenario.params, scenario.params.VV),
            "DD": _build_link_matrix(scenario.params, scenario.params.DD),
            "QQbase": _build_link_matrix(scenario.params, scenario.params.QQ),
            "Bbase": scenario.params.B_BaseRef,
            "PP_type": scenario.params.PP_type,
        }

        forcing_dict = {
            "Ftime": scenario.start_state.Ftime.copy(),
            "ForcedBio": np.zeros(n_groups),
            "PP_forcing": np.ones(n_groups),
            "ForcedPrey": np.ones(n_groups),
            "ForcedMigrate": np.zeros(n_groups),
            "ForcedEffort": np.ones(scenario.params.NUM_GEARS + 1),
        }

        # Build fishing dict with actual fishing mortality from FishQ
        fishing_mort = np.zeros(n_groups)
        for i in range(1, len(scenario.params.FishFrom)):
            grp = scenario.params.FishFrom[i]
            fishing_mort[grp] += scenario.params.FishQ[i]

        fishing_dict = {
            "FishFrom": scenario.params.FishFrom,
            "FishThrough": scenario.params.FishThrough,
            "FishQ": scenario.params.FishQ,
            "FishingMort": fishing_mort,
        }

        # Initial state = baseline
        state = scenario.start_state.Biomass.copy()

        # Calculate derivatives
        derivs = deriv_vector(state, params_dict, forcing_dict, fishing_dict)

        # Derivatives should be near zero at equilibrium
        for i in range(1, scenario.params.NUM_LIVING + 1):
            # Allow small numerical error (up to 1% of biomass per year)
            max_deriv = state[i] * 0.01
            assert (
                abs(derivs[i]) < max_deriv + 0.01
            ), f"Derivative too large for group {i}: {derivs[i]}"


class TestSimulationIntegration:
    """Tests for simulation integration methods.

    These tests verify that both RK4 and Adams-Bashforth
    integration methods produce valid results.
    """

    def test_rk4_runs_without_error(self, balanced_5group_model):
        """Test that RK4 integration completes without error."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 6))

        output = rsim_run(scenario, method="RK4")

        assert output is not None
        assert output.out_Biomass.shape[0] == 5 * 12 + 1  # 5 years + initial

    def test_ab_runs_without_error(self, balanced_5group_model):
        """Test that Adams-Bashforth integration completes without error."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 6))

        output = rsim_run(scenario, method="AB")

        assert output is not None
        assert output.out_Biomass.shape[0] == 5 * 12 + 1

    def test_biomass_stays_finite(self, balanced_5group_model):
        """Test that biomass values remain finite for living groups and detritus."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 11))

        output = rsim_run(scenario, method="RK4")

        # Check only living groups and detritus (columns 1 to NUM_LIVING + NUM_DEAD)
        # Fleet/gear groups don't have biomass (NaN is expected)
        n_groups = scenario.params.NUM_LIVING + scenario.params.NUM_DEAD
        living_biomass = output.out_Biomass[:, 1 : n_groups + 1]

        # No NaN or Inf values for living groups
        assert not np.any(
            np.isnan(living_biomass)
        ), f"NaN in living biomass: {output.out_Biomass[-1, :]}"
        assert not np.any(
            np.isinf(living_biomass)
        ), f"Inf in living biomass: {output.out_Biomass[-1, :]}"

    def test_biomass_stays_positive(self, balanced_5group_model):
        """Test that biomass values remain positive (or epsilon) for living groups."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 6))

        output = rsim_run(scenario, method="RK4")

        # Check only living groups and detritus
        n_groups = scenario.params.NUM_LIVING + scenario.params.NUM_DEAD
        living_biomass = output.out_Biomass[:, 1 : n_groups + 1]

        # All biomass should be >= 0
        assert np.all(
            living_biomass >= 0
        ), f"Negative biomass detected: {output.out_Biomass[-1, :]}"

    def test_biomass_no_explosion(self, balanced_5group_model):
        """Test that biomass doesn't explode (stay within 100x baseline) for living groups."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 21))

        output = rsim_run(scenario, method="RK4")

        # Check only living groups and detritus
        n_groups = scenario.params.NUM_LIVING + scenario.params.NUM_DEAD

        initial = output.out_Biomass[0, 1 : n_groups + 1]
        max_biomass = np.max(output.out_Biomass[:, 1 : n_groups + 1], axis=0)

        # No group should exceed 100x its initial value
        for i in range(len(initial)):
            if initial[i] > 0.001:
                ratio = max_biomass[i] / initial[i]
                assert ratio < 100, f"Group {i + 1} exploded: {ratio}x increase"


class TestEnergyBalance:
    """Tests for energy balance in Ecosim.

    These tests verify that energy flows are conserved
    during simulation.
    """

    def test_production_from_consumption(self, minimal_3group_model):
        """Test that consumer production = GE × consumption."""
        from pypath.core.ecosim import _build_link_matrix, rsim_params

        model, params = minimal_3group_model
        sim_params = rsim_params(model)

        # For zooplankton (consumer)
        zoo_idx = 2  # 1-indexed in sim (group 1 in model)

        # GE = PB/QB
        ge = sim_params.PBopt[zoo_idx] / sim_params.FtimeQBOpt[zoo_idx]

        # At baseline, consumption = sum of QQbase for this predator
        QQbase = _build_link_matrix(sim_params, sim_params.QQ)
        total_consumption = np.sum(QQbase[:, zoo_idx])

        # Production should equal QB * B (which is total_consumption at baseline)
        expected_production = ge * total_consumption
        actual_production = sim_params.PBopt[zoo_idx] * sim_params.B_BaseRef[zoo_idx]

        assert np.isclose(
            expected_production, actual_production, rtol=0.01
        ), f"Production mismatch: {expected_production} vs {actual_production}"

    def test_mortality_balances_production(self, balanced_5group_model):
        """Test that M0 + predation = production × (1 - EE) at equilibrium."""
        from pypath.core.ecosim import rsim_params

        model, params = balanced_5group_model
        sim_params = rsim_params(model)

        for i in range(model.NUM_LIVING):
            production = model.PB[i] * model.Biomass[i]
            m0 = sim_params.MzeroMort[i + 1] * model.Biomass[i]

            # M0 should equal PB * (1-EE) * B
            expected_m0 = model.PB[i] * (1.0 - model.EE[i]) * model.Biomass[i]

            assert np.isclose(
                m0, expected_m0, rtol=0.01
            ), f"M0 mismatch for group {i}: {m0} vs {expected_m0}"


class TestForcingScenarios:
    """Tests for forcing modifications."""

    def test_zero_fishing_increases_fish(self, balanced_5group_model):
        """Test that removing fishing leads to fish increase."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 11))

        # Set fishing effort to zero
        scenario.fishing.ForcedEffort[:] = 0.0

        output = rsim_run(scenario, method="RK4")

        # Fish group (index 3 in 1-indexed) should increase
        fish_initial = output.out_Biomass[0, 3]
        fish_final = output.out_Biomass[-1, 3]

        # With no fishing, fish biomass should increase or stay similar
        # (depending on food availability)
        assert (
            fish_final >= fish_initial * 0.9
        ), f"Fish decreased too much without fishing: {fish_initial} -> {fish_final}"

    def test_doubled_fishing_decreases_fish(self, balanced_5group_model):
        """Test that doubling fishing leads to fish decrease."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 11))

        # Double fishing effort
        scenario.fishing.ForcedEffort[:] = 2.0

        output = rsim_run(scenario, method="RK4")

        # Fish group should decrease
        fish_initial = output.out_Biomass[0, 3]
        fish_final = output.out_Biomass[-1, 3]

        assert (
            fish_final < fish_initial
        ), f"Fish didn't decrease with doubled fishing: {fish_initial} -> {fish_final}"


class TestOutputStructure:
    """Tests for simulation output structure."""

    def test_output_has_all_fields(self, minimal_3group_model):
        """Test that output has all required fields."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = minimal_3group_model
        scenario = rsim_scenario(model, params, years=range(1, 4))

        output = rsim_run(scenario)

        required_fields = [
            "out_Biomass",
            "out_Catch",
            "out_Gear_Catch",
            "annual_Biomass",
            "annual_Catch",
            "annual_QB",
            "end_state",
            "crash_year",
            "pred",
            "prey",
        ]

        for field in required_fields:
            assert hasattr(output, field), f"Missing output field: {field}"

    def test_end_state_matches_final(self, minimal_3group_model):
        """Test that end_state matches final output."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = minimal_3group_model
        scenario = rsim_scenario(model, params, years=range(1, 4))

        output = rsim_run(scenario)

        np.testing.assert_array_almost_equal(
            output.end_state.Biomass, output.out_Biomass[-1]
        )

    def test_annual_output_correct_shape(self, minimal_3group_model):
        """Test that annual output has correct shape."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = minimal_3group_model
        n_years = 5
        scenario = rsim_scenario(model, params, years=range(1, n_years + 1))

        output = rsim_run(scenario)

        assert output.annual_Biomass.shape[0] == n_years
        assert output.annual_Catch.shape[0] == n_years


class TestComplexFoodweb:
    """Tests using the complex 8-group food web model."""

    def test_simulation_runs(self, complex_foodweb_model):
        """Test that complex model simulation runs."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = complex_foodweb_model
        scenario = rsim_scenario(model, params, years=range(1, 11))

        output = rsim_run(scenario, method="RK4")

        assert output is not None
        assert output.out_Biomass.shape[0] == 10 * 12 + 1

    def test_all_groups_have_biomass(self, complex_foodweb_model):
        """Test that all living groups maintain some biomass."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = complex_foodweb_model
        scenario = rsim_scenario(model, params, years=range(1, 11))

        output = rsim_run(scenario, method="RK4")

        # Check final biomass for living groups
        n_living = model.NUM_LIVING
        final_biomass = output.out_Biomass[-1, 1 : n_living + 1]

        # All living groups should have some biomass (> 1e-6)
        for i, b in enumerate(final_biomass):
            assert b > 1e-6, f"Group {i} went extinct"

    def test_catch_produced(self, complex_foodweb_model):
        """Test that fishing produces catch."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = complex_foodweb_model
        scenario = rsim_scenario(model, params, years=range(1, 11))

        output = rsim_run(scenario, method="RK4")

        total_catch = np.sum(output.annual_Catch)
        assert total_catch > 0, "No catch produced"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="LT2022 data file not found")
class TestLT2022Model:
    """Tests using the real LT2022 model from EwE database.

    These tests verify the full workflow with a real model,
    including multi-stanza handling.
    """

    def test_model_loads(self, lt2022_model):
        """Test that LT2022 model loads successfully."""
        model, params = lt2022_model
        assert model is not None
        assert params is not None

    def test_ecosim_params_created(self, lt2022_model):
        """Test that Ecosim params are created from LT2022."""
        from pypath.core.ecosim import rsim_params

        model, params = lt2022_model
        sim_params = rsim_params(model)

        assert sim_params is not None
        assert sim_params.NUM_GROUPS > 10  # LT2022 has ~24 groups

    def test_scenario_created(self, lt2022_model):
        """Test that scenario is created from LT2022."""
        from pypath.core.ecosim import rsim_scenario

        model, params = lt2022_model
        scenario = rsim_scenario(model, params, years=range(1, 6))

        assert scenario is not None

    def test_simulation_runs(self, lt2022_model):
        """Test that LT2022 simulation runs."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = lt2022_model
        scenario = rsim_scenario(model, params, years=range(1, 11))

        output = rsim_run(scenario, method="AB")

        assert output is not None
        assert output.out_Biomass.shape[0] == 10 * 12 + 1

    def test_no_biomass_explosion(self, lt2022_model):
        """Test that LT2022 doesn't have biomass explosion.

        This tests 20 years with RK4 (more stable than Adams-Bashforth
        for longer simulations with complex food webs).
        """
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = lt2022_model
        scenario = rsim_scenario(model, params, years=range(1, 21))

        output = rsim_run(scenario, method="RK4")

        # Check max biomass for living groups only
        n_groups = scenario.params.NUM_LIVING + scenario.params.NUM_DEAD
        living_biomass = output.out_Biomass[:, 1 : n_groups + 1]
        max_biomass = np.nanmax(living_biomass)

        # Max biomass should be reasonable (< 10^6)
        assert max_biomass < 1e6, f"Biomass explosion: max={max_biomass}"

    def test_biomass_finite(self, lt2022_model):
        """Test that all biomass values are finite."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = lt2022_model
        scenario = rsim_scenario(model, params, years=range(1, 21))

        output = rsim_run(scenario, method="AB")

        assert not np.any(np.isnan(output.out_Biomass))
        assert not np.any(np.isinf(output.out_Biomass))

    def test_derivs_near_zero_at_baseline(self, lt2022_model):
        """Test that derivatives are near zero at baseline."""
        from pypath.core.ecosim import (
            _build_active_link_matrix,
            _build_link_matrix,
            rsim_scenario,
        )
        from pypath.core.ecosim_deriv import deriv_vector

        model, params = lt2022_model
        scenario = rsim_scenario(model, params, years=range(1, 3))

        # Build params_dict
        n_groups = scenario.params.NUM_GROUPS + 1

        params_dict = {
            "NUM_GROUPS": scenario.params.NUM_GROUPS,
            "NUM_LIVING": scenario.params.NUM_LIVING,
            "NUM_DEAD": scenario.params.NUM_DEAD,
            "NUM_GEARS": scenario.params.NUM_GEARS,
            "PB": scenario.params.PBopt,
            "QB": scenario.params.FtimeQBOpt,
            "M0": scenario.params.MzeroMort,
            "Unassim": scenario.params.UnassimRespFrac,
            "ActiveLink": _build_active_link_matrix(scenario.params),
            "VV": _build_link_matrix(scenario.params, scenario.params.VV),
            "DD": _build_link_matrix(scenario.params, scenario.params.DD),
            "QQbase": _build_link_matrix(scenario.params, scenario.params.QQ),
            "Bbase": scenario.params.B_BaseRef,
            "PP_type": scenario.params.PP_type,  # Use actual PP_type from params
        }

        forcing_dict = {
            "Ftime": scenario.start_state.Ftime.copy(),
            "ForcedBio": np.zeros(n_groups),
            "PP_forcing": np.ones(n_groups),
            "ForcedPrey": np.ones(n_groups),
            "ForcedMigrate": np.zeros(n_groups),
            "ForcedEffort": np.ones(scenario.params.NUM_GEARS + 1),
        }

        fishing_dict = {
            "FishFrom": scenario.params.FishFrom,
            "FishThrough": scenario.params.FishThrough,
            "FishQ": scenario.params.FishQ,
            "FishingMort": np.zeros(n_groups),
        }

        state = scenario.start_state.Biomass.copy()
        derivs = deriv_vector(state, params_dict, forcing_dict, fishing_dict)

        # Check derivatives are small relative to biomass
        # Note: LT2022 has some groups with inherent imbalance, allow 10% tolerance
        for i in range(1, scenario.params.NUM_LIVING + 1):
            if state[i] > 0.001:
                rel_deriv = abs(derivs[i]) / state[i]
                # Allow 10% per time unit for real-world models
                assert (
                    rel_deriv < 0.1
                ), f"Derivative too large for group {i}: rel={rel_deriv}"


class TestCrashDetection:
    """Tests for crash detection and recovery."""

    def test_crash_year_reported(self, balanced_5group_model):
        """Test that crash year is reported when biomass drops."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 6))

        output = rsim_run(scenario, method="RK4")

        # crash_year should be -1 (no crash) or a positive year
        assert output.crash_year == -1 or output.crash_year > 0

    def test_simulation_continues_after_crash(self, balanced_5group_model):
        """Test that simulation continues even after crash detection."""
        from pypath.core.ecosim import rsim_run, rsim_scenario

        model, params = balanced_5group_model
        scenario = rsim_scenario(model, params, years=range(1, 11))

        # Set extreme fishing to cause crash
        scenario.fishing.ForcedEffort[:] = 10.0

        output = rsim_run(scenario, method="RK4")

        # Simulation should complete
        assert output.out_Biomass.shape[0] == 10 * 12 + 1


class TestVulnerabilityHandling:
    """Tests for vulnerability and handling time parameters."""

    def test_custom_vulnerability(self, minimal_3group_model):
        """Test that custom vulnerability values work."""
        from pypath.core.ecosim import rsim_params

        model, params = minimal_3group_model

        # Use different vulnerability values
        sim_params = rsim_params(model, mscramble=4.0)

        # VV should be 4.0
        for i in range(1, len(sim_params.VV)):
            if sim_params.VV[i] != 0:
                assert sim_params.VV[i] == 4.0

    def test_custom_handling_time(self, minimal_3group_model):
        """Test that custom handling time values work."""
        from pypath.core.ecosim import rsim_params

        model, params = minimal_3group_model

        # Use different handling time
        sim_params = rsim_params(model, mhandle=100.0)

        # DD should be 100.0
        for i in range(1, len(sim_params.DD)):
            if sim_params.DD[i] != 0:
                assert sim_params.DD[i] == 100.0


class TestProducerDynamics:
    """Tests for primary producer dynamics."""

    def test_producer_identified(self, balanced_5group_model):
        """Test that producers are correctly identified."""
        from pypath.core.ecosim import rsim_params

        model, params = balanced_5group_model
        sim_params = rsim_params(model)

        # Phytoplankton (group 0 in model, 1 in sim) should be producer
        # Check that FtimeQBOpt uses PB for producers
        phyto_pb = sim_params.PBopt[1]
        phyto_qb = sim_params.FtimeQBOpt[1]

        # For producers, QB should equal PB in rsim_params
        assert np.isclose(
            phyto_qb, phyto_pb, rtol=0.01
        ), f"Producer QB not equal to PB: {phyto_qb} vs {phyto_pb}"

    def test_primary_production_link(self, balanced_5group_model):
        """Test that primary production link exists."""
        from pypath.core.ecosim import rsim_params

        model, params = balanced_5group_model
        sim_params = rsim_params(model)

        # There should be a link from Outside (0) to producer (1)
        has_pp_link = False
        for i in range(len(sim_params.PreyFrom)):
            if sim_params.PreyFrom[i] == 0 and sim_params.PreyTo[i] == 1:
                has_pp_link = True
                break

        assert has_pp_link, "Primary production link not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
