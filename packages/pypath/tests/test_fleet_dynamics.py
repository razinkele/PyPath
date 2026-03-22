"""Tests for fleet_dynamics module: FleetEconParams, FleetDynamicsResult,
create_fleet_econ_params, fleet_dynamics_step, apply_quota_caps."""

import numpy as np

from pypath.core.fleet_dynamics import (
    _CAPACITY_FLOOR,
    FleetDynamicsResult,
    FleetEconParams,
    apply_quota_caps,
    create_fleet_econ_params,
    fleet_dynamics_step,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# 1 fleet, 2 fishing links
# fish_through: index 0 unused; links 1 and 2 go through gear group 4 (1-based)
# fish_from:   index 0 unused; link 1 harvests group 1, link 2 harvests group 2
# fleet_lookup: gear_0based=3 -> gear_idx=1 (so fleet_0=0)
N_FLEETS = 1
N_LINKS = 2
N_GROUPS = 2

FISH_THROUGH = np.array([0, 4, 4], dtype=float)  # length 3 (index 0 unused)
FISH_FROM = np.array([0, 1, 2], dtype=float)  # length 3 (index 0 unused)
FLEET_LOOKUP = {3: 1}  # gear_0based=3 -> gear_idx=1


def _make_params(
    fixed_cost=0.0,
    variable_cost=0.0,
    sailing_cost=0.0,
    price=None,
    cap_depreciate=0.1,
    cap_base_growth=0.5,
    eff_power=1.0,
    tac=None,
):
    """Build a 1-fleet / 2-link FleetEconParams for testing."""
    if price is None:
        price = np.array([0.0, 1.0, 1.0])  # per link (index 0 unused), length N_LINKS
    return FleetEconParams(
        fixed_cost=np.array([fixed_cost]),
        variable_cost=np.array([variable_cost]),
        sailing_cost=np.array([sailing_cost]),
        price=price,
        cap_depreciate=np.array([cap_depreciate]),
        cap_base_growth=np.array([cap_base_growth]),
        eff_power=np.array([eff_power]),
        tac=tac,
    )


def _make_cumulative():
    """Zero cumulative catch (n_fleets, n_groups)."""
    return np.zeros((N_FLEETS, N_GROUPS))


# ---------------------------------------------------------------------------
# TestFleetEconParams
# ---------------------------------------------------------------------------


class TestFleetEconParams:
    def test_construction(self):
        """Create with explicit arrays and verify stored values."""
        fixed = np.array([10.0, 20.0])
        variable = np.array([1.0, 2.0])
        sailing = np.array([0.5, 0.5])
        price = np.array([100.0, 80.0, 60.0])
        cap_d = np.array([0.05, 0.08])
        cap_g = np.array([0.3, 0.4])
        eff_p = np.array([1.0, 0.8])

        params = FleetEconParams(
            fixed_cost=fixed,
            variable_cost=variable,
            sailing_cost=sailing,
            price=price,
            cap_depreciate=cap_d,
            cap_base_growth=cap_g,
            eff_power=eff_p,
        )

        np.testing.assert_array_equal(params.fixed_cost, fixed)
        np.testing.assert_array_equal(params.variable_cost, variable)
        np.testing.assert_array_equal(params.sailing_cost, sailing)
        np.testing.assert_array_equal(params.price, price)
        np.testing.assert_array_equal(params.cap_depreciate, cap_d)
        np.testing.assert_array_equal(params.cap_base_growth, cap_g)
        np.testing.assert_array_equal(params.eff_power, eff_p)
        assert params.tac is None

    def test_with_tac(self):
        """Create with tac array and verify shape."""
        tac = np.array([[10.0, 20.0], [5.0, 15.0]])
        params = FleetEconParams(
            fixed_cost=np.zeros(2),
            variable_cost=np.zeros(2),
            sailing_cost=np.zeros(2),
            price=np.zeros(3),
            cap_depreciate=np.zeros(2),
            cap_base_growth=np.zeros(2),
            eff_power=np.ones(2),
            tac=tac,
        )
        assert params.tac is not None
        assert params.tac.shape == (2, 2)
        np.testing.assert_array_equal(params.tac, tac)


# ---------------------------------------------------------------------------
# TestFleetDynamicsResult
# ---------------------------------------------------------------------------


class TestFleetDynamicsResult:
    def test_construction(self):
        """Create with zero arrays and verify shapes."""
        n_months = 12
        n_years = 1
        n_fleets = 3

        result = FleetDynamicsResult(
            out_Effort=np.zeros((n_months, n_fleets)),
            out_Revenue=np.zeros((n_months, n_fleets)),
            out_Cost=np.zeros((n_months, n_fleets)),
            out_Profit=np.zeros((n_months, n_fleets)),
            annual_Effort=np.zeros((n_years, n_fleets)),
            annual_Profit=np.zeros((n_years, n_fleets)),
            fleet_names=["Fleet A", "Fleet B", "Fleet C"],
        )

        assert result.out_Effort.shape == (n_months, n_fleets)
        assert result.out_Revenue.shape == (n_months, n_fleets)
        assert result.out_Cost.shape == (n_months, n_fleets)
        assert result.out_Profit.shape == (n_months, n_fleets)
        assert result.annual_Effort.shape == (n_years, n_fleets)
        assert result.annual_Profit.shape == (n_years, n_fleets)
        assert len(result.fleet_names) == n_fleets


# ---------------------------------------------------------------------------
# TestCreateFleetEconParams
# ---------------------------------------------------------------------------


class TestCreateFleetEconParams:
    def test_defaults(self):
        """Verify all zeros, eff_power=1, tac=None."""
        params = create_fleet_econ_params(n_fleets=2, n_links=4)
        np.testing.assert_array_equal(params.fixed_cost, np.zeros(2))
        np.testing.assert_array_equal(params.variable_cost, np.zeros(2))
        np.testing.assert_array_equal(params.sailing_cost, np.zeros(2))
        np.testing.assert_array_equal(params.price, np.zeros(4))
        np.testing.assert_array_equal(params.cap_depreciate, np.zeros(2))
        np.testing.assert_array_equal(params.cap_base_growth, np.zeros(2))
        np.testing.assert_array_equal(params.eff_power, np.ones(2))
        assert params.tac is None

    def test_shapes(self):
        """Verify array dimensions match n_fleets and n_links."""
        n_fleets, n_links = 5, 10
        params = create_fleet_econ_params(n_fleets=n_fleets, n_links=n_links)
        assert params.fixed_cost.shape == (n_fleets,)
        assert params.variable_cost.shape == (n_fleets,)
        assert params.sailing_cost.shape == (n_fleets,)
        assert params.price.shape == (n_links,)
        assert params.cap_depreciate.shape == (n_fleets,)
        assert params.cap_base_growth.shape == (n_fleets,)
        assert params.eff_power.shape == (n_fleets,)


# ---------------------------------------------------------------------------
# TestFleetDynamicsStep
# ---------------------------------------------------------------------------


class TestFleetDynamicsStep:
    """Tests for fleet_dynamics_step()."""

    def test_profitable_fleet_grows(self):
        """Revenue > cost → capacity increases."""
        params = _make_params(
            price=np.array([0.0, 10.0, 10.0]),  # high price (index 0 unused)
            variable_cost=0.0,
            sailing_cost=0.0,
            fixed_cost=0.0,
            cap_depreciate=0.05,
            cap_base_growth=1.0,
        )
        capacity = np.array([1.0])
        # monthly_catch: index 0 unused, links 1 and 2 have positive catch
        monthly_catch = np.array([0.0, 5.0, 5.0])
        cumul = _make_cumulative()

        new_cap, effort = fleet_dynamics_step(
            capacity=capacity,
            monthly_catch=monthly_catch,
            cumulative_catch=cumul,
            params=params,
            fish_through=FISH_THROUGH,
            fish_from=FISH_FROM,
            fleet_lookup=FLEET_LOOKUP,
            n_fleets=N_FLEETS,
        )

        assert new_cap[0] > capacity[0], "Profitable fleet should grow"
        assert effort[0] > 0.0

    def test_unprofitable_fleet_shrinks(self):
        """No catch (zero revenue) → capacity depreciates."""
        params = _make_params(
            price=np.array([0.0, 1.0, 1.0]),  # index 0 unused
            variable_cost=0.0,
            sailing_cost=0.0,
            fixed_cost=0.0,
            cap_depreciate=0.2,  # 20% annual depreciation
            cap_base_growth=1.0,
        )
        capacity = np.array([1.0])
        monthly_catch = np.array([0.0, 0.0, 0.0])  # no catch → no revenue
        cumul = _make_cumulative()

        new_cap, effort = fleet_dynamics_step(
            capacity=capacity,
            monthly_catch=monthly_catch,
            cumulative_catch=cumul,
            params=params,
            fish_through=FISH_THROUGH,
            fish_from=FISH_FROM,
            fleet_lookup=FLEET_LOOKUP,
            n_fleets=N_FLEETS,
        )

        assert new_cap[0] < capacity[0], "Unprofitable fleet should shrink"

    def test_capacity_floor(self):
        """Very small capacity should not drop below _CAPACITY_FLOOR."""
        params = _make_params(
            cap_depreciate=100.0,  # extreme depreciation
            cap_base_growth=0.0,
            variable_cost=0.0,
            sailing_cost=0.0,
            fixed_cost=0.0,
        )
        capacity = np.array([_CAPACITY_FLOOR * 2])
        monthly_catch = np.array([0.0, 0.0, 0.0])
        cumul = _make_cumulative()

        new_cap, _ = fleet_dynamics_step(
            capacity=capacity,
            monthly_catch=monthly_catch,
            cumulative_catch=cumul,
            params=params,
            fish_through=FISH_THROUGH,
            fish_from=FISH_FROM,
            fleet_lookup=FLEET_LOOKUP,
            n_fleets=N_FLEETS,
        )

        assert new_cap[0] >= _CAPACITY_FLOOR, (
            f"Capacity {new_cap[0]} dropped below floor {_CAPACITY_FLOOR}"
        )

    def test_eff_power(self):
        """With eff_power=0.5, effort = sqrt(capacity)."""
        params = _make_params(
            cap_depreciate=0.0,
            cap_base_growth=0.0,
            eff_power=0.5,
            variable_cost=0.0,
            sailing_cost=0.0,
            fixed_cost=0.0,
        )
        capacity = np.array([4.0])  # sqrt(4) = 2
        monthly_catch = np.array([0.0, 0.0, 0.0])
        cumul = _make_cumulative()

        new_cap, effort = fleet_dynamics_step(
            capacity=capacity,
            monthly_catch=monthly_catch,
            cumulative_catch=cumul,
            params=params,
            fish_through=FISH_THROUGH,
            fish_from=FISH_FROM,
            fleet_lookup=FLEET_LOOKUP,
            n_fleets=N_FLEETS,
        )

        # With zero depreciation and zero growth, capacity stays at 4.0
        # (no catch → profit_signal ≤ 0, max(profit_signal, 0) = 0, decay=0)
        expected_effort = new_cap[0] ** 0.5
        np.testing.assert_allclose(effort[0], expected_effort, rtol=1e-10)

    def test_zero_profit_only_depreciation(self):
        """Zero catch and zero costs → only depreciation acts on capacity."""
        dep_rate = 0.12  # annual
        params = _make_params(
            fixed_cost=0.0,
            variable_cost=0.0,
            sailing_cost=0.0,
            cap_depreciate=dep_rate,
            cap_base_growth=1.0,
        )
        capacity = np.array([1.0])
        monthly_catch = np.array([0.0, 0.0, 0.0])
        cumul = _make_cumulative()
        dt = 1.0 / 12

        new_cap, _ = fleet_dynamics_step(
            capacity=capacity,
            monthly_catch=monthly_catch,
            cumulative_catch=cumul,
            params=params,
            fish_through=FISH_THROUGH,
            fish_from=FISH_FROM,
            fleet_lookup=FLEET_LOOKUP,
            n_fleets=N_FLEETS,
            dt=dt,
        )

        # With zero revenue and zero cost, both revenue and cost = 0
        # denom = epsilon → profit_signal = 0/epsilon = 0
        # growth_term = cap_base_growth * max(0, 0) = 0
        # dC = (0 - dep_rate) * capacity * dt
        expected = max(1.0 + (0.0 - dep_rate) * 1.0 * dt, _CAPACITY_FLOOR)
        np.testing.assert_allclose(new_cap[0], expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# TestApplyQuotaCaps
# ---------------------------------------------------------------------------


class TestApplyQuotaCaps:
    """Tests for apply_quota_caps()."""

    def _make_fish_q(self):
        """FishQ array: index 0 unused, links 1 and 2 have value 1.0."""
        return np.array([0.0, 1.0, 1.0])

    def test_below_tac_unchanged(self):
        """Cumulative catch below TAC → FishQ unchanged."""
        tac = np.array([[100.0, 100.0]])  # (n_fleets=1, n_groups=2)
        cumul = np.array([[10.0, 10.0]])  # well below TAC
        fish_q = self._make_fish_q()

        result = apply_quota_caps(
            fish_q=fish_q,
            cumulative_catch=cumul,
            tac=tac,
            fish_through=FISH_THROUGH,
            fish_from=FISH_FROM,
            fleet_lookup=FLEET_LOOKUP,
        )

        np.testing.assert_array_equal(result, fish_q)

    def test_at_tac_zeroed(self):
        """Cumulative catch >= TAC → FishQ zeroed for that link."""
        tac = np.array([[5.0, 5.0]])  # (n_fleets=1, n_groups=2)
        cumul = np.array([[5.0, 5.0]])  # exactly at TAC
        fish_q = self._make_fish_q()

        result = apply_quota_caps(
            fish_q=fish_q,
            cumulative_catch=cumul,
            tac=tac,
            fish_through=FISH_THROUGH,
            fish_from=FISH_FROM,
            fleet_lookup=FLEET_LOOKUP,
        )

        # Both links should be zeroed (cumul >= tac for groups 0 and 1)
        assert result[1] == 0.0, "Link 1 should be zeroed at TAC"
        assert result[2] == 0.0, "Link 2 should be zeroed at TAC"

    def test_no_tac_link_unaffected(self):
        """Links for groups not in fleet_lookup stay unchanged."""
        # Use a fleet_lookup that doesn't map the link's gear
        empty_lookup = {}
        tac = np.array([[5.0, 5.0]])
        cumul = np.array([[10.0, 10.0]])  # over TAC
        fish_q = self._make_fish_q()

        result = apply_quota_caps(
            fish_q=fish_q,
            cumulative_catch=cumul,
            tac=tac,
            fish_through=FISH_THROUGH,
            fish_from=FISH_FROM,
            fleet_lookup=empty_lookup,  # no mapping → gear_idx=0 → skip
        )

        # With empty fleet_lookup, no link gets zeroed
        np.testing.assert_array_equal(result, fish_q)
