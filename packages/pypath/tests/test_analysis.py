"""
Unit tests for the analysis module.

Tests for Mixed Trophic Impacts, network indices,
and other analysis functions.

Rpath arrays are 0-indexed (``Group[0]`` is the first group) and ``DC`` has
one row per prey group plus a trailing import row, and one column per living
predator. The mocks below follow that layout.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pypath.core.analysis import (
    EcosimSummary,
    NetworkIndices,
    calculate_network_indices,
    check_ecopath_balance,
    export_ecopath_to_dataframe,
    export_ecosim_to_dataframe,
    keystoneness_index,
    mixed_trophic_impacts,
    summarize_ecosim_output,
)

EXAMPLE_DATA = Path(__file__).parent.parent / "example_model_data"


def _mock_rpath(
    n_living, n_dead, *, dc=None, pb=None, qb=None, biomass=None, tl=None, ee=None
):
    """Build a 0-indexed MagicMock Rpath with consistent array shapes."""
    n = n_living + n_dead
    rpath = MagicMock()
    rpath.NUM_LIVING = n_living
    rpath.NUM_DEAD = n_dead
    rpath.NUM_GROUPS = n
    rpath.NUM_GEARS = 0
    rpath.Group = np.array([f"G{i}" for i in range(n)])
    rpath.DC = (
        np.zeros((n + 1, n_living)) if dc is None else np.asarray(dc, dtype=float)
    )
    rpath.PB = np.ones(n) if pb is None else np.asarray(pb, dtype=float)
    rpath.QB = np.zeros(n) if qb is None else np.asarray(qb, dtype=float)
    rpath.Biomass = np.ones(n) if biomass is None else np.asarray(biomass, dtype=float)
    rpath.TL = np.ones(n) if tl is None else np.asarray(tl, dtype=float)
    rpath.EE = np.full(n, 0.5) if ee is None else np.asarray(ee, dtype=float)
    return rpath


class TestMixedTrophicImpacts:
    """Tests for mixed_trophic_impacts function."""

    def test_mti_returns_square_matrix(self):
        """MTI should return square matrix of living + detritus groups."""
        rpath = _mock_rpath(
            3,
            1,
            # rows: G0, G1, G2, Detritus, Import; cols: predators G0, G1, G2
            dc=[
                [0, 0.5, 0],  # G0 eaten by G1
                [0, 0, 0.5],  # G1 eaten by G2
                [0, 0, 0],
                [0.5, 0.5, 0],  # Detritus eaten by G0 and G1
                [0, 0, 0],
            ],
            pb=[1.0, 0.5, 0.2, 0],
            qb=[5, 3, 1, 0],
            biomass=[10, 5, 2, 3],
        )

        mti = mixed_trophic_impacts(rpath)

        assert mti.shape == (4, 4)  # n_groups x n_groups

    def test_mti_with_no_diet(self):
        """MTI with zero diet should produce valid matrix."""
        rpath = _mock_rpath(2, 0, pb=[1.0, 0.5], qb=[5, 5], biomass=[1, 1])

        mti = mixed_trophic_impacts(rpath)

        assert mti.shape == (2, 2)
        assert np.all(np.isfinite(mti))


class TestKeystonenessIndex:
    """Tests for keystoneness_index function."""

    def test_returns_array_aligned_with_groups(self):
        """Should return one value per living + detritus group (0-indexed)."""
        rpath = _mock_rpath(
            3,
            1,
            dc=[
                [0, 0.5, 0],
                [0, 0, 0.5],
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0, 0, 0],
            ],
            pb=[1.0, 0.5, 0.2, 0],
            qb=[5, 3, 1, 0],
            biomass=[10, 5, 2, 3],
        )

        ks = keystoneness_index(rpath)

        assert len(ks) == 4

    def test_accepts_precomputed_mti(self):
        """Should use provided MTI matrix."""
        rpath = _mock_rpath(2, 0, biomass=[10, 5])
        mti = np.array([[0, 0.5], [0.5, 0]])

        ks = keystoneness_index(rpath, mti=mti)

        assert len(ks) == 2
        # Rarer group (lower biomass proportion) scores higher for equal impact
        assert ks[1] > ks[0]


class TestNetworkIndices:
    """Tests for calculate_network_indices function."""

    def test_returns_network_indices_dataclass(self):
        """Should return NetworkIndices dataclass."""
        rpath = _mock_rpath(
            3,
            1,
            dc=[
                [0, 0.3, 0],
                [0, 0, 0.4],
                [0, 0, 0],
                [0.7, 0.6, 0],
                [0, 0, 0],
            ],
            tl=[1.0, 2.0, 3.0, 1.0],
            pb=[1.0, 0.5, 0.2, 0],
            qb=[5, 3, 1, 0],
            biomass=[10, 5, 2, 3],
            ee=[0.9, 0.8, 0.7, 0.5],
        )

        indices = calculate_network_indices(rpath)

        assert isinstance(indices, NetworkIndices)
        assert indices.n_living == 3

    def test_connectance_calculation(self):
        """Connectance should be links / possible_links."""
        rpath = _mock_rpath(
            3,
            0,
            # 2 links in a 3-species system
            dc=[
                [0, 0.5, 0],  # 1 link
                [0, 0, 0.5],  # 1 link
                [0, 0, 0],
                [0, 0, 0],
            ],
            tl=[1.0, 2.0, 3.0],
            pb=[1.0, 0.5, 0.2],
            qb=[5, 3, 1],
            biomass=[10, 5, 2],
            ee=[0.9, 0.8, 0.7],
        )

        indices = calculate_network_indices(rpath)

        assert indices.n_links == 2
        assert indices.connectance == pytest.approx(2 / 9)

    def test_total_biomass(self):
        """Total biomass should sum all groups including detritus."""
        rpath = _mock_rpath(
            3,
            1,
            tl=[1.0, 2.0, 3.0, 1.0],
            pb=[1.0, 0.5, 0.2, 0],
            qb=[5, 3, 1, 0],
            biomass=[10, 5, 2, 3],  # Total = 20
            ee=[0.9, 0.8, 0.7, 0.5],
        )

        indices = calculate_network_indices(rpath)

        assert indices.total_biomass == 20


class TestSummarizeEcosimOutput:
    """Tests for summarize_ecosim_output function."""

    def test_returns_ecosim_summary(self):
        """Should return EcosimSummary dataclass with summary statistics."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 5)
        output.out_Biomass_annual[:, 0] = 0
        output.out_Catch_annual = np.random.rand(10, 5)
        output.out_Catch_annual[:, 0] = 0

        summary = summarize_ecosim_output(output)

        assert isinstance(summary, EcosimSummary)
        assert summary.years == 10


class TestCheckEcopathBalance:
    """Tests for check_ecopath_balance function."""

    def test_balanced_model(self):
        """Balanced model should pass checks."""
        rpath = _mock_rpath(
            2,
            0,
            biomass=[10.0, 5.0],
            pb=[1.0, 0.5],
            qb=[0, 3.0],
            ee=[0.9, 0.8],
            tl=[1.0, 2.0],
        )
        rpath.DC[0, 1] = 1.0  # Consumer eats producer
        rpath.Group = np.array(["Producer", "Consumer"])

        result = check_ecopath_balance(rpath)

        assert result["is_balanced"] is True
        assert result["diet_issues"] == []
        assert result["messages"] == ["Model is properly balanced"]

    def test_reports_issues_by_group_name(self):
        """EE > 1 and an incomplete diet are reported using group names."""
        rpath = _mock_rpath(
            2,
            0,
            biomass=[10.0, 5.0],
            pb=[1.0, 0.5],
            qb=[0, 3.0],
            ee=[1.5, 0.8],
            tl=[1.0, 2.0],
        )
        rpath.DC[0, 1] = 0.5  # diet sums to 0.5
        rpath.Group = np.array(["Producer", "Consumer"])

        result = check_ecopath_balance(rpath)

        assert result["is_balanced"] is False
        assert result["ee_issues"] == [0]
        assert result["diet_issues"] == [1]
        assert any(m.startswith("Producer: EE") for m in result["messages"])
        assert any(m.startswith("Consumer: Diet sum") for m in result["messages"])


class TestExportEcopathToDataframe:
    """Tests for export_ecopath_to_dataframe function."""

    def test_returns_dict_of_dataframes(self):
        """Should return dictionary of DataFrames keyed by group name."""
        rpath = _mock_rpath(
            3,
            1,
            biomass=[10, 5, 2, 3],
            pb=[1.0, 0.5, 0.2, 0],
            qb=[0, 3, 1, 0],
            ee=[0.9, 0.8, 0.7, 0.5],
            tl=[1.0, 2.0, 3.0, 1.0],
        )
        rpath.DC[0, 1] = 1.0

        result = export_ecopath_to_dataframe(rpath)

        assert isinstance(result, dict)
        assert "groups" in result
        assert list(result["groups"]["Group"]) == ["G0", "G1", "G2", "G3"]
        assert list(result["groups"]["Type"]) == ["Living"] * 3 + ["Detritus"]
        assert result["diet"].shape == (4, 3)
        assert list(result["flows"]["From"]) == ["G0"]
        assert list(result["flows"]["To"]) == ["G1"]


class TestExportEcosimToDataframe:
    """Tests for export_ecosim_to_dataframe function."""

    def test_returns_dict_of_dataframes(self):
        """Should return dictionary of DataFrames."""
        output = MagicMock()
        output.out_Biomass_annual = np.random.rand(10, 5)
        output.out_Catch_annual = np.random.rand(10, 5)
        output.out_Biomass = None

        result = export_ecosim_to_dataframe(output)

        assert isinstance(result, dict)
        assert "biomass_annual" in result
        assert "catch_annual" in result


class TestNetworkIndicesDataclass:
    """Tests for NetworkIndices dataclass."""

    def test_fields(self):
        """NetworkIndices should have all required fields."""
        ni = NetworkIndices(
            n_groups=10,
            n_living=8,
            n_links=20,
            connectance=0.25,
            linkage_density=2.5,
            omnivory_index=0.3,
            system_omnivory=0.4,
            mean_trophic_level=2.5,
            max_trophic_level=4.0,
            total_biomass=100.0,
            total_throughput=500.0,
            transfer_efficiency=0.1,
        )

        assert ni.n_groups == 10
        assert ni.n_living == 8
        assert ni.connectance == 0.25

    def test_default_values(self):
        """NetworkIndices should have sensible defaults."""
        ni = NetworkIndices()

        assert ni.n_groups == 0
        assert ni.n_living == 0
        assert ni.connectance == 0.0
        assert ni.finn_cycling_index == 0.0


@pytest.mark.skipif(
    not (EXAMPLE_DATA / "model.csv").exists(), reason="example model data missing"
)
class TestRealModel:
    """Regression tests against a real balanced Rpath (0-indexed arrays)."""

    @pytest.fixture(scope="class")
    def model(self):
        from pypath.core.ecopath import rpath
        from pypath.core.params import read_rpath_params

        params = read_rpath_params(
            str(EXAMPLE_DATA / "model.csv"), str(EXAMPLE_DATA / "diet.csv")
        )
        return rpath(params)

    def test_mti_and_keystoneness_shapes(self, model):
        n = model.NUM_LIVING + model.NUM_DEAD
        mti = mixed_trophic_impacts(model)
        ks = keystoneness_index(model, mti)
        assert mti.shape == (n, n)
        assert np.all(np.isfinite(mti))
        assert len(ks) == n
        assert np.all(np.isfinite(ks))

    def test_balance_check_runs(self, model):
        result = check_ecopath_balance(model)
        assert isinstance(result["is_balanced"], bool)
        assert isinstance(result["messages"], list)

    def test_network_indices_run(self, model):
        indices = calculate_network_indices(model)
        n = model.NUM_LIVING + model.NUM_DEAD
        assert indices.n_groups == n
        assert indices.total_biomass == pytest.approx(float(np.sum(model.Biomass[:n])))
        assert 0 <= indices.connectance <= 1

    def test_export_uses_group_names(self, model):
        dfs = export_ecopath_to_dataframe(model)
        n = model.NUM_LIVING + model.NUM_DEAD
        assert list(dfs["groups"]["Group"]) == [str(g) for g in model.Group[:n]]
        assert dfs["diet"].shape == (n, model.NUM_LIVING)
