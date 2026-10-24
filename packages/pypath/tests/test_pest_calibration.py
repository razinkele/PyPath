"""Unit tests for PEST calibration framework."""

import numpy as np

from pypath.core.ecopath import rpath
from pypath.core.pest_calibration import (
    PestCalibrationResult,
    PestCalibrator,
    PestObservation,
    PestParameter,
)


class TestPestParameter:
    """Test PestParameter dataclass."""

    def test_create_parameter(self):
        """Test parameter creation."""
        param = PestParameter(
            name="VV_0_1",
            initial=1.0,
            lower=0.1,
            upper=10.0,
        )
        assert param.name == "VV_0_1"
        assert param.initial == 1.0
        assert param.lower == 0.1
        assert param.upper == 10.0
        assert param.p_type == "vv"

    def test_parameter_with_indices(self):
        """Test parameter with group indices."""
        param = PestParameter(
            name="VV_2_3",
            initial=1.0,
            lower=0.1,
            upper=10.0,
            group_indices=(2, 3),
        )
        assert param.group_indices == (2, 3)


class TestPestObservation:
    """Test PestObservation dataclass."""

    def test_create_observation(self):
        """Test observation creation."""
        obs = PestObservation(
            name="Biomass_Fish_Y5",
            value=100.0,
            weight=1.0,
            group_index=0,
            time_year=5.0,
        )
        assert obs.name == "Biomass_Fish_Y5"
        assert obs.value == 100.0
        assert obs.weight == 1.0
        assert obs.group_index == 0
        assert obs.time_year == 5.0

    def test_observation_types(self):
        """Test different observation types."""
        for o_type in ["biomass", "catch", "catch_rate"]:
            obs = PestObservation(
                name=f"Obs_{o_type}",
                value=1.0,
                o_type=o_type,
            )
            assert obs.o_type == o_type


class TestPestCalibrator:
    """Test PestCalibrator class."""

    def test_add_parameter(self, spatial_scenario):
        """Test adding parameters."""
        params = spatial_scenario[1]  # Extract RpathParams from tuple
        rpath_model = rpath(params, debug=False)
        calibrator = PestCalibrator(rpath_model, params)

        param = PestParameter(
            name="VV_0_1",
            initial=1.0,
            lower=0.1,
            upper=10.0,
        )
        calibrator.add_parameter(param)

        assert len(calibrator.pest_parameters) == 1
        assert calibrator.pest_parameters[0].name == "VV_0_1"

    def test_add_observation(self, spatial_scenario):
        """Test adding observations."""
        params = spatial_scenario[1]
        rpath_model = rpath(params, debug=False)
        calibrator = PestCalibrator(rpath_model, params)

        obs = PestObservation(
            name="Biomass_Fish_Y5",
            value=100.0,
            weight=1.0,
        )
        calibrator.add_observation(obs)

        assert len(calibrator.pest_observations) == 1
        assert calibrator.pest_observations[0].name == "Biomass_Fish_Y5"

    def test_add_parameters_from_dict(self, spatial_scenario):
        """Test adding parameters from dictionary."""
        params = spatial_scenario[1]
        rpath_model = rpath(params, debug=False)
        calibrator = PestCalibrator(rpath_model, params)

        params_dict = {
            "VV_0_1": (1.0, 0.1, 10.0),
            "VV_1_2": (1.5, 0.5, 5.0),
        }
        calibrator.add_parameters_from_dict(params_dict)

        assert len(calibrator.pest_parameters) == 2

    def test_apply_bounds(self, spatial_scenario):
        """Test parameter bounds enforcement."""
        params = spatial_scenario[1]
        rpath_model = rpath(params, debug=False)
        calibrator = PestCalibrator(rpath_model, params)

        calibrator.add_parameter(PestParameter(
            name="VV_0_1",
            initial=1.0,
            lower=0.1,
            upper=10.0,
        ))

        # Test out-of-bounds parameter
        parmvec = np.array([50.0])  # > upper bound
        bounded = calibrator._apply_bounds(parmvec)

        assert bounded[0] == 10.0  # Clipped to upper

    def test_observation_out_of_bounds(self, spatial_scenario):
        """Test handling of out-of-bounds observations."""
        params = spatial_scenario[1]
        rpath_model = rpath(params, debug=False)
        calibrator = PestCalibrator(rpath_model, params)

        obs = PestObservation(
            name="Test",
            value=1.0,
            group_index=999,  # Invalid group
            time_year=50.0,   # Beyond simulation
        )

        pred = calibrator._extract_prediction(None, obs)
        assert pred is None

    def test_compute_uncertainty(self, spatial_scenario):
        """Test uncertainty computation."""
        params = spatial_scenario[1]
        rpath_model = rpath(params, debug=False)
        calibrator = PestCalibrator(rpath_model, params)

        # Add parameters so correlation matrix can be constructed
        for i in range(3):
            calibrator.add_parameter(PestParameter(
                name=f"param_{i}",
                initial=1.0,
                lower=0.5,
                upper=2.0,
            ))

        # Create dummy Jacobian
        n_obs, n_parms = 10, 3
        jacobian = np.random.randn(n_obs, n_parms)
        ssr = 1.0

        cov, corr, uncertainty = calibrator._compute_uncertainty(
            jacobian, ssr, n_obs, n_parms
        )

        # Should return valid uncertainty estimates
        assert cov is not None
        assert corr is not None
        assert uncertainty is not None
        assert len(uncertainty) == n_parms


class TestPestCalibrationResult:
    """Test PestCalibrationResult dataclass."""

    def test_create_result(self):
        """Test result creation."""
        best_params = {"VV_0_1": 1.5}
        best_parmvec = np.array([1.5])
        residuals = {"Obs_1": 0.1, "Obs_2": -0.05}

        result = PestCalibrationResult(
            best_params=best_params,
            best_parmvec=best_parmvec,
            residuals=residuals,
            sum_of_squared_residuals=0.015,
            n_iterations=5,
            converged=True,
        )

        assert result.best_params == best_params
        assert result.converged is True
        assert result.n_iterations == 5


class TestPestCalibrationIntegration:
    """Integration tests for full calibration workflow."""

    def test_glm_calibration_setup(self, spatial_scenario):
        """Test GLM calibration can be set up."""
        params = spatial_scenario[1]
        rpath_model = rpath(params, debug=False)
        calibrator = PestCalibrator(rpath_model, params)

        # Add parameters
        for i in range(3):
            calibrator.add_parameter(PestParameter(
                name=f"param_{i}",
                initial=1.0,
                lower=0.5,
                upper=2.0,
            ))

        # Add observations
        for i in range(5):
            calibrator.add_observation(PestObservation(
                name=f"obs_{i}",
                value=float(i + 1),
                weight=1.0,
                group_index=0,
                time_year=float(i + 1),
            ))

        assert len(calibrator.pest_parameters) == 3
        assert len(calibrator.pest_observations) == 5

    def test_report_generation(self, spatial_scenario):
        """Test PEST report generation."""
        params = spatial_scenario[1]
        rpath_model = rpath(params, debug=False)
        calibrator = PestCalibrator(rpath_model, params)

        result = PestCalibrationResult(
            best_params={"VV_0_1": 1.5, "VV_1_2": 0.8},
            best_parmvec=np.array([1.5, 0.8]),
            residuals={"Obs_1": 0.1, "Obs_2": -0.05},
            sum_of_squared_residuals=0.015,
            n_iterations=10,
            converged=True,
            uncertainty={"VV_0_1": 0.2, "VV_1_2": 0.1},
        )

        report = calibrator.report(result)
        assert "PEST CALIBRATION REPORT" in report
        assert "VV_0_1" in report
        assert "1.5" in report
