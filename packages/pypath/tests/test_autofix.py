"""Tests for core autofix module."""

import numpy as np

from pypath.core.autofix import AutofixResult, autofix_parameters, diagnose_crash_causes


class MockRpath:
    """Minimal mock Rpath for autofix tests."""

    def __init__(self, num_living=3, num_dead=1):
        n = num_living + num_dead + 1  # +1 for 0-index
        self.NUM_LIVING = num_living
        self.NUM_DEAD = num_dead
        self.NUM_GROUPS = num_living + num_dead
        self.Group = ["Outside"] + [f"G{i}" for i in range(1, n)]
        self.EE = np.array([0.0] + [0.8] * num_living + [0.0] * num_dead)
        self.QB = np.array([0.0] + [5.0] * num_living + [0.0] * num_dead)
        self.PB = np.array([0.0] + [1.0] * num_living + [0.0] * num_dead)
        self.DC = np.zeros((n, n))
        # Give each consumer some diet
        for i in range(1, num_living + 1):
            if i > 1:
                self.DC[i, i - 1] = 1.0  # eats previous group


class MockRsimParams:
    """Minimal mock RsimParams for autofix tests."""

    def __init__(self, n_links=5, n_groups=5):
        self.B_BaseRef = np.ones(n_groups) * 10.0
        self.VV = np.ones(n_links) * 2.0
        self.QQ = np.ones(n_links) * 1.0
        self.DD = np.ones(n_links) * 1.0
        self.PreyFrom = np.arange(n_links)
        self.PreyTo = np.arange(n_links)


class TestDiagnoseCrashCauses:
    """Tests for diagnose_crash_causes."""

    def test_healthy_model_no_critical(self):
        rpath = MockRpath()
        rpath.QB[1] = 0.0  # G1 is a producer (no diet needed)
        rpath.DC[2, 1] = 1.0  # G2 eats G1
        rpath.DC[3, 1] = 0.5  # G3 eats G1 + G2
        rpath.DC[3, 2] = 0.5
        params = MockRsimParams(n_links=3, n_groups=5)
        result = diagnose_crash_causes(rpath, params)
        assert len(result["critical"]) == 0

    def test_ee_greater_than_one_detected(self):
        rpath = MockRpath()
        rpath.EE[2] = 1.5  # Group 2 over-exploited
        rpath.DC[2, 1] = 1.0
        rpath.DC[3, 1] = 1.0
        params = MockRsimParams(n_links=3, n_groups=5)
        result = diagnose_crash_causes(rpath, params)
        ee_issues = [i for i in result["critical"] if i["type"] == "ee_too_high"]
        assert len(ee_issues) == 1
        assert ee_issues[0]["group"] == 2

    def test_low_biomass_warned(self):
        rpath = MockRpath()
        rpath.DC[2, 1] = 1.0
        rpath.DC[3, 1] = 1.0
        params = MockRsimParams(n_links=3, n_groups=5)
        params.B_BaseRef[1] = 0.0001  # Very low
        result = diagnose_crash_causes(rpath, params)
        low_bio = [i for i in result["warnings"] if i["type"] == "low_biomass"]
        assert len(low_bio) >= 1

    def test_high_vulnerability_warned(self):
        rpath = MockRpath()
        rpath.DC[2, 1] = 1.0
        rpath.DC[3, 1] = 1.0
        params = MockRsimParams(n_links=3, n_groups=5)
        params.VV[0] = 100.0  # Very high
        result = diagnose_crash_causes(rpath, params)
        high_vv = [i for i in result["warnings"] if i["type"] == "high_vulnerability"]
        assert len(high_vv) >= 1


class TestAutofixParameters:
    """Tests for autofix_parameters."""

    def test_caps_high_vulnerability(self):
        rpath = MockRpath()
        params = MockRsimParams(n_links=3, n_groups=5)
        params.VV[0] = 100.0
        fixed, result = autofix_parameters(rpath, params)
        assert fixed.VV[0] <= 5.0
        assert len(result.fixes_applied) > 0

    def test_aggressive_mode_lower_caps(self):
        rpath = MockRpath()
        params = MockRsimParams(n_links=3, n_groups=5)
        params.VV[0] = 4.0  # Above aggressive cap (3) but below default (5)
        fixed, result = autofix_parameters(rpath, params, aggressive=True)
        assert fixed.VV[0] <= 3.0

    def test_no_fixes_needed(self):
        rpath = MockRpath()
        params = MockRsimParams(n_links=3, n_groups=5)
        fixed, result = autofix_parameters(rpath, params)
        assert len(result.fixes_applied) == 0
        assert result.success is True

    def test_result_is_autofix_result(self):
        rpath = MockRpath()
        params = MockRsimParams(n_links=3, n_groups=5)
        _, result = autofix_parameters(rpath, params)
        assert isinstance(result, AutofixResult)
