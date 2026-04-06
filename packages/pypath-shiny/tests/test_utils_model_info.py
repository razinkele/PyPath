"""Tests for get_model_info and load_rpath_diagnostics."""

import json

from pypath_shiny.pages.utils import get_model_info, load_rpath_diagnostics


class TestGetModelInfo:
    def test_none_returns_none(self):
        assert get_model_info(None) is None

    def test_unknown_object_returns_none(self):
        class Fake:
            pass

        assert get_model_info(Fake()) is None

    def test_rpath_params_is_not_balanced(self, rpath_params):
        info = get_model_info(rpath_params)
        assert info is not None
        assert info["is_balanced"] is False
        assert info["trophic_level"] is None

    def test_rpath_params_counts(self, rpath_params):
        info = get_model_info(rpath_params)
        assert info["num_groups"] == 3
        assert info["num_living"] == 2
        assert info["num_dead"] == 1

    def test_rpath_params_groups(self, rpath_params):
        info = get_model_info(rpath_params)
        assert info["groups"] == ["Fish", "Plankton", "Detritus"]

    def test_balanced_model_is_balanced(self, balanced_rpath_model):
        info = get_model_info(balanced_rpath_model)
        assert info["is_balanced"] is True

    def test_balanced_model_has_trophic_level(self, balanced_rpath_model):
        info = get_model_info(balanced_rpath_model)
        assert info["trophic_level"] is not None
        assert len(info["trophic_level"]) == 3

    def test_balanced_model_groups(self, balanced_rpath_model):
        info = get_model_info(balanced_rpath_model)
        assert info["groups"] == ["Fish", "Plankton", "Detritus"]

    def test_balanced_model_num_groups(self, balanced_rpath_model):
        info = get_model_info(balanced_rpath_model)
        assert info["num_groups"] == 3
        assert info["num_living"] == 2
        assert info["num_dead"] == 1

    def test_return_has_all_keys(self, rpath_params):
        info = get_model_info(rpath_params)
        for key in [
            "groups",
            "num_living",
            "num_dead",
            "num_groups",
            "trophic_level",
            "biomass",
            "type_codes",
            "eco_name",
            "is_balanced",
            "params",
        ]:
            assert key in info


class TestLoadRpathDiagnostics:
    def test_valid_dir(self, tmp_diag_dir):
        out = load_rpath_diagnostics(tmp_diag_dir)
        assert out["meta"] is not None
        assert out["qq_provided"] is True
        assert out["qq_df"] is not None
        assert out["comps_df"] is not None
        assert out["note"] == "test note"
        assert out["errors"] == []

    def test_missing_dir_reports_error(self, tmp_path):
        out = load_rpath_diagnostics(tmp_path / "nonexistent")
        assert len(out["errors"]) > 0
        assert "meta.json" in out["errors"][0]

    def test_corrupted_json(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "meta.json").write_text("{ not valid json }")
        out = load_rpath_diagnostics(d)
        assert len(out["errors"]) > 0

    def test_missing_csvs_returns_dict(self, tmp_path):
        d = tmp_path / "nometa"
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"qq_provided": True}))
        out = load_rpath_diagnostics(d)
        assert isinstance(out, dict)
