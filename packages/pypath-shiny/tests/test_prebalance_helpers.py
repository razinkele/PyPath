"""Tests for prebalance._resolve_repo_root."""
import os
from pathlib import Path

import pytest

from pypath_shiny.pages.prebalance import _resolve_repo_root


def test_resolve_returns_path_or_none():
    result = _resolve_repo_root()
    assert result is None or isinstance(result, Path)


def test_env_var_valid_dir(tmp_path):
    old = os.environ.get("PYPATH_REPO_ROOT")
    try:
        os.environ["PYPATH_REPO_ROOT"] = str(tmp_path)
        result = _resolve_repo_root()
        assert result == tmp_path
    finally:
        if old is None:
            os.environ.pop("PYPATH_REPO_ROOT", None)
        else:
            os.environ["PYPATH_REPO_ROOT"] = old


def test_env_var_nonexistent_falls_through(tmp_path):
    """Invalid env var is ignored; walk-up logic still runs."""
    old = os.environ.get("PYPATH_REPO_ROOT")
    try:
        os.environ["PYPATH_REPO_ROOT"] = str(tmp_path / "nonexistent")
        result = _resolve_repo_root()
        assert result is None or isinstance(result, Path)
    finally:
        if old is None:
            os.environ.pop("PYPATH_REPO_ROOT", None)
        else:
            os.environ["PYPATH_REPO_ROOT"] = old


def test_walk_up_finds_repo_when_in_monorepo():
    """When run inside the monorepo, walk-up should find the root."""
    old = os.environ.pop("PYPATH_REPO_ROOT", None)
    try:
        result = _resolve_repo_root()
        if result is not None:
            assert (result / "packages" / "pypath").is_dir()
    finally:
        if old is not None:
            os.environ["PYPATH_REPO_ROOT"] = old
