"""Release-discipline checks for version consistency across package metadata and docs."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _project_version(pyproject_path: Path) -> str:
    text = _read_text(pyproject_path)
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and stripped.startswith("version"):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise AssertionError(f"Could not find [project].version in {pyproject_path}")


def _dunder_version(init_path: Path) -> str:
    text = _read_text(init_path)
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise AssertionError(f"Could not find __version__ in {init_path}")
    return match.group(1)


def _shiny_requires_core_min_version(shiny_pyproject_path: Path) -> str:
    text = _read_text(shiny_pyproject_path)
    match = re.search(r"pypath-ewe\[[^\]]+\]>=([0-9]+\.[0-9]+\.[0-9]+)", text)
    if not match:
        raise AssertionError(
            "Could not find pypath-ewe minimum version in pypath-shiny dependencies"
        )
    return match.group(1)


def test_package_versions_match_dunder_versions():
    core_pyproject = REPO_ROOT / "packages" / "pypath" / "pyproject.toml"
    core_init = REPO_ROOT / "packages" / "pypath" / "src" / "pypath" / "__init__.py"
    shiny_pyproject = REPO_ROOT / "packages" / "pypath-shiny" / "pyproject.toml"
    shiny_init = (
        REPO_ROOT / "packages" / "pypath-shiny" / "src" / "pypath_shiny" / "__init__.py"
    )

    assert _project_version(core_pyproject) == _dunder_version(core_init)
    assert _project_version(shiny_pyproject) == _dunder_version(shiny_init)


def test_shiny_core_dependency_floor_matches_core_version():
    core_version = _project_version(
        REPO_ROOT / "packages" / "pypath" / "pyproject.toml"
    )
    shiny_floor = _shiny_requires_core_min_version(
        REPO_ROOT / "packages" / "pypath-shiny" / "pyproject.toml"
    )

    assert shiny_floor == core_version


def test_readme_core_version_matches_core_package_version():
    core_version = _project_version(
        REPO_ROOT / "packages" / "pypath" / "pyproject.toml"
    )
    readme = _read_text(REPO_ROOT / "README.md")

    assert f"badge/version-{core_version}-blue" in readme
    assert f"### Current Version: {core_version}" in readme


def test_claude_version_table_matches_package_versions():
    core_version = _project_version(
        REPO_ROOT / "packages" / "pypath" / "pyproject.toml"
    )
    shiny_version = _project_version(
        REPO_ROOT / "packages" / "pypath-shiny" / "pyproject.toml"
    )
    claude = _read_text(REPO_ROOT / "CLAUDE.md")

    expected_row = f"| **Version** | {core_version} | {shiny_version} |"
    assert expected_row in claude
