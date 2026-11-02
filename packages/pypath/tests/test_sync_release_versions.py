"""scripts/sync_release_versions.py keeps the release-discipline files in step.

The Release & Publish workflow runs it after semantic-release bumps the
versions, so a release commit does not break test_release_discipline.py.
"""

import importlib.util
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "sync_release_versions.py"
FILES = [
    "packages/pypath/pyproject.toml",
    "packages/pypath-shiny/pyproject.toml",
    "README.md",
    "CLAUDE.md",
    "conda-recipes/pypath-ewe/meta.yaml",
    "conda-recipes/pypath-shiny/meta.yaml",
]


def _load():
    spec = importlib.util.spec_from_file_location("sync_release_versions", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_repo(tmp_path: Path) -> Path:
    for rel in FILES:
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / rel, dst)
    return tmp_path


def _set_version(pyproject: Path, version: str) -> None:
    lines = pyproject.read_text(encoding="utf-8").splitlines(keepends=True)
    in_project = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
        elif in_project and stripped.startswith("version"):
            lines[i] = f'version = "{version}"\n'
            break
    pyproject.write_text("".join(lines), encoding="utf-8")


def test_current_repo_is_already_in_sync(tmp_path):
    root = _copy_repo(tmp_path)
    assert _load().sync(root) == []


def test_core_bump_updates_floor_readme_and_claude(tmp_path):
    mod = _load()
    root = _copy_repo(tmp_path)
    _set_version(root / "packages/pypath/pyproject.toml", "9.8.7")
    shiny = mod.project_version(root / "packages/pypath-shiny/pyproject.toml")

    changed = mod.sync(root)

    assert set(changed) == {
        "packages/pypath-shiny/pyproject.toml",
        "README.md",
        "CLAUDE.md",
        "conda-recipes/pypath-ewe/meta.yaml",
        "conda-recipes/pypath-shiny/meta.yaml",
    }
    text = (root / "packages/pypath-shiny/pyproject.toml").read_text(encoding="utf-8")
    assert "]>=9.8.7" in text
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "badge/version-9.8.7-blue" in readme
    assert "### Current Version: 9.8.7" in readme
    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert f"| **Version** | 9.8.7 | {shiny} |" in claude
    ewe_recipe = (root / "conda-recipes/pypath-ewe/meta.yaml").read_text(
        encoding="utf-8"
    )
    assert 'set version = "9.8.7"' in ewe_recipe
    shiny_recipe = (root / "conda-recipes/pypath-shiny/meta.yaml").read_text(
        encoding="utf-8"
    )
    assert "- pypath-ewe >=9.8.7" in shiny_recipe
    assert f'set version = "{shiny}"' in shiny_recipe
    # idempotent
    assert mod.sync(root) == []


def test_optional_extra_floor_is_left_alone(tmp_path):
    mod = _load()
    root = _copy_repo(tmp_path)
    shiny = root / "packages/pypath-shiny/pyproject.toml"
    before = shiny.read_text(encoding="utf-8")
    _set_version(root / "packages/pypath/pyproject.toml", "9.8.7")
    mod.sync(root)
    after = shiny.read_text(encoding="utf-8")
    assert "pypath-ewe[biodata]>=0.4.0" in before
    assert "pypath-ewe[biodata]>=0.4.0" in after


def test_shiny_bump_updates_claude_and_shiny_recipe(tmp_path):
    mod = _load()
    root = _copy_repo(tmp_path)
    _set_version(root / "packages/pypath-shiny/pyproject.toml", "7.6.5")
    core = mod.project_version(root / "packages/pypath/pyproject.toml")

    changed = mod.sync(root)

    assert set(changed) == {"CLAUDE.md", "conda-recipes/pypath-shiny/meta.yaml"}
    claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert f"| **Version** | {core} | 7.6.5 |" in claude
    recipe = (root / "conda-recipes/pypath-shiny/meta.yaml").read_text(encoding="utf-8")
    assert 'set version = "7.6.5"' in recipe
    assert mod.sync(root) == []
