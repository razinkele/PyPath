"""Copy the package versions from pyproject.toml into the places that repeat them.

semantic-release bumps ``[project].version`` and ``__version__``; the
release-discipline tests (packages/pypath/tests/test_release_discipline.py)
also require, for the core version X.Y.Z and Shiny version A.B.C:

- packages/pypath-shiny/pyproject.toml: ``pypath-ewe[...]>=X.Y.Z``
- README.md: ``badge/version-X.Y.Z-blue`` and ``### Current Version: X.Y.Z``
- CLAUDE.md: ``| **Version** | X.Y.Z | A.B.C |``
- conda-recipes/*/meta.yaml: ``set version`` and pypath-shiny's
  ``pypath-ewe >=X.Y.Z`` (the conda-build workflow checks the versions)

The Release & Publish workflow runs this after semantic-release so the
release commit keeps main's tests green. Idempotent; prints changed files.

Usage: python scripts/sync_release_versions.py [repo_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SEMVER = r"[0-9]+\.[0-9]+\.[0-9]+"


def project_version(pyproject: Path) -> str:
    """``[project].version`` from a pyproject.toml."""
    in_project = False
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project and stripped.startswith("version"):
            return stripped.split("=", 1)[1].strip().strip('"')
    raise ValueError(f"No [project].version in {pyproject}")


def _sub(path: Path, pattern: str, repl: str, count: int = 0) -> bool:
    text = path.read_text(encoding="utf-8")
    new, replaced = re.subn(pattern, repl, text, count=count)
    if replaced == 0:
        raise ValueError(f"Pattern {pattern!r} not found in {path}")
    if new != text:
        path.write_text(new, encoding="utf-8", newline="")
        return True
    return False


def sync(root: Path) -> list[str]:
    """Rewrite the repeated versions under ``root``; return changed files."""
    core = project_version(root / "packages" / "pypath" / "pyproject.toml")
    shiny = project_version(root / "packages" / "pypath-shiny" / "pyproject.toml")
    changes = {
        # Only the first match: the main dependency, as the test reads it.
        # Optional extras (e.g. pypath-ewe[biodata]>=0.4.0) keep their floor.
        "packages/pypath-shiny/pyproject.toml": [
            (rf"(pypath-ewe\[[^\]]+\]>=){SEMVER}", rf"\g<1>{core}", 1),
        ],
        "README.md": [
            (rf"badge/version-{SEMVER}-blue", f"badge/version-{core}-blue"),
            (rf"### Current Version: {SEMVER}", f"### Current Version: {core}"),
        ],
        "CLAUDE.md": [
            (
                rf"\| \*\*Version\*\* \| {SEMVER} \| {SEMVER} \|",
                f"| **Version** | {core} | {shiny} |",
            ),
        ],
        "conda-recipes/pypath-ewe/meta.yaml": [
            (rf'set version = "{SEMVER}"', f'set version = "{core}"'),
        ],
        "conda-recipes/pypath-shiny/meta.yaml": [
            (rf'set version = "{SEMVER}"', f'set version = "{shiny}"'),
            (rf"(- pypath-ewe >=){SEMVER}", rf"\g<1>{core}"),
        ],
    }
    changed = []
    for rel, subs in changes.items():
        path = root / rel
        if any([_sub(path, *sub) for sub in subs]):
            changed.append(rel)
    return changed


if __name__ == "__main__":
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1]
    for rel in sync(repo):
        print(rel)
