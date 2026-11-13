"""The deploy's rsync exclude list must not drop the packages' own source.

``.github/deploy-excludes.txt`` is read by rsync (``--exclude-from``): a
pattern without a slash matches a file or directory name at any depth, one
with a leading slash only at the top of the repository. An unanchored
``examples`` once dropped ``pypath/examples`` (the shipped example models),
and the pre-deploy import check failed.
"""

import fnmatch
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXCLUDES = ROOT / ".github" / "deploy-excludes.txt"
SOURCES = [
    ROOT / "packages" / "pypath" / "src",
    ROOT / "packages" / "pypath-shiny" / "src",
]
# not source: build output, and the app's runtime store (logs/feedback.db),
# which the server keeps its own copy of
NOT_SOURCE = {"__pycache__", "logs"}


def _patterns():
    lines = EXCLUDES.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _excluded(relative: Path, patterns) -> str:
    """The pattern that excludes ``relative`` (repo-relative), or ''."""
    parts = relative.parts
    for pattern in patterns:
        if pattern.startswith("/"):
            if fnmatch.fnmatch(parts[0], pattern[1:]):
                return pattern
        elif any(fnmatch.fnmatch(part, pattern.rstrip("/")) for part in parts):
            return pattern
    return ""


def test_no_package_source_file_is_excluded():
    patterns = _patterns()
    dropped = []
    for source in SOURCES:
        for folder, dirs, files in os.walk(source):
            dirs[:] = [
                d for d in dirs if d not in NOT_SOURCE and not d.endswith(".egg-info")
            ]
            for name in files:
                relative = (Path(folder) / name).relative_to(ROOT)
                pattern = _excluded(relative, patterns)
                if pattern:
                    dropped.append(f"{relative} (by {pattern!r})")
    assert not dropped, "deploy would drop:\n" + "\n".join(dropped[:10])


def test_the_top_level_examples_folder_is_still_excluded():
    assert _excluded(Path("examples") / "basic.py", _patterns())
