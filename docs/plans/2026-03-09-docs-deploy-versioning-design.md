# Documentation, Deployment & Versioning Design

**Date:** 2026-03-09

**Goal:** Automate versioning with python-semantic-release, add fully automated PyPI publishing, populate API documentation with GitHub Pages deployment, and update deployment routines.

## 1. Versioning System (python-semantic-release)

**Single source of truth:** Version lives only in each package's `pyproject.toml`. The `__version__` in `__init__.py` is auto-synced by semantic-release.

**Per-package tags:** `pypath-ewe-v{version}` and `pypath-shiny-v{version}` — allows independent releases.

**Per-package CHANGELOGs:** `packages/pypath/CHANGELOG.md` and `packages/pypath-shiny/CHANGELOG.md`. Root `CHANGELOG.md` becomes a pointer to both.

**Configuration** added to each package's `pyproject.toml`:

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
version_variables = ["src/pypath/__init__.py:__version__"]
branch = "main"
commit_message = "chore(release): {package} v{version}"
tag_format = "pypath-ewe-v{version}"  # or pypath-shiny-v{version}
changelog_file = "CHANGELOG.md"
build_command = "python -m build"
```

**Conventional commits** already in use (`feat:`, `fix:`, `perf:`, `style:`, etc.) — no workflow change needed.

## 2. PyPI Publishing (Full Automation)

**New workflow:** `.github/workflows/release.yml`

**Flow:** On push to `main`, semantic-release:
1. Analyzes commits since last tag
2. If `feat:`/`fix:`/`breaking:` found → bumps version
3. Updates `pyproject.toml` + `__init__.py`
4. Generates CHANGELOG entry
5. Creates commit + tag
6. Builds sdist + wheel via `python -m build`
7. Publishes to PyPI via trusted publisher (OIDC — no API tokens)
8. Creates GitHub Release with auto-generated notes

**Sequencing:** Core (`pypath-ewe`) publishes first, then frontend (`pypath-shiny`) depends on that job completing.

**PyPI Trusted Publisher:** GitHub Actions OIDC — configure once per package on pypi.org. More secure than API tokens.

## 3. API Documentation (GitHub Pages + Shiny Link)

**New workflow:** `.github/workflows/docs.yml`
- Triggers on push to `main` when docs or source change
- Runs `mkdocs build` from `packages/pypath/docs/`
- Deploys to `gh-pages` branch via `peaceiris/actions-gh-pages`
- Live at `razinkele.github.io/PyPath/`

**Populate API doc stubs** with `mkdocstrings` directives:
```markdown
::: pypath.core.ecopath
::: pypath.core.ecosim
```

**Fill content:** `getting-started.md` and `index.md` with real content from README.md and docs/archive/.

**Shiny app link:** Add "Documentation" nav entry opening GitHub Pages URL in new tab.

## 4. Deployment Routine Updates

- Update `deploy.yml` tag pattern: `['v*']` → `['pypath-*-v*']`
- Add version display after deploy install step
- Update `deploy/prepare_package.ps1` to read version from pyproject.toml

## Summary

| Component | Changes | New Files |
|-----------|---------|-----------|
| Versioning | semantic-release config in both pyproject.tomls | `packages/*/CHANGELOG.md` |
| Publishing | Automated PyPI via OIDC | `.github/workflows/release.yml` |
| Docs | Populated stubs, real content, GH Pages CI | `.github/workflows/docs.yml`, 9 doc files |
| Shiny link | Nav entry to docs | Edit `app.py` |
| Deploy | Tag pattern + version logging | Edit `deploy.yml` |
| Root CHANGELOG | Pointer to per-package changelogs | Edit `CHANGELOG.md` |

**Dependencies added:** `python-semantic-release` (dev), `build` (dev)
