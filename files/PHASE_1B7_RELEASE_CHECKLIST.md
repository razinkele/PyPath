# Phase 1B.7 Release Checklist v0.5.3

## Pre-Release Validation

### Code Quality
- [x] All Phase 1B tests passing (336/336)
- [x] Ruff linting passed
- [x] MyPy type checking passed
- [x] No pre-existing test failures
- [ ] Full regression suite passing (1500+)

### Documentation
- [ ] Jupyter notebooks created (6 files)
- [ ] API documentation updated (core.md)
- [ ] Integration guide completed
- [ ] README guide completed
- [ ] Phase 1B module docstrings verified
- [ ] API reference complete

### Testing
- [ ] EwE export integration tests created
- [ ] Round-trip tests passing
- [ ] Parity verification tests passing
- [ ] Performance benchmarks documented

### Version Updates
- [ ] packages/pypath/__init__.py → 0.5.3
- [ ] packages/pypath/pyproject.toml → 0.5.3
- [ ] packages/pypath-shiny/__init__.py → 0.5.3 (if needed)
- [ ] CLAUDE.md version table updated
- [ ] CHANGELOG.md finalized for v0.5.3

## Git Workflow

- [ ] Stage all changes
- [ ] Create final commit with release message
- [ ] Create git tag: pypath-ewe-v0.5.3
- [ ] Push to origin

## Release to PyPI

- [ ] Build distribution: python -m build
- [ ] Check build artifacts: twine check dist/*
- [ ] Test upload: twine upload --repository testpypi dist/*
- [ ] Production upload: twine upload dist/*

## Release to Conda

- [ ] Update conda recipe version
- [ ] Build package: conda build -c razinka
- [ ] Index channel
- [ ] Verify installation: conda install -c razinka pypath-ewe=0.5.3

## Post-Release

- [ ] Verify PyPI package (pip install --upgrade pypath-ewe==0.5.3)
- [ ] Verify conda package
- [ ] Update documentation site (if applicable)
- [ ] Create GitHub release notes
- [ ] Post to community forums/issues

## Rollback Plan

If critical issues found post-release:
1. Stop distribution on PyPI
2. Revoke conda builds
3. Tag as v0.5.3-hotfix or revert to v0.5.2
4. Notify users

## Phase 1B.7 Completion Milestones

### Checkpoint 1: Documentation Complete
- Jupyter notebooks: All 6 created and verified
- API docs: Updated and complete
- Guides: Integration, README
- Status: Documentation ready for release

### Checkpoint 2: Testing & Validation
- Export tests: Created and passing
- Regression suite: Full 1500+ tests passing
- Performance benchmarks: Documented
- Status: Ready for version bump

### Checkpoint 3: Release Preparation
- All version numbers updated
- CHANGELOG finalized
- Git tags created
- Status: Ready for PyPI/conda upload

### Checkpoint 4: Release Complete
- PyPI upload successful
- Conda build successful
- Documentation deployed
- Status: v0.5.3 released to production

---

## Expected Timeline

- Jupyter notebooks: ~30-45 minutes
- API docs update: ~15 minutes
- Integration guide: ~20 minutes
- Export tests: ~25 minutes
- README guide: ~20 minutes (waiting for slot)
- Version bump & tag: ~10 minutes
- PyPI/conda upload: ~15 minutes
- Total: ~2-3 hours

## Success Criteria

✅ All 336 Phase 1B tests passing
✅ Full regression suite (1500+) passing
✅ All documentation created and linked
✅ EwE export round-trip tests passing
✅ Version consistency across packages
✅ PyPI and conda packages available
✅ No regressions from Phase 1A

