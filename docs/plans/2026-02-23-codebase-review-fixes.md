# Codebase Review Fixes — 2026-02-23

## Critical Bugs (9)

1. [ ] `io/ewemdb.py:113-134` — `_read_mdb_with_tools()` stub, returns None
2. [ ] `core/ecosim_deriv.py:644` — IBM dt hardcoded to 1/12 in RK4
3. [ ] `spatial/integration.py:143-169` — B_BaseRef mutation not in finally
4. [ ] `core/optimization.py:113-116` — NRMSE div by zero on flat series
5. [ ] `spatial/external_flux.py:335-374` — rescale_flux always halves
6. [ ] `core/ecopath.py:181-225` — Duplicate docstring in rpath()
7. [ ] `io/ewemdb.py:297-304` — SQL injection via table/column names
8. [ ] `io/biodata.py:524-595` — OBIS zero-records returns None
9. [ ] `io/ewemdb.py:2637-2644` — balance=False still calls rpath()

## Performance (5)

10. [ ] `core/ecosim_deriv.py:96-107` — O(n²) prey switching
11. [ ] `core/ecopath.py:707-710` — Redundant np.where in TL loop
12. [ ] `core/ecosim.py:1380-1700` — Debug scaffolding in rsim_run
13. [ ] `spatial/fishing.py:248-264,310-324` — Duplicated distance loops
14. [ ] `spatial/dispersal.py:450-465` — External flux ignores connectivity format

## Convention Fixes (4 systematic)

15. [ ] Missing logger — 15+ modules
16. [ ] f-string logging — 50+ instances in ecosim_deriv, ecosim, ewemdb
17. [ ] Bare except Exception: pass — 15+ locations
18. [ ] Version inconsistencies — spatial/ibm __init__ say 0.1.0

## Other Important (12)

19. [ ] `ibm/reproduction.py:222` — Recruit energy reserve 50x too high
20. [ ] `ibm/smelt.py:497` — Non-reproducible RNG in spatial movement
21. [ ] `spatial/integration.py:127-136` — food_density == predator_density
22. [ ] `spatial/integration.py:246` — ForcedPrey vs ForcedBio attribute
23. [ ] `io/ewemdb.py:295-308` — DB connection leak on exception
24. [ ] `core/plotting.py:185,208` — plt.cm.get_cmap() removed in mpl 3.9+
25. [ ] `io/biodata.py:568` — datetime.utcnow() deprecated
26. [ ] `io/ecobase.py:43-48` — HTTP URLs (should be HTTPS)
27. [ ] `io/ewemdb.py:1587-1589` — Identical condition on both sides of or
28. [ ] `core/params.py:418-422` — print() instead of logger
29. [ ] `spatial/dispersal.py:166-172` — Distance matrix monkey-patched
30. [ ] `io/biodata.py:1186-1187` — iterrows() index used as position
