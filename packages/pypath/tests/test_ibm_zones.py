"""Tests for IBM zonal spatial model (Curonian Lagoon 3-zone system).

Package 5 of the smelt ELS implementation: zone-forcing resolution,
passive drift, ontogenetic habitat constraints, and spawning migration.
"""
import numpy as np
import pytest

from pypath.ibm.base import SuperIndividual
from pypath.ibm.development import ZoneParams
from pypath.ibm.smelt import SmeltIBM, SmeltParams


# =====================================================================
# Task 5.1: Zone-forcing resolution
# =====================================================================


def test_zone_forcing_overrides():
    """Zone-specific forcing overrides global defaults."""
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    env = {
        "temperature": 12.0,
        "zoo_density": 80.0,
        "zone_forcing": {
            0: {"temperature": 8.0, "zoo_density": 30.0},
            1: {"temperature": 15.0, "zoo_density": 120.0},
        },
    }
    r0 = ibm._resolve_forcing(env, 0)
    assert r0["temperature"] == 8.0
    assert r0["zoo_density"] == 30.0
    r1 = ibm._resolve_forcing(env, 1)
    assert r1["temperature"] == 15.0
    r2 = ibm._resolve_forcing(env, 2)  # no zone override
    assert r2["temperature"] == 12.0  # falls back to global


def test_no_zone_forcing_uses_global():
    """Without zone_forcing key, global values are used."""
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    env = {"temperature": 10.0, "zoo_density": 80.0}
    r = ibm._resolve_forcing(env, 0)
    assert r["temperature"] == 10.0


def test_zone_forcing_preserves_non_overridden_keys():
    """Zone override doesn't remove keys not present in the zone dict."""
    params = SmeltParams.baltic_defaults_els()
    ibm = SmeltIBM(group_index=2, n_groups=6, params=params)
    env = {
        "temperature": 12.0,
        "month": 5,
        "zoo_density": 80.0,
        "zone_forcing": {
            0: {"temperature": 8.0},
        },
    }
    r0 = ibm._resolve_forcing(env, 0)
    assert r0["temperature"] == 8.0
    assert r0["month"] == 5  # preserved from global
    assert r0["zoo_density"] == 80.0  # preserved from global
