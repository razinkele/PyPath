"""The bundled example model (example_model_data/) must be a valid food web.

create_example_model.py built the diet matrix the wrong way round: its
``diet_data`` maps each prey to the shares every predator takes of it, but the
script stored each key as a predator column (and normalised those columns).
The saved diet.csv also carried pandas' index column and a stray
"Phytoplankton" row. Result: producers ate other groups and sat at TL ~3.7,
and pre-balance flagged "Phytoplankton" as a predator.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pypath.core.ecopath import rpath
from pypath.core.params import read_rpath_params

EXAMPLE_DATA = Path(__file__).parent.parent / "example_model_data"


@pytest.fixture(scope="module")
def params():
    return read_rpath_params(
        str(EXAMPLE_DATA / "model.csv"), str(EXAMPLE_DATA / "diet.csv")
    )


def test_diet_csv_has_no_index_column_or_stray_rows():
    raw = pd.read_csv(EXAMPLE_DATA / "diet.csv")
    assert raw.columns[0] == "Group"
    assert not any(str(c).startswith("Unnamed") for c in raw.columns)
    groups = set(pd.read_csv(EXAMPLE_DATA / "model.csv")["Group"]) | {"Import"}
    assert set(raw["Group"]) <= groups
    assert raw["Group"].is_unique


def test_only_consumers_have_diets_and_they_sum_to_one(params):
    diet = params.diet.set_index("Group")
    types = dict(zip(params.model["Group"], params.model["Type"]))
    for pred in diet.columns:
        total = diet[pred].fillna(0).sum()
        if types[pred] == 0:
            assert total == pytest.approx(1.0, abs=1e-6), pred
        else:
            assert total == pytest.approx(0.0, abs=1e-12), pred  # producers


def test_producers_and_detritus_at_tl_one_and_model_balances(params):
    m = rpath(params)
    n = m.NUM_LIVING + m.NUM_DEAD
    tl = dict(zip(m.Group[:n], m.TL[:n]))
    for g, t in zip(m.Group[:n], m.type[:n]):
        if t in (1, 2):
            assert tl[g] == pytest.approx(1.0), g
        else:
            assert tl[g] >= 2.0, g
    ee = np.asarray(m.EE[: m.NUM_LIVING], dtype=float)
    assert np.all((ee >= 0) & (ee <= 1))
