"""The traits auto_populate_taxonomy writes for a species."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pypath.io.biodata import SpeciesInfo, auto_populate_taxonomy


def _taxon(growth_params):
    info = SpeciesInfo(
        common_name="cod",
        scientific_name="Gadus morhua",
        aphia_id=126436,
        authority="Linnaeus, 1758",
        max_length=200.0,
        growth_params=growth_params,
    )
    rpath = SimpleNamespace(Group=np.array(["Cod"]))
    with (
        patch("pypath.io.biodata.get_species_info", return_value=info),
        patch("pypath.io.biodata._fetch_worms_accepted", return_value={}),
    ):
        data = auto_populate_taxonomy(rpath, {"Cod": ["cod"]})
    (taxon,) = data.taxa
    return taxon


def test_asymptotic_length_is_not_written_as_asymptotic_weight():
    """Winf is a weight; the lookup has only the length Loo (cm)."""
    traits = _taxon({"Loo": 110.0, "K": 0.2}).traits
    assert traits["winf"] is None
    assert traits["vbgf_k"] == 0.2
    assert traits["max_length"] == 200.0


def test_traits_without_a_growth_study():
    traits = _taxon(None).traits
    assert traits["winf"] is None
    assert traits["vbgf_k"] is None
