from __future__ import annotations

import numpy as np

from pypath.spatial.habitat import HabitatMap, HabitatQuality, SubstrateType


def test_habitat_quality_bounds():
    quality = HabitatQuality(
        substrate_quality=0.8,
        structural_complexity=1.2,
        environmental_suitability=0.9,
    )
    assert quality.overall() == 0.864


def test_habitat_map_species_quality_and_masks():
    substrate = np.array(
        [["sand", "mud"], ["rock", "vegetation"]],
        dtype=object,
    )
    quality = np.array([[1.0, 0.5], [0.8, 0.7]])
    habitat = HabitatMap(substrate=substrate, quality=quality)

    mask = habitat.get_substrate_mask(SubstrateType.ROCK)
    assert mask.sum() == 1

    pref = {SubstrateType.SAND: 0.6, SubstrateType.ROCK: 1.0}
    species = habitat.compute_species_quality(pref)
    np.testing.assert_allclose(species, np.array([[0.6, 0.0], [0.8, 0.0]]))


def test_habitat_map_from_csv(tmp_path):
    substrate_path = tmp_path / "substrate.csv"
    quality_path = tmp_path / "quality.csv"
    substrate_path.write_text("sand,mud\nrock,sand\n", encoding="utf-8")
    quality_path.write_text("1.0,0.5\n0.8,0.9\n", encoding="utf-8")

    habitat = HabitatMap.from_csv(substrate_path, quality_path)
    assert habitat.shape == (2, 2)
    assert habitat.metadata["source"] == "csv"
