"""Test that duplicate diet entries in EcoBase XML are accumulated, not overwritten."""

from unittest.mock import patch

import pytest

from pypath.io.ecobase import get_ecobase_model


DUPLICATE_DIET_XML = """<?xml version="1.0"?>
<ecobase>
    <group>
        <group_name>Predator</group_name>
        <group_seq>1</group_seq>
        <group_type>0</group_type>
    </group>
    <group>
        <group_name>Prey</group_name>
        <group_seq>2</group_seq>
        <group_type>0</group_type>
    </group>
    <group>
        <group_name>Predator</group_name>
        <group_seq>1</group_seq>
        <diet_descr>
            <diet><prey_seq>2</prey_seq><proportion>0.3</proportion></diet>
            <diet><prey_seq>2</prey_seq><proportion>0.2</proportion></diet>
        </diet_descr>
    </group>
</ecobase>
"""


def test_diet_duplicate_prey_accumulated():
    """If prey appears twice in diet_descr, fractions should accumulate."""
    with patch("pypath.io.ecobase.fetch_url", return_value=DUPLICATE_DIET_XML):
        result = get_ecobase_model(999)

    diet = result["diet"]
    assert "Predator" in diet
    pred_diet = diet["Predator"]
    assert "Prey" in pred_diet
    # Should accumulate: 0.3 + 0.2 = 0.5, not overwrite to 0.2
    assert abs(pred_diet["Prey"] - 0.5) < 1e-10, (
        f"Expected accumulated 0.5, got {pred_diet['Prey']} (likely overwritten)"
    )


def test_diet_single_prey_unchanged():
    """Single diet entry should work normally."""
    xml = """<?xml version="1.0"?>
    <ecobase>
        <group><group_name>A</group_name><group_seq>1</group_seq></group>
        <group><group_name>B</group_name><group_seq>2</group_seq></group>
        <group>
            <group_name>A</group_name><group_seq>1</group_seq>
            <diet_descr>
                <diet><prey_seq>2</prey_seq><proportion>0.7</proportion></diet>
            </diet_descr>
        </group>
    </ecobase>
    """
    with patch("pypath.io.ecobase.fetch_url", return_value=xml):
        result = get_ecobase_model(999)

    assert abs(result["diet"]["A"]["B"] - 0.7) < 1e-10
