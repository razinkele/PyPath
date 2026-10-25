from pypath_shiny.services.ui_state import (
    WIZARD_DRAFT_VERSION,
    deserialize_wizard_draft,
    serialize_wizard_draft,
)


def test_wizard_draft_roundtrip():
    payload = {
        "step": 3,
        "polygon": {"type": "Polygon", "coords": [1, 2]},
        "grid_type": "hex",
    }
    serialized = serialize_wizard_draft(payload)
    restored = deserialize_wizard_draft(serialized)

    assert restored["draft_version"] == WIZARD_DRAFT_VERSION
    assert restored["step"] == 3
    assert restored["grid_type"] == "hex"
