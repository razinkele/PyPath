"""Tests for pypath-ewe health payload."""

from pypath import __version__, get_health_status


def test_get_health_status_payload_shape_and_values():
    payload = get_health_status()
    assert payload == {
        "status": "ok",
        "service": "pypath-ewe",
        "version": __version__,
    }
