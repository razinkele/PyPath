from pathlib import Path

from app.pages.prebalance import make_rpath_status_badge, rpath_diagnostics_summary


def test_rpath_diagnostics_summary_detects_provided():
    diag = Path("tests/data/rpath_reference/ecosim/diagnostics")
    s = rpath_diagnostics_summary(diag)
    assert isinstance(s, str)
    assert "provided" in s or "not provided" in s
    assert s == "Rpath QQ diagnostics provided"


def test_rpath_diagnostics_summary_handles_missing(tmp_path):
    d = tmp_path / "diag"
    d.mkdir()
    s = rpath_diagnostics_summary(d)
    assert s == "No diagnostics available" or s.startswith("Diagnostics incomplete")


def test_make_rpath_status_badge_outputs_ui():
    # Verify badge creation returns an object whose string contains the status and appropriate class
    provided = "Rpath QQ diagnostics provided"
    badge = make_rpath_status_badge(provided)
    # When Shiny is available the badge should be a Tag convertible to string
    bstr = str(badge)
    assert "Rpath QQ diagnostics provided" in bstr
    assert "bg-success" in bstr

    incomplete = "Diagnostics incomplete: 1 error(s)"
    badge2 = make_rpath_status_badge(incomplete)
    assert "Diagnostics incomplete" in str(badge2)
    assert "bg-warning" in str(badge2) or "bg-secondary" in str(badge2)

    none = "No diagnostics available"
    badge3 = make_rpath_status_badge(none)
    assert "No diagnostics available" in str(badge3)
    assert "bg-secondary" in str(badge3)

    # Verify note and link are included when provided
    badge4 = make_rpath_status_badge(provided, note="meta-note-123", link="file:///tmp/diag")
    b4 = str(badge4)
    assert "meta-note-123" in b4
    assert "href=\"file:///tmp/diag\"" in b4 or "href='file:///tmp/diag'" in b4
