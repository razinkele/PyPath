from datetime import datetime, timezone

from pypath_shiny.services.feedback_store import (
    add_feedback,
    feedback_stats,
    get_feedback,
    list_feedback,
    resolve_feedback,
    weekly_feedback_counts,
)


def test_add_and_get_feedback_roundtrip(tmp_path):
    db = tmp_path / "feedback.db"
    feedback_id = add_feedback(
        category="bug",
        title="Validation error",
        message="Validation crashes on empty model",
        steps="1. Open app\n2. Run validation",
        rating=2,
        page="Ecopath Model",
        app_version="0.5.0",
        browser="pytest-agent",
        session_id="sess-1",
        db_path=db,
    )

    entry = get_feedback(feedback_id, db_path=db)
    assert entry is not None
    assert entry["title"] == "Validation error"
    assert entry["status"] == "open"
    assert entry["category"] == "bug"


def test_list_feedback_filters_by_status_and_category(tmp_path):
    db = tmp_path / "feedback.db"
    first = add_feedback(
        category="general",
        title="Looks good",
        message="Great release",
        steps=None,
        rating=5,
        page="Home",
        app_version="0.5.0",
        browser="pytest-agent",
        session_id="sess-1",
        db_path=db,
    )
    add_feedback(
        category="bug",
        title="Bug",
        message="Something breaks",
        steps="step",
        rating=1,
        page="Ecosim",
        app_version="0.5.0",
        browser="pytest-agent",
        session_id="sess-2",
        db_path=db,
    )

    resolved = resolve_feedback(
        feedback_id=first,
        status="wontfix",
        note="Expected behavior",
        db_path=db,
    )
    assert resolved

    open_items = list_feedback(statuses=("open",), db_path=db)
    assert len(open_items) == 1
    assert open_items[0]["category"] == "bug"

    bug_items = list_feedback(statuses=("open", "wontfix"), category="bug", db_path=db)
    assert len(bug_items) == 1
    assert bug_items[0]["title"] == "Bug"


def test_resolve_feedback_duplicate_requires_valid_target(tmp_path):
    db = tmp_path / "feedback.db"
    entry_a = add_feedback(
        category="bug",
        title="A",
        message="first",
        steps=None,
        rating=None,
        page="Home",
        app_version="0.5.0",
        browser="pytest-agent",
        session_id="s1",
        db_path=db,
    )
    entry_b = add_feedback(
        category="bug",
        title="B",
        message="second",
        steps=None,
        rating=None,
        page="Home",
        app_version="0.5.0",
        browser="pytest-agent",
        session_id="s2",
        db_path=db,
    )

    assert not resolve_feedback(
        feedback_id=entry_a,
        status="duplicate",
        note="duplicate",
        duplicate_of=9999,
        db_path=db,
    )

    assert resolve_feedback(
        feedback_id=entry_b,
        status="duplicate",
        note="same issue",
        duplicate_of=entry_a,
        db_path=db,
    )


def test_feedback_stats_and_weekly_counts(tmp_path):
    db = tmp_path / "feedback.db"
    add_feedback(
        category="general",
        title="n1",
        message="m1",
        steps=None,
        rating=None,
        page="Home",
        app_version="0.5.0",
        browser="pytest",
        session_id="s1",
        db_path=db,
    )
    add_feedback(
        category="bug",
        title="n2",
        message="m2",
        steps="x",
        rating=1,
        page="Ecosim",
        app_version="0.5.0",
        browser="pytest",
        session_id="s2",
        db_path=db,
    )

    stats = feedback_stats(db_path=db)
    assert stats["total"] == 2
    assert stats["open"] == 2
    assert stats["bugs"] == 1

    weeks = weekly_feedback_counts(db_path=db)
    assert weeks
    this_week = datetime.now(timezone.utc).isocalendar()
    expected_prefix = f"{this_week[0]}-W"
    assert weeks[-1][0].startswith(expected_prefix)
