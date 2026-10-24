from pypath_shiny.services.validation import (
    ValidationIssue,
    ValidationSeverity,
    sort_issues,
    summarize_issues,
)


def test_sort_issues_deterministic_order():
    issues = [
        ValidationIssue(field="z", message="late", severity=ValidationSeverity.ERROR, section="B"),
        ValidationIssue(field="a", message="warn", severity=ValidationSeverity.WARNING, section="A"),
        ValidationIssue(field="b", message="err", severity=ValidationSeverity.ERROR, section="A"),
    ]

    sorted_issues = sort_issues(issues)
    assert [i.field for i in sorted_issues] == ["b", "a", "z"]


def test_summarize_issues_counts_by_severity():
    issues = [
        ValidationIssue(field="a", message="m1", severity=ValidationSeverity.ERROR, section="A"),
        ValidationIssue(field="b", message="m2", severity=ValidationSeverity.ERROR, section="A"),
        ValidationIssue(field="c", message="m3", severity=ValidationSeverity.WARNING, section="A"),
    ]

    summary = summarize_issues(issues)
    assert "2 error" in summary
    assert "1 warning" in summary
