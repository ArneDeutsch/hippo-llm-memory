from scripts import audit_datasets


def test_audit_datasets_passes() -> None:
    """All required datasets and configs are present."""
    ok, issues = audit_datasets.audit()
    assert ok, f"Audit reported issues: {issues}"
