"""Release-security regression tests (B6 S4A).

Verifies that no account-specific Evolus identifiers are committed as default
values and that they are supplied only via environment variables.
"""

from boundary.evolus_http import (
    DEFAULT_APPLICATION_ID,
    DEFAULT_WORKFLOW_ID,
    ENV_APPLICATION_ID,
    ENV_WORKFLOW_ID,
    EvolusHttpConfig,
)


def test_no_hardcoded_evolus_identifiers():
    assert DEFAULT_APPLICATION_ID == ""
    assert DEFAULT_WORKFLOW_ID == ""


def test_from_env_defaults_empty_without_env(monkeypatch):
    monkeypatch.delenv(ENV_APPLICATION_ID, raising=False)
    monkeypatch.delenv(ENV_WORKFLOW_ID, raising=False)
    cfg = EvolusHttpConfig.from_env()
    assert cfg.application_id == ""
    assert cfg.workflow_id == ""


def test_from_env_reads_identifiers(monkeypatch):
    monkeypatch.setenv(ENV_APPLICATION_ID, "app-123")
    monkeypatch.setenv(ENV_WORKFLOW_ID, "wf-456")
    cfg = EvolusHttpConfig.from_env()
    assert cfg.application_id == "app-123"
    assert cfg.workflow_id == "wf-456"
