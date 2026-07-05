"""Shared test fixtures."""

import pytest

from nvidia_smartroute.config import settings


@pytest.fixture(autouse=True)
def _isolate_models_file(monkeypatch):
    """Ensure tests don't pick up a locally-generated discovered_models.json."""
    monkeypatch.setattr(settings, "models_file", "/nonexistent-discovered-models.json")
