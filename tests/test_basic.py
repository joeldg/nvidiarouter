"""
Basic tests for NVIDIA-SmartRoute-CLI setup.
"""

from fastapi.testclient import TestClient

from nvidia_smartroute.gateway.server import app


def test_health_endpoint():
    """Test that the health endpoint returns 200."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert data["service"] == "nvidia-smartroute-cli"


def test_readiness_endpoint():
    """Test that the readiness endpoint returns 200."""
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ready", "not_ready"]  # Accept either state
    assert "timestamp" in data
    assert "checks" in data


def test_root_endpoint():
    """Test that the root endpoint returns service information."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "NVIDIA-SmartRoute-CLI"
    assert data["version"] == "0.1.0"
    assert "endpoints" in data


def test_settings_import():
    """Test that settings can be imported without error."""
    from nvidia_smartroute.config import settings
    assert settings is not None
    assert hasattr(settings, 'host')
    assert hasattr(settings, 'port')


def test_version_import():
    """Test that version can be imported."""
    from nvidia_smartroute import __version__
    assert __version__ == "0.1.0"
