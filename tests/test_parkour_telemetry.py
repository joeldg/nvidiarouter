from nvidia_smartroute.parkour.telemetry import ParkourTelemetry
from nvidia_smartroute.prometheus import render_prometheus
from tests.test_parkour_gateway import _result


def test_parkour_snapshot_and_prometheus_parity():
    tracker = ParkourTelemetry()
    tracker.start()
    tracker.finish("run", _result(), 12.3)
    snapshot = tracker.snapshot()
    assert snapshot["runs"] == 1
    assert snapshot["total_calls"] == 1
    assert snapshot["total_tokens"] == 8
    assert snapshot["recent_runs"][0]["run_id"] == "run"
    assert snapshot["role_tokens"] == {
        "conductor": 0, "worker": 5, "synthesizer": 3
    }
    assert snapshot["recent_runs"][0]["nodes_summary"][0]["model"] == "worker-model"
    text = render_prometheus({"parkour": snapshot})
    assert "nsr_parkour_runs 1" in text
    assert "nsr_parkour_tokens 8" in text


def test_parkour_failure_and_limit_metrics():
    tracker = ParkourTelemetry()
    tracker.start()
    tracker.fail(limit=True)
    assert tracker.snapshot()["failures"] == 1
    assert tracker.snapshot()["limit_stops"] == 1
    assert tracker.snapshot()["active_runs"] == 0
