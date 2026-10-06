import time
import urllib.error

import numpy as np
import pytest

from lerobot_3d.yam.runtime.models import YamModel
from lerobot_3d.yam.runtime.transport import (
    Client,
    CommandGate,
    ObservationServer,
    decode,
    encode,
    save_capture,
)


def observation(now=10):
    return {
        "schema_version": 1,
        "mode": "live",
        "observation_id": "session:1",
        "server_monotonic_s": now,
        "cameras": {},
        "arms": {
            "left": {
                "position_rad": [0, 1, 1, 0, 0, 0],
                "gripper_open": 0.5,
                "velocity_rad_s": [0] * 6,
                "host_monotonic_s": now,
            }
        },
    }


def command():
    return {
        "side": "left",
        "position_rad": [0.01, 1, 1, 0, 0, 0],
        "gripper_open": 0.5,
        "observation_id": "session:1",
    }


@pytest.mark.parametrize(
    "change",
    [
        "nan",
        "shape",
        "limit",
        "step",
        "gripper",
        "old_id",
        "replay",
        "stale_state",
        "disabled",
    ],
)
def test_gate_rejects_unsafe_commands(bundle, change):
    gate = CommandGate(YamModel(bundle / "yam_bimanual.urdf").limits, True)
    obs = observation()
    cmd = command()
    if change == "nan":
        cmd["position_rad"][0] = float("nan")
    elif change == "shape":
        cmd["position_rad"] = [0]
    elif change == "limit":
        cmd["position_rad"][1] = -1
    elif change == "step":
        cmd["position_rad"][0] = 0.1
    elif change == "gripper":
        cmd["gripper_open"] = 1.2
    elif change == "old_id":
        cmd["observation_id"] = "old"
    elif change == "replay":
        obs["mode"] = "replay"
    elif change == "stale_state":
        obs["arms"]["left"]["host_monotonic_s"] = 9
    elif change == "disabled":
        gate.enabled = False
    with pytest.raises(ValueError):
        gate.validate(cmd, obs, 10.05)


def test_command_replay_rate_and_watchdog(bundle):
    gate = CommandGate(YamModel(bundle / "yam_bimanual.urdf").limits, True)
    gate.validate(command(), observation(), 10.05)
    with pytest.raises(ValueError, match="repeated"):
        gate.validate(command(), observation(), 10.06)
    obs = observation(10.06)
    obs["observation_id"] = "session:2"
    cmd = command()
    cmd["observation_id"] = "session:2"
    cmd["position_rad"][0] = 0.02
    with pytest.raises(ValueError, match="rate"):
        gate.validate(cmd, obs, 10.06)
    assert gate.expire(11.1) and not gate.enabled


class FakeProvider:
    def __init__(self):
        self.commands = []
        self.closed = False
        self.fail = False

    def snapshot(self):
        if self.fail:
            raise RuntimeError("disconnected")
        m = observation(time.monotonic())
        m["mode"] = "simulation"
        return m, {
            "left_rgb": np.full((4, 5, 3), 123, np.uint8),
            "left_depth_raw": np.full((4, 5), 10000, np.uint16),
        }

    def command(self, side, target):
        self.commands.append((side, target))

    def hold(self):
        pass

    def close(self):
        self.closed = True


def test_real_http_roundtrip_recording_and_disconnect(tmp_path, bundle):
    provider = FakeProvider()
    server = ObservationServer(provider, 0).start()
    client = Client(f"http://127.0.0.1:{server.port}")
    try:
        for _ in range(100):
            try:
                m, a = client.observation()
                break
            except urllib.error.HTTPError:
                time.sleep(0.01)
        assert m["mode"] == "simulation" and a["left_depth_raw"].dtype == np.uint16
        path = tmp_path / "capture"
        save_capture(client, path)
        _stored, arr = decode((path / "observation.npz").read_bytes())
        np.testing.assert_array_equal(arr["left_rgb"], a["left_rgb"])
        with pytest.raises(urllib.error.HTTPError):
            client.command("left", [0.01, 1, 1, 0, 0, 0], 0.5, m["observation_id"])
        provider.fail = True
        time.sleep(0.1)
        with pytest.raises(urllib.error.HTTPError):
            client.observation()
    finally:
        server.close()
    assert provider.closed


def test_bad_schema_and_nan_metadata_rejected():
    with pytest.raises(ValueError):
        decode(encode({"schema_version": 99}, {}))
    with pytest.raises(ValueError):
        encode({"schema_version": 1, "value": float("nan")}, {})
    with pytest.raises(ValueError, match="array names"):
        decode(encode({"schema_version": 1}, {"../escape": np.zeros(1)}))


def test_authenticated_command_and_acquisition_failure_latch(bundle):
    provider = FakeProvider()
    gate = CommandGate(YamModel(bundle / "yam_bimanual.urdf").limits, True)
    token = "test-session-token-32-characters-long"
    server = ObservationServer(provider, 0, gate, token).start()
    client = Client(f"http://127.0.0.1:{server.port}", token)
    try:
        for _ in range(100):
            try:
                state = client.state()
                break
            except urllib.error.HTTPError:
                time.sleep(0.01)
        result = client.command("left", [0.01, 1, 1, 0, 0, 0], 0.5, state["observation_id"])
        assert result["accepted"] and len(provider.commands) == 1
        with pytest.raises(urllib.error.HTTPError):
            client.command("left", [0.01, 1, 1, 0, 0, 0], 0.5, state["observation_id"])
        provider.fail = True
        time.sleep(0.1)
        assert not gate.enabled
    finally:
        server.close()


def test_state_refresh_does_not_wait_for_image_compression(monkeypatch):
    from lerobot_3d.yam.runtime import transport

    original = transport.encode

    def slow_encode(*args):
        time.sleep(0.25)
        return original(*args)

    monkeypatch.setattr(transport, "encode", slow_encode)
    server = ObservationServer(FakeProvider(), 0).start()
    client = Client(f"http://127.0.0.1:{server.port}")
    try:
        first = client.state()["observation_id"]
        time.sleep(0.1)
        assert client.state()["observation_id"] != first
        assert server.packet is None
    finally:
        server.close()


def test_motor_initialization_requires_ack_before_driver_import():
    from lerobot_3d.yam.runtime.acquisition import LiveProvider

    with pytest.raises(ValueError, match="may calibrate/move grippers"):
        LiveProvider({}, channels={"left": "can_left"})
