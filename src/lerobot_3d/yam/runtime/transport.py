"""Loopback HTTP observations over an SSH tunnel. No pickle, no direct CAN client."""

from __future__ import annotations

import copy
import io
import json
import re
import secrets
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

MAX_PACKET = 64 * 1024 * 1024


def encode(metadata, arrays):
    out = io.BytesIO()
    np.savez_compressed(
        out,
        metadata=np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8),
        **arrays,
    )
    return out.getvalue()


def decode(packet):
    if len(packet) > MAX_PACKET:
        raise ValueError("Observation too large")
    import zipfile

    with zipfile.ZipFile(io.BytesIO(packet)) as archive:
        if sum(x.file_size for x in archive.infolist()) > MAX_PACKET:
            raise ValueError("Expanded observation too large")
    with np.load(io.BytesIO(packet), allow_pickle=False) as f:
        if len(set(f.files)) != len(f.files) or any(
            key != "metadata" and not re.fullmatch(r"(left|right|overhead)_(rgb|depth_raw)", key)
            for key in f.files
        ):
            raise ValueError("Invalid observation array names")
        metadata = json.loads(f["metadata"].tobytes())
        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported observation schema")
        return metadata, {k: f[k].copy() for k in f.files if k != "metadata"}


class Client:
    def __init__(self, url="http://127.0.0.1:8765", token=None, timeout=5):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode()
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        req = urllib.request.Request(self.url + path, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            result = r.read(MAX_PACKET + 1)
            if len(result) > MAX_PACKET:
                raise ValueError("Response too large")
            return result

    def observation(self):
        return decode(self.request("/v1/observation.npz"))

    def state(self):
        """Small, fresh joint-state response independent of image compression."""
        return json.loads(self.request("/v1/status"))

    def command(self, side, position_rad, gripper_open, observation_id):
        return json.loads(
            self.request(
                "/v1/command",
                {
                    "side": side,
                    "position_rad": list(position_rad),
                    "gripper_open": float(gripper_open),
                    "observation_id": observation_id,
                },
            )
        )


class CommandGate:
    """Supervised small-step control, not trajectory planning or collision avoidance.

    Reject stale/repeated observations, large steps and fast target changes. An
    expired command session latches off until the local server is restarted.
    """

    def __init__(self, limits, enabled=False, timeout=1.0):
        self.limits = limits
        self.enabled = enabled
        self.timeout = timeout
        self.last_time = None
        self.used = {}
        self.last_target = {}
        self.reason = ""

    def expire(self, now):
        if self.enabled and self.last_time is not None and now - self.last_time > self.timeout:
            self.enabled = False
            self.reason = "command watchdog expired; local restart required"
            return True
        return False

    def validate(self, command, observation, now):
        self.expire(now)
        if not self.enabled:
            raise ValueError(self.reason or "Motion disabled on server")
        if observation["mode"] not in ("live", "simulation"):
            raise ValueError("Replay cannot control hardware")
        if now - observation["server_monotonic_s"] > 0.2:
            raise ValueError("Stale server observation")
        side = command["side"]
        if side not in ("left", "right") or side not in observation["arms"]:
            raise ValueError("Missing measured arm state")
        if (
            command["observation_id"] != observation["observation_id"]
            or self.used.get(side) == command["observation_id"]
        ):
            raise ValueError("Stale or repeated command observation")
        current = observation["arms"][side]
        if now - current["host_monotonic_s"] > 0.2:
            raise ValueError("Stale arm state")
        q = np.asarray(command["position_rad"], float)
        g = float(command["gripper_open"])
        if q.shape != (6,) or not np.isfinite(q).all() or not np.isfinite(g) or not 0 <= g <= 1:
            raise ValueError("Invalid command shape/value")
        for i, value in enumerate(q, 1):
            lo, hi = self.limits[f"{side}_joint{i}"]
            if not lo <= value <= hi:
                raise ValueError("Command outside vendor joint limit")
        if (
            np.max(abs(q - np.asarray(current["position_rad"]))) > np.deg2rad(2)
            or abs(g - current["gripper_open"]) > 0.02
        ):
            raise ValueError("Command exceeds 2 degree / 2% gripper step bound")
        target = np.r_[q, g]
        if side in self.last_target:
            previous, t = self.last_target[side]
            allowance = np.r_[np.full(6, 0.1), 0.05] * min(now - t, 0.35)
            if np.any(abs(target - previous) > allowance + 1e-8):
                raise ValueError("Command target rate exceeds 0.1 rad/s or 5% gripper/s")
        self.used[side] = command["observation_id"]
        self.last_time = now
        self.last_target[side] = (target.copy(), now)
        return side, target


class ObservationServer:
    def __init__(self, provider, port=8765, gate=None, token=None):
        self.provider = provider
        self.gate = gate
        self.token = token
        if gate and gate.enabled and (not token or len(token) < 24):
            raise ValueError("Motion requires a random token of at least 24 characters")
        self.lock = threading.Lock()
        self.packet = None
        self.latest = None
        self.latest_arrays = None
        self.error = None
        self.history = {}
        self.stop_event = threading.Event()
        self.counter = 0
        self.session = secrets.token_hex(8)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, body, kind="application/json"):
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path not in ("/v1/observation.npz", "/v1/status"):
                    return self.reply(404, b"{}")
                with owner.lock:
                    if (
                        owner.error
                        or owner.latest is None
                        or (self.path == "/v1/observation.npz" and owner.packet is None)
                    ):
                        return self.reply(
                            503,
                            json.dumps({"error": owner.error or "waiting for acquisition"}).encode(),
                        )
                    if self.path == "/v1/status":
                        data = json.dumps(
                            {
                                **owner.latest,
                                "control_enabled": bool(owner.gate and owner.gate.enabled),
                            }
                        ).encode()
                        return self.reply(200, data)
                    data = owner.packet
                self.reply(200, data, "application/octet-stream")

            def do_POST(self):
                if self.path != "/v1/command":
                    return self.reply(404, b"{}")
                if (
                    self.headers.get("Origin")
                    or not owner.token
                    or not secrets.compare_digest(
                        self.headers.get("Authorization", ""), "Bearer " + owner.token
                    )
                ):
                    return self.reply(403, b'{"error":"authorization rejected"}')
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length < 4096:
                        raise ValueError("Invalid request size")
                    command = json.loads(self.rfile.read(length))
                    with owner.lock:
                        if owner.error or owner.latest is None or owner.gate is None:
                            raise ValueError("Acquisition unavailable or motion disabled")
                        observed = owner.history.get(command.get("observation_id"))
                        if observed is None:
                            raise ValueError("Unknown or expired observation")
                        side, target = owner.gate.validate(command, observed, time.monotonic())
                        current = owner.latest["arms"][side]
                        if (
                            np.max(abs(target[:6] - np.asarray(current["position_rad"]))) > np.deg2rad(2)
                            or abs(target[6] - current["gripper_open"]) > 0.02
                        ):
                            raise ValueError("Robot moved too far since the referenced observation")
                        owner.provider.command(side, target)
                    self.reply(200, b'{"accepted":true}')
                except (ValueError, KeyError, TypeError) as e:
                    self.reply(400, json.dumps({"error": str(e)}).encode())
                except Exception as e:  # noqa: BLE001 — any unexpected hardware failure disables commands.
                    with owner.lock:
                        if owner.gate:
                            owner.gate.enabled = False
                        owner.error = f"Command failed: {e}"
                    self.reply(503, b'{"error":"command failure; control disabled"}')

        # Binding deliberately restricted to loopback. Use SSH for remote access.
        self.http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.http.daemon_threads = True
        self.thread = None
        self.poller = None
        self.encoder = None

    @property
    def port(self):
        return self.http.server_port

    def acquire(self):
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                meta, arrays = self.provider.snapshot()
                meta = copy.deepcopy(meta)
                meta.update(
                    schema_version=1,
                    observation_id=f"{self.session}:{self.counter}",
                    server_monotonic_s=time.monotonic(),
                    server_unix_ns=time.time_ns(),
                )
                self.counter += 1
                with self.lock:
                    if self.gate and self.gate.expire(time.monotonic()):
                        self.provider.hold()
                    self.latest = meta
                    self.latest_arrays = arrays
                    self.history = {
                        key: value
                        for key, value in self.history.items()
                        if time.monotonic() - value["server_monotonic_s"] < 0.2
                    }
                    self.history[meta["observation_id"]] = meta
                    self.error = None
            except Exception as e:  # noqa: BLE001 — acquisition thread failures must latch commands off.
                with self.lock:
                    self.error = f"{type(e).__name__}: {e}"
                    if self.gate and self.gate.enabled:
                        self.gate.enabled = False
                        # Never command a possibly stale feedback pose after acquisition failure.
                        # Existing bounded SDK target remains; operator handles physical stop.
            period = getattr(self.provider, "poll_period_s", 0.02)
            self.stop_event.wait(max(0, period - (time.monotonic() - started)))

    def compress_images(self):
        while not self.stop_event.is_set():
            with self.lock:
                meta, arrays = self.latest, self.latest_arrays
            if meta is not None:
                try:
                    packet = encode(meta, arrays)
                    with self.lock:
                        self.packet = packet
                except Exception as e:  # noqa: BLE001 — report codec failures across the worker boundary.
                    with self.lock:
                        self.error = f"Image encoding failed: {e}"
                        if self.gate:
                            self.gate.enabled = False
            self.stop_event.wait(0.01)

    def start(self):
        self.poller = threading.Thread(target=self.acquire, daemon=True)
        self.poller.start()
        self.encoder = threading.Thread(target=self.compress_images, daemon=True)
        self.encoder.start()
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        self.http.shutdown()
        self.http.server_close()
        if self.poller:
            self.poller.join(timeout=6)
        if self.encoder:
            self.encoder.join(timeout=6)
        self.provider.close()


class ReplayProvider:
    def __init__(self, directory):
        p = Path(directory)
        if (p / "observation.npz").exists():
            self.meta, self.arrays = decode((p / "observation.npz").read_bytes())
        else:
            capture = json.loads((p / "metadata.json").read_text())
            self.meta = {
                "cameras": capture["cameras"],
                "arms": {},
                "capture": capture.get("capture"),
            }
            self.arrays = {
                f"{side}_{kind}": np.load(p / f"{side}_{kind}.npy", allow_pickle=False)
                for side in self.meta["cameras"]
                for kind in ("rgb", "depth_raw")
            }
        self.meta["source_mode"] = self.meta.get("mode", "saved_camera_capture")
        self.meta["mode"] = "replay"

    def snapshot(self):
        return self.meta, self.arrays

    def close(self):
        pass


def save_capture(client, directory, require_joints=False, max_skew_ms=50):
    meta, arrays = client.observation()
    if require_joints:
        if meta["mode"] != "live" or set(meta["arms"]) != {"left", "right"}:
            raise ValueError("Paired calibration capture requires LIVE state from both arms")
        times = [v["host_monotonic_s"] for v in meta["arms"].values()]
        for state in meta["arms"].values():
            if np.max(abs(np.asarray(state["velocity_rad_s"]))) > 0.01:
                raise ValueError("Stop both arms for a static calibration capture")
        for camera in meta["cameras"].values():
            if "host_monotonic_s" not in camera:
                raise ValueError("Camera host timing unavailable")
            times.append(camera["host_monotonic_s"])
        if not times or (max(times) - min(times)) * 1000 > max_skew_ms:
            raise ValueError("Host arrival skew exceeds capture limit; this is not hardware synchronization")
    p = Path(directory)
    p.mkdir(parents=True, exist_ok=False)
    meta["client_received_unix_ns"] = time.time_ns()
    (p / "observation.npz").write_bytes(encode(meta, arrays))
    (p / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, array in arrays.items():
        np.save(p / f"{name}.npy", array, allow_pickle=False)
    return meta
