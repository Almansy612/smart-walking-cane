#!/usr/bin/env python3
"""
yolo_detector.py — vision-based obstacle detection for the Smart Walking Cane
==============================================================================

FYP: Smart Walking Cane with Vision-Based Perception and Finger-Allocated
Haptic Directional Feedback.

Pipeline position
-----------------
camera frame ──► Roboflow Inference Server (Docker, local network)
             ──► RF-DETR Small fine-tuned on SOD (Boussihmed et al., 2025)
             ──► confidence filter (0.35)
             ──► per detection: forward-path corridor + ground-plane distance
             ──► single binary obstacle flag (drives finger 5 stop cue),
                 with 1.5 s obstacle memory across cane-sweep occlusion gaps.

Interface for main.py
---------------------
    det = ObstacleDetector(SERVER_URL, MODEL_ID, API_KEY)
    status = det.process_frame(frame_bgr, cane_yaw_deg=imu.yaw)
    if status.obstacle_active:  haptic.fire(FINGER_5)
    if det.is_stale():          haptic.fire(FINGER_5)   # fail-safe, Ch.3 §3.2

process_frame() blocks for one inference round trip, so calling it in a loop
is naturally closed-loop paced to the server's throughput. If main.py needs
GPS/IMU cycles faster than the vision rate, run the detector in its own
thread and read the latest status — do not queue frames.

Bench testing
-------------
    python yolo_detector.py --selftest                       # no hardware
    python yolo_detector.py --camera 0 --server http://<host>:9001 \
                            --model <workspace/project/version> [--display]

On the Raspberry Pi, swap cv2.VideoCapture for Picamera2:
    picam2.capture_array() returns the latest frame directly (two-line swap
    in run_bench(), marked below).

Revisions vs. the Phase 1 specification — record these in Chapter 4
-------------------------------------------------------------------
R1  Distance estimation no longer uses the class-size pinhole model
    Z = f·W_real/W_px (Eq. 22). The binary class collapse discards the class
    label, so there is no W_real to look up. Distance now comes from
    projecting the bounding-box BOTTOM edge onto the ground plane using the
    camera mount geometry already defined in §3.4.9. No class knowledge
    needed, and it behaves correctly for SOD classes that have no canonical
    real-world size (potholes, kerbs, bollards). Class labels are retained
    in logs only, for test evidence — never in the decision.

R2  §3.4.9 trigonometry corrected. For a camera h metres above ground,
    optical axis tilted θ below the horizon, a ray depressed δ below the
    horizon meets the ground at d = h / tan(δ). The submitted report used
    h·tan(δ), giving a ground window of 0.10–1.24 m — inside which a 3 m
    stop cue is geometrically impossible. Corrected window with the same
    assumed mount (h = 0.90 m, θ = 30°, FOV_v = 48°): 0.65 m … 8.56 m.

R3  Confidence threshold 0.35 (report said 0.5): deliberate bias toward
    false positives for a safety-critical alert.

R4  Closed-loop pacing: exactly one inference request in flight, always on
    the newest frame. End-to-end latency stays ≈ (1/fps_server + RTT)
    instead of growing a stale-frame queue at the server.

R5  Latency and effective-fps instrumentation, feeding the Chapter 5
    measured-vs-theoretical comparison against Table 2 (§3.4.7).

Constants tagged [*] are pre-build assumptions. Measure on the physical
prototype and update, per the asterisk convention in §3.4.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

SERVER_URL = os.environ.get("CANE_INFER_URL", "http://192.168.0.100:9001")
#   Docker host on the LAN. Use http://localhost:9001 only if the container
#   actually runs on the Pi itself (check achievable fps there first).
MODEL_ID = os.environ.get("CANE_MODEL_ID", "WORKSPACE/sod-project/1")
#   Roboflow model id "workspace/project/version" — set to the trained
#   RF-DETR Small SOD deployment.
API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

CONFIDENCE_THRESHOLD = 0.35          # R3 — was 0.5 in the Phase 1 text

CAMERA_HEIGHT_M = 0.90               # [*] mount height above ground
CAMERA_TILT_DEG = 30.0               # [*] optical axis below the horizon
VERTICAL_FOV_DEG = 48.0              # [*]
HORIZONTAL_FOV_DEG = 62.0            # [*] used by Eq. 26 angular offset

STOP_TRIGGER_M = 3.0                 # Eq. 21 — re-derive once end-to-end
                                     # latency is measured on the prototype
CORRIDOR_HALF_ANGLE_DEG = 20.0       # [*] forward-path gate on α_body
MEMORY_HOLD_S = 1.5                  # Eq. 29 — sweep occlusion hold
VISION_TIMEOUT_S = 3.0               # fail-safe window (Ch.3 §3.2)

HORIZON_MARGIN_DEG = 1.0             # rays this close to the horizon have no
                                     # usable ground intersection
FALLBACK_BBOX_HEIGHT_FRAC = 0.45     # [*] if a box bottom sits above the
                                     # horizon (overhanging obstacle), treat
                                     # it as in range when it fills ≥ this
                                     # fraction of frame height — conservative
LATENCY_WINDOW = 120                 # rolling samples for stats()


# ─────────────────────────────────────────────────────────────────────────────
# Geometry — corrected §3.4.9 and Eq. 26
# ─────────────────────────────────────────────────────────────────────────────

def ground_distance_m(
    y_bottom_px: float,
    frame_h_px: int,
    h: float = CAMERA_HEIGHT_M,
    tilt_deg: float = CAMERA_TILT_DEG,
    fov_v_deg: float = VERTICAL_FOV_DEG,
) -> Optional[float]:
    """Horizontal distance to where the bbox bottom edge meets the ground.

    Pinhole row→angle: a pixel row a fraction r = y/H down the frame sits
    ε = (r − 0.5)·FOV_v away from the optical axis (positive = further below
    the axis). Total depression below the horizon: δ = θ_tilt + ε. Ground
    intersection (R2):  d = h / tan(δ).

    Returns None when δ ≤ HORIZON_MARGIN_DEG — the ray skims or clears the
    horizon, so no ground-plane distance exists (overhanging obstacle or a
    detection whose base is out of frame).
    """
    eps_deg = ((y_bottom_px / frame_h_px) - 0.5) * fov_v_deg
    delta_deg = tilt_deg + eps_deg
    if delta_deg <= HORIZON_MARGIN_DEG:
        return None
    return h / math.tan(math.radians(delta_deg))


def camera_angle_deg(
    x_center_px: float,
    frame_w_px: int,
    fov_h_deg: float = HORIZONTAL_FOV_DEG,
) -> float:
    """Eq. 26 — angular offset of a detection from the camera centre axis."""
    return ((x_center_px / frame_w_px) - 0.5) * fov_h_deg


def _row_fraction_for_distance(d: float) -> float:
    """Inverse of ground_distance_m, used by the self-test only."""
    delta = math.degrees(math.atan(CAMERA_HEIGHT_M / d))
    eps = delta - CAMERA_TILT_DEG
    return 0.5 + eps / VERTICAL_FOV_DEG


# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """One filtered detection — kept for logging/overlay, never for control."""
    x: float                 # bbox centre x (px)
    y: float                 # bbox centre y (px)
    w: float                 # bbox width (px)
    h: float                 # bbox height (px)
    confidence: float
    cls: str                 # logged only — binary collapse happens downstream
    distance_m: Optional[float] = None
    alpha_body_deg: float = 0.0
    triggered: bool = False


@dataclass
class ObstacleStatus:
    """What main.py consumes. The control signal is obstacle_active alone."""
    ok: bool                          # inference round trip succeeded
    obstacle_now: bool                # ≥1 detection in corridor and in range
    obstacle_active: bool             # obstacle_now OR within memory hold
    nearest_distance_m: Optional[float]
    n_detections: int
    latency_ms: float
    detections: List[Detection] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class ObstacleDetector:
    def __init__(
        self,
        server_url: str = SERVER_URL,
        model_id: str = MODEL_ID,
        api_key: str = API_KEY,
        confidence: float = CONFIDENCE_THRESHOLD,
    ):
        # Lazy import so --selftest runs on machines without the SDK.
        from inference_sdk import InferenceHTTPClient  # pip install inference-sdk

        self._client = InferenceHTTPClient(api_url=server_url, api_key=api_key)
        self._model_id = model_id
        self._confidence = confidence

        self._last_success_t = time.monotonic()
        self._last_obstacle_t: Optional[float] = None
        self._latencies = deque(maxlen=LATENCY_WINDOW)
        self._frame_times = deque(maxlen=LATENCY_WINDOW)

    # -- core ----------------------------------------------------------------

    def process_frame(self, frame_bgr, cane_yaw_deg: float = 0.0) -> ObstacleStatus:
        """One closed-loop inference cycle on the newest frame (R4).

        cane_yaw_deg: IMU-measured cane yaw relative to the body centreline
        (Eq. 27, α_body = α_cam + θ_yaw). Pass 0.0 during bench tests before
        imu_filter.py exists.
        """
        now = time.monotonic()
        frame_h, frame_w = frame_bgr.shape[:2]

        t0 = now
        try:
            result = self._client.infer(frame_bgr, model_id=self._model_id)
        except Exception:
            # Server unreachable / decode failure. Report not-ok; main.py's
            # fail-safe path covers the conservative stop cue via is_stale().
            return ObstacleStatus(
                ok=False,
                obstacle_now=False,
                obstacle_active=self._memory_active(time.monotonic()),
                nearest_distance_m=None,
                n_detections=0,
                latency_ms=(time.monotonic() - t0) * 1000.0,
            )
        latency_ms = (time.monotonic() - t0) * 1000.0

        self._last_success_t = time.monotonic()
        self._latencies.append(latency_ms)
        self._frame_times.append(self._last_success_t)

        detections: List[Detection] = []
        obstacle_now = False
        nearest: Optional[float] = None

        for p in result.get("predictions", []):
            conf = float(p.get("confidence", 0.0))
            if conf < self._confidence:
                continue

            det = Detection(
                x=float(p["x"]), y=float(p["y"]),
                w=float(p["width"]), h=float(p["height"]),
                confidence=conf,
                cls=str(p.get("class", "?")),
            )

            # Forward-path corridor after sweep correction (Eq. 26–27).
            alpha_cam = camera_angle_deg(det.x, frame_w)
            det.alpha_body_deg = alpha_cam + cane_yaw_deg
            in_path = abs(det.alpha_body_deg) <= CORRIDOR_HALF_ANGLE_DEG

            # Ground-plane range from the bbox bottom edge (R1, R2).
            y_bottom = det.y + det.h / 2.0
            det.distance_m = ground_distance_m(y_bottom, frame_h)
            if det.distance_m is not None:
                in_range = det.distance_m <= STOP_TRIGGER_M
            else:
                # Base above the horizon: fall back to apparent size — biased
                # to alerting, consistent with the 0.35 threshold philosophy.
                in_range = (det.h / frame_h) >= FALLBACK_BBOX_HEIGHT_FRAC

            det.triggered = in_path and in_range
            if det.triggered:
                obstacle_now = True
                if det.distance_m is not None:
                    nearest = (det.distance_m if nearest is None
                               else min(nearest, det.distance_m))
            detections.append(det)

        if obstacle_now:
            self._last_obstacle_t = self._last_success_t

        return ObstacleStatus(
            ok=True,
            obstacle_now=obstacle_now,
            obstacle_active=obstacle_now or self._memory_active(self._last_success_t),
            nearest_distance_m=nearest,
            n_detections=len(detections),
            latency_ms=latency_ms,
            detections=detections,
        )

    # -- health & instrumentation ---------------------------------------------

    def is_stale(self, timeout_s: float = VISION_TIMEOUT_S) -> bool:
        """True when no successful inference for timeout_s — main.py should
        hold the finger-5 stop cue as the conservative safe state."""
        return (time.monotonic() - self._last_success_t) > timeout_s

    def stats(self) -> dict:
        """Rolling latency / throughput for the Chapter 5 measured table."""
        if not self._latencies:
            return {"samples": 0}
        lat = sorted(self._latencies)
        fps = None
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                fps = (len(self._frame_times) - 1) / span
        return {
            "samples": len(lat),
            "latency_mean_ms": sum(lat) / len(lat),
            "latency_p95_ms": lat[int(0.95 * (len(lat) - 1))],
            "effective_fps": fps,
        }

    def _memory_active(self, now: float) -> bool:
        return (self._last_obstacle_t is not None
                and (now - self._last_obstacle_t) <= MEMORY_HOLD_S)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — pure geometry, no camera, no server
# ─────────────────────────────────────────────────────────────────────────────

def selftest() -> None:
    H, W = 480, 640
    approx = lambda a, b, tol=0.02: abs(a - b) <= tol * max(1.0, abs(b))

    # Corrected §3.4.9 window: bottom edge of frame → nearest visible ground.
    d_near = ground_distance_m(H, H)
    assert d_near is not None and approx(d_near, 0.654), d_near
    print(f"PASS  nearest visible ground  {d_near:.3f} m  (expected ≈ 0.654)")

    # Top edge of frame → farthest visible ground.
    d_far = ground_distance_m(0, H)
    assert d_far is not None and approx(d_far, 8.563), d_far
    print(f"PASS  farthest visible ground {d_far:.3f} m  (expected ≈ 8.563)")

    # Stop-cue sanity: the 3 m trigger lies inside the corrected window and
    # round-trips through the inverse mapping.
    r = _row_fraction_for_distance(STOP_TRIGGER_M)
    d_back = ground_distance_m(r * H, H)
    assert 0.0 < r < 1.0 and d_back is not None and approx(d_back, 3.0), (r, d_back)
    print(f"PASS  3 m trigger ↔ row fraction {r:.3f} (≈ 0.223), round-trip {d_back:.3f} m")

    # Eq. 26 worked example from the Phase 1 report: x = 400 px → +7.75°.
    a = camera_angle_deg(400, W)
    assert approx(a, 7.75), a
    print(f"PASS  Eq.26 worked example    {a:+.2f}°  (expected +7.75°)")

    # Horizon guard: a bbox bottom above the horizon returns None.
    r_horizon = 0.5 - CAMERA_TILT_DEG / VERTICAL_FOV_DEG          # δ = 0
    assert ground_distance_m((r_horizon - 0.02) * H, H) is None
    print("PASS  above-horizon rows correctly yield no ground distance")

    # Monotonicity: lower in frame ⇒ nearer.
    ds = [ground_distance_m(y, H) for y in range(int(0.3 * H), H, 20)]
    ds = [d for d in ds if d is not None]
    assert all(ds[i] > ds[i + 1] for i in range(len(ds) - 1))
    print("PASS  distance decreases monotonically toward the frame bottom")

    print("\nAll geometry self-tests passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Bench runner — live camera against the Docker server
# ─────────────────────────────────────────────────────────────────────────────

def run_bench(camera_index: int, server: str, model: str, display: bool) -> None:
    import cv2  # pip install opencv-python

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # newest-frame bias (R4)
    if not cap.isOpened():
        raise SystemExit(f"camera {camera_index} not available")
    # Raspberry Pi: replace the two lines above with
    #   from picamera2 import Picamera2; picam2 = Picamera2(); picam2.start()
    # and read frames with: frame = picam2.capture_array()

    det = ObstacleDetector(server_url=server, model_id=model)
    print(f"bench: {server}  model={model}  conf≥{CONFIDENCE_THRESHOLD}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            status = det.process_frame(frame, cane_yaw_deg=0.0)

            flag = "OBSTACLE" if status.obstacle_active else "clear   "
            near = (f"{status.nearest_distance_m:4.1f} m"
                    if status.nearest_distance_m is not None else "  --  ")
            s = det.stats()
            fps = f"{s.get('effective_fps') or 0:4.1f}"
            print(f"{flag}  nearest {near}  dets {status.n_detections}  "
                  f"lat {status.latency_ms:6.1f} ms  fps {fps}", flush=True)

            if display:
                for d in status.detections:
                    p1 = (int(d.x - d.w / 2), int(d.y - d.h / 2))
                    p2 = (int(d.x + d.w / 2), int(d.y + d.h / 2))
                    col = (0, 0, 255) if d.triggered else (0, 200, 0)
                    cv2.rectangle(frame, p1, p2, col, 2)
                    tag = f"{d.distance_m:.1f}m" if d.distance_m else "n/a"
                    cv2.putText(frame, tag, (p1[0], p1[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                cv2.imshow("yolo_detector bench", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()
        print("\nfinal stats:", det.stats())


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--selftest", action="store_true",
                    help="run geometry checks, no hardware needed")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--server", default=SERVER_URL)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--display", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    else:
        run_bench(args.camera, args.server, args.model, args.display)
