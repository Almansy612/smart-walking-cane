#!/usr/bin/env python3
"""
main.py — system orchestrator for the Smart Walking Cane
========================================================

Wires the four subsystem modules into the Figure 3.1 data flow and owns
the decision/arbitration layer:

    gps_navigator   GpsReader + GpsNavigator   → fingers 1–4 (macro nav)
    imu_filter      ImuService                 → heading_slow / sweep_yaw
    yolo_detector   ObstacleDetector           → finger 5 (stop cue)
    haptic_output   HapticDriver               → the five ERM motors

Arbitration policy (§3.4.4 fail-safe table, plus priorities)
------------------------------------------------------------
* STOP OVERRIDES NAV: while the obstacle flag is active — or the vision
  pipeline is stale > 3 s (conservative safe state) — finger 5 re-cues
  every OBSTACLE_RECUE_S and turn cues are suppressed for that tick.
* GPS LOST > 10 s: NavEvent.NAV_LOST → all-finger alert pattern, nav cues
  silent; the alert's own 8 s refractory paces repeats while lost. On
  recovery, a thumb pulse confirms navigation has resumed.
* IMU UNSTABLE (magnetometer innovation std > 15°): left/right/turn-back
  cues are suppressed — heading can't be trusted to pick a direction —
  while forward confirmations and the stop cue stay live.
* Every tick is appended to a CSV (logs/run_*.csv or sim_run.csv): the
  raw material for the Chapter 5 results tables.

Running it
----------
    python main.py --sim                 # full-system simulation, no hardware
    python main.py --sim --duration 200
    python main.py                       # hardware run (Pi 5, all sensors)
    python main.py --route route.csv     # waypoints: lat,lon[,name] per line

The simulation drives the REAL HeadingFilter, GpsNavigator and HapticDriver
at logical time with a seeded random walk, GPS dropout, a disorientation
kick, an obstacle window and a vision-stale window — so the whole decision
layer is exercised and asserted before a single wire is soldered. Use it
as In-Progress Demo evidence.

Constants tagged [*] are pre-build assumptions — tune on the prototype.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from gps_navigator import (Fix, GpsNavigator, NavEvent, Waypoint,
                           bearing_deg, wrap180)
from haptic_output import Finger, HapticDriver, RecordingBackend
from imu_filter import HeadingFilter, ImuState

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

CONTROL_RATE_HZ = 10.0           # §3.4.7 decision-layer rate
OBSTACLE_RECUE_S = 2.0           # [*] finger-5 repeat while obstacle persists
VISION_STALE_S = 3.0             # §3.2 conservative safe state boundary
LOG_DIR = "logs"

EVENT_FINGER: Dict[NavEvent, Finger] = {
    NavEvent.TURN_RIGHT: Finger.INDEX,
    NavEvent.TURN_LEFT: Finger.MIDDLE,
    NavEvent.TURN_BACK: Finger.RING,
    NavEvent.WAYPOINT_REACHED: Finger.THUMB,
    NavEvent.NAV_RECOVERED: Finger.THUMB,
}
TURN_EVENTS = {NavEvent.TURN_RIGHT, NavEvent.TURN_LEFT, NavEvent.TURN_BACK}


@dataclass
class VisionState:
    active: bool = False         # obstacle in corridor and in range
    stale: bool = False          # no successful inference > VISION_STALE_S
    latency_ms: float = 0.0
    nearest_m: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Decision layer — shared verbatim by simulation and hardware
# ─────────────────────────────────────────────────────────────────────────────

class ControlLoop:
    def __init__(self, nav: GpsNavigator, hap: HapticDriver,
                 clock: Callable[[], float], log_path: Optional[str] = None):
        self.nav, self.hap, self.clock = nav, hap, clock
        self.gps_lost = False
        self._t_last_obstacle_cue = -1e9
        self.counts: Dict[str, int] = {}
        self._log = None
        if log_path:
            self._log = open(log_path, "w", newline="")
            self._csv = csv.writer(self._log)
            self._csv.writerow(["t", "lat", "lon", "gps_valid", "heading_slow",
                                "err_deg", "dist_m", "wp", "events",
                                "obstacle", "vision_stale", "imu_unstable"])

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def step(self, fix: Fix, imu: ImuState, vision: VisionState) -> List[str]:
        now = self.clock()
        fired: List[str] = []

        # 1 — stop cue first: it outranks everything (§3.3.2 priority).
        stop_state = vision.active or vision.stale
        if stop_state and now - self._t_last_obstacle_cue >= OBSTACLE_RECUE_S:
            if self.hap.pulse(Finger.PINKY):
                self._t_last_obstacle_cue = now
                tag = "STOP(stale)" if (vision.stale and not vision.active) \
                    else "STOP(obstacle)"
                fired.append(tag)
                self._count(tag)

        # 2 — navigation events through the fail-safe filters.
        events = self.nav.update(fix, imu.heading_slow, now=now)
        for ev in events:
            if ev is NavEvent.NAV_LOST:
                self.gps_lost = True
                if self.hap.alert():
                    fired.append("ALERT(gps-lost)")
                    self._count("ALERT")
            elif ev is NavEvent.NAV_RECOVERED:
                self.gps_lost = False
                if self.hap.pulse(Finger.THUMB):
                    fired.append("THUMB(recovered)")
                    self._count("RECOVERED")
            elif ev is NavEvent.ROUTE_COMPLETE:
                if self.hap.double_pulse(Finger.THUMB):
                    fired.append("THUMBx2(route-complete)")
                    self._count("COMPLETE")
            elif ev in TURN_EVENTS and imu.unstable:
                fired.append(f"suppressed({ev.name}:imu-unstable)")
                self._count("SUPPRESSED_IMU")
            elif ev in TURN_EVENTS and stop_state:
                fired.append(f"suppressed({ev.name}:stop-override)")
                self._count("SUPPRESSED_STOP")
            else:
                # Same-finger events landing in one tick (e.g. NAV_RECOVERED +
                # WAYPOINT_REACHED) merge into a single physical pulse via the
                # refractory — correct device behaviour. Count the navigator
                # event either way; the pulse ledger lives in the backend.
                pulsed = self.hap.pulse(EVENT_FINGER[ev])
                fired.append(ev.name if pulsed else f"{ev.name}(merged)")
                self._count(ev.name)

        # 3 — while lost, keep offering the alert; its refractory paces it.
        if self.gps_lost and self.hap.alert():
            fired.append("ALERT(repeat)")
            self._count("ALERT")

        if self._log:
            self._csv.writerow([
                f"{now:.1f}", f"{fix.lat:.7f}", f"{fix.lon:.7f}",
                int(fix.valid), f"{imu.heading_slow:.1f}",
                "" if self.nav.last_error is None else f"{self.nav.last_error:.1f}",
                "" if self.nav.last_distance_m is None else f"{self.nav.last_distance_m:.1f}",
                self.nav.wp_index, ";".join(fired),
                int(vision.active), int(vision.stale), int(imu.unstable)])
        return fired

    def close(self) -> None:
        if self._log:
            self._log.close()


# ─────────────────────────────────────────────────────────────────────────────
# Simulation world — exercises the real filter/navigator/haptics, no hardware
# ─────────────────────────────────────────────────────────────────────────────

M_PER_DEG_LAT = 111_195.0


def square_route(lat0: float, lon0: float, leg_m: float = 40.0) -> List[Waypoint]:
    m_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
    dx, dy = leg_m / m_lon, leg_m / M_PER_DEG_LAT
    return [Waypoint(lat0, lon0 + dx, "E corner"),
            Waypoint(lat0 + dy, lon0 + dx, "NE corner"),
            Waypoint(lat0 + dy, lon0, "NW corner"),
            Waypoint(lat0, lon0, "home")]


class SimBackend(RecordingBackend):
    """Recording backend that also narrates with simulation timestamps."""

    def pulse(self, finger: Finger, duration_s: float) -> None:
        super().pulse(finger, duration_s)
        print(f"  t={self.clock():6.1f}s  [HAPTIC] finger {int(finger)} "
              f"{finger.name:<6} pulse {duration_s*1000:.0f} ms")

    def alert(self) -> None:
        super().alert()
        print(f"  t={self.clock():6.1f}s  [HAPTIC] ALL-FINGER ALERT — GPS lost")


class SimWorld:
    """Walker with lagging steering, cane sweep, gyro bias, mag noise."""

    SPEED_MPS = 1.4              # §3.4: assumed walking speed
    STEER_GAIN = 1.0             # /s — heading relaxes toward the bearing
    SWEEP_DEG, SWEEP_HZ = 30.0, 0.9
    GYRO_BIAS, GYRO_NOISE = 0.4, 0.3
    MAG_NOISE = 3.0
    KICK_T, KICK_LEN, KICK_OFF = 40.0, 6.0, -150.0   # disorientation:
    # the walker locks onto (bearing − 150°) for 6 s — a held wrong heading,
    # which an impulsive spike cannot represent because steering cancels it.

    def __init__(self, lat0: float, lon0: float, filt: HeadingFilter):
        self.lat0, self.lon0 = lat0, lon0
        self.m_lon = M_PER_DEG_LAT * math.cos(math.radians(lat0))
        self.east = self.north = 0.0
        self.walker_heading = 90.0
        self.filt = filt
        self._cane_prev = 90.0
        self.t = 0.0

    def latlon(self):
        return (self.lat0 + self.north / M_PER_DEG_LAT,
                self.lon0 + self.east / self.m_lon)

    def advance(self, dt_ctrl: float, target: Waypoint) -> None:
        steps = 10                                  # 100 Hz IMU inside 10 Hz
        dt = dt_ctrl / steps
        for _ in range(steps):
            self.t += dt
            lat, lon = self.latlon()
            brg = bearing_deg(lat, lon, target.lat, target.lon)
            desired = brg
            if self.KICK_T <= self.t < self.KICK_T + self.KICK_LEN:
                desired = wrap180(brg + self.KICK_OFF)
            err = wrap180(desired - self.walker_heading)
            self.walker_heading = wrap180(
                self.walker_heading + self.STEER_GAIN * err * dt)

            h = math.radians(self.walker_heading)
            self.east += self.SPEED_MPS * math.sin(h) * dt
            self.north += self.SPEED_MPS * math.cos(h) * dt

            cane = wrap180(self.walker_heading + self.SWEEP_DEG
                           * math.sin(2 * math.pi * self.SWEEP_HZ * self.t))
            true_rate = wrap180(cane - self._cane_prev) / dt
            self._cane_prev = cane
            gz = true_rate + self.GYRO_BIAS + random.gauss(0, self.GYRO_NOISE)
            mag = wrap180(cane + random.gauss(0, self.MAG_NOISE))
            self.filt.update(gz, mag, dt)

    def gps_fix(self, t: float) -> Fix:
        lat, lon = self.latlon()
        return Fix(lat=lat + random.gauss(0, 1.5) / M_PER_DEG_LAT,
                   lon=lon + random.gauss(0, 1.5) / self.m_lon,
                   valid=True, n_sats=9, t_mono=t)


def run_sim(duration_s: float = 220.0, seed: int = 42) -> None:
    random.seed(seed)
    print(f"=== SIMULATION  seed={seed}  duration≤{duration_s:.0f}s "
          f"(40 m square, 1.4 m/s) ===\n"
          "    obstacle 20–27 s · disorientation −150° held 40–46 s ·\n"
          "    vision-stale 52–58 s · GPS dropout 70–83 s\n")

    route = square_route(3.10000, 101.60000)
    nav = GpsNavigator(route)
    filt = HeadingFilter(initial_deg=90.0)
    tstate = {"now": 0.0}
    clock = lambda: tstate["now"]

    backend = SimBackend(clock)
    hap = HapticDriver(backend=backend, clock=clock)
    hap.add_pulse_listener(filt.notify_pulse)        # B1 — real blanking path

    world = SimWorld(3.10000, 101.60000, filt)
    ctl = ControlLoop(nav, hap, clock, log_path="sim_run.csv")

    fix = world.gps_fix(0.0)
    t_last_fix = 0.0
    t_complete = None
    dt = 1.0 / CONTROL_RATE_HZ

    steps = int(duration_s / dt)
    for i in range(steps):
        t = i * dt
        tstate["now"] = t
        world.advance(dt, nav.target())

        if not (70.0 <= t < 83.0) and t - t_last_fix >= 0.2:   # 5 Hz GPS
            fix, t_last_fix = world.gps_fix(t), t

        vision = VisionState(active=(20.0 <= t < 27.0),
                             stale=(52.0 <= t < 58.0))

        imu = ImuState(heading=filt.heading, heading_slow=filt.heading_slow,
                       sweep_yaw=filt.sweep_yaw, unstable=filt.unstable,
                       ok=True, t_mono=t)
        ctl.step(fix, imu, vision)

        if nav.complete and t_complete is None:
            t_complete = t
        if t_complete is not None and t - t_complete > 3.0:
            break

    ctl.close()
    c = ctl.counts
    print(f"\n=== SUMMARY @ t={tstate['now']:.1f}s — sim_run.csv written ===")
    for k in sorted(c):
        print(f"    {c[k]:3d} × {k}")

    checks = [
        ("route completed", nav.complete),
        ("turn-left cues at corners", c.get("TURN_LEFT", 0) >= 2),
        ("turn-right after disorientation", c.get("TURN_RIGHT", 0) >= 1),
        ("turn-back at |err| ≥ 135°", c.get("TURN_BACK", 0) >= 1),
        ("waypoint confirmations", c.get("WAYPOINT_REACHED", 0) >= 3),
        ("GPS-loss alert fired", c.get("ALERT", 0) >= 1),
        ("recovery thumb pulse", c.get("RECOVERED", 0) >= 1),
        ("obstacle stop cues", c.get("STOP(obstacle)", 0) >= 3),
        ("vision-stale conservative cues", c.get("STOP(stale)", 0) >= 2),
        ("route-complete double pulse", c.get("COMPLETE", 0) == 1),
    ]
    print()
    ok = True
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    if not ok:
        raise SystemExit("simulation checks FAILED")
    print("\nAll end-to-end simulation checks passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Hardware path (Raspberry Pi 5 — bring-up order in the docstring)
# ─────────────────────────────────────────────────────────────────────────────

def _vision_worker(shared: VisionState, lock: threading.Lock,
                   imu_svc, stop: threading.Event) -> None:
    import cv2
    from yolo_detector import ObstacleDetector
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Raspberry Pi: replace with Picamera2 (see yolo_detector.run_bench note).
    det = ObstacleDetector()
    while not stop.is_set():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        yaw = imu_svc.snapshot().sweep_yaw
        st = det.process_frame(frame, cane_yaw_deg=yaw)
        with lock:
            shared.active = st.obstacle_active
            shared.stale = det.is_stale(VISION_STALE_S)
            shared.latency_ms = st.latency_ms
            shared.nearest_m = st.nearest_distance_m
    cap.release()


def load_route(path: Optional[str]) -> List[Waypoint]:
    if not path:
        print("[main] no --route file: using a 40 m demo square at the "
              "default origin — replace with surveyed campus waypoints")
        return square_route(3.10000, 101.60000)
    wps = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            wps.append(Waypoint(float(row[0]), float(row[1]),
                                row[2] if len(row) > 2 else ""))
    return wps


def run_hardware(route_path: Optional[str]) -> None:
    from gps_navigator import GpsReader
    from imu_filter import ImuService

    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, time.strftime("run_%Y%m%d_%H%M%S.csv"))

    nav = GpsNavigator(load_route(route_path))
    hap = HapticDriver()                                  # real GPIO on Pi
    imu_svc = ImuService().start()
    hap.add_pulse_listener(imu_svc.filter.notify_pulse)   # B1 blanking
    gps = GpsReader().start()

    shared, vlock, stop = VisionState(), threading.Lock(), threading.Event()
    threading.Thread(target=_vision_worker,
                     args=(shared, vlock, imu_svc, stop), daemon=True).start()

    ctl = ControlLoop(nav, hap, time.monotonic, log_path=log)
    print(f"[main] running — logging to {log}  (Ctrl+C to stop)")
    try:
        while not nav.complete:
            t0 = time.monotonic()
            with vlock:
                vision = VisionState(**vars(shared))
            fired = ctl.step(gps.fix(), imu_svc.snapshot(), vision)
            for f in fired:
                print(f"  {time.strftime('%H:%M:%S')}  {f}")
            time.sleep(max(0.0, 1.0 / CONTROL_RATE_HZ
                           - (time.monotonic() - t0)))
        print("[main] route complete")
    except KeyboardInterrupt:
        print("\n[main] stopped by user")
    finally:
        stop.set()
        gps.stop()
        imu_svc.stop()
        hap.stop_all()
        ctl.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Smart Walking Cane orchestrator")
    ap.add_argument("--sim", action="store_true",
                    help="full-system simulation, no hardware required")
    ap.add_argument("--duration", type=float, default=220.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--route", help="CSV of waypoints: lat,lon[,name]")
    args = ap.parse_args()

    if args.sim:
        run_sim(args.duration, args.seed)
    else:
        run_hardware(args.route)
