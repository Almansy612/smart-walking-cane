#!/usr/bin/env python3
"""
gps_navigator.py — macro-navigation for the Smart Walking Cane
==============================================================

Implements the GPS pipeline of Chapter 3 (§3.3.2): Haversine distance
(Eq. 9), forward-azimuth bearing (Eq. 10), heading error normalised to
±180° (Eq. 11–12), the 5 m arrival radius (Eq. 13–14), and the
threshold-plus-hysteresis trigger logic (§3.4.4), emitting one event per
trigger so main.py can fire the matching finger.

Hardware notes (NEO-M8N)
------------------------
* The module ships transmitting NMEA at 1 Hz / 9600 baud. GpsReader sends a
  UBX-CFG-RATE frame on connect to raise this — default 5 Hz, which keeps
  multi-constellation reception (GPS+GLONASS+Galileo+BeiDou). 10 Hz is
  possible but forces GPS-only and costs the accuracy benefit the module was
  selected for; at 1.4 m/s walking speed, 5 Hz already resolves 0.28 m per
  fix against a 5 m arrival radius, so 5 Hz is the justified default.
  Record this trade-off in Chapter 4.
* Raspberry Pi 5 wiring: GPS TX → GPIO15 (RXD), GPS RX → GPIO14 (TXD),
  enable the UART with `dtparam=uart0=on` in /boot/firmware/config.txt;
  the port then appears as /dev/ttyAMA0 (aliased /dev/serial0).

Sign convention (resolves the ambiguous wording in the Phase 1 text)
--------------------------------------------------------------------
    err = wrap180(required_bearing − current_heading)
Worked example: bearing 090° (east), heading 060° → err = +30°. The target
sits clockwise of the user's nose, so a POSITIVE error means TURN RIGHT
(index finger); a NEGATIVE error means TURN LEFT (middle finger). |err| ≥
135° means the user is walking away from the waypoint → TURN BACK (ring).

Interface for main.py
---------------------
    reader = GpsReader("/dev/ttyAMA0")          # background serial thread
    nav    = GpsNavigator(route)
    events = nav.update(reader.fix(), heading_deg=imu.heading_slow)
    for ev in events: ...                       # map NavEvent → finger

Run `python gps_navigator.py --selftest` for the no-hardware checks.

Constants tagged [*] are pre-build assumptions — measure / tune on the
prototype and update, per the §3.4 asterisk convention.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

EARTH_RADIUS_M = 6_371_000.0          # Eq. 9

ARRIVAL_RADIUS_M = 5.0                # Eq. 14 — 1.5× safety on 2.5 m CEP
TRIGGER_DEG = 20.0                    # §3.4.4: θ_th 15° + hysteresis e 5°
RELEASE_DEG = 10.0                    # θ_th − e: cue clears inside this band
TURNBACK_DEG = 135.0                  # ring-finger "walking away" boundary [*]
TURNBACK_REFRACTORY_S = 5.0           # [*] min gap between turn-back cues
REPEAT_CUE_S = 4.0                    # [*] re-cue period while a correction
                                      # state persists (missed-cue recovery —
                                      # deviation from strict one-pulse spec;
                                      # justify in Chapter 4)
GPS_STALE_S = 10.0                    # §3.4.4 fail-safe: nav cues suspended

SERIAL_PORT = "/dev/ttyAMA0"
SERIAL_BAUD = 9600
GPS_RATE_HZ = 5                       # see hardware notes; 10 ⇒ GPS-only


# ─────────────────────────────────────────────────────────────────────────────
# Pure geodesy — Eq. 9 / 10 / 11–12 (unit-tested in selftest)
# ─────────────────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres (Eq. 9)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth from point 1 to point 2, 0–360°, 0 = north (Eq. 10)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlmb = math.radians(lon2 - lon1)
    x = math.sin(dlmb) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlmb)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def wrap180(deg: float) -> float:
    """Normalise an angle to (−180, +180] (Eq. 12)."""
    return ((deg + 180.0) % 360.0) - 180.0


def heading_error_deg(required_bearing: float, current_heading: float) -> float:
    """Eq. 11–12. Positive ⇒ turn right, negative ⇒ turn left."""
    return wrap180(required_bearing - current_heading)


# ─────────────────────────────────────────────────────────────────────────────
# NMEA parsing (RMC / GGA) with checksum validation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fix:
    lat: float = 0.0
    lon: float = 0.0
    valid: bool = False
    speed_mps: float = 0.0
    n_sats: int = 0
    t_mono: float = 0.0               # monotonic stamp of last VALID fix

    def age_s(self) -> float:
        return time.monotonic() - self.t_mono if self.t_mono else float("inf")


def nmea_checksum_ok(sentence: str) -> bool:
    """XOR of all chars between '$' and '*' must equal the hex suffix."""
    if not sentence.startswith("$") or "*" not in sentence:
        return False
    body, _, cs = sentence[1:].partition("*")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(cs[:2], 16)
    except ValueError:
        return False


def _dm_to_deg(dm: str, hemi: str) -> Optional[float]:
    """ddmm.mmmm / dddmm.mmmm → signed decimal degrees."""
    if not dm or "." not in dm:
        return None
    head, _ = dm.split(".", 1)
    deg_digits = len(head) - 2
    deg = float(dm[:deg_digits])
    minutes = float(dm[deg_digits:])
    val = deg + minutes / 60.0
    return -val if hemi in ("S", "W") else val


def parse_nmea(line: str, fix: Fix) -> bool:
    """Update `fix` in place from one RMC or GGA sentence.
    Returns True when the sentence carried a new valid position."""
    line = line.strip()
    if not nmea_checksum_ok(line):
        return False
    f = line[1:line.index("*")].split(",")
    talker = f[0]

    if talker.endswith("RMC") and len(f) >= 8:
        # $GxRMC,time,status,lat,NS,lon,EW,speed_kn,course,date,...
        if f[2] != "A":
            fix.valid = False
            return False
        lat = _dm_to_deg(f[3], f[4])
        lon = _dm_to_deg(f[5], f[6])
        if lat is None or lon is None:
            return False
        fix.lat, fix.lon, fix.valid = lat, lon, True
        fix.speed_mps = float(f[7]) * 0.514444 if f[7] else 0.0
        fix.t_mono = time.monotonic()
        return True

    if talker.endswith("GGA") and len(f) >= 8:
        # $GxGGA,time,lat,NS,lon,EW,quality,numsats,...
        if f[7]:
            fix.n_sats = int(f[7])
        if f[6] in ("0", ""):
            return False
        lat = _dm_to_deg(f[2], f[3])
        lon = _dm_to_deg(f[4], f[5])
        if lat is None or lon is None:
            return False
        fix.lat, fix.lon, fix.valid = lat, lon, True
        fix.t_mono = time.monotonic()
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# UBX-CFG-RATE — lift the NEO-M8N above its 1 Hz default
# ─────────────────────────────────────────────────────────────────────────────

def build_ubx_cfg_rate(rate_hz: int = GPS_RATE_HZ) -> bytes:
    """Binary UBX frame: class 0x06 id 0x08, payload measRate(ms) u2,
    navRate=1 u2, timeRef=1(GPS) u2, 8-bit Fletcher checksum."""
    meas_ms = max(50, int(round(1000.0 / rate_hz)))
    payload = (meas_ms.to_bytes(2, "little")
               + (1).to_bytes(2, "little")
               + (1).to_bytes(2, "little"))
    frame = bytes([0x06, 0x08]) + len(payload).to_bytes(2, "little") + payload
    ck_a = ck_b = 0
    for b in frame:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return b"\xb5\x62" + frame + bytes([ck_a, ck_b])


class GpsReader:
    """Background serial reader. Lazy pyserial import keeps laptop selftests
    dependency-free. Reconnects automatically on port errors."""

    def __init__(self, port: str = SERIAL_PORT, baud: int = SERIAL_BAUD,
                 rate_hz: int = GPS_RATE_HZ):
        self._port, self._baud, self._rate = port, baud, rate_hz
        self._fix = Fix()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "GpsReader":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def fix(self) -> Fix:
        with self._lock:
            return Fix(**vars(self._fix))

    # internal ---------------------------------------------------------------

    def _run(self) -> None:
        import serial  # pip install pyserial
        while not self._stop.is_set():
            try:
                with serial.Serial(self._port, self._baud, timeout=1.0) as s:
                    s.write(build_ubx_cfg_rate(self._rate))   # 1 Hz → rate_hz
                    s.flush()
                    while not self._stop.is_set():
                        raw = s.readline().decode("ascii", errors="replace")
                        if raw.startswith("$"):
                            with self._lock:
                                parse_nmea(raw, self._fix)
            except Exception:
                time.sleep(1.0)        # port unplugged / not yet enumerated


# ─────────────────────────────────────────────────────────────────────────────
# Navigation state machine — §3.4.4 threshold + hysteresis
# ─────────────────────────────────────────────────────────────────────────────

class NavEvent(Enum):
    TURN_RIGHT = auto()        # index finger  (err > 0)
    TURN_LEFT = auto()         # middle finger (err < 0)
    TURN_BACK = auto()         # ring finger   (|err| ≥ 135°)
    WAYPOINT_REACHED = auto()  # thumb — forward confirmation
    ROUTE_COMPLETE = auto()    # thumb ×2 (main.py decides the pattern)
    NAV_LOST = auto()          # GPS stale > 10 s → suspend 1–4, alert
    NAV_RECOVERED = auto()


@dataclass
class Waypoint:
    lat: float
    lon: float
    name: str = ""


class _CueState(Enum):
    ALIGNED = auto()
    RIGHT = auto()
    LEFT = auto()


class GpsNavigator:
    """Consumes (fix, heading) at the 10 Hz control rate; emits NavEvents."""

    def __init__(self, route: List[Waypoint]):
        if not route:
            raise ValueError("route must contain at least one waypoint")
        self.route = route
        self.wp_index = 0
        self.complete = False
        self._cue = _CueState.ALIGNED
        self._lost = False
        self._t_last_cue = 0.0
        self._t_last_turnback = 0.0
        # telemetry for logging / Chapter 5
        self.last_distance_m: Optional[float] = None
        self.last_bearing: Optional[float] = None
        self.last_error: Optional[float] = None

    def target(self) -> Waypoint:
        return self.route[min(self.wp_index, len(self.route) - 1)]

    def update(self, fix: Fix, heading_deg: float,
               now: Optional[float] = None) -> List[NavEvent]:
        now = time.monotonic() if now is None else now
        events: List[NavEvent] = []

        # -- fail-safe: GPS staleness (§3.4.4) --------------------------------
        stale = (not fix.valid) or (now - fix.t_mono) > GPS_STALE_S
        if stale and not self._lost:
            self._lost = True
            self._cue = _CueState.ALIGNED
            events.append(NavEvent.NAV_LOST)
            return events
        if stale:
            return events                       # remain silent while lost
        if self._lost:
            self._lost = False
            events.append(NavEvent.NAV_RECOVERED)

        if self.complete:
            return events

        wp = self.target()
        dist = haversine_m(fix.lat, fix.lon, wp.lat, wp.lon)
        self.last_distance_m = dist

        # -- arrival (Eq. 13–14) ----------------------------------------------
        if dist < ARRIVAL_RADIUS_M:
            self.wp_index += 1
            self._cue = _CueState.ALIGNED
            if self.wp_index >= len(self.route):
                self.complete = True
                events.append(NavEvent.ROUTE_COMPLETE)
            else:
                events.append(NavEvent.WAYPOINT_REACHED)
            return events

        # -- heading error (Eq. 10–12) ----------------------------------------
        brg = bearing_deg(fix.lat, fix.lon, wp.lat, wp.lon)
        err = heading_error_deg(brg, heading_deg)
        self.last_bearing, self.last_error = brg, err

        # walking away entirely → turn-back cue with its own refractory
        if abs(err) >= TURNBACK_DEG:
            if now - self._t_last_turnback >= TURNBACK_REFRACTORY_S:
                self._t_last_turnback = now
                events.append(NavEvent.TURN_BACK)
            self._cue = _CueState.ALIGNED
            return events

        # hysteresis: trigger at ±TRIGGER_DEG, release inside ±RELEASE_DEG
        if self._cue is _CueState.ALIGNED:
            if err >= TRIGGER_DEG:
                self._cue, self._t_last_cue = _CueState.RIGHT, now
                events.append(NavEvent.TURN_RIGHT)
            elif err <= -TRIGGER_DEG:
                self._cue, self._t_last_cue = _CueState.LEFT, now
                events.append(NavEvent.TURN_LEFT)
        else:
            if abs(err) <= RELEASE_DEG:
                self._cue = _CueState.ALIGNED
            elif now - self._t_last_cue >= REPEAT_CUE_S:
                self._t_last_cue = now
                events.append(NavEvent.TURN_RIGHT if self._cue is _CueState.RIGHT
                              else NavEvent.TURN_LEFT)
        return events


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — pure logic, no serial port required
# ─────────────────────────────────────────────────────────────────────────────

def selftest() -> None:
    approx = lambda a, b, tol: abs(a - b) <= tol

    # Eq. 9: one degree of longitude on the equator = πR/180 ≈ 111 195 m.
    d = haversine_m(0, 0, 0, 1)
    assert approx(d, 111_195, 30), d
    print(f"PASS  Haversine 1° @ equator      {d:9.1f} m (≈ 111195)")

    # Eq. 10: cardinal bearings.
    assert approx(bearing_deg(0, 0, 0, 1), 90.0, 0.01)
    assert approx(bearing_deg(0, 0, 1, 0), 0.0, 0.01)
    assert approx(bearing_deg(0, 1, 0, 0), 270.0, 0.01)
    print("PASS  bearings: E=90  N=0  W=270")

    # Eq. 11–12 incl. the 0/360 seam: bearing 350, heading 10 → −20 (left).
    assert approx(heading_error_deg(350, 10), -20.0, 1e-9)
    assert approx(heading_error_deg(90, 60), +30.0, 1e-9)   # docstring example
    print("PASS  heading error wraps the 0/360 seam, sign convention holds")

    # NMEA: checksum + coordinate conversion on a synthetic RMC sentence.
    body = "GPRMC,123519,A,0309.474,N,10142.696,E,1.0,084.4,100626,,"
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    fx = Fix()
    assert parse_nmea(f"${body}*{cs:02X}", fx) and fx.valid
    assert approx(fx.lat, 3.1579, 0.0001) and approx(fx.lon, 101.7116, 0.0001)
    bad = f"${body}*00"
    assert not parse_nmea(bad, Fix())
    print(f"PASS  NMEA RMC parse → ({fx.lat:.4f}, {fx.lon:.4f}); bad checksum rejected")

    # UBX frame: known-good CFG-RATE @ 5 Hz (measRate 200 ms).
    frame = build_ubx_cfg_rate(5)
    assert frame.hex() == "b56206080600c80001000100de6a", frame.hex()
    print("PASS  UBX-CFG-RATE frame matches reference bytes")

    # Hysteresis state machine: fire once at 25°, hold at 18°, release at 8°,
    # re-arm and fire again at 25°.
    nav = GpsNavigator([Waypoint(0.0, 1.0)])
    fx = Fix(lat=0.0, lon=0.0, valid=True, t_mono=time.monotonic())
    seq = [(90 - 25, [NavEvent.TURN_RIGHT]), (90 - 18, []), (90 - 8, []),
           (90 - 25, [NavEvent.TURN_RIGHT])]
    t = 0.0
    for heading, expect in seq:
        fx.t_mono = time.monotonic()
        got = nav.update(fx, heading, now=t)
        assert got == expect, (heading, got)
        t += 0.1
    print("PASS  hysteresis: trigger 20° / release 10°, one event per entry")

    # Turn-back at |err| ≥ 135°, arrival advances the waypoint, route completes.
    nav = GpsNavigator([Waypoint(0.0, 0.00009), Waypoint(0.0, 0.00018)])
    fx.t_mono = time.monotonic()
    assert NavEvent.TURN_BACK in nav.update(fx, 270.0, now=100.0)
    fx2 = Fix(lat=0.0, lon=0.000088, valid=True, t_mono=time.monotonic())
    assert nav.update(fx2, 90.0, now=101.0) == [NavEvent.WAYPOINT_REACHED]
    fx3 = Fix(lat=0.0, lon=0.000179, valid=True, t_mono=time.monotonic())
    assert nav.update(fx3, 90.0, now=102.0) == [NavEvent.ROUTE_COMPLETE]
    print("PASS  turn-back, waypoint arrival, route completion")

    # Fail-safe: stale fix → single NAV_LOST; fresh fix → NAV_RECOVERED.
    nav = GpsNavigator([Waypoint(0.0, 1.0)])
    old = Fix(lat=0, lon=0, valid=True, t_mono=time.monotonic() - 30.0)
    assert nav.update(old, 90.0) == [NavEvent.NAV_LOST]
    assert nav.update(old, 90.0) == []
    fresh = Fix(lat=0, lon=0, valid=True, t_mono=time.monotonic())
    assert NavEvent.NAV_RECOVERED in nav.update(fresh, 90.0)
    print("PASS  GPS-loss fail-safe: NAV_LOST once, NAV_RECOVERED on return")

    print("\nAll gps_navigator self-tests passed.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--monitor", action="store_true",
                    help="print live fixes from the serial port")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.monitor:
        r = GpsReader().start()
        try:
            while True:
                f = r.fix()
                print(f"valid={f.valid} sats={f.n_sats} "
                      f"({f.lat:.6f},{f.lon:.6f}) age={f.age_s():.1f}s")
                time.sleep(1.0)
        except KeyboardInterrupt:
            r.stop()
    else:
        print(__doc__)
