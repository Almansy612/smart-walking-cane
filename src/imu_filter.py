#!/usr/bin/env python3
"""
imu_filter.py — heading estimation for the Smart Walking Cane
=============================================================

Implements the sensor-fusion column of Figure 3.1: hard-iron magnetometer
calibration (§3.4.5), the complementary filter θ = α(θ_prev + ω·dt) +
(1−α)·θ_mag with α = 0.98 (Eq. 16), and two additions agreed for Phase 2:

B1  Motor-pulse blanking. The five ERM motors contain rotating magnets that
    energise precisely when a cue fires; hard-iron calibration removes only
    FIXED offsets. notify_pulse() therefore freezes magnetometer corrections
    (gyro-only integration) for the pulse duration plus a spin-down tail,
    so cues cannot corrupt the heading that triggered them.

B2  Sweep-yaw separation. The cane sweeps ±30° while the body walks a
    steady line. A 2 s wrap-aware low-pass of the fused heading gives the
    walking direction (heading_slow — what gps_navigator consumes), and
    heading − heading_slow gives the instantaneous sweep yaw (what
    yolo_detector's forward-path correction consumes, Eq. 27).

Fail-safe (§3.4.4): a rolling std-dev of magnetometer innovations sets
`unstable` (hysteresis 15° set / 8° clear); main.py suspends left/right
cues while it holds, leaving forward and stop cues live.

Hardware notes (MPU-9250 @ I2C 0x68, AK8963 magnetometer @ 0x0C)
----------------------------------------------------------------
* Enable I2C bypass (INT_PIN_CFG bit1) to reach the AK8963 directly.
* AK8963 data is LITTLE-endian (MPU registers are big-endian) and ST2 must
  be read to release the data latch — both are classic silent-failure traps.
* AK8963 axes are rotated relative to the gyro/accel frame:
  X_mag→Y_body, Y_mag→X_body, Z_mag→−Z_body — remapped in read().
* Final compass sign/offset depends on how the board sits in the shaft:
  point the cane at a known bearing during bench-up and set
  MOUNT_OFFSET_DEG / AXIS_SIGN accordingly. Magnetic declination in
  Peninsular Malaysia is ≈ 0°, so MAG_DECLINATION_DEG defaults to 0.

Interface for main.py
---------------------
    svc = ImuService(cal=MagCalibration.load()).start()
    haptic.add_pulse_listener(svc.filter.notify_pulse)     # B1 wiring
    s = svc.snapshot()    # .heading  .heading_slow  .sweep_yaw  .unstable

`python imu_filter.py --selftest`   — deterministic, no hardware
`python imu_filter.py --calibrate`  — 30 s figure-8 hard-iron routine
`python imu_filter.py --monitor`    — live fused heading printout

Constants tagged [*] are pre-build assumptions — tune on the prototype.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from statistics import pstdev
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ALPHA = 0.98                 # Eq. 16 — gyro weight
IMU_RATE_HZ = 100            # §3.4.7 sampling rate
BLANK_TAIL_S = 0.10          # [*] ERM spin-down after the 200 ms pulse
# Innovation std-dev thresholds. Raised from the pre-walk 15/8 after the walk
# test (Section 5.6.3): environmental magnetic jitter during ordinary smooth
# walking pushed the innovation std-dev to ~23 deg, tripping the old 15 deg
# gate and suppressing turn cues even though the heading itself was smooth
# (18-40 deg/s). Measured regimes: smooth-walk-with-jitter ~23 deg; genuine
# erratic instability ~50 deg; a fast but CLEAN turn ~1.5 deg (low, because the
# gyro prediction and magnetometer agree). The 40/25 gate sits cleanly above
# walk jitter and below real instability, so it flags genuine heading chaos
# without suppressing cues during normal walking.
UNSTABLE_SET_DEG = 40.0      # innovation std-dev to flag instability
UNSTABLE_CLEAR_DEG = 25.0
INNOV_WINDOW = 100           # 1 s of innovations at 100 Hz
SLOW_TAU_S = 2.0             # [*] body-direction low-pass time constant
MOUNT_OFFSET_DEG = 0.0       # [*] set at bench-up against a known bearing
MAG_DECLINATION_DEG = 0.0    # ≈ 0° in Peninsular Malaysia
AXIS_SIGN = (1, 1, 1)        # [*] flip during bench-up if heading runs mirror

CAL_FILE = "imu_cal.json"

I2C_BUS = 1
MPU_ADDR = 0x68              # MPU-6050/6500/9250 gyro + accel
GYRO_LSB_PER_DPS = 131.0     # ±250 dps full scale
ACCEL_LSB_PER_G = 16384.0    # ±2 g full scale

# External magnetometer (separate chip on the same I2C bus). The board may be a
# true HMC5883L (Honeywell, addr 0x1E) or the QMC5883L clone (addr 0x0D); they
# are different silicon with different registers, so the driver auto-detects.
HMC_ADDR = 0x1E
QMC_ADDR = 0x0D
HMC_UT_PER_LSB = 100.0 / 1090.0     # 1090 LSB/Gauss (±1.3 G) -> uT
QMC_UT_PER_LSB = 100.0 / 12000.0    # 12000 LSB/Gauss (±2 G)  -> uT


def wrap180(deg: float) -> float:
    return ((deg + 180.0) % 360.0) - 180.0


# ─────────────────────────────────────────────────────────────────────────────
# Hard-iron calibration (§3.4.5: offset_i = (max_i + min_i) / 2)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MagCalibration:
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)   # soft-iron, optional

    def apply(self, m: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return tuple((m[i] - self.offset[i]) * self.scale[i] for i in range(3))

    def save(self, path: str = CAL_FILE) -> None:
        with open(path, "w") as f:
            json.dump({"offset": self.offset, "scale": self.scale}, f, indent=2)

    @classmethod
    def load(cls, path: str = CAL_FILE) -> "MagCalibration":
        try:
            with open(path) as f:
                d = json.load(f)
            return cls(tuple(d["offset"]), tuple(d["scale"]))
        except (OSError, KeyError, ValueError):
            print(f"[imu] no calibration at {path} — run --calibrate "
                  "before trusting absolute headings")
            return cls()


def compute_hard_iron(samples) -> MagCalibration:
    """samples: iterable of raw (mx, my, mz) gathered while sweeping the cane
    through a slow figure-8. Offset is the per-axis range midpoint; the
    per-axis scale equalises ranges (a cheap soft-iron first correction)."""
    xs, ys, zs = zip(*samples)
    mins = (min(xs), min(ys), min(zs))
    maxs = (max(xs), max(ys), max(zs))
    offset = tuple((maxs[i] + mins[i]) / 2.0 for i in range(3))
    ranges = [max(1e-6, maxs[i] - mins[i]) for i in range(3)]
    avg = sum(ranges) / 3.0
    scale = tuple(avg / r for r in ranges)
    return MagCalibration(offset, scale)


# ─────────────────────────────────────────────────────────────────────────────
# Tilt-compensated compass heading
# ─────────────────────────────────────────────────────────────────────────────

def tilt_compensated_heading(accel, mag) -> float:
    """Standard roll/pitch de-rotation of the magnetic vector, returning a
    0–360° heading after mount offset and declination. The cane is never
    held level, so plain atan2(my, mx) is not acceptable here. Verify the
    sign conventions against the physical mount during bench-up."""
    ax, ay, az = accel
    mx, my, mz = (AXIS_SIGN[i] * mag[i] for i in range(3))
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.hypot(ay, az))
    mxc = (mx * math.cos(pitch)
           + my * math.sin(roll) * math.sin(pitch)
           + mz * math.cos(roll) * math.sin(pitch))
    myc = my * math.cos(roll) - mz * math.sin(roll)
    hdg = math.degrees(math.atan2(-myc, mxc))
    return (hdg + MOUNT_OFFSET_DEG + MAG_DECLINATION_DEG) % 360.0


# ─────────────────────────────────────────────────────────────────────────────
# Complementary filter — Eq. 16, with wrap-aware blending
# ─────────────────────────────────────────────────────────────────────────────

class HeadingFilter:
    """Fuses gyro-Z rate with magnetometer heading. All blending happens on
    the shortest arc so the 0/360 seam never produces a 360° transient —
    the naive θ = αθ_g + (1−α)θ_m form fails there."""

    def __init__(self, alpha: float = ALPHA, initial_deg: float = 0.0):
        self.alpha = alpha
        self.heading = wrap180(initial_deg)
        self.heading_slow = self.heading
        self.sweep_yaw = 0.0
        self.unstable = False
        self._innov = deque(maxlen=INNOV_WINDOW)
        self._blank_remaining = 0.0

    # B1 — called by haptic_output whenever any motor fires
    def notify_pulse(self, duration_s: float = 0.2) -> None:
        self._blank_remaining = max(self._blank_remaining,
                                    duration_s + BLANK_TAIL_S)

    @property
    def blanked(self) -> bool:
        return self._blank_remaining > 0.0

    def update(self, gyro_z_dps: float, mag_heading_deg: Optional[float],
               dt: float) -> float:
        pred = wrap180(self.heading + gyro_z_dps * dt)

        if mag_heading_deg is not None and not self.blanked:
            innov = wrap180(mag_heading_deg - pred)         # shortest arc
            self._innov.append(innov)
            pred = wrap180(pred + (1.0 - self.alpha) * innov)
            if len(self._innov) >= 30:
                s = pstdev(self._innov)
                # Threshold set from measured walk data (Section 5.6.3): smooth
                # walking with magnetic jitter sits near 23 deg, genuine
                # instability near 50 deg, so 40/25 separates them cleanly.
                if s > UNSTABLE_SET_DEG:
                    self.unstable = True
                elif s < UNSTABLE_CLEAR_DEG:
                    self.unstable = False

        self._blank_remaining = max(0.0, self._blank_remaining - dt)
        self.heading = pred

        # B2 — body direction vs sweep yaw
        k = min(1.0, dt / SLOW_TAU_S)
        self.heading_slow = wrap180(
            self.heading_slow + k * wrap180(self.heading - self.heading_slow))
        self.sweep_yaw = wrap180(self.heading - self.heading_slow)
        return self.heading


# ─────────────────────────────────────────────────────────────────────────────
# MPU-9250 hardware driver (lazy smbus2 import — selftest needs none of it)
# ─────────────────────────────────────────────────────────────────────────────

WHO_AM_I_REG = 0x75
WHO_AM_I_EXPECTED = {0x71: "MPU-9250", 0x73: "MPU-9255"}
WHO_AM_I_KNOWN_BAD = {0x70: "MPU-6500 (no magnetometer)",
                      0x68: "MPU-6050 (no magnetometer)",
                      0x75: "MPU-9250 clone/relabelled"}


class _Magnetometer:
    """Auto-detecting driver for an external HMC5883L or QMC5883L on the I2C
    bus. Both return a 3-axis field; read() yields microtesla in the sensor's
    own frame (axis remap to the body frame happens in Imu.read())."""

    def __init__(self, bus):
        self._bus = bus
        self.kind = None
        # Probe QMC first (0x0D): its chip-ID reg 0x0D reads 0xFF.
        try:
            if bus.read_byte_data(QMC_ADDR, 0x0D) == 0xFF:
                self.kind, self.addr = "QMC5883L", QMC_ADDR
        except OSError:
            pass
        # Then HMC (0x1E): ID regs A/B/C = 'H','4','3' (0x48,0x34,0x33).
        if self.kind is None:
            try:
                ida = bus.read_i2c_block_data(HMC_ADDR, 0x0A, 3)
                if tuple(ida) == (0x48, 0x34, 0x33):
                    self.kind, self.addr = "HMC5883L", HMC_ADDR
            except OSError:
                pass
        if self.kind is None:
            raise RuntimeError(
                "no magnetometer found at 0x1E (HMC5883L) or 0x0D (QMC5883L). "
                "Check wiring/solder and that i2cdetect -y 1 shows one of them.")

        if self.kind == "HMC5883L":
            bus.write_byte_data(HMC_ADDR, 0x00, 0x78)   # CRA: 8-avg, 75 Hz
            bus.write_byte_data(HMC_ADDR, 0x01, 0x20)   # CRB: gain ±1.3 G
            bus.write_byte_data(HMC_ADDR, 0x02, 0x00)   # continuous measure
            self._scale = HMC_UT_PER_LSB
        else:                                           # QMC5883L
            bus.write_byte_data(QMC_ADDR, 0x0B, 0x01)   # SET/RESET period
            bus.write_byte_data(QMC_ADDR, 0x09, 0x1D)   # continuous,200Hz,±2G,512x
            self._scale = QMC_UT_PER_LSB
        time.sleep(0.01)
        print(f"[imu] magnetometer: {self.kind} at 0x{self.addr:02X}")

    @staticmethod
    def _s16(hi, lo):
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def read(self):
        """-> (mx, my, mz) in microtesla, sensor frame, or None on failure."""
        try:
            if self.kind == "HMC5883L":
                d = self._bus.read_i2c_block_data(HMC_ADDR, 0x03, 6)  # X,Z,Y  BE
                x = self._s16(d[0], d[1]) * self._scale
                z = self._s16(d[2], d[3]) * self._scale
                y = self._s16(d[4], d[5]) * self._scale
                return (x, y, z)
            else:                                                     # QMC LE, X,Y,Z
                d = self._bus.read_i2c_block_data(QMC_ADDR, 0x00, 6)
                x = self._s16(d[1], d[0]) * self._scale
                y = self._s16(d[3], d[2]) * self._scale
                z = self._s16(d[5], d[4]) * self._scale
                return (x, y, z)
        except OSError:
            return None


class Imu:
    """MPU-6050/6500 gyro+accel plus a separate HMC/QMC magnetometer.

    Replaces the former single-package Mpu9250 driver: the gyro/accel come from
    the MPU at 0x68 and the heading reference from an external magnetometer, so
    a magnetometer-less MPU (the 6500 sold as a 9250) no longer blocks heading.
    """

    def __init__(self, bus_id: int = I2C_BUS):
        from smbus2 import SMBus           # pip install smbus2
        self._bus = SMBus(bus_id)
        b = self._bus

        # MPU identity: accept anything in the 6050/6500/9250 family for
        # gyro+accel (the magnetometer is now external, so 0x70 is fine here).
        try:
            who = b.read_byte_data(MPU_ADDR, 0x75)
        except OSError as exc:
            raise RuntimeError(
                f"no I2C response at 0x{MPU_ADDR:02X} ({exc}). Check wiring and "
                f"that the MPU appears in i2cdetect -y 1") from exc
        names = {0x68: "MPU-6050", 0x70: "MPU-6500", 0x71: "MPU-9250",
                 0x73: "MPU-9255", 0x75: "MPU-9250(clone)"}
        print(f"[imu] MPU: {names.get(who, 'unknown 0x%02X' % who)} "
              f"(WHO_AM_I=0x{who:02X}) - using external magnetometer for heading")

        b.write_byte_data(MPU_ADDR, 0x6B, 0x01)   # PWR_MGMT_1: PLL clock
        time.sleep(0.05)
        b.write_byte_data(MPU_ADDR, 0x1B, 0x00)   # gyro ±250 dps
        b.write_byte_data(MPU_ADDR, 0x1C, 0x00)   # accel ±2 g
        b.write_byte_data(MPU_ADDR, 0x37, 0x02)   # INT_PIN_CFG: bypass on
        b.write_byte_data(MPU_ADDR, 0x6A, 0x00)   # USER_CTRL: I2C master off
        time.sleep(0.01)

        self._mag = _Magnetometer(b)              # auto-detects HMC vs QMC

    @staticmethod
    def _s16(hi: int, lo: int) -> int:
        v = (hi << 8) | lo
        return v - 65536 if v & 0x8000 else v

    def read(self):
        """-> (gyro_z_dps, accel_g(3), mag_uT(3) in BODY frame, ok)."""
        try:
            b = self._bus
            a = b.read_i2c_block_data(MPU_ADDR, 0x3B, 6)        # accel BE
            ax = self._s16(a[0], a[1]) / ACCEL_LSB_PER_G
            ay = self._s16(a[2], a[3]) / ACCEL_LSB_PER_G
            az = self._s16(a[4], a[5]) / ACCEL_LSB_PER_G
            g = b.read_i2c_block_data(MPU_ADDR, 0x47, 2)        # gyro Z BE
            gz = self._s16(g[0], g[1]) / GYRO_LSB_PER_DPS
            m = self._mag.read()
            if m is None:
                return gz, (ax, ay, az), None, True
            # Axis remap: mount the magnetometer with its X forward-aligned to
            # the MPU X. VERIFY against cardinals at bench-up (--monitor); if
            # heading runs mirrored or 90-off, fix AXIS_SIGN / MOUNT_OFFSET_DEG.
            mag_body = (m[0], m[1], m[2])
            return gz, (ax, ay, az), mag_body, True
        except OSError:
            return 0.0, (0.0, 0.0, 1.0), None, False


# Back-compat alias so existing imports (ImuService uses Mpu9250) keep working.
Mpu9250 = Imu


def run_calibration(duration_s: float = 30.0) -> MagCalibration:
    imu = Mpu9250()
    print(f"Sweep the cane in slow figure-8s for {duration_s:.0f} s …")
    samples, t0 = [], time.monotonic()
    while time.monotonic() - t0 < duration_s:
        _, _, mag, ok = imu.read()
        if ok and mag is not None:
            samples.append(mag)
        time.sleep(0.01)
    if len(samples) < 200:
        raise SystemExit(
            f"only {len(samples)} magnetometer samples in {duration_s:.0f}s - "
            f"expected ~{int(duration_s*100)}. The sensor is not streaming; "
            f"calibration NOT saved.")
    cal = compute_hard_iron(samples)
    rng = [max(s[i] for s in samples) - min(s[i] for s in samples)
           for i in range(3)]
    if min(rng) < 10.0:                       # uT: a real figure-8 sweeps far more
        print(f"[imu] WARNING: axis ranges {tuple(round(r,1) for r in rng)} uT "
              f"are small - did the cane actually rotate through all axes? "
              f"Re-run --calibrate with fuller figure-8s.")
    cal.save()
    print(f"saved {CAL_FILE}: offset={tuple(round(o,1) for o in cal.offset)} "
          f"scale={tuple(round(s,3) for s in cal.scale)} "
          f"({len(samples)} samples)")
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# 100 Hz service thread for main.py
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ImuState:
    heading: float = 0.0
    heading_slow: float = 0.0
    sweep_yaw: float = 0.0
    unstable: bool = False
    ok: bool = False
    t_mono: float = 0.0

    def age_s(self) -> float:
        return time.monotonic() - self.t_mono if self.t_mono else float("inf")


class ImuService:
    def __init__(self, cal: Optional[MagCalibration] = None,
                 rate_hz: int = IMU_RATE_HZ):
        self.cal = cal or MagCalibration.load()
        self.filter = HeadingFilter()
        self._dt = 1.0 / rate_hz
        self._state = ImuState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "ImuService":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def snapshot(self) -> ImuState:
        with self._lock:
            return ImuState(**vars(self._state))

    def _run(self) -> None:
        try:
            imu = Mpu9250()
        except Exception as exc:                       # noqa: BLE001
            print(f"\n[imu] SENSOR INIT FAILED: {exc}")
            print("[imu] holding ok=False so main.py suppresses turn cues "
                  "(forward/stop stay live)\n")
            while not self._stop.is_set():             # keep publishing ok=False
                with self._lock:
                    self._state = ImuState(ok=False, t_mono=time.monotonic())
                time.sleep(0.2)
            return
        t_prev = time.monotonic()
        while not self._stop.is_set():
            gz, accel, mag, ok = imu.read()
            now = time.monotonic()
            dt = min(0.05, now - t_prev)        # clamp scheduler hiccups
            t_prev = now
            mag_hdg = None
            if ok and mag is not None:
                mag_hdg = tilt_compensated_heading(accel, self.cal.apply(mag))
            self.filter.update(gz, mag_hdg, dt)
            with self._lock:
                f = self.filter
                self._state = ImuState(f.heading, f.heading_slow, f.sweep_yaw,
                                       f.unstable, ok, now)
            time.sleep(max(0.0, self._dt - (time.monotonic() - now)))


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — deterministic, no hardware, no third-party imports
# ─────────────────────────────────────────────────────────────────────────────

def selftest() -> None:
    import random
    random.seed(7)
    DT = 1.0 / IMU_RATE_HZ

    # A — wrap-aware blending at the ±180 seam: innovation must be +2°,
    # never −358°.
    f = HeadingFilter(initial_deg=179.0)
    h = f.update(0.0, -179.0, DT)
    assert 179.0 < h < 179.1, h
    print(f"PASS  seam blend: 179° toward −179° moves +{h-179:.3f}° (not −358°)")

    # B — tracking a full 360° rotation with 0.5 dps gyro bias: the
    # complementary filter must beat gyro-only integration decisively.
    true, fc, fg = 0.0, HeadingFilter(initial_deg=0), HeadingFilter(initial_deg=0)
    for _ in range(2000):                                   # 20 s
        true = wrap180(true + 18.0 * DT)
        gz = 18.0 + 0.5 + random.gauss(0, 0.2)
        mag = wrap180(true + random.gauss(0, 3.0))
        fc.update(gz, mag, DT)
        fg.update(gz, None, DT)
    e_c = abs(wrap180(fc.heading - true))
    e_g = abs(wrap180(fg.heading - true))
    assert e_c < 3.0 and e_g > 6.0, (e_c, e_g)
    print(f"PASS  complementary error {e_c:.2f}° vs gyro-only drift {e_g:.2f}°")

    # C — motor-pulse blanking: a +60° magnetic lie during the blank window
    # must not move the heading at all; it may pull only after the window.
    f = HeadingFilter(initial_deg=90.0)
    for _ in range(200):
        f.update(0.0, 90.0, DT)                             # settle
    f.notify_pulse(0.2)                                     # 0.2 + 0.1 tail
    for _ in range(30):                                     # 0.30 s blanked
        f.update(0.0, 150.0, DT)
    assert abs(f.heading - 90.0) < 1e-9, f.heading
    f.update(0.0, 150.0, DT)                                # blank expired
    assert f.heading > 90.5, f.heading
    print("PASS  blanking freezes mag influence for pulse + spin-down tail")

    # D — instability fail-safe, thresholds set from the walk test (5.6.3).
    # Genuine instability = erratic heading where gyro and magnetometer disagree
    # widely (innovation std ~50 deg) -> must flag. Smooth walking with magnetic
    # jitter (innovation std ~23 deg) -> must NOT flag (the walk-test fix).
    f = HeadingFilter(initial_deg=0.0)
    for _ in range(200):                       # erratic: fast random gyro, noisy mag
        f.update(random.uniform(-150, 150), random.uniform(-80, 80), DT)
    assert f.unstable, "erratic heading (innov~50deg) should flag unstable"
    for _ in range(300):
        f.update(0.0, f.heading, DT)           # clean, still
    assert not f.unstable, "clean heading should clear the flag"
    # regression guard: smooth walk + magnetic jitter must stay stable
    f2 = HeadingFilter(initial_deg=0.0)
    for _ in range(200):                       # slow gyro, jittery mag around heading
        f2.update(random.uniform(-20, 20), f2.heading + random.uniform(-40, 40), DT)
    assert not f2.unstable, ("smooth walking with magnetic jitter must NOT flag "
                             "unstable - this is the walk-test fix (Section 5.6.3)")
    print("PASS  instability gate: flags erratic heading, ignores walk jitter")

    # E — sweep separation: a ±30° @ 0.9 Hz cane sweep about a 90° walk
    # direction must land heading_slow near 90° and sweep_yaw near ±30°.
    f = HeadingFilter(initial_deg=90.0)
    t, peak = 0.0, 0.0
    slow_tail = []
    for _ in range(3000):                                   # 30 s
        t += DT
        cane = 90.0 + 30.0 * math.sin(2 * math.pi * 0.9 * t)
        rate = 30.0 * 2 * math.pi * 0.9 * math.cos(2 * math.pi * 0.9 * t)
        f.update(rate, cane, DT)
        if t > 20.0:
            slow_tail.append(f.heading_slow)
            peak = max(peak, abs(f.sweep_yaw))
    mean_slow = sum(slow_tail) / len(slow_tail)
    assert abs(mean_slow - 90.0) < 4.0, mean_slow
    assert 24.0 < peak < 36.0, peak
    print(f"PASS  sweep separation: body {mean_slow:.1f}° (≈90), "
          f"sweep peak {peak:.1f}° (≈30)")

    # F — hard-iron recovery from a synthetic offset sphere.
    random.seed(8)
    off = (30.0, -20.0, 10.0)
    pts = []
    for _ in range(800):
        th, ph = random.uniform(0, 2 * math.pi), random.uniform(0, math.pi)
        r = 45.0
        pts.append((off[0] + r * math.sin(ph) * math.cos(th),
                    off[1] + r * math.sin(ph) * math.sin(th),
                    off[2] + r * math.cos(ph)))
    cal = compute_hard_iron(pts)
    assert all(abs(cal.offset[i] - off[i]) < 1.0 for i in range(3)), cal.offset
    print(f"PASS  hard-iron offsets recovered: "
          f"{tuple(round(o,1) for o in cal.offset)}")

    print("\nAll imu_filter self-tests passed.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--monitor", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.calibrate:
        run_calibration()
    elif args.monitor:
        svc = ImuService().start()
        try:
            while True:
                s = svc.snapshot()
                print(f"hdg {s.heading:7.1f}°  body {s.heading_slow:7.1f}°  "
                      f"sweep {s.sweep_yaw:6.1f}°  "
                      f"{'UNSTABLE' if s.unstable else 'stable'}")
                time.sleep(0.2)
        except KeyboardInterrupt:
            svc.stop()
    else:
        print(__doc__)
