#!/usr/bin/env python3
"""
haptic_output.py — finger-allocated ERM motor control
=====================================================

Implements the output column of Figure 3.1: five flat coin ERM motors,
one per finger, driven by NPN transistors (S8050/2N2222) from Raspberry Pi
GPIO. Spatial mapping per §3.3.2 — the finger IS the message:

    finger 1  THUMB    GPIO 5    move forward / waypoint confirmation
    finger 2  INDEX    GPIO 26   turn right            (heading error > 0)
    finger 3  MIDDLE   GPIO 19   turn left             (heading error < 0)
    finger 4  RING     GPIO 6    turn back / reroute   (|error| ≥ 135°)
    finger 5  PINKY    GPIO 13   stop — obstacle ahead (vision / fail-safe)

Cues are event-driven 200 ms pulses (§3.4.4), with a per-finger refractory
interval so repeated triggers cannot blur into continuous vibration and
provoke sensory adaptation. The all-finger alert pattern (3 × 150 ms) is
the GPS-loss fail-safe signature — deliberately unlike any single cue.

Electrical notes (record in Chapter 4)
--------------------------------------
* Each GPIO drives a transistor BASE through ~1 kΩ; the motor sits on the
  collector with a flyback diode (1N4007 or similar) across its terminals.
  GPIO pins NEVER source motor current directly.
* Pins 5/6/13/19/26 are plain GPIOs with no boot-time alternate function
  (clear of I2C 2/3, UART 14/15, SPI) — chosen so a reboot can't twitch
  the motors. [*] Adjust PIN_MAP to match the actual wiring loom.
* Raspberry Pi 5: the legacy RPi.GPIO library does NOT work (new RP1 I/O
  controller). This module uses gpiozero, which drives the Pi 5 through
  lgpio — `pip install gpiozero lgpio`. Off the Pi it falls back to a
  console backend automatically, so every other module develops unplugged.

Interface for main.py
---------------------
    hap = HapticDriver()                       # auto backend
    hap.add_pulse_listener(imu.filter.notify_pulse)   # B1 blanking hook
    hap.pulse(Finger.INDEX)                    # one 200 ms cue
    hap.alert()                                # GPS-loss pattern
    hap.double_pulse(Finger.THUMB)             # route-complete signature

`python haptic_output.py --selftest` — deterministic, injected fake clock
`python haptic_output.py --demo`     — fires each finger in order (hardware
                                       bring-up test for the wiring loom)

Constants tagged [*] are pre-build assumptions — tune on the prototype.
"""

from __future__ import annotations

import atexit
import threading
import time
from enum import IntEnum
from typing import Callable, Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


class Finger(IntEnum):
    THUMB = 1
    INDEX = 2
    MIDDLE = 3
    RING = 4
    PINKY = 5


PIN_MAP: Dict[Finger, int] = {        # BCM numbering, as-built loom
    Finger.THUMB: 5,                   # (design order was 5/6/13/19/26;
    Finger.INDEX: 26,                  #  corrected to match the soldered
    Finger.MIDDLE: 19,                 #  harness at assembly, verified
    Finger.RING: 6,                    #  cue by cue on the finished cane)
    Finger.PINKY: 13,
}

PULSE_S = 0.20                # §3.4.4 event pulse width
REFRACTORY_S = 0.50           # [*] min gap between pulses on one finger
DOUBLE_GAP_S = 0.35           # [*] gap inside the route-complete double pulse
ALERT_ON_S = 0.15             # all-finger alert pattern: 3 × (on/off)
ALERT_OFF_S = 0.15
ALERT_CYCLES = 3
ALERT_REFRACTORY_S = 8.0      # [*] min gap between alert patterns


# ─────────────────────────────────────────────────────────────────────────────
# Backends
# ─────────────────────────────────────────────────────────────────────────────

class ConsoleBackend:
    """Development / simulation backend: prints instead of vibrating."""

    name = "console"

    def pulse(self, finger: Finger, duration_s: float) -> None:
        print(f"        [HAPTIC] finger {int(finger)} {finger.name:<6} "
              f"pulse {duration_s*1000:.0f} ms")

    def alert(self) -> None:
        print(f"        [HAPTIC] ALL-FINGER ALERT "
              f"({ALERT_CYCLES} × {ALERT_ON_S*1000:.0f} ms) — GPS lost")

    def all_off(self) -> None:
        pass


class GpioZeroBackend:
    """Real motors via gpiozero (lgpio under the hood on Pi 5)."""

    name = "gpiozero"

    def __init__(self):
        from gpiozero import DigitalOutputDevice   # pip install gpiozero lgpio
        self._dev = {f: DigitalOutputDevice(pin, initial_value=False)
                     for f, pin in PIN_MAP.items()}

    def pulse(self, finger: Finger, duration_s: float) -> None:
        d = self._dev[finger]
        d.on()
        threading.Timer(duration_s, d.off).start()

    def alert(self) -> None:
        def _pattern():
            for _ in range(ALERT_CYCLES):
                for d in self._dev.values():
                    d.on()
                time.sleep(ALERT_ON_S)
                for d in self._dev.values():
                    d.off()
                time.sleep(ALERT_OFF_S)
        threading.Thread(target=_pattern, daemon=True).start()

    def all_off(self) -> None:
        for d in self._dev.values():
            d.off()


class RecordingBackend:
    """Selftest backend: captures every command with the driver's clock."""

    name = "recording"

    def __init__(self, clock: Callable[[], float]):
        self.clock = clock
        self.pulses: List[Tuple[float, Finger, float]] = []
        self.alerts: List[float] = []

    def pulse(self, finger: Finger, duration_s: float) -> None:
        self.pulses.append((self.clock(), finger, duration_s))

    def alert(self) -> None:
        self.alerts.append(self.clock())

    def all_off(self) -> None:
        pass


def auto_backend():
    try:
        b = GpioZeroBackend()
        print("[haptic] gpiozero backend — driving real GPIO")
        return b
    except Exception:
        print("[haptic] gpiozero unavailable — console backend "
              "(normal on a development laptop)")
        return ConsoleBackend()


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

class HapticDriver:
    """Owns cue policy: refractory intervals, patterns, pulse listeners.
    The clock is injectable so the simulator can run at logical time and
    the selftest stays deterministic."""

    def __init__(self, backend=None, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.backend = backend if backend is not None else auto_backend()
        self._last_pulse: Dict[Finger, float] = {f: -1e9 for f in Finger}
        self._last_alert = -1e9
        self._listeners: List[Callable[[float], None]] = []
        self._lock = threading.Lock()
        atexit.register(self.backend.all_off)

    # B1 wiring — imu_filter.notify_pulse goes here
    def add_pulse_listener(self, fn: Callable[[float], None]) -> None:
        self._listeners.append(fn)

    def pulse(self, finger: Finger, duration_s: float = PULSE_S) -> bool:
        """Fire one cue. Returns False when the per-finger refractory
        suppressed it (event absorbed, not queued — §3.4.4 one-event-one-
        pulse policy)."""
        with self._lock:
            now = self.clock()
            if now - self._last_pulse[finger] < REFRACTORY_S:
                return False
            self._last_pulse[finger] = now
        self.backend.pulse(finger, duration_s)
        for fn in self._listeners:
            fn(duration_s)
        return True

    def double_pulse(self, finger: Finger) -> bool:
        """Route-complete signature: two pulses, distinct from single cues."""
        if not self.pulse(finger):
            return False
        if isinstance(self.backend, (ConsoleBackend, RecordingBackend)):
            self._last_pulse[finger] = -1e9          # logical time: no sleep
            self.pulse(finger)
        else:
            def _second():
                self._last_pulse[finger] = -1e9
                self.pulse(finger)
            threading.Timer(PULSE_S + DOUBLE_GAP_S, _second).start()
        return True

    def alert(self) -> bool:
        """All-finger GPS-loss pattern, rate-limited by its own refractory."""
        with self._lock:
            now = self.clock()
            if now - self._last_alert < ALERT_REFRACTORY_S:
                return False
            self._last_alert = now
        self.backend.alert()
        for fn in self._listeners:
            fn(ALERT_CYCLES * (ALERT_ON_S + ALERT_OFF_S))
        return True

    def stop_all(self) -> None:
        self.backend.all_off()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test — fake clock, instant and deterministic
# ─────────────────────────────────────────────────────────────────────────────

def selftest() -> None:
    t = {"now": 0.0}
    clock = lambda: t["now"]
    rec = RecordingBackend(clock)
    hap = HapticDriver(backend=rec, clock=clock)

    blank_calls: List[float] = []
    hap.add_pulse_listener(blank_calls.append)

    # Pin map sanity: five distinct plain-GPIO pins.
    pins = list(PIN_MAP.values())
    assert len(set(pins)) == 5 and not set(pins) & {2, 3, 14, 15}, pins
    print(f"PASS  pin map: 5 distinct GPIOs {pins}, clear of I2C/UART")

    # Refractory: first cue fires, an immediate repeat is absorbed,
    # a different finger is independent, and 0.6 s later it re-arms.
    assert hap.pulse(Finger.INDEX) is True
    assert hap.pulse(Finger.INDEX) is False
    assert hap.pulse(Finger.MIDDLE) is True
    t["now"] = 0.6
    assert hap.pulse(Finger.INDEX) is True
    fired = [f for (_, f, _) in rec.pulses]
    assert fired == [Finger.INDEX, Finger.MIDDLE, Finger.INDEX], fired
    print("PASS  refractory: per-finger 0.5 s, fingers independent")

    # Pulse listener (IMU blanking hook) saw every fired cue.
    assert len(blank_calls) == 3 and all(d == PULSE_S for d in blank_calls)
    print("PASS  pulse listeners notified with pulse duration (B1 hook)")

    # Alert pattern fires once, is suppressed inside its refractory,
    # and re-arms after 8 s.
    t["now"] = 10.0
    assert hap.alert() is True
    assert hap.alert() is False
    t["now"] = 19.0
    assert hap.alert() is True
    assert len(rec.alerts) == 2
    print("PASS  alert pattern: own 8 s refractory, one event per trigger")

    # Route-complete double pulse records exactly two cues.
    t["now"] = 30.0
    n0 = len(rec.pulses)
    assert hap.double_pulse(Finger.THUMB) is True
    assert len(rec.pulses) - n0 == 2
    print("PASS  route-complete double pulse on the thumb")

    print("\nAll haptic_output self-tests passed.")


def demo() -> None:
    """Hardware bring-up: fires each finger in order, then the alert."""
    hap = HapticDriver()
    for f in Finger:
        print(f"finger {int(f)} — {f.name}")
        hap.pulse(f)
        time.sleep(1.0)
    print("all-finger alert pattern")
    hap.alert()
    time.sleep(1.5)
    hap.stop_all()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.demo:
        demo()
    else:
        print(__doc__)
