# Core logic (abbreviated)

These listings are the project-specific decision logic, abbreviated from Appendix F of the final report. Library calls (Picamera2, Roboflow inference server, pyserial, smbus2, gpiozero) are shown as single lines.

## main.py — orchestrator and control loop

```python
CONTROL_RATE_HZ = 10; ARRIVAL_RADIUS = 5.0; STOP_TRIGGER = 3.0

init camera (Picamera2), gps (pyserial), imu (smbus2), haptics (gpiozero)
route = load_waypoints(route_file)

while not route.complete:
    fix, heading, frame = read_sensors()
    if obstacle_detector.active(frame, cane_yaw=imu.sweep_yaw):
        haptics.fire(PINKY); continue           # stop overrides nav
    events = navigator.update(fix, heading)
    for e in events: haptics.fire(finger_for(e))
    sleep_to_maintain_rate()
```

## gps_navigator.py — macro-navigation

```python
EARTH_R = 6_371_000; TRIGGER = 20; RELEASE = 10; TURNBACK = 135

def haversine(lat1, lon1, lat2, lon2): ...   # great-circle distance (m)
def bearing(lat1, lon1, lat2, lon2): ...     # forward azimuth 0..360
def heading_error(req, cur): return wrap180(req - cur)

def update(fix, heading):
    if stale(fix): return [NAV_LOST]
    d = haversine(fix, target)
    if d < ARRIVAL_RADIUS: advance(); return [WAYPOINT_REACHED]
    err = heading_error(bearing(fix, target), heading)
    if abs(err) >= TURNBACK: return [TURN_BACK]
    apply hysteresis(TRIGGER, RELEASE) -> [TURN_LEFT] or [TURN_RIGHT]
```

## imu_filter.py — heading estimation

```python
ALPHA = 0.98   # gyro weight

def update(gyro_z, mag_heading, dt):
    predicted = wrap180(heading + gyro_z * dt)
    if mag_heading is not None and not blanked:
        innov = wrap180(mag_heading - predicted)
        predicted = wrap180(predicted + (1 - ALPHA) * innov)
    heading = predicted
    # magnetometer is blanked while a motor pulses (magnet interference)
    return heading
```

## yolo_detector.py — vision and obstacle gating

```python
CONFIDENCE = 0.35; STOP_TRIGGER = 3.0

def process_frame(frame, cane_yaw):
    preds = infer(frame)   # Roboflow server on localhost:9001, base64 JPEG
    obstacle = False
    for p in preds:
        if p.conf < CONFIDENCE: continue
        angle = ((p.x / W) - 0.5) * H_FOV
        d = CAM_HEIGHT / tan(radians(CAM_TILT + row_angle(p)))
        if abs(angle + cane_yaw) <= CORRIDOR and d <= STOP_TRIGGER:
            obstacle = True
    return obstacle
```

Calibration and evidence tooling added in Phase 2:

- `calibrate_tilt(d_known)` — computes the camera mounting angle from a target at a known distance
- `aim_preview(target_tilt)` — draws guide lines where chosen ground distances should appear, so the mount is set by construction
- `annotate_frame()` — red (triggered) / green (gated out) boxes labelled with class, confidence, range and body angle
- `EvidenceRecorder` — writes `frames/*.jpg`, `frames.csv`, `detections.csv` and `summary.txt`; `--save-mode all|detect|trigger|interval|none`

## haptic_output.py — finger-allocated driver

```python
PIN = {THUMB: 5, INDEX: 26, MIDDLE: 19, RING: 6, PINKY: 13}   # BCM, as-built loom
PULSE = 0.20; REFRACTORY = 0.50

def fire(finger):
    if now - last[finger] < REFRACTORY: return    # no chatter
    motor[finger].on(); after(PULSE): motor[finger].off()

def alert():   # GPS-lost signature
    pulse all five motors 3 times (150 ms on / 150 ms off)
```
