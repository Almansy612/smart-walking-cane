@echo off
echo Running FYP smart walking cane tests...
echo.

echo Test 1: yolo_detector.py --selftest
python yolo_detector.py --selftest
if %errorlevel% neq 0 exit /b %errorlevel%
echo.

echo Test 2: gps_navigator.py --selftest
python gps_navigator.py --selftest
if %errorlevel% neq 0 exit /b %errorlevel%
echo.

echo Test 3: imu_filter.py --selftest
python imu_filter.py --selftest
if %errorlevel% neq 0 exit /b %errorlevel%
echo.

echo Test 4: haptic_output.py --selftest
python haptic_output.py --selftest
if %errorlevel% neq 0 exit /b %errorlevel%
echo.

echo Test 5: main.py --sim
python main.py --sim
if %errorlevel% neq 0 exit /b %errorlevel%
echo.

echo All tests passed!
