# Enable shell command tracing for debugging
set -x

# Enable pointer location debug mode to show touch point coordinates on screen
adb shell settings put system pointer_location 1
sleep 0.2

# Check if screen is on by checking DisplaySuspendBlocker status
# If screen is on (mHoldingDisplaySuspendBlocker is true), turn it off
if adb shell dumpsys power | grep 'mHoldingDisplaySuspendBlocker' | grep -q 'true'; then
    # Send power key event (KEYCODE_POWER = 26) to turn off screen
    adb shell input keyevent 26
    sleep 0.2
fi

adb shell input keyevent 26
sleep 0.2

# Simulate vertical swipe gesture to unlock screen
# Swipe from (300, 1000) to (300, 500)
adb shell input swipe 300 1000 300 500
sleep 0.2

# Press Home key to return to home screen (KEYCODE_HOME = 3)
adb shell input keyevent 3
sleep 0.2

# Press Home key to return to home screen (KEYCODE_HOME = 3)
adb shell input keyevent 3
sleep 0.2

# Disable pointer location debug mode
adb shell settings put system pointer_location 0
