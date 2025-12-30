export PYTHONUNBUFFERED=1
export ANDROID_SERIAL=eda783d3
export ANDROID_ADB_SERVER_PORT=5040
export CUDA_VISIBLE_DEVICES=4,5,6,7

init_adb() {
    . scripts/mobile/adb.sh

    if [ $? -eq 0 ]; then
        echo "adb connect successfully"
    else
        echo "error: adb bad connection"
        exit
    fi
}

# init_adb


# adb shell screencap -p /sdcard/__screen.png
# adb pull /sdcard/__screen.png screenshot/screenshot_eda783d3.png
# adb shell rm /sdcard/__screen.png

python -m tests.test_screenshot