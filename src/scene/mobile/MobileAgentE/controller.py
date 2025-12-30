import os
import subprocess
import time
from time import sleep

from PIL import Image, UnidentifiedImageError


def check_file_exists(adb_path, file_path):
    command = f"""{adb_path} shell '[ -f {file_path} ] && echo "1" || echo "0"'"""
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout.strip() == "1"


def get_screenshot(
    adb_path,
    device_serial,
    SCREENSHOT_DIR="./screenshot",
    max_retry=10,
):
    assert device_serial is not None, "Device serial cannot be None"

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    local_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{device_serial}.png")
    remote_path = f"/sdcard/__screenshot_{device_serial}.png"

    last_error = None

    for retry in range(max_retry + 1):
        try:
            print(f"Retry count: {retry}/{max_retry}")

            # screenshot process: 1. screencap on phone,  2. pull to local,  3. validate,  4. clean up on phone
            subprocess.run(
                f"{adb_path} shell screencap -p {remote_path}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=False,
            )

            subprocess.run(
                f"{adb_path} pull {remote_path} {local_path}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=False,
            )

            if not os.path.exists(local_path):
                raise RuntimeError("Screenshot file not found after adb pull")

            if os.path.getsize(local_path) < 1024:
                raise RuntimeError("Screenshot file too small, capture likely failed")

            try:
                with Image.open(local_path) as img:
                    img.load()
            except UnidentifiedImageError as e:
                raise RuntimeError("Pulled file is not a valid PNG") from e

            subprocess.run(
                f"{adb_path} shell rm -f {remote_path}",
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=False,
            )

            print(f"Screenshot successfully captured after {retry} retries")
            return local_path

        except Exception as e:
            last_error = e
            print(f"Error: {e}")
            time.sleep(0.2)

    raise RuntimeError(
        f"Failed to capture screenshot after {max_retry} attempts. "
        f"Last error: {last_error}"
    )


def start_recording(adb_path):
    print("Remove existing screenrecord.mp4")
    command = adb_path + " shell rm /sdcard/screenrecord.mp4"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    print("Start!")
    # Use subprocess.Popen to allow terminating the recording process later
    command = adb_path + " shell screenrecord /sdcard/screenrecord.mp4"
    process = subprocess.Popen(command, shell=True)
    return process


def end_recording(adb_path, output_recording_path):
    print("Stopping recording...")
    # Send SIGINT to stop the screenrecord process gracefully
    stop_command = adb_path + " shell pkill -SIGINT screenrecord"
    subprocess.run(stop_command, capture_output=True, text=True, shell=True)
    sleep(1)  # Allow some time to ensure the recording is stopped

    print("Pulling recorded file from device...")
    pull_command = f"{adb_path} pull /sdcard/screenrecord.mp4 {output_recording_path}"
    subprocess.run(pull_command, capture_output=True, text=True, shell=True)
    print(f"Recording saved to {output_recording_path}")


def save_screenshot_to_file(adb_path, file_path="screenshot.png", max_retry=10):
    """
    Captures a screenshot from an Android device using ADB, saves it locally, and removes the screenshot from the device.

    Args:
        adb_path (str): The path to the adb executable.

    Returns:
        str: The path to the saved screenshot, or raises an exception on failure.
    """
    # Define the local filename for the screenshot
    local_file = file_path

    if os.path.dirname(local_file) != "":
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

    retry = 0
    success = False
    last_error = None

    while retry <= max_retry:
        try:
            print(f"Retry count: {retry}/{max_retry}")

            screencap_command = adb_path + " exec-out screencap -p >  {}".format(
                local_file,
            )
            print(screencap_command)
            subprocess.run(
                screencap_command,
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )
            success = True
            print(f"\tAtomic Operation Screenshot saved to {local_file}")
            break

        except Exception as e:
            last_error = e
            retry += 1
            print(f"Error: {str(e)}")

    if not success:
        raise RuntimeError(
            f"Failed to capture screenshot after {max_retry} attempts. Last error: {str(last_error)}"
        )
    else:
        return local_file


def tap(adb_path, x, y):
    command = adb_path + f" shell input tap {x} {y}"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def type(adb_path, text):
    text = text.replace("\\n", "_").replace("\n", "_")
    for char in text:
        if char == " ":
            command = adb_path + " shell input text %s"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char == "_":
            command = adb_path + " shell input keyevent 66"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif "a" <= char <= "z" or "A" <= char <= "Z" or char.isdigit():
            command = adb_path + f" shell input text {char}"
            subprocess.run(command, capture_output=True, text=True, shell=True)
        elif char in "-.,!?@'°/:;()":
            command = adb_path + f' shell input text "{char}"'
            subprocess.run(command, capture_output=True, text=True, shell=True)
        else:
            command = (
                adb_path + f' shell am broadcast -a ADB_INPUT_TEXT --es msg "{char}"'
            )
            subprocess.run(command, capture_output=True, text=True, shell=True)


def enter(adb_path):
    command = adb_path + " shell input keyevent KEYCODE_ENTER"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def swipe(adb_path, x1, y1, x2, y2):
    command = adb_path + f" shell input swipe {x1} {y1} {x2} {y2} 500"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def back(adb_path):
    command = adb_path + " shell input keyevent 4"
    subprocess.run(command, capture_output=True, text=True, shell=True)


def home(adb_path):
    # command = adb_path + f" shell am start -a android.intent.action.MAIN -c android.intent.category.HOME"
    command = adb_path + " shell input keyevent KEYCODE_HOME"
    subprocess.run(command, capture_output=True, text=True, shell=True)
    subprocess.run(command, capture_output=True, text=True, shell=True)


def switch_app(adb_path):
    command = adb_path + " shell input keyevent KEYCODE_APP_SWITCH"
    subprocess.run(command, capture_output=True, text=True, shell=True)
