import os
import subprocess
from time import sleep


def check_file_exists(adb_path, file_path):
    command = f"""{adb_path} shell '[ -f {file_path} ] && echo "1" || echo "0"'"""
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    return result.stdout.strip() == "1"


def get_screenshot(
    adb_path, device_serial, SCREENSHOT_DIR="./screenshot", max_retry=10
):
    assert device_serial is not None, "Device serial cannot be None"

    os.makedirs(SCREENSHOT_DIR, exist_ok=True)

    save_path = os.path.join(SCREENSHOT_DIR, "screenshot_{}.png".format(device_serial))

    # file_on_phone = "/sdcard/screenshot_{}.png".format(device_serial)
    # image_path = os.path.join(SCREENSHOT_DIR, "screenshot_{}.png".format(device_serial))
    # save_path = os.path.join(SCREENSHOT_DIR, "screenshot_{}.jpg".format(device_serial))

    retry = 0
    success = False
    last_error = None

    while retry <= max_retry:
        try:
            print(f"Retry count: {retry}/{max_retry}")

            screencap_command = adb_path + " exec-out screencap -p > {}".format(
                save_path
            )
            print(screencap_command)
            subprocess.run(
                screencap_command,
                capture_output=True,
                text=True,
                shell=True,
                check=True,
            )

            # delete_command = adb_path + " shell rm -f {}".format(file_on_phone)
            # print(delete_command)
            # subprocess.run(
            #     delete_command, capture_output=True, text=True, shell=True, check=True
            # )

            # if os.path.exists(image_path):
            #     os.remove(image_path)
            # if os.path.exists(save_path):
            #     os.remove(save_path)

            # screencap_command = adb_path + " shell screencap -p {}".format(
            #     file_on_phone
            # )
            # subprocess.run(
            #     screencap_command,
            #     capture_output=True,
            #     text=True,
            #     shell=True,
            #     check=True,
            # )
            # print(screencap_command)

            # timeout = 5
            # start_time = time.time()
            # while not check_file_exists(adb_path=adb_path, file_path=file_on_phone):
            #     time.sleep(0.1)
            #     if time.time() - start_time > timeout:
            #         raise TimeoutError(
            #             f"Screenshot file not created on device after {timeout} seconds"
            #         )

            # size_check_command = adb_path + f" shell ls -l {file_on_phone}"
            # subprocess.run(
            #     size_check_command,
            #     capture_output=True,
            #     text=True,
            #     shell=True,
            #     check=True,
            # )
            # print(size_check_command)

            # pull_command = adb_path + " pull {} {}".format(
            #     file_on_phone, SCREENSHOT_DIR
            # )
            # subprocess.run(
            #     pull_command, capture_output=True, text=True, shell=True, check=True
            # )
            # print(pull_command)

            # timeout = 5
            # start_time = time.time()
            # while not os.path.exists(image_path):
            #     time.sleep(0.1)
            #     if time.time() - start_time > timeout:
            #         raise TimeoutError(
            #             f"Screenshot file not available locally after {timeout} seconds"
            #         )

            # if os.path.getsize(image_path) == 0:
            #     raise ValueError("Local screenshot file has zero size")

            # Image.open(image_path).convert("RGB").save(save_path, "JPEG")
            # os.remove(image_path)

            # delete_command = adb_path + " shell rm -f {}".format(file_on_phone)
            # subprocess.run(delete_command, capture_output=True, text=True, shell=True)

            success = True
            print(f"Screenshot successfully captured after {retry} retries")
            break

        except Exception as e:
            last_error = e
            retry += 1
            print(f"Error: {str(e)}")

    if not success:
        raise RuntimeError(
            f"Failed to capture screenshot after {max_retry} attempts. Last error: {str(last_error)}"
        )

    return save_path


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
