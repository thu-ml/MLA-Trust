from src.scene.mobile.MobileAgentE.controller import get_screenshot

if __name__ == "__main__":
    adb_path = "adb"  # Adjust this path if adb is located elsewhere
    device_serial = "eda783d3"  # Replace with your device serial number
    while True:
        screenshot_path = get_screenshot(adb_path, device_serial)
        print(f"Screenshot saved to: {screenshot_path}")
