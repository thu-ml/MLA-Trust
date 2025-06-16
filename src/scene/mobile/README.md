# 💻 Environment Setup

## Device Specifications
- Device: Redmi Note 13 Pro
- Operating System: Xiaomi HyperOS 2.0.6.0


## ADB Setup and Configuration
> Reference: [Mobile-Agent-E Repository](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-E)

1. **Install Android Debug Bridge (ADB)**
   - Windows: Download from [Android Developer Platform Tools](https://developer.android.com/tools/releases/platform-tools)
   - MacOS: `brew install android-platform-tools`
   - Linux: `sudo apt-get install android-tools-adb`

2. **Enable Developer Options**
   - Go to Settings → About phone
   - Tap "MIUI version" multiple times until developer options are enabled (take Xiaomi for example)
   - Navigate to Settings → Additional Settings → Developer options

3. **Enable USB Debugging**
   - Enable "USB debugging" in Developer options
   - Connect phone via USB cable
   - Select "File Transfer" mode when prompted

4. **Verify ADB Connection**
   ```bash
   # Check connected devices
   adb devices
   ```


## Environment Configuration
1. Create `.env` file in root directory
2. Refer to `src/scene/mobile/inference_agent_E.py` and configure environment variables

## Preconditions for Tasks
1. Change `scripts/mobile/adb.sh` script for device setup
    - Script functions: 1) Unlock device; 2) Return to home screen;
    - Must be run before each task
    - Customize according to your device specifications
2. Change ANDROID_SERIAL in `scripts/mobile/run_task.sh` to match your device


# 🚀 Quick Start
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Execute main task
bash scripts/mobile/run_task.sh

# 3. Run evaluation
bash scripts/mobile/eval.sh

# 4. Generate statistics
python src/scene/mobile/eval/stat.py
```
