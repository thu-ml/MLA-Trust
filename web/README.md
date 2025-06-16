# SeeAct

A multimodal AI agent framework for web automation and interaction tasks.

## Preparation

Since many tasks require login to function properly, we provide cookie loading functionality to enable the agent to work correctly. You only need to run the following command (must be run on a machine with a visual web interface), then perform your personal login, and finally close the popup website to save cookies.

```bash
python load_cookies.py
```

Then put the `*.json` into `cookies` folder


## Quick Start

1. **Activate virtual environment**
   ```bash
   source ../.venv/bin/activate
   ```

2. **Execute main task**
   > Consider configuring a network proxy if encountering issues like network inaccessibility.
   ```bash
   bash ./demo.sh
   ```

3. **Run evaluation**
   ```bash
   python tests/test.py
   ```

## Usage

The main entry point is through the `demo.sh` script, which contains various model configurations and task execution commands. You can modify the script to run specific models or tasks according to your needs.

## Project Structure

```
├── src/                    # Source code
│   ├── seeact.py          # Main SeeAct implementation
│   ├── config/            # Configuration files
│   ├── data_utils/        # Data utilities
│   └── demo_utils/        # Demo utilities
├── data/                  # Data files
├── tests/                 # Test files
├── models/                # Model files
├── demo.sh               # Demo execution script
└── load_cookies.py       # Cookie loading utility
```

## Acknowledgement

This project is primarily based on the original [SeeAct](https://github.com/OSU-NLP-Group/SeeAct) framework developed by the OSU NLP Group.
