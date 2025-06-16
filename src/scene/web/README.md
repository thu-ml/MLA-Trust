# SeeAct

A multimodal AI agent framework for web automation and interaction tasks.

## Preparation

Since many tasks require login to function properly, we provide cookie loading functionality to enable the agent to work correctly. You only need to run the following command (must be run on a machine with a visual web interface), then perform your personal login, and finally close the popup website to save cookies.

```bash
python src/scene/web/load_cookies.py
```

Then put the `*.json` into `src/scene/web/cookies` folder


## Quick Start

1. **Activate virtual environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Execute main task**
   > Consider configuring a network proxy if encountering issues like network inaccessibility.
   Modify the script to run specific models or tasks according to your needs.
   ```bash
   bash scripts/web/run_task.sh
   ```

3. **Run evaluation**
   ```bash
   python src/scene/web/eval/test.py
   ```
