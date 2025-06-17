

<h1 align="center">
MLA-Trust: Benchmarking Trustworthiness of Multimodal LLM Agents in GUI Environments

</h1>


<div align="center">
    <img src="https://img.shields.io/badge/Benchmark-Truthfulness-yellow" alt="Truthfulness" />
    <img src="https://img.shields.io/badge/Benchmark-Safety-red" alt="Safety" />
    <img src="https://img.shields.io/badge/Benchmark-Controllability-blue" alt="Controllability" />
    <img src="https://img.shields.io/badge/Benchmark-Privacy-green" alt="Privacy" />


</div>


---

🛡️ **MLA-Trust** is a comprehensive and unified framework that evaluates the MLA trustworthiness across four principled dimensions: **truthfulness**, **controllability**, **safety** and **privacy**. The framework includes 34 high-risk interactive tasks to expose new trustworthiness challenges in GUI environments.

![Framework](assets/framework.jpg)

- **Truthfulness** captures whether the agent correctly interprets visual or DOM-based elements on the GUI, and whether it produces factual outputs based on those perceptions. 

- **Controllability** assesses whether the agent introduces unnecessary steps, drifts from the intended goal, or triggers side effects not specified by the user.

- **Safety** demonstrates whether the agent's actions are free from harmful or irreversible consequences, which encompasses the prevention of behaviors that cause financial loss, data corruption, or system failures.

- **Privacy** evaluates whether the agent respects the confidentiality of sensitive information. MLAs often capture screenshots, handle form data, and interact with files.


## 🎯 Main Findings

<table>
<tr>
<td align="center">🚨</td>
<td><strong>Severe vulnerabilities in GUI environments</strong>: Both proprietary and open-source MLAs that interact with GUIs exhibit more severe trustworthiness risks compared to traditional MLLMs, particularly in high-stakes scenarios such as financial transactions.</td>
</tr>
<tr>
<td align="center">🔄</td>
<td><strong>Multi-step dynamic interactions amplify vulnerabilities</strong>: The transformation of MLLMs into GUI-based MLAs significantly compromises their trustworthiness. In multi-step interactive settings, these agents can execute harmful content that standalone MLLMs would typically reject.</td>
</tr>
<tr>
<td align="center">⚡</td>
<td><strong>Emergence of derived risks from iterative autonomy</strong>: Multi-step execution enhances adaptability but introduces latent and nonlinear risk accumulation across decision cycles, leading to unpredictable derived risks.</td>
</tr>
<tr>
<td align="center">📈</td>
<td><strong>Trustworthiness correlation</strong>: Open-source models employing structured fine-tuning strategies (e.g., SFT and RLHF) demonstrate improved controllability and safety. Larger models generally exhibit higher trustworthiness across multiple sub-aspects.</td>
</tr>
</table>



## 💻 Installation
1. Install `uv` by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer). Ensure the PATH environment variable is configured as prompted.
2. Install dependencies:
    ```bash
    uv sync
    uv sync --extra flash-attn
    ```

<details> 
<summary> 📱 Mobile Setup</summary>

#### A. Device Specifications 
- Device: Redmi Note 13 Pro
- Operating System: Xiaomi HyperOS 2.0.6.0


#### B. ADB Setup and Configuration
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
   ## Check connected devices
   adb devices
   ```


#### C. Environment Configuration
1. Create `.env` file in root directory
2. Refer to `src/scene/mobile/inference_agent_E.py` and configure environment variables

#### D. Task Preconditions
1. Modify `scripts/mobile/adb.sh` script for device setup
    - Script functions: 1) Unlock device; 2) Return to home screen;
    - Must execute before each task
    - Customize according to your device specifications
2. Update ANDROID_SERIAL in `scripts/mobile/run_task.sh` to match your device


#### E. Quick Start
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

</details> 

<details> 
<summary> 🌐 Web Setup</summary>


#### A. Preparation

Since many tasks require a login to function properly, we provide cookie loading functionality to enable the agent to work correctly. You only need to run the following command (must be run on a machine with a visual web interface), then perform your login, and finally close the popup website to save cookies.

```bash
python src/scene/web/load_cookies.py
```

Then save the generated `*.json` files to `src/scene/web/cookies`


#### B. Quick Start

1. **Activate virtual environment**
   ```bash
   source .venv/bin/activate
   ```

2. **Execute main task**
   > Consider configuring a network proxy if you encounter issues like network inaccessibility.
   Modify the script to run specific models or tasks according to your needs.
   ```bash
   bash scripts/web/run_task.sh
   ```

3. **Run evaluation**
   ```bash
   python src/scene/web/eval/test.py
   ```
</details>

## 🚀 Supported Models

The following models are supported:

- `gpt-4o-2024-11-20`
- `gpt-4-turbo`
- `gemini-2.0-flash`
- `gemini-2.0-pro-exp-02-05`
- `claude-3-7-sonnet-20250219`
- `llava-hf/llava-v1.6-mistral-7b-hf`
- `lmms-lab/llava-onevision-qwen2-72b-ov-sft`
- `lmms-lab/llava-onevision-qwen2-72b-ov-chat`
- `microsoft/Magma-8B`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `deepseek-ai/deepseek-vl2`
- `openbmb/MiniCPM-o-2_6`
- `mistral-community/pixtral-12b`
- `microsoft/Phi-4-multimodal-instruct`
- `OpenGVLab/InternVL2-8B`

</div>

## 📋 Task Overview

<div align="center">

![Task List](assets/task_list.jpg)

*Our comprehensive task suite covers 34 high-risk interactive scenarios across multiple domains*

</div>

## 🏆 Results

<div align="center">

![Results](assets/rank.png)

*Performance ranking of different MLAs across trustworthiness dimensions*

</div>

---

<div align="left">





## 🤝 Acknowledgement
We acknowledge and thank the projects [Mobile-Agent-E](https://github.com/X-PLUG/MobileAgent/tree/main/Mobile-Agent-E) and [SeeAct](https://github.com/OSU-NLP-Group/SeeAct), whose foundational work has supported the development of this project.

## 📞 Contact

We welcome contributions! Please feel free to submit issues and pull requests.
For questions or collaboration opportunities, please contact us at jankinfmail@gmail.com, 52285904015@stu.ecnu.edu.cn, yangxiao19@tsinghua.org.cn

## 🌟 Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{yang2025mla,
  title={MLA-Trust: Benchmarking Trustworthiness of Multimodal LLM Agents in GUI Environments},
  author={Yang, Xiao and Chen, Jiawei and Luo, Jun and Fang, Zhengwei and Dong, Yinpeng and Su, Hang and Zhu, Jun},
  journal={arXiv preprint arXiv:2506.01616},
  year={2025}
}
```

</div> 
