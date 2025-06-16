# 🛡️ MLA-Trust: Benchmarking Trustworthiness of Multimodal LLM Agents in GUI Environments

<div align="center">

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)
![Research](https://img.shields.io/badge/Research-AI%20Safety-blue.svg?style=for-the-badge)

📊 **First Comprehensive Framework** | 🔒 **4 Trust Dimensions** | ⚠️ **34 High-Risk Tasks** | 🤖 **GUI Agent Evaluation**

</div>

---

🚀 **MLA-Trust** is the first comprehensive and unified framework that evaluates the MLA trustworthiness across four principled dimensions: **truthfulness**, **controllability**, **safety** and **privacy**. The framework includes 34 high-risk interactive tasks to expose new trustworthiness challenges in GUI environments.

![Framework](assets/framework.jpg)


## 🎯 Main Contributions

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

## 🔍 Key Findings

> 💡 **Our comprehensive evaluation reveals critical insights about MLA trustworthiness across different dimensions and scenarios.**

### 🚨 1. Severe Vulnerabilities in GUI Environments
Both proprietary and open-source MLAs exhibit more severe trustworthiness risks compared to traditional MLLMs. This discrepancy stems from MLAs' interactions with external environments and real-world executions, which introduce actual risks and hazards during execution beyond the passive text generation of LLMs, particularly in high-stakes scenarios such as financial transactions.

### 🧠 2. Deficiencies in Complex Contextual Reasoning
In agentic GUI tasks, complex contextual reasoning tasks increase MLAs trustworthiness risks compared to predefined step-by-step tasks. Long-chain reasoning processes spanning multiple scenarios and environmental states struggle to maintain semantic consistency due to their inherent nonlinearity.

### 🛡️ 3. Superior Safety Alignment in Proprietary Models
Proprietary MLAs demonstrate enhanced trustworthiness compared to open-source alternatives. Their refined multilayered security protocols, such as toxicity detection in GPT-4o, enable superior risk identification and prevention of malicious actions.

### 🔄 4. Multi-step Dynamic Interactions Amplify Vulnerabilities
The transformation of MLLMs into GUI-based agents significantly compromises their trustworthiness. In multi-step execution, these agents can execute harmful content that standalone MLLMs would typically reject, even without explicit jailbreak prompts.

### ⚡ 5. Emergence of Derived Risks from Iterative Autonomy
Multi-step execution enhances system autonomy and adaptability but introduces latent and nonlinear risk accumulation across decision cycles. Continuous interactions can trigger uncontrolled self-evolution and concealed vulnerabilities, leading to unpredictable derived risks.

### 🎯 6. Training Paradigm Effectiveness
Open-source models employing structured fine-tuning strategies, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), exhibit enhanced controllability and safety in real-world tasks. Models like DeepSeek-VL2, which adopt three-stage pretraining with full-parameter fine-tuning, rank higher in trustworthiness metrics.

### 📊 7. Model Scale and Trustworthiness Correlation
Larger-scale models generally exhibit higher trustworthiness across various sub-aspects. LLaVA-OneVision (72B), DeepSeek-VL2 (27B) and Pixtral-12B outperform smaller models in overall rankings, suggesting that increased model capacity enables better alignment with safety mechanisms.

## 💡 Research Implications

> 🔄 **The trustworthiness landscape has evolved from "information risk" to "behavior risk"** as MLAs become more autonomous and capable of performing actions within diverse environments.

### 🎯 Key Recommendations:

<div align="left">

🔧 **a) Draw lessons from system engineering**: Consider the entire lifecycle of an intelligent agent, from design and development to deployment and operation in real-world environments. A system approach ensures that security measures are integrated at every stage, emphasizing robustness, transparency, and controllability.

🚀 **b) Expanding focus on action learning in MLAs**: Beyond enhancing execution capabilities, there's a need to prioritize behavioral intention understanding, context-aware reasoning capabilities, maintenance of inherent alignment relationships in foundational MLLMs, and improved coordination for aligning content behavior.


## 💻 Installation
1. First, you need to refer to [uv installation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) to install uv and set up the PATH environment variable as prompted.
2. Use the following command to install dependencies
    ```bash
    uv sync
    uv sync --extra flash-attn
    ```

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

<div align="center">



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

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📞 Contact

For questions or collaboration opportunities, please contact us at [yangxiao19@tsinghua.org.cn,52285904015@stu.ecnu.edu.cn]

---

💡 This work represents a significant step forward in understanding and evaluating the trustworthiness of multimodal LLM agents in practical GUI environments, providing crucial insights for the development of safer and more reliable AI agents.

<sub>Made with ❤️ for AI Safety Research</sub>

</div> 