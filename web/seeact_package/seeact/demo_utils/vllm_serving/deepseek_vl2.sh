
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/home/fangzhengwei/miniconda3/envs/seeact/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# refer to https://github.com/deepseek-ai/DeepSeek-VL2/issues/75
CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/deepseek-vl2 --dtype auto --api-key token-abc123 --trust-remote-code --chat-template seeact_package/seeact/demo_utils/vllm_chat_templates/template_deepseek_vl2.jinja \
    --hf-overrides '{"architectures": ["DeepseekVLV2ForCausalLM"]}'