
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=/home/fangzhengwei/miniconda3/envs/seeact/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=0 vllm serve llava-hf/llava-1.5-7b-hf --dtype auto --api-key token-abc123 --trust-remote-code --chat-template seeact_package/seeact/demo_utils/vllm_chat_templates/template_llava.jinja