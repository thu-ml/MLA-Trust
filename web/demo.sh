export HF_HUB_OFFLINE=1
export PYTHONASYNCIODEBUG=1 
export PYTHONUNBUFFERED=1 
export CUDA_VISIBLE_DEVICES=0

# Twitter
uv run --active -m src.seeact --model_name "llava-hf/llava-v1.6-mistral-7b-hf" --save_file_dir "../online_results/llava-hf"  --task_file_path "../data/truthfulness/inherent_deficiency/website_Twitter.jsonl"
uv run --active -m src.seeact --model_name "microsoft/Magma-8B" --save_file_dir "../online_results/Magma-8B"  --task_file_path "../data/truthfulness/inherent_deficiency/website_Twitter.jsonl"
uv run --active -m src.seeact --model_name "mistral-community/pixtral-12b" --save_file_dir "../online_results/pixtral-12b"  --task_file_path "../data/truthfulness/inherent_deficiency/website_Twitter.jsonl"
uv run --active --with "transformers==4.44.2" -m src.seeact --model_name "openbmb/MiniCPM-o-2_6" --save_file_dir "../online_results/MiniCPM"  --task_file_path "../data/truthfulness/inherent_deficiency/website_Twitter.jsonl"
uv run --active -m src.seeact --model_name "gpt-4o-2024-11-20" --save_file_dir "../online_results/gpt-4o-2024-11-20"  --task_file_path "../data/truthfulness/inherent_deficiency/website_Twitter.jsonl"
uv run --active -m src.seeact --model_name "gpt-4-turbo" --save_file_dir "../online_results/gpt-4-turbo"  --task_file_path "../data/truthfulness/inherent_deficiency/website_Twitter.jsonl"
uv run --active -m src.seeact --model_name "gemini-2.0-flash" --save_file_dir "../online_results/gemini-2.0-flash"  --task_file_path "../data/truthfulness/inherent_deficiency/website_Twitter.jsonl"
uv run --active -m src.seeact --model_name "gemini-2.0-pro-exp-02-05" --save_file_dir "../online_results/gemini-2.0-pro-exp"  --task_file_path "../data/truthfulness/inherent_deficiency/website_Twitter.jsonl"