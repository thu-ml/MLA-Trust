export CUDA_VISIBLE_DEVICES=7

# CUDA_VISIBLE_DEVICES=7 uv run python -m tests.test_web_models --model "Qwen/Qwen2.5-VL-7B-Instruct" --temperature 0.0
# CUDA_VISIBLE_DEVICES=7 uv run --with "transformers==4.49.0" python -m tests.test_web_models --model "OpenGVLab/InternVL2-8B" --temperature 0.0
# CUDA_VISIBLE_DEVICES=7 uv run python -m tests.test_web_models --model "OpenGVLab/InternVL3_5-8B" --temperature 0.0
# CUDA_VISIBLE_DEVICES=7 uv run python -m tests.test_web_models --model "Qwen/Qwen3-VL-8B-Instruct" --temperature 0.0

CUDA_VISIBLE_DEVICES=7 uv run python -m tests.test_web_models --model "gemini-3-pro-preview" --temperature 0.0
CUDA_VISIBLE_DEVICES=7 uv run python -m tests.test_web_models --model "gpt-5.2-2025-12-11" --temperature 0.0
CUDA_VISIBLE_DEVICES=7 uv run python -m tests.test_web_models --model "claude-sonnet-4-5-20250929" --temperature 0.0
CUDA_VISIBLE_DEVICES=7 uv run python -m tests.test_web_models --model "gpt-4o-mini" --temperature 0.0
