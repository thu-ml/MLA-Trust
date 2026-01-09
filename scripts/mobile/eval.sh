
export PYTHONUNBUFFERED=1

# run_name=demo
run_name=ijcv


model_names=(
    # claude-3-7-sonnet-20250219
    # deepseek-vl2
    # gemini-2.0-flash
    # gemini-2.0-pro-exp-02-05
    # gpt-4-turbo
    # gpt-4o-2024-11-20
    # llava-onevision-qwen2-72b-ov-chat
    # llava-v1.6-mistral-7b-hf
    # MiniCPM-o-2_6
    # pixtral-12b
    # Qwen2.5-VL-7B-Instruct
    # InternVL2-8B
    # Phi-4-multimodal-instruct

    # InternVL3_5-8B
    # Qwen3-VL-8B-Instruct
    # "gpt-5.2-2025-12-11"
    "gemini-3-pro-preview"
    # "claude-sonnet-4-5-20250929"
)

tasks_jsons=(
    mobile_Chrome
    mobile_Notes
    mobile_cross_app
    mobile_misleading
    mobile_unclear
    sampled_realtoxicityprompts

    advbench
    autobreach
    jailbreakbench
    strongreject
    various_apps_jailbreak

    sampled_dynahate
    mobile_indirect_leakage
    mobile_indirect_awareness
    mobile_direct_leakage
    mobile_direct_awareness

    mobile_gmail_overcompletion
    mobile_gmail_speculative_risk
    mobile_note_overcompletion
    mobile_note_speculative_risk
)


for model_name in ${model_names[@]}; do
    for tasks_json in ${tasks_jsons[@]}; do
        echo "Eval logs/mobile/${model_name}/${run_name}/${tasks_json}"
        python -m src.scene.mobile.eval.test --path logs/mobile/${model_name}/${run_name}/${tasks_json} --filetype csv
    done
done