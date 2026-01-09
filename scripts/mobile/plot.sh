run_name=ijcv

task_names=(
    sampled_dynahate
    sampled_realtoxicityprompts
    mobile_Chrome
    mobile_cross_app
    mobile_Notes
    mobile_misleading
    mobile_unclear

    advbench
    autobreach
    jailbreakbench
    strongreject
    various_apps_jailbreak

    mobile_indirect_leakage
    mobile_indirect_awareness
    mobile_direct_leakage
    mobile_direct_awareness

    mobile_gmail_overcompletion_gcg
    mobile_gmail_overcompletion_other
    mobile_gmail_speculative_risk_gcg
    mobile_gmail_speculative_risk_other
    mobile_note_overcompletion_gcg
    mobile_note_overcompletion_other
    mobile_note_speculative_risk_gcg
    mobile_note_speculative_risk_other
)
model_names=(
    # gpt-4o-2024-11-20
    # gpt-4-turbo
    # gemini-2.0-flash
    # gemini-2.0-pro-exp-02-05
    # claude-3-7-sonnet-20250219
    # llava-v1.6-mistral-7b-hf
    # llava-onevision-qwen2-72b-ov-chat
    # Qwen2.5-VL-7B-Instruct
    # deepseek-vl2
    # MiniCPM-o-2_6
    # pixtral-12b
    # InternVL2-8B
    # Phi-4-multimodal-instruct
    # llava-onevision-qwen2-72b-ov-sft
    # Magma-8B

    InternVL3_5-8B
    Qwen3-VL-8B-Instruct
    # gpt-5.2-2025-12-11
    # gemini-3-pro-preview
    # claude-sonnet-4-5-20250929
)

for task_name in ${task_names[@]}; do
    for model_name in ${model_names[@]}; do
        python -m src.scene.mobile.tools.plots.plot --path "logs/mobile/${model_name}/${run_name}" --task_name $task_name 
        # python -m src.scene.mobile.tools.plots.plot --path "logs/mobile/${model_name}/${run_name}" --task_name $task_name --rewrite_html
        # exit
    done
done