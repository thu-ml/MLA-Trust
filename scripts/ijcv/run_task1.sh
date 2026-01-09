export PYTHONUNBUFFERED=1
export ANDROID_SERIAL=a9675c1b
export ANDROID_ADB_SERVER_PORT=5040
export CUDA_VISIBLE_DEVICES=0,1,2,3

run_name=ijcv

init_adb() {
    . scripts/mobile/adb.sh

    if [ $? -eq 0 ]; then
        echo "adb connect successfully"
    else
        echo "error: adb bad connection"
        exit
    fi
}

model_names=(
    # "lmms-lab/llava-onevision-qwen2-72b-ov-chat"
    # "deepseek-ai/deepseek-vl2"
    # "llava-hf/llava-v1.6-mistral-7b-hf"
    # "mistral-community/pixtral-12b"
    # "Qwen/Qwen2.5-VL-7B-Instruct"
    # "openbmb/MiniCPM-o-2_6"
    # "gpt-4o-2024-11-20"
    # "gemini-2.0-flash"
    # "gemini-2.0-pro-exp-02-05"
    # "claude-3-7-sonnet-20250219"
    # "gpt-4-turbo"
    # "OpenGVLab/InternVL2-8B"
    # "microsoft/Phi-4-multimodal-instruct"
    
    # "OpenGVLab/InternVL3_5-8B"
    # "Qwen/Qwen3-VL-8B-Instruct"
    # "gpt-5.2-2025-12-11"
    "gemini-3-pro-preview"
    # "claude-sonnet-4-5-20250929"
)

tasks_jsons=(
    # "data/mobile/truthfulness/inherent_deficiency/mobile_Chrome.jsonl"
    # "data/mobile/truthfulness/inherent_deficiency/mobile_cross_app.jsonl"
    # "data/mobile/truthfulness/inherent_deficiency/mobile_Notes.jsonl"
    # "data/mobile/truthfulness/misguided_mistakes/mobile_misleading.jsonl"
    # "data/mobile/truthfulness/misguided_mistakes/mobile_unclear.jsonl"

    # "data/mobile/safe_data/advbench.jsonl"
    # "data/mobile/safe_data/autobreach.jsonl"
    # "data/mobile/safe_data/jailbreakbench.jsonl"
    # "data/mobile/safe_data/sampled_dynahate.jsonl"
    # "data/mobile/safe_data/sampled_realtoxicityprompts.jsonl"
    # "data/mobile/safe_data/strongreject.jsonl"
    # "data/mobile/safe_data/various_apps_jailbreak.jsonl"

    # "data/mobile/privacy/privacy_leakage/mobile_indirect_leakage.jsonl"
    # "data/mobile/privacy/privacy_leakage/mobile_direct_leakage.jsonl"
    # "data/mobile/privacy/privacy_awareness/mobile_indirect_awareness.jsonl"
    # "data/mobile/privacy/privacy_awareness/mobile_direct_awareness.jsonl"
    
    # "data/mobile/controllability/overcompletion/mobile_gmail_overcompletion.jsonl"
    # "data/mobile/controllability/overcompletion/mobile_note_overcompletion.jsonl"
    # "data/mobile/controllability/speculative_risk/mobile_gmail_speculative_risk.jsonl"
    "data/mobile/controllability/speculative_risk/mobile_note_speculative_risk.jsonl"
)

get_task_name() {
    echo $(basename $1) | cut -d. -f1
}

get_model_name() {
    echo $(basename $1) | cut -d/ -f2
}


for model_name in ${model_names[@]}; do
    for tasks_json in ${tasks_jsons[@]}; do
        task_name=$(get_task_name $tasks_json)
        short_model_name=$(get_model_name $model_name)
        echo "run task $task_name with model $short_model_name"
        mkdir -p buglog/$task_name

        init_adb

        uv run --active -m src.scene.mobile.run --run_name "$run_name" --manager_model_name "$model_name" \
            --tasks_json "$tasks_json" 2>&1 | tee buglog/$task_name/${short_model_name}.log
    done
done