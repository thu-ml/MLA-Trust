export PYTHONUNBUFFERED=1
export ANDROID_SERIAL=a9675c1b
export ANDROID_ADB_SERVER_PORT=5037

run_name=demo

init_adb() {
    . adb.sh

    if [ $? -eq 0 ]; then
        echo "adb connect successfully"
    else
        echo "error: adb bad connection"
        exit
    fi
}

model_names=(
    "lmms-lab/llava-onevision-qwen2-72b-ov-chat"
    "deepseek-ai/deepseek-vl2"
    "llava-hf/llava-v1.6-mistral-7b-hf"
    "mistral-community/pixtral-12b"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "openbmb/MiniCPM-o-2_6"
    "gpt-4o-2024-11-20"
    "gemini-2.0-flash"
    "gemini-2.0-pro-exp-02-05"
    "claude-3-7-sonnet-20250219"
    "gpt-4-turbo"
    "OpenGVLab/InternVL2-8B"
    "microsoft/Phi-4-multimodal-instruct"
)

tasks_jsons=(
    "data/truthfulness/inherent_deficiency/mobile_Chrome.jsonl"
    "data/truthfulness/inherent_deficiency/mobile_cross_app.jsonl"
    "data/truthfulness/inherent_deficiency/mobile_Notes.jsonl"
    "data/truthfulness/misguided_mistakes/mobile_misleading.jsonl"
    "data/truthfulness/misguided_mistakes/mobile_unclear.jsonl"

    "data/safe_data/advbench.jsonl"
    "data/safe_data/autobreach.jsonl"
    "data/safe_data/jailbreakbench.jsonl"
    "data/safe_data/sampled_dynahate.jsonl"
    "data/safe_data/sampled_realtoxicityprompts.jsonl"
    "data/safe_data/strongreject.jsonl"
    "data/safe_data/various_apps_jailbreak.jsonl"

    "data/privacy/privacy_leakage/mobile_indirect_leakage.jsonl"
    "data/privacy/privacy_leakage/mobile_direct_leakage.jsonl"
    "data/privacy/privacy_awareness/mobile_indirect_awareness.jsonl"
    "data/privacy/privacy_awareness/mobile_direct_awareness.jsonl"

    "data/controllability/overcompletion/mobile_gmail_overcompletion_gcg.jsonl"
    "data/controllability/overcompletion/mobile_note_overcompletion_gcg.jsonl"
    "data/controllability/speculative_risk/mobile_gmail_speculative_risk_gcg.jsonl"
    "data/controllability/speculative_risk/mobile_note_speculative_risk_gcg.jsonl"
    
    "data/controllability/overcompletion/mobile_gmail_overcompletion_other.jsonl"
    "data/controllability/overcompletion/mobile_note_overcompletion_other.jsonl"
    "data/controllability/speculative_risk/mobile_gmail_speculative_risk_other.jsonl"
    "data/controllability/speculative_risk/mobile_note_speculative_risk_other.jsonl"
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

        init_adb

        if [ "$model_name" == "openbmb/MiniCPM-o-2_6" ]; then
            export CUDA_VISIBLE_DEVICES="0"
            
            uv run --active --with "transformers==4.44.2" run.py --run_name "$run_name" --manager_model_name "$model_name" \
                --tasks_json "$tasks_json"

        else
            if [ "$model_name" == "deepseek-ai/deepseek-vl2" ]; then
                export CUDA_VISIBLE_DEVICES="0,1"
            elif [ "$model_name" == "lmms-lab/llava-onevision-qwen2-72b-ov-chat" ]; then
                export CUDA_VISIBLE_DEVICES="0,1,2"
            else
                export CUDA_VISIBLE_DEVICES="0"
            fi
            
            uv run --active run.py --run_name "$run_name" --manager_model_name "$model_name" \
                --tasks_json "$tasks_json"
        fi
    done
done