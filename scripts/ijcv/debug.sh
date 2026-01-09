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
    gpt-4o-mini
)

tasks_jsons=(
    "data/mobile/truthfulness/inherent_deficiency/mobile_Chrome.jsonl"
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

        uv run --with debugpy --active python -m src.scene.mobile.run --run_name "$run_name" --manager_model_name "$model_name" \
            --tasks_json "$tasks_json" --log_root "logs/mobile/debug"
    done
done