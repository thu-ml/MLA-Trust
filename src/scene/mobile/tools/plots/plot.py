import argparse
import json
import logging
import os
import shutil

import pandas as pd
from natsort import natsorted

from .jinja_template import image_to_base64, template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

datasets_mapping = {
    "advbench": "data/mobile/safe_data/advbench.jsonl",
    "autobreach": "data/mobile/safe_data/autobreach.jsonl",
    "jailbreakbench": "data/mobile/safe_data/jailbreakbench.jsonl",
    "sampled_dynahate": "data/mobile/safe_data/sampled_dynahate.jsonl",
    "sampled_realtoxicityprompts": "data/mobile/safe_data/sampled_realtoxicityprompts.jsonl",
    "strongreject": "data/mobile/safe_data/strongreject.jsonl",
    "various_apps_jailbreak": "data/mobile/safe_data/various_apps_jailbreak.jsonl",
    "mobile_Chrome": "data/mobile/truthfulness/inherent_deficiency/mobile_Chrome.jsonl",
    "mobile_cross_app": "data/mobile/truthfulness/inherent_deficiency/mobile_cross_app.jsonl",
    "mobile_Notes": "data/mobile/truthfulness/inherent_deficiency/mobile_Notes.jsonl",
    "mobile_misleading": "data/mobile/truthfulness/misguided_mistakes/mobile_misleading.jsonl",
    "mobile_unclear": "data/mobile/truthfulness/misguided_mistakes/mobile_unclear.jsonl",
    "mobile_direct_awareness": "data/mobile/privacy/privacy_awareness/mobile_direct_awareness.jsonl",
    "mobile_direct_leakage": "data/mobile/privacy/privacy_leakage/mobile_direct_leakage.jsonl",
    "mobile_indirect_leakage": "data/mobile/privacy/privacy_leakage/mobile_indirect_leakage.jsonl",
    "mobile_indirect_awareness": "data/mobile/privacy/privacy_awareness/mobile_indirect_awareness.jsonl",
    "mobile_gmail_overcompletion_gcg": "data/mobile/controllability/overcompletion/mobile_gmail_overcompletion_gcg.jsonl",
    "mobile_gmail_overcompletion_other": "data/mobile/controllability/overcompletion/mobile_gmail_overcompletion_other.jsonl",
    "mobile_gmail_speculative_risk_gcg": "data/mobile/controllability/speculative_risk/mobile_gmail_speculative_risk_gcg.jsonl",
    "mobile_gmail_speculative_risk_other": "data/mobile/controllability/speculative_risk/mobile_gmail_speculative_risk_other.jsonl",
    "mobile_note_overcompletion_gcg": "data/mobile/controllability/overcompletion/mobile_note_overcompletion_gcg.jsonl",
    "mobile_note_overcompletion_other": "data/mobile/controllability/overcompletion/mobile_note_overcompletion_other.jsonl",
    "mobile_note_speculative_risk_gcg": "data/mobile/controllability/speculative_risk/mobile_note_speculative_risk_gcg.jsonl",
    "mobile_note_speculative_risk_other": "data/mobile/controllability/speculative_risk/mobile_note_speculative_risk_other.jsonl",
}


def clean_and_check_results(main_result_path, log_file):
    assert os.path.isdir(main_result_path)
    image_inputs_path = os.path.join(main_result_path, "screenshots")
    result_json_path = os.path.join(main_result_path, "steps.json")

    # Determine whether the program is successfully executed
    if not (os.path.isdir(image_inputs_path) and os.path.isfile(result_json_path)):
        print(f"Removing: {main_result_path}")
        shutil.rmtree(main_result_path)
        log_file.write(f"Removing: {main_result_path}")

    else:
        image_files = [
            f
            for f in os.listdir(image_inputs_path)
            if f.lower().endswith((".jpg", ".png"))
        ]

        image_files = natsorted(image_files)

        if image_files:
            latest_image = os.path.join(image_inputs_path, image_files[-1])
            print("Latest image:", latest_image)
        else:
            latest_image = None
            logging.log(level=logging.ERROR, msg=f"No image for {main_result_path}")

        with open(result_json_path, "r", encoding="utf-8") as f:
            result_data = json.load(f)

        final_info_pool = result_data[-1].get("final_info_pool")
        if final_info_pool is None:
            # logging.log(
            #     level=logging.INFO, msg=f"No final_info_pool for {main_result_path}"
            # )
            summary_history = []
            important_notes = ""
            finish_thought = ""
        else:
            summary_history = final_info_pool.get("summary_history")
            important_notes = final_info_pool.get("important_notes")
            finish_thought = final_info_pool.get("finish_thought")

        instruction = result_data[0].get("instruction")
        latest_image = latest_image
        refuse = result_data[-1].get("refuse", 0)
        finish_flag = result_data[-1].get("finish_flag")

        return {
            "task": instruction,
            "action_history": summary_history,
            "screenshot": latest_image,
            "important_notes": important_notes,
            "finish_thought": finish_thought,
            "refuse": refuse,
            "finish_flag": finish_flag,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="logs/mobile/Magma-8B/home_at_first_page",
    )
    parser.add_argument(
        "--task_name",
        type=str,
    )
    parser.add_argument(
        "--rewrite_html",
        action="store_true",
        help="Whether to rewrite the html file",
    )

    args = parser.parse_args()
    if not os.path.exists(args.path):
        print(f"Path {args.path} does not exist")
        exit(1)

    model_name = args.path.split("/")[-2]
    run_name = args.path.split("/")[-1]
    task_name = args.task_name

    rendered_file = f"../mla_trust_plots/{task_name}/{model_name}.html"

    if os.path.exists(rendered_file) and not args.rewrite_html:
        print(f"File {rendered_file} already exists, skipping...")
        exit(0)

    main_task_path = os.path.join(args.path, task_name)

    log_path = os.path.join(main_task_path, "plot_results.log")
    log_exists = os.path.exists(log_path)
    if log_exists:
        os.remove(log_path)

    if not list(os.walk(main_task_path)) or not next(os.walk(main_task_path))[1]:
        print(f"No subdirectories found in {main_task_path}")
        exit(1)

    # load metrics results from csv
    assert os.path.exists(os.path.join(main_task_path, "result.csv")), (
        f"No result.csv found in {main_task_path}"
    )
    result_df = pd.read_csv(os.path.join(main_task_path, "result.csv"))

    subdirs = next(os.walk(main_task_path))[1]
    examples = []
    with open(log_path, "a", encoding="utf-8") as log_file:
        for folder in natsorted(subdirs):
            main_result_path = os.path.join(main_task_path, folder)
            dataset_name = folder.rpartition("_")[0]
            task_id = int(folder.rpartition("_")[2])
            results = clean_and_check_results(main_result_path, log_file)

            metric_results = result_df[result_df["task_id"] == task_id]
            assert len(metric_results) == 1, f"{len(metric_results)} != 1"
            refuse_answer, is_success_attack, toxicity, is_success_execute = (
                metric_results.iloc[0][
                    [
                        "refuse_answer",
                        "is_success_attack",
                        "toxicity",
                        "is_success_execute",
                    ]
                ]
            )

            assert results is not None

            (
                task,
                action_history,
                screenshot,
                important_notes,
                finish_thought,
                refuse_flag,
                finish_flag,
            ) = results.values()

            jsonl_path = datasets_mapping[task_name]
            with open(jsonl_path, "r", encoding="utf-8") as f:
                line = f.readlines()[task_id]
                data = json.loads(line)
                ans = data.get("ans", None)

            if action_history is None:
                action_history = ["Error: No action history"]
                logging.log(level=logging.ERROR, msg=f"No action history for {folder}")

            example = {
                "task": task,
                "action_history": action_history,
                "screenshot": image_to_base64(screenshot),
                "important_notes": important_notes,
                "finish_thought": finish_thought,
                "refuse_flag": refuse_flag,
                "ans": ans,
                "metrics": {
                    "refuse_answer": refuse_answer,
                    "is_success_attack": is_success_attack,
                    "toxicity": toxicity,
                    "is_success_execute": is_success_execute,
                },
            }
            examples.append(example)

    rendered_html = template.render(examples=examples)
    os.makedirs(os.path.dirname(rendered_file), exist_ok=True)
    with open(rendered_file, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    print(f"Saving results to {os.path.abspath(rendered_file)}")
