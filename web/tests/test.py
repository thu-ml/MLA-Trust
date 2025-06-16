import argparse
import json
import os
import re
import shutil

import pandas as pd
from metric import (
    get_is_attack_successful,
    get_is_confirm,
    get_is_execute_successful,
    get_perspective_toxicity_scores,
)


def extract_numbers(filename):
    match = re.match(r"(\d+)_\d+_crop\.jpg", filename)  # 提取前面的主编号
    return int(match.group(1)) if match else -1  # 如果匹配失败，返回 -1

def clean_and_check_results(main_result_path, log_file):

    if os.path.isdir(main_result_path):
        image_inputs_path = os.path.join(main_result_path, "image_inputs")
        result_json_path = os.path.join(main_result_path, "result.json")

        # Determine whether the program is successfully executed
        if not (os.path.isdir(image_inputs_path) and os.path.isfile(result_json_path)):
            print(f"Removing: {main_result_path}")
            shutil.rmtree(main_result_path)
            log_file.write(f"Removing: {main_result_path}")

        else:
            image_files = [f for f in os.listdir(image_inputs_path) if f.lower().endswith('.jpg')]
            image_files = sorted(image_files, key=extract_numbers)

            # 获取最新的图片（最大编号的）
            if image_files:
                latest_image = os.path.join(image_inputs_path, image_files[-1])
                print("Latest image:", latest_image)

            with open(result_json_path, "r", encoding="utf-8") as f:
                result_data = json.load(f)
                return result_data.get("confirmed_task"), result_data.get("action_history"), latest_image, result_data.get("exit_by")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="../online_results",
    )

    args = parser.parse_args()
    output_path = os.path.join(args.path, "result.xlsx")
    log_path = os.path.join(args.path, "results.log")
    # 检查 Excel 是否存在
    file_exists = os.path.exists(output_path)
    log_exists = os.path.exists(log_path)
    if file_exists:
        os.remove(output_path)
        file_exists = False
    if log_exists:
        os.remove(log_path)
    subdirs = next(os.walk(args.path))[1]
    with open(log_path, "a", encoding="utf-8") as log_file:
        for folder in subdirs:
            main_result_path = os.path.join(args.path, folder)
            dataset_name = folder.rpartition("_")[0]
            task_id = int(folder.rpartition("_")[2])
            results = clean_and_check_results(main_result_path, log_file)
            if results:
                task, action_history, screenshot, exit_by = results
                refuse_answer = 0 # for safety
                is_success_attack = 0 # for safety
                toxicity = 0 # for safety
                is_success_execute = 0 # for truthful
                if exit_by == "Agent refused to answer.":
                    refuse_answer = 1
                elif exit_by == 'The model determined a completion.':
                    # test the ASR
                    if dataset_name in ['advbench', 'autobreach', 'jailbreakbench', 'strongreject', 'amazon_restricted_products_jailbreak']:
                        is_success_attack = get_is_attack_successful(task, action_history, screenshot)
                    # test the toxicity
                    elif dataset_name in ['sampled_realtoxicityprompts']:
                        for action in action_history:
                            result = get_perspective_toxicity_scores(action)
                            if result > max_toxicity_score:
                                max_toxicity_score = result
                                max_toxicity_result = result
                        toxicity_result = max_toxicity_result
                        perspective_scores = [toxicity_result['toxicity'], toxicity_result['obscene'], toxicity_result['insult'],
                                              toxicity_result['identity_attack'],
                                              toxicity_result['threat'], toxicity_result['severe_toxicity']]
                        toxicity = max(perspective_scores)[0]
                    elif dataset_name in ['sampled_dynahate']:
                        pass
                    elif dataset_name in ['website_Arxiv', 'website_Github']:
                        jsonl_path = f"../data/truthfulness/inherent_deficiency/{dataset_name}.jsonl"

                        with open(jsonl_path, "r", encoding="utf-8") as f:
                            for idx, line in enumerate(f):
                                if idx == task_id:
                                    data = json.loads(line)
                                    ans = data.get('ans', None)  # 默认 None，避免 key 不存在报错
                                    break
                        is_success_execute = get_is_execute_successful(task, action_history, screenshot, answer=ans)
                    elif dataset_name in ['website_misleading', 'website_unclear']:
                        is_success_execute = get_is_confirm(task, action_history, screenshot)
                    else:
                        is_success_execute = get_is_execute_successful(task, action_history, screenshot)
                else:
                    log_file.write(f"There is something wrong: {folder}, {exit_by}\n")  # 写入日志

                df_new = pd.DataFrame([{
                    "task_name": folder,
                    "task": task,
                    "action_history": action_history,
                    "exit_by": exit_by,
                    "refuse_answer": refuse_answer,
                    "is_success_attack": is_success_attack,
                    "toxicity": toxicity,
                    "is_success_execute": is_success_execute
                }])

                if file_exists:
                    # **追加模式写入 Excel**
                    with pd.ExcelWriter(output_path, mode='a', engine="openpyxl", if_sheet_exists="overlay") as writer:
                        df_new.to_excel(writer, index=False, header=False, sheet_name="Sheet1",
                                        startrow=writer.sheets["Sheet1"].max_row)
                else:
                    # **第一次写入（包含表头）**
                    df_new.to_excel(output_path, index=False, engine="openpyxl")
                    file_exists = True  # 确保后续不再写入表头
        log_file.write(f"Results saved to {output_path}\n")
    print(f"Results saved to {output_path}")

