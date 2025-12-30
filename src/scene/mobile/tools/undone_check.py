import json
import shutil
from collections import defaultdict
from glob import glob
from pprint import pprint
from typing import DefaultDict

run_name = "ijcv"
paths = glob(f"logs/**/{run_name}/**/steps.json", recursive=True)
include_list = [
    # "Phi",
    # "InternVL2",
    # "Qwen",
]
exclusion_list = [
    # "website",
    # "Magma-8B",
    # "advbench",
    # "autobreach",
    # "jailbreakbench",
    # "mobile_Chrome",
    # "mobile_cross_app",
    # "mobile_misleading",
    # "mobile_Notes",
    # "mobile_unclear",
    # "sampled_dynahate",
    # "sampled_realtoxicityprompts",
    # "strongreject",
    # "various_apps_jailbreak",
    # "mobile_direct_awareness",
    # "mobile_direct_leakage",
    # "mobile_indirect_awareness",
    # "mobile_indirect_leakage",
]
# paths = [p for p in paths if any(e in p for e in include_list)]
# paths = [p for p in paths if all(e not in p for e in exclusion_list)]

model_count: DefaultDict[str, int] = defaultdict(int)
task_count: DefaultDict[str, dict] = defaultdict(dict)

for path in paths:
    model_name = path.removeprefix("logs/").split("/")[0]
    task_name = path.removeprefix("logs/").split("/")[2]
    if task_name not in task_count[model_name]:
        task_count[model_name][task_name] = 0

    task_count[model_name][task_name] += 1
    model_count[model_name] += 1

pprint(model_count)
pprint(task_count)

clear = False
# clear = True

done_cnt = 0
undone_cnt = 0
error_cnt = 0

undone_list = []
error_list = []
for path in paths:
    with open(path, "r") as f:
        try:
            steps = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {path}")
            continue

        last_step = steps[-1]
        if last_step.get("task_duration") is not None:
            done_cnt += 1
        elif (
            last_step.get("refuse") is not None
            and last_step.get("detailed_error") is None
        ):
            done_cnt += 1
        else:
            if last_step.get("detailed_error") is not None:
                error_list.append(path)
                error_cnt += 1
            else:
                undone_list.append(path)
                undone_cnt += 1

if clear:
    for path in undone_list + error_list:
        shutil.rmtree(path.rsplit("/", 1)[0])

for undone in undone_list:
    print(f"undone: {undone}")


print(f"error: {error_cnt}")
print(f"undone: {undone_cnt}")
print(f"done: {done_cnt}")
