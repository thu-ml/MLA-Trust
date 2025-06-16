from collections import defaultdict
from glob import glob

import pandas as pd

paths = glob("logs/mobile/**/result.csv", recursive=True)

for path in paths:
    print(path)

key_cols = ["refuse_answer", "is_success_attack", "toxicity", "is_success_execute"]
stat = defaultdict(dict)
support = defaultdict(dict)

for path in paths:
    model, run_name, task = path.split("/")[1:4]
    if run_name != "two_device":
        continue
    # df = pd.read_excel(path)
    df = pd.read_csv(path)
    support[task][model] = len(df)

    def str2int(x):
        if isinstance(x, str):
            if "0" in x:
                return 0
            elif "1" in x:
                return 1
            else:
                raise ValueError
        else:
            return x

    df["is_success_attack"] = df["is_success_attack"].apply(str2int)
    stat_df = df[key_cols].mean().to_dict()

    stat[task][model] = stat_df

stats = {}
for task, models in stat.items():
    print(task.center(40, "="))
    stat_df = pd.DataFrame(models).T

    for col in key_cols:
        if col in stat_df.columns:
            stat_df[col] = stat_df[col].apply(lambda x: f"{x * 100:.1f}%")

    stat_df["support"] = pd.Series(
        {model: support[task][model] for model in stat_df.index}
    )

    print(stat_df)
    stats[task] = stat_df

# save to excel
order = [
    "mobile_Chrome",
    "mobile_Notes",
    "mobile_cross_app",
    "mobile_unclear",
    "mobile_misleading",
    "sampled_dynahate",
    "sampled_realtoxicityprompts",
    "various_apps_jailbreak",
    "jailbreakbench",
    "strongreject",
    "advbench",
    "autobreach",
    "mobile_direct_awareness",
    "mobile_indirect_awareness",
    "mobile_direct_leakage",
    "mobile_indirect_leakage",
    "mobile_note_speculative_risk_other",
    "mobile_note_overcompletion_gcg",
    "mobile_note_overcompletion_other",
    "mobile_note_speculative_risk_gcg",
    "mobile_gmail_overcompletion_gcg",
    "mobile_gmail_speculative_risk_other",
    "mobile_gmail_overcompletion_other",
    "mobile_gmail_speculative_risk_gcg",
]

order_index = {model: i for i, model in enumerate(order)}

sorted_stats = {
    task: stats[task] for task in sorted(stats.keys(), key=lambda x: order_index[x])
}

writer = pd.ExcelWriter("mobile_stat.xlsx")
for task, stat_df in sorted_stats.items():
    stat_df.to_excel(writer, sheet_name=task)
writer.close()
