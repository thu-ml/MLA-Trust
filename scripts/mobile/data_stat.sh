#!/usr/bin/env bash

files=(
  data/mobile/controllability/overcompletion/mobile_gmail_overcompletion.jsonl
  data/mobile/controllability/overcompletion/mobile_note_overcompletion.jsonl
  data/mobile/controllability/speculative_risk/mobile_gmail_speculative_risk.jsonl
  data/mobile/controllability/speculative_risk/mobile_note_speculative_risk.jsonl
  data/mobile/privacy/privacy_awareness/mobile_direct_awareness.jsonl
  data/mobile/privacy/privacy_awareness/mobile_indirect_awareness.jsonl
  data/mobile/privacy/privacy_leakage/mobile_direct_leakage.jsonl
  data/mobile/privacy/privacy_leakage/mobile_indirect_leakage.jsonl
  data/mobile/safe_data/advbench.jsonl
  data/mobile/safe_data/autobreach.jsonl
  data/mobile/safe_data/jailbreakbench.jsonl
  data/mobile/safe_data/sampled_dynahate.jsonl
  data/mobile/safe_data/sampled_realtoxicityprompts.jsonl
  data/mobile/safe_data/strongreject.jsonl
  data/mobile/safe_data/various_apps_jailbreak.jsonl
  data/mobile/truthfulness/inherent_deficiency/mobile_Chrome.jsonl
  data/mobile/truthfulness/inherent_deficiency/mobile_cross_app.jsonl
  data/mobile/truthfulness/inherent_deficiency/mobile_Notes.jsonl
  data/mobile/truthfulness/misguided_mistakes/mobile_misleading.jsonl
  data/mobile/truthfulness/misguided_mistakes/mobile_unclear.jsonl
)

for f in "${files[@]}"; do
  if [[ -f "$f" ]]; then
    lines=$(awk 'END {print NR}' "$f")
    printf "%7d  %s\n" "$lines" "$f"
  else
    echo "MISSING  $f"
  fi
done
