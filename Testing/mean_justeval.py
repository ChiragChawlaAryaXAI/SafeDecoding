import json
import numpy as np

file_path = "/home/jovyan/SafeDecoding/exp_outputs/SafeDecoding_llama2_Just-Eval_50_2025-12-16 05:44:01/SafeDecoding_llama2_Just-Eval_50_2025-12-16 05:44:01_safe_eval.json"

with open(file_path, "r") as f:
    data = json.load(f)

per_sample_means = []

for item in data:
    scores = [
        int(item["parsed_result"]["helpfulness"]["score"]),
        int(item["parsed_result"]["clarity"]["score"]),
        int(item["parsed_result"]["factuality"]["score"]),
        int(item["parsed_result"]["depth"]["score"]),
        int(item["parsed_result"]["engagement"]["score"]),
    ]
    per_sample_means.append(np.mean(scores))

overall_mean = np.mean(per_sample_means)

print(f"Per-sample mean scores (first 5): {per_sample_means[:5]}")
print(f"Overall Just-Eval score: {overall_mean:.4f}")
