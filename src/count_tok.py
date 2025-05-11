import os
import json
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("eci-io/ClimateGPT-7B")

def count_tokens_in_jsonl(file_path):
    total_tokens = 0
    df_file = pd.read_json(file_path, lines=True)
    text = df_file["text"].to_list()[0]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens += len(tokens)
    return total_tokens

def process_directory(directory_path):
    all_tokens_in_dir = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jsonl"):
            full_path = os.path.join(directory_path, filename)
            token_count = count_tokens_in_jsonl(full_path)
            all_tokens_in_dir.append(token_count)
            print(f"{filename}: {token_count} tokens")
    return all_tokens_in_dir

jsonl_directory = "./climate_reports/ccrm_2022_olmocr/"
ccrm_2022_olmocr = process_directory(jsonl_directory)

jsonl_directory = "./climate_reports/ccrm_2023_olmocr/"
ccrm_2023_olmocr = process_directory(jsonl_directory)

jsonl_directory = "./climate_reports/ccrm_2024_olmocr/"
ccrm_2024_olmocr = process_directory(jsonl_directory)

jsonl_directory = "./climate_reports/round_2_reports_olmocr/2022/"
round_2_reports_olmocr_2022 = process_directory(jsonl_directory)

jsonl_directory = "./climate_reports/round_2_reports_olmocr/2023/"
round_2_reports_olmocr_2023 = process_directory(jsonl_directory)

jsonl_directory = "./climate_reports/round_2_reports_olmocr/2024/"
round_2_reports_olmocr_2024 = process_directory(jsonl_directory)

jsonl_directory = "./climate_reports/tp_reports_olmocr/"
tp_reports_olmocr = process_directory(jsonl_directory)

all_counts = ccrm_2022_olmocr + ccrm_2023_olmocr + ccrm_2024_olmocr + round_2_reports_olmocr_2022 + round_2_reports_olmocr_2023 + round_2_reports_olmocr_2024 + tp_reports_olmocr

print("average", np.average(all_counts))
print("median", np.median(all_counts))
print("max", max(all_counts))
print("min", min(all_counts))

df_describe = pd.DataFrame(all_counts)
print(df_describe.describe())