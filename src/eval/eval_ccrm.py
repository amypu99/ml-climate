import glob
import pandas as pd
import numpy as np
import regex as re
import os
from sklearn.metrics import accuracy_score,average_precision_score, precision_recall_curve,recall_score,precision_score, f1_score
import csv


LABELS_FILE = "../merge_data/final_w_training.csv"

label_mapping = {
    "very poor": 0,
    "poor": 1,
    "moderate": 2,
    "reasonable": 3,
    "high": 4,
    "unknown": -1
}

score_columns = [
        'Overall transparency score',
        'Overall integrity score',
        '1. Transparency and Integrity',
        '2. Transparency',
        '2. Integrity',
        '3. Transparency',
        '3. Integrity',
        '4. Transparency',
        '4. Integrity'
    ]

label_columns = [
    'Overall transparency score (numeric)',
    'Overall integrity score (numeric)',
    '1. Transparency and Integrity (numeric)',
    '2. Transparency (numeric)',
    '2. Integrity (numeric)',
    '3. Transparency (numeric)',
    '3. Integrity (numeric)',
    '4. Transparency (numeric)',
    '4. Integrity (numeric)'
]
pred_columns = [
    'Overall transparency score (pred)',
    'Overall integrity score (pred)',
    '1. Transparency and Integrity (pred)',
    '2. Transparency (pred)',
    '2. Integrity (pred)',
    '3. Transparency (pred)',
    '3. Integrity (pred)',
    '4. Transparency (pred)',
    '4. Integrity (pred)'
]

# Helper function to extract the first label found
def extract_label(text):
    # Normalize text
    text = text.lower()
    # Try to find a label inside the text
    for label in label_mapping.keys():
        if re.search(r'\b' + re.escape(label) + r'\b', text):
            return label_mapping[label]
    return -2

def run_model(model):
    results_folder = f"../all_results/{model}/ccrm/"
    mapped_results = {}
    temp_df = pd.DataFrame(columns=["source"] + pred_columns)

    for file in os.listdir(results_folder):
        # print(results_folder+file)
        file_results = pd.read_json(results_folder+file, lines=True)
        sources = file_results.source.to_list()
        mapped_results["source"] = sources
        # ["0", "1", "2", "3", "4", "5", "6","7", "8", "9"]?
        for i, question_idx in enumerate(["0", "1", "2", "3", "4", "5", "6","7", "8"]):
            mapped_results[pred_columns[i]]  = file_results[question_idx].map(extract_label).to_list()
        temp_df = temp_df.merge(pd.DataFrame.from_dict(mapped_results), how='outer')
    
    return temp_df


def create_new_labels_file():

    # Load the original CSV
    input_file = LABELS_FILE  # <-- change if your file has a different name
    output_file = "../merge_data/final_converted_scores.csv"

    df = pd.read_csv(input_file)

    # Function to map the labels safely
    def map_label(value):
        if pd.isnull(value):
            return -1
        return label_mapping.get(str(value).strip().lower(),-2)

    # For each score column, create a new column with the numeric value
    for col in score_columns:
        new_col = col + " (numeric)"
        df[new_col] = df[col].apply(map_label)

    # Save the updated DataFrame
    df.to_csv(output_file, index=False)

    print(f"Finished writing output to {output_file}")


def compare_ccrm_model(model):
    predictions_df = run_model(model)
    labels_df = pd.read_csv("../merge_data/final_converted_scores.csv")
    joined_df = predictions_df.merge(labels_df, how='inner', left_on='source', right_on='training_idx',)
    accuracy = {}
    macro_precision = {}
    micro_precision = {}
    micro_recall = {}
    macro_recall = {}
    micro_f1 = {}
    macro_f1 = {}
    for i, col in enumerate(score_columns):
        predictions = joined_df[pred_columns[i]].to_list()
        labels = joined_df[label_columns[i]].to_list()
        accuracy[col] = accuracy_score(labels, predictions)
        macro_precision[col] = precision_score(labels, predictions, average='macro', zero_division=0)
        micro_precision[col] = precision_score(labels, predictions, average='micro', zero_division=0)
        macro_recall[col] = recall_score(labels, predictions, average='macro', zero_division=0)
        micro_recall[col] = recall_score(labels, predictions, average='micro', zero_division=0)
        micro_f1[col] = f1_score(labels, predictions, average='micro', zero_division=0)
        macro_f1[col] = f1_score(labels, predictions, average='macro', zero_division=0)

    
    with open(f"ccrm_eval_{model}.csv", "w") as f:
        writer = csv.writer(f)
        for key,item in accuracy.items():
            writer.writerow([key, "accuracy",item])
        for key,item in macro_precision.items():
            writer.writerow([key, "macro_precision",item])
        for key,item in micro_precision.items():
            writer.writerow([key, "micro_precision",item])
        for key,item in macro_recall.items():
            writer.writerow([key, "macro_recall",item])
        for key,item in micro_recall.items():
            writer.writerow([key, "micro_recall",item])
        for key,item in macro_f1.items():
            writer.writerow([key, "macro_f1",item])
        for key,item in micro_f1.items():
            writer.writerow([key, "micro_f1",item])




if __name__ == "__main__":
    # convert_labels()
    # predictions_df = run_model("climategpt-7b")
    # create_new_labels_file()
    # compare_ccrm_model("climategpt-7b")
    compare_ccrm_model("ministral-8B")
    compare_ccrm_model("qwen")