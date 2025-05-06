import glob
import pandas as pd
import numpy as np
import regex as re
import os
from sklearn.metrics import accuracy_score,average_precision_score, precision_recall_curve,recall_score,precision_score, f1_score
import csv


LABELS_FILE = "../merge_data/updated_final_interpolated.csv"

label_columns = ["Q1L0 (numeric)","Q2L1 (numeric)","Q3L1 (numeric)","Q4L2 (numeric)","Q5L2 (numeric)","Q6L3 (numeric)", "Q7L3 (numeric)","Q8L3 (numeric)","Q9L3 (numeric)","Q10L3 (numeric)","Q11L3 (numeric)","Q12L3 (numeric)","Q13L4 (numeric)","Q14L4 (numeric)","Q15L4 (numeric)","Q16L4 (numeric)","Q17L4 (numeric)","Q18L4 (numeric)","Q19L5 (numeric)","Q20L5 (numeric)","Q21L5 (numeric)","Q22L5 (numeric)","Q23L5 (numeric)"]
pred_columns = ["Q1L0","Q2L1","Q3L1","Q4L2","Q5L2","Q6L3", "Q7L3","Q8L3","Q9L3","Q10L3","Q11L3","Q12L3","Q13L4","Q14L4","Q15L4","Q16L4","Q17L4","Q18L4","Q19L5","Q20L5","Q21L5","Q22L5","Q23L5"]

label_mapping = {
    "yes": 1,
    "no": 0,
    "not applicable": -1,
}
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
    results_folder = f"../all_results/{model}/tp/"
    mapped_results = {}
    temp_df = pd.DataFrame(columns=["source"] + pred_columns)

    for file in os.listdir(results_folder):
        # print(results_folder+file)
        file_results = pd.read_json(results_folder+file, lines=True)
        sources = file_results.source.to_list()
        mapped_results["source"] = sources
        for i, question_idx in enumerate([str(x) for x in range(23)]):
            mapped_results[pred_columns[i]]  = file_results[question_idx].map(extract_label).to_list()
        temp_df = temp_df.merge(pd.DataFrame.from_dict(mapped_results), how='outer')
    
    # print(temp_df)
    
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


def compare_tp_model(model):
    predictions_df = run_model(model)
    labels_df = pd.read_csv("../merge_data/updated_final_interpolated.csv")
    joined_df = predictions_df.merge(labels_df, how='inner', left_on='source', right_on='doc_idx',)
    accuracy = {}
    macro_precision = {}
    micro_precision = {}
    micro_recall = {}
    macro_recall = {}
    micro_f1 = {}
    macro_f1 = {}
    for i, col in enumerate(label_columns):
        predictions = joined_df[pred_columns[i]].to_list()
        labels = joined_df[label_columns[i]].to_list()
        accuracy[col] = accuracy_score(labels, predictions)
        macro_precision[col] = precision_score(labels, predictions, average='macro', zero_division=0)
        micro_precision[col] = precision_score(labels, predictions, average='micro', zero_division=0)
        macro_recall[col] = recall_score(labels, predictions, average='macro', zero_division=0)
        micro_recall[col] = recall_score(labels, predictions, average='micro', zero_division=0)
        micro_f1[col] = f1_score(labels, predictions, average='micro', zero_division=0)
        macro_f1[col] = f1_score(labels, predictions, average='macro', zero_division=0)
        

    with open(f"tp_eval_{model}.csv", "w") as f:
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
    compare_tp_model("qwen")
    compare_tp_model("ministral-8B")
    # compare_tp_model("climategpt-7b")