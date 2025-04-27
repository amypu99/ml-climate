import glob
import pandas as pd
import numpy as np
import regex as re
import os

LABELS_FILE = "../merge_data/final_w_training.csv"

label_mapping = {
    "very poor": 0,
    "poor": 1,
    "moderate": 2,
    "reasonable": 3,
    "high": 4,
    "unknown": np.nan  # or use -1 if you prefer
}

# Helper function to extract the first label found
def extract_label(text):
    # Normalize text
    text = text.lower()
    # Try to find a label inside the text
    for label in label_mapping.keys():
        if re.search(r'\b' + re.escape(label) + r'\b', text):
            return label_mapping[label]
    return np.nan


def convert_labels():
    ccrm_files = glob.glob('../all_results/ccrm*', recursive=True)
    mapped_results = {}
    temp_df = pd.DataFrame(columns=["source", "0", "1", "2", "3", "4", "5", "6","7", "8", "9"])
    for ccrm_file in ccrm_files:
        print(ccrm_file)
        with open(ccrm_file, "r") as file:
            file_results = pd.read_json(file, lines=True)
            sources = file_results.source.to_list()
            mapped_results["source"] = sources
            for question_idx in ["0", "1", "2", "3", "4", "5", "6","7", "8", "9"]:
                mapped_results[question_idx] = file_results[question_idx].map(extract_label).to_list()
            temp_df = temp_df.merge(pd.DataFrame.from_dict(mapped_results), how='outer')
    return temp_df

def run_model(model):
    results_folder = f"../all_results/{model}/"
    mapped_results = {}
    temp_df = pd.DataFrame(columns=["source", "0", "1", "2", "3", "4", "5", "6","7", "8", "9"])

    for file in os.listdir(results_folder):
        # print(results_folder+file)
        file_results = pd.read_json(results_folder+file, lines=True)
        sources = file_results.source.to_list()
        mapped_results["source"] = sources
        for question_idx in ["0", "1", "2", "3", "4", "5", "6","7", "8", "9"]:
            mapped_results[question_idx] = file_results[question_idx].map(extract_label).to_list()
        temp_df = temp_df.merge(pd.DataFrame.from_dict(mapped_results), how='outer')
        print(temp_df)
        break

def create_new_labels_file():

    # Load the original CSV
    input_file = LABELS_FILE  # <-- change if your file has a different name
    output_file = "../merge_data/final_converted_scores.csv"

    df = pd.read_csv(input_file)

    # Function to map the labels safely
    def map_label(value):
        if pd.isnull(value):
            return np.nan
        return label_mapping.get(str(value).strip().lower(), np.nan)

    # Columns you want to map
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

    # For each score column, create a new column with the numeric value
    for col in score_columns:
        new_col = col + " (numeric)"
        df[new_col] = df[col].apply(map_label)

    # Save the updated DataFrame
    df.to_csv(output_file, index=False)

    print(f"Finished writing output to {output_file}")


if __name__ == "__main__":
    # convert_labels()
    # run_model("climategpt-7b")
    create_new_labels_file()