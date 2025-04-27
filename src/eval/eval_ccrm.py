import glob
import pandas as pd
import numpy as np
import regex as re

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
    ccrm_files = glob.glob('../results/ccrm*', recursive=True)
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

if __name__ == "__main__":
    convert_labels()