import os
import pandas as pd
import json
import csv

def create_indices_ccrm():
    for year in ["2022", "2023", "2024"]:
        document_folder = f"climate_reports/ccrm_{year}_olmocr/results/"

        for filename in os.listdir(document_folder):
            with open(document_folder+filename, "r") as f:
                data = json.load(f)
                source = data["metadata"]["Source-File"].replace(".pdf", "").replace(f"climate_reports/ccrm_{year}/", "")
                data["source"] = source
                with open(f"climate_reports/ccrm_{year}_olmocr/indexed/"+filename, "w") as outfile:
                    outfile.write(json.dumps(data))

def create_indices_round_2_reports():
    document_folder = f"climate_reports/round_2_reports_olmocr/results/"

    for filename in os.listdir(document_folder):
        with open(document_folder+filename, "r") as f:
            data = json.load(f)
            source = data["metadata"]["Source-File"].replace(".pdf", "").replace(f"climate_reports/additional_reports/", "")
            data["source"] = source
            with open(f"climate_reports/round_2_reports_olmocr_indexed/"+filename, "w") as outfile:
                outfile.write(json.dumps(data))

def count_training_data():
    training_examples = set()
    for year in ["2022", "2023", "2024"]:
        document_folder = f"climate_reports/ccrm_{year}_olmocr/indexed/"
        for filename in os.listdir(document_folder):
            training_examples.add(filename.replace(document_folder, "").replace(".jsonl", ""))
        document_folder = f"climate_reports/round_2_reports_olmocr_indexed/{year}/"
        for filename in os.listdir(document_folder):
            training_examples.add(filename.replace(document_folder, "").replace(".jsonl", ""))
    document_folder = f"climate_reports/tp_reports/"
    for filename in os.listdir(document_folder):
        training_examples.add(filename.replace(document_folder, "").replace(".pdf", ""))
    print(training_examples)
    import csv

    with open('training_idx.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for item in training_examples:
            writer.writerow([item])
    # print(len(training_examples))

def update_labels():
    # with open("merge_data/final.csv", "r") as f:
    # def map_func(company, year)


    df = pd.read_csv("merge_data/final.csv")
    # x = df["Company"].str.lower()
    df['training_idx'] = df['Company'].str.lower() + '_' + (df['Year'] - 2).astype(str)
    df.to_csv("merge_data/final_w_training.csv", index=False)
    # print(df['training_idx'])



if __name__ == "__main__":
    # create_indices_ccrm()
    # create_indices_round_2_reports()
    count_training_data()
    # update_labels()