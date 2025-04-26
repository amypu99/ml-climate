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

if __name__ == "__main__":
    # create_indices_ccrm()
    create_indices_round_2_reports()