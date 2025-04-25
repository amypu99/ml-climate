import os
import pandas as pd
import csv

def get_remaining_companies_by_year():
    
    all_existing_sources = set()
    for year in ["2022", "2023", "2024"]:
      document_folder = f"climate_reports/ccrm_{year}_olmocr/results/"

      for filename in os.listdir(document_folder):
          with open(document_folder+filename, "r") as f:
              df = pd.read_json(f)
              source = df.metadata["Source-File"].replace(".pdf", "").replace(f"climate_reports/ccrm_{year}/", "")
              all_existing_sources.add(source)


    all_merged_companies = set()

    with open("merge_data/merged_ccrm_tp_filtered.csv", "r") as f:
        csvreader = csv.reader(f)
        header = next(csvreader)
        for i, row in enumerate(csvreader):
            company = "_".join(row[0].lower().split(" "))
            year = str(int(row[1])-2)
            full_source = "_".join([company,year])
            all_merged_companies.add(full_source)

    return all_merged_companies.difference(all_existing_sources)

if __name__ == "__main__":

    missing_companies = get_remaining_companies_by_year()
    print(sorted(missing_companies))