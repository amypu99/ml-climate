#!/usr/bin/env bash

# Loop over every .pdf file in cases_temp/DNMS
for file in climate_reports/ccrm_2024/*.pdf
do
  # Run the Python pipeline command on the current file
  python3 -m olmocr.pipeline climate_reports/ccrm_2024_olmocr --pdfs "$file"
done