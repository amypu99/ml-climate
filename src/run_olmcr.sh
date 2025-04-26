#!/usr/bin/env bash

# Loop over every .pdf file in cases_temp/DNMS
for file in climate_reports/round_3_reports/*.pdf
do
  # Run the Python pipeline command on the current file
  python3 -m olmocr.pipeline climate_reports/round_3_reports_olmocr --pdfs "$file"
done