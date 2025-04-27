#!/usr/bin/env bash

# Loop over every .pdf file in cases_temp/DNMS
for file in climate_reports/tp_reports/*.pdf
do
  # Run the Python pipeline command on the current file
  python3 -m olmocr.pipeline climate_reports/tp_reports_olmocr --pdfs "$file" --workers=6
done