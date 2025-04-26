#!/usr/bin/env bash

# Loop over every .pdf file in cases_temp/DNMS
for file in climate_reports/round_3_reports/unilever_2020.pdf
do
  # Run the Python pipeline command on the current file
  python3 -m olmocr.pipeline climate_reports/round_3_reports_olmocr/unilever_2020.jsonl --pdfs climate_reports/round_3_reports/unilever_2020.pdf
done