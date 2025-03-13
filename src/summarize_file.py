import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
import json
import gc
import re

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from transformers import pipeline

f = open("./climate_reports/extracted_text/amazon-2021-sustainability-report_cleaned.txt")
truncated = ' '.join(f.read().split(" ")[:1000])

messages = [
    {"role": "user", "content": truncated + " \nSummarize the document above and include all important information related to sustainability and climate."},
]
pipe = pipeline("text-generation", model="eci-io/climategpt-7b", max_new_tokens=200)
print(pipe(messages))