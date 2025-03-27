import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
import json
import gc
import re
import math

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from transformers import pipeline

CHUNK_LEN = 2000
MAX_SEQ_LEN = 8000

def climategpt_setup():
    model_name = "eci-io/climategpt-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


model, tokenizer = climategpt_setup()
pipe = pipeline("text-generation", model=model, max_new_tokens=200, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
pipe.model = pipe.model.to('cuda')


f = open("./climate_reports/extracted_text/amazon-2021-sustainability-report_cleaned.txt")
text = f.read()
tokenized_document = tokenizer(text, return_tensors='pt' ).to('cuda')
document_length = len(tokenized_document['input_ids'][0][1:-1])
num_chunks = math.ceil(document_length/CHUNK_LEN)
summary_len = MAX_SEQ_LEN//num_chunks

for i in range(num_chunks):
    tokenized_chunk = tokenized_document['input_ids'][0][i*CHUNK_LEN:(i+1)*CHUNK_LEN]
    decoded_chunk = tokenizer.decode(tokenized_chunk)
    messages = [{"role": "user", "content": f"Read the following corporate climate report, summarize the document, and include all important information and statistics related to sustainability and climate. \n\n{decoded_chunk}"},
    ]
    print(pipe(messages, max_new_tokens=summary_len))
    print("\n\n\n\n\n")