import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import pipeline
import json
import gc
import re
import math
from glob import glob
import os

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


def apply_prompt(chunk_text, question):
    full_prompt = (
        f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
    )

    return full_prompt

path = 'climate_reports/ccrm_2024_olmocr/results/'
files = list(glob(os.path.join(path, "*.jsonl")))


questions_csv = pd.read_csv('ccrm_2024_questions.csv')  
questions = questions_csv.Question.to_list()


for file_path in files:

    text = pd.read_json(file_path).text
    chunked_content = text.split("\n\n")
    for question
            
    all_prompts = [apply_prompt(chunk) for chunk in chunked_content]
    # print("\n\n\n".join(all_prompts))
    all_messages = [[{"role": "user", "content": x}] for x in all_prompts]



    tokenized_document = tokenizer(text, return_tensors='pt' ).to('cuda')
    
    document_length = len(tokenized_document['input_ids'][0][1:-1])
    num_chunks = math.ceil(document_length/CHUNK_LEN)
    summary_len = MAX_SEQ_LEN//num_chunks

    for i in range(num_chunks):
        tokenized_chunk = tokenized_document['input_ids'][0][i*CHUNK_LEN:(i+1)*CHUNK_LEN]
        decoded_chunk = tokenizer.decode(tokenized_chunk)
        for question in questions:
            content = f"Context information is below.\n---------------------\n{decoded_chunk}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
            # content = f"Read the following corporate climate report and answer the question. {question}\n\n{decoded_chunk}"
            messages = [{"role": "user", "content": content},
            ]
            print(file_path)
            print(pipe(messages, max_new_tokens=summary_len))
            print("\n\n\n\n\n")
            break
        break