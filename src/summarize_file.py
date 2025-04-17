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

# Comment if not running on GPU. Alternatively, change your visible device
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from transformers import pipeline

MAX_SEQ_LEN = 25000
BATCH_SIZE = 10

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
        f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nResponse:"
    )

    return full_prompt

path = 'climate_reports/ccrm_2024_olmocr/results/'
files = list(glob(os.path.join(path, "*.jsonl")))


from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter


document_folder = "/home/amy_pu/ml-climate/src/climate_reports/ccrm_2024_olmocr/results/"
# output_3d593a7d821f2abbf4d54ee7fd048d77e27af537.jsonl'
loader = DirectoryLoader(document_folder, glob='output_3d593a7d821f2abbf4d54ee7fd048d77e27af537.jsonl', show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'content_key': 'text', 'json_lines': True,})
# loader = DirectoryLoader(document_folder, glob='*.jsonl', show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'content_key': 'text', 'json_lines': True,})
documents = loader.load()
print(f'document count: {len(documents)}')
text_splitter = CharacterTextSplitter(separator="\n\n",
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
from collections import defaultdict
chunked_documents = text_splitter.split_documents(documents)
question = "Summarize the document above in 100 tokens or less and include all important information related to sustainability and climate. If there is no information about sustainability and cilmate, return an empty response."

# print(chunked_documents[0])
groupings = defaultdict(list)
for chunk in chunked_documents:
    groupings[chunk.metadata['source'].replace('/home/amy_pu/ml-climate/src/climate_reports/ccrm_2024_olmocr/results/', '')].append(chunk.page_content)
# print(len(chunked_documents))
# print(len(chunked_documents.page_content))
# print(groupings['output_3d593a7d821f2abbf4d54ee7fd048d77e27af537.jsonl'][0])

NUM_CHUNKS = len(chunked_documents)
OUTPUT_SEQ_LEN = max(50, MAX_SEQ_LEN//NUM_CHUNKS)

# for each training example

def replace_after_last_period(text):
    last_period_index = text.rfind('.')
    if last_period_index == -1:
        return text
    else:
        return text[:last_period_index + 1]
    
from datetime import datetime

# current_time = datetime.now()
# print("Current Time:", current_time.strftime("%H:%M:%S"))

all_results = []
for key in groupings.keys():
    chunked_content = groupings[key] 
    all_prompts = [apply_prompt(chunk, question) for chunk in chunked_content]
    all_messages = [[{"role": "user", "content": x}] for x in all_prompts]
    print(len(all_messages))
    for i in range(0, len(all_messages), BATCH_SIZE):
        batch_messages = all_messages[i:i+BATCH_SIZE]
        results = pipe(batch_messages, max_new_tokens=OUTPUT_SEQ_LEN)
        # print(results[0])
        # print(results[0][0]['generated_text'][1]['content'])
        generated_texts = [result[0]['generated_text'][1]['content'] for result in results]
        all_results += generated_texts

cleaned_results = list(map(replace_after_last_period, all_results))
summarized_document = '\n'.join(cleaned_results)

with open("test_summarized_file.txt", "w") as file:
    file.write(summarized_document)
# current_time = datetime.now()
# print("Current Time:", current_time.strftime("%H:%M:%S"))

#takes 2 min to do 20 examples