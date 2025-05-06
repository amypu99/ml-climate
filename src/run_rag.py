# for environment, 
# pip install -U langchain sentence-transformers faiss-cpu openpyxl pacmap datasets ragatouille
# pip install -U langchain-community
# pip install jq
# pip install chromadb
# pip install -U langchain-core


import os
import pandas as pd
import json
import torch
from itertools import accumulate
from transformers import (
    pipeline
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from model_setup import climategpt_7b_setup, climategpt_13b_setup, climatellmama_8b_setup, qwen_setup, ministral_8b_it_setup, mistral_7b_it_setup, climte_nlp_longformer_detect_evidence_setup, longformer_setup, led_base_setup, gemma_setup


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders import DirectoryLoader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CHUNK_LEN = 2200

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def filter_by_company_year(document, index):
    if document.metadata["index"] == index: 
        return True
    return False

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["index"] = record.get("source")
    return metadata

def length_function(text: str, tokenizer) -> int:
    return len(tokenizer(text)["input_ids"])

def setup_vector_store(document_folder, glob, tokenizer):
    loader = DirectoryLoader(document_folder, glob=glob, show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'content_key': 'text', 'json_lines': True,'metadata_func': metadata_func})
    documents = loader.load()
    print(f'document count: {len(documents)}')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_LEN,
        chunk_overlap=200,
        length_function=lambda text: length_function(text, tokenizer=tokenizer),
        is_separator_regex=False,
        strip_whitespace=True,
    )
    chunked_documents = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunked_documents)
    return vector_store

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_report(all_chunk_text):
    full_text = f"\n\n\n\n\n—— REPORT START ——\n{all_chunk_text}—— REPORT END ——"
    return full_text

def query_model(model, tokenizer, query, docs):
    pipe = pipeline("text-generation", model=model, max_new_tokens=256, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer,temperature=0.7, do_sample=True)
    # pipe = pipeline("text-generation", model=model, max_new_tokens=256, torch_dtype=torch.bfloat16,  device_map='auto',tokenizer=tokenizer,temperature=0.7, attn_implementation="flash_attention_2", do_sample=True)
    # pipe = pipeline("text-generation", model=model, max_new_tokens=256, torch_dtype=torch.bfloat16,  device_map='auto',tokenizer=tokenizer)
    # pipe.model = pipe.model.to('auto')

    all_chunk_text = format_docs(docs)

    messages = [{"role": "system", "content": "You are a meticulous emissions-disclosure analyst. Your job is to read a corporate sustainability report (provided) and judge the company's emissions tracking and reporting."},{"role": "user", "content": f"{query}"},{"role": "user", "content": format_report(all_chunk_text)}]
            
    results = pipe(messages, max_new_tokens=256, temperature=0.7, do_sample=True)
    return results


def truncate_documents(documents, max_seq_len):
    cut = next((i for i, total in enumerate(accumulate(d.metadata["tok_len"] for d in documents))
                if total > max_seq_len), len(documents))
    return documents[:cut]

def get_query_len(query, tokenizer):
    return len(tokenizer(query)["input_ids"])

def run_year(model_params, year):
    model, tokenizer, max_seq_len, model_name = model_params["model"], model_params["tokenizer"], model_params["max_seq_len"], model_params["name"]
    assert max_seq_len > CHUNK_LEN
    document_folder = f"climate_reports/ccrm_{year}_olmocr/indexed/"
    glob = "*.jsonl"

    sources = []
    for filename in os.listdir(document_folder):
        with open(document_folder+filename, "r") as f:
            df = pd.read_json(f, lines=True)
            assert len(df.source.to_list()) == 1
            source = df.source.to_list()[0]
            sources.append(source)

    vector_store = setup_vector_store(document_folder, glob, tokenizer)

    query_df = load_jsonl("CCRM/questions.jsonl")
    queries = query_df.question.to_list()
    all_results = []
    for i, source in enumerate(sources):
        company_answers = {"source": source}
        for j, query in enumerate(queries):
            docs = vector_store.similarity_search(query, k=30,filter=lambda doc: filter_by_company_year(doc, index=source))
            docs = [add_token_count(doc, tokenizer) for doc in docs]
            query_len = get_query_len(query, tokenizer)
            max_docs_len = max_seq_len - query_len
            docs = truncate_documents(docs, max_docs_len)
            assert len(docs) > 0, len(docs)
            full_chat = query_model(model, tokenizer, query, docs)
            llm_response = full_chat[0]['generated_text'][3]['content']
            company_answers[j] = llm_response

        all_results.append(company_answers)
        with open(f"results/ccrm_{year}_{model_name}_results.jsonl.temp", "a") as f:
                json.dump(company_answers,f)
                f.write("\n")
        if i % 5 == 0:
            print(i)

        
    with open(f"results/ccrm_{year}_{model_name}_results.jsonl", "w", newline='') as f:
         for d in all_results:
            json.dump(d, f)
            f.write('\n')

def add_token_count(document, tokenizer):
    document.metadata["tok_len"] = len(tokenizer(document.page_content)['input_ids'])
    document.metadata["str_len"] = len(document.page_content)
    return document

if __name__ == "__main__":
    # model_params_1 = climategpt_7b_setup()
    # model_params_2 = qwen_setup()
    # model_params_3 = ministral_8b_it_setup()
    # model_params_4 = climategpt_13b_setup()
    # for model_params in [model_params_1]:
    #     for year in ["2023", "2024"]:
    #         run_year(model_params, year=year)
    #     gc.collect()
    #     torch.cuda.empty_cache()
    # run_company(model_params_1, company="e_on", year="2022")