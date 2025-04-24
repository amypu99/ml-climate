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
import csv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from model_setup import climategpt_7b_setup, climategpt_13b_setup, climatellmama_8b_setup, qwen_setup, ministral_8b_it_setup, mistral_7b_it_setup, climte_nlp_longformer_detect_evidence_setup, longformer_setup, led_base_setup, gemma_setup


from datasets import load_dataset
from peft import LoraConfig, PeftModel

from langchain.text_splitter import CharacterTextSplitter


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain_core.vectorstores import InMemoryVectorStore

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders import DirectoryLoader

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

CHUNK_LEN = 1000

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

def filter(document, index):
    if document.metadata["index"] == index: 
        return True
    return False

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["index"] = record.get("metadata")["Source-File"].replace(".pdf", "").replace("climate_reports/ccrm_2022/", "")
    return metadata

def length_function(text: str, tokenizer) -> int:
    return len(tokenizer(text)["input_ids"])

# 'output_1f175a880ed5a09c6f77e3658258c7740416decf.jsonl'
def setup_vector_store(document_folder, glob, tokenizer):
    loader = DirectoryLoader(document_folder, glob=glob, show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'content_key': 'text', 'json_lines': True,'metadata_func': metadata_func})
    documents = loader.load()
    print(f'document count: {len(documents)}')
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=CHUNK_LEN,
        chunk_overlap=200,
        # length_function=len,
        length_function=lambda text: length_function(text, tokenizer=tokenizer),
        is_separator_regex=False,
        strip_whitespace=True,
    )
    chunked_documents = text_splitter.split_documents(documents)
    # db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))
    # retriever = db.as_retriever()

    embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunked_documents)
    return vector_store


def apply_prompt(chunk_text, question):
    # full_prompt = (
    #     f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\n\n\n\n\n\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}"
    # )
    # full_prompt = f"Instruction: Read the corporate climate report below and answer the query.\n---------------------\n{chunk_text}\n---------------------\n\n\n\n\n\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}"
    full_prompt = f"{question}"

    return full_prompt

prompt_template = """
Context information is below.\n---------------------\n{context}\n---------------------\n\n\n\n\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}"
"""

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def invoke_ragchain(llm, prompt_template, retriever, question):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
    # Create llm chain 
    # llm_chain = LLMChain(llm=llm, prompt=prompt)

    rag_chain = ( 
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    ans = rag_chain.invoke(question)
    return ans

def apply_text(all_chunk_text):
    full_text = f"\n\n\n\n\n—— REPORT START ——\n{all_chunk_text}—— REPORT END ——"
    return full_text

def query_model(model, tokenizer, query, docs):
    pipe = pipeline("text-generation", model=model, max_new_tokens=200, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer,)
    pipe.model = pipe.model.to('cuda')

    all_chunk_text = format_docs(docs)

    full_prompt = apply_prompt(all_chunk_text, query)
    messages = [{"role": "system", "content": "You are a meticulous emissions-disclosure analyst. Your job is to read a corporate sustainability report (provided) and judge the company's emissions tracking and reporting."},{"role": "user", "content": full_prompt},{"role": "user", "content": apply_text(all_chunk_text)}]
            
    results = pipe(messages, max_new_tokens=256)
    return results


def run_year(model_params, year):
    model, tokenizer, max_seq_len = model_params["model"], model_params["tokenizer"], model_params["max_seq_len"]
    document_folder = f"/home/amy_pu/ml-climate/src/climate_reports/ccrm_{year}_olmocr/results/"
    glob = "*.jsonl"

    sources = []
    for filename in os.listdir(document_folder):
        with open(document_folder+filename, "r") as f:
            df = pd.read_json(f)
            source = df.metadata["Source-File"].replace(".pdf", "").replace(f"climate_reports/ccrm_{year}/", "")
            sources.append(source)

    vector_store = setup_vector_store(document_folder, glob, tokenizer)

    query_df = load_jsonl("CCRM/questions.jsonl")
    queries = query_df.question.to_list()
    # k_docs = query_df.k_docs.to_list()
    k_docs = max_seq_len // CHUNK_LEN
    all_results = []
    for i, source in enumerate(sources):
        company_answers = [source]
        for j, query in enumerate(queries):
            docs = vector_store.similarity_search(query, k=k_docs,filter=lambda doc: filter(doc, index=source))
            full_chat = query_model(model, tokenizer, query, docs)
            llm_response = full_chat[0]['generated_text'][3]['content']
            company_answers.append(llm_response)
            # print("QUESTION ", j)
            # print("\n")
            # print(llm_response)
            # print("\n\n")
        all_results.append(company_answers)
        if i % 5 == 0:
            print(i)
            with open(f"ccrm_{year}_results.csv.temp", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(company_answers)


    with open(f"ccrm_{year}_results.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(all_results)

def add_token_count(document, tokenizer):
    document.metadata["tok_len"] = len(tokenizer(document.page_content)['input_ids'])
    document.metadata["str_len"] = len(document.page_content)
    return document

def run_company(model_params, company, year):
    model, tokenizer, max_seq_len = model_params["model"],model_params["tokenizer"],model_params["max_seq_len"]
    document_folder = f"/home/amy_pu/ml-climate/src/climate_reports/ccrm_{year}_olmocr/results/"
    glob = f"{company}*.jsonl"
    company_year = str(int(year) - 2)
    filename = f"{company}_{company_year}.jsonl"

    sources = []
    with open(document_folder+filename, "r") as f:
        df = pd.read_json(f)
        source = df.metadata["Source-File"].replace(".pdf", "").replace(f"climate_reports/ccrm_{year}/", "")
        sources.append(source)

    vector_store = setup_vector_store(document_folder, glob, tokenizer)

    query_df = load_jsonl("CCRM/questions.jsonl")
    queries = query_df.question.to_list()
    # k_docs = query_df.k_docs.to_list()
    k_docs = max_seq_len // CHUNK_LEN
    company_answers = [source]
    for j, query in enumerate(queries):
        docs = vector_store.similarity_search(query, k=k_docs,filter=lambda doc: filter(doc, index=source))
        docs = [add_token_count(doc, tokenizer) for doc in docs]

        all_chunk_text = format_docs(docs)
        print("all chunk text")
        print(all_chunk_text)
        full_chat = query_model(model, tokenizer, query, docs)
        llm_response = full_chat[0]['generated_text'][3]['content']
        print("\n\nllm response")
        print(llm_response)
        company_answers.append(llm_response)
        break
        # print("QUESTION ", j)
        # print("\n")
        # print(llm_response)
        # print("\n\n")
        if j % 5 == 0:
            print(j)
            with open(f"{company}_{year}_results.csv.temp", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(company_answers)


if __name__ == "__main__":
    model_params = qwen_setup()
    run_company(model_params, company="walmart", year="2022")