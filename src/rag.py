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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM


from datasets import load_dataset
from peft import LoraConfig, PeftModel

from langchain.text_splitter import CharacterTextSplitter


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader
from langchain_core.vectorstores import InMemoryVectorStore

from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain.vectorstores import Chroma

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)

# 'output_1f175a880ed5a09c6f77e3658258c7740416decf.jsonl'
def setup_vector_store(document_folder, glob):
    loader = DirectoryLoader(document_folder, glob=glob, show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'content_key': 'text', 'json_lines': True,})
    documents = loader.load()
    print(f'document count: {len(documents)}')
    text_splitter = CharacterTextSplitter(separator="\n\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = text_splitter.split_documents(documents)
    # db = FAISS.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))
    # retriever = db.as_retriever()

    embeddings = HuggingFaceEmbeddings(model_name ='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=chunked_documents)
    return vector_store


def climategpt_7b_setup():
    model_name = "eci-io/climategpt-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def climategpt_13b_setup():
    model_name = "eci-io/climategpt-13b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def qwen_setup():
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def gemma_setup():
    model_name = "google/gemma-7b-itM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def ministral_8b_it_setup():
    model_name = "mistralai/Ministral-8B-Instruct-2410"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def mistral_7b_it_setup():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def climatellmama_8b_setup():
    model_name = "suayptalha/ClimateLlama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer

def longformer_setup():
    model_name = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16,
        device_map="auto",)
    return model, tokenizer

# cannot use pipeline with following model
def led_base_setup():
    model_name = "allenai/led-base-16384"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def climte_nlp_longformer_detect_evidence_setup():
    model_name = "climate-nlp/longformer-base-4096-1-detect-evidence"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16,
        device_map="auto",)

    return model, tokenizer



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
    pipe = pipeline("text-generation", model=model, max_new_tokens=200, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')

    all_chunk_text = format_docs(docs)

    full_prompt = apply_prompt(all_chunk_text, query)
    messages = [{"role": "system", "content": "You are a meticulous emissions-disclosure analyst. Your job is to read a corporate sustainability report (provided) and judge the company's emissions tracking and reporting"},{"role": "user", "content": full_prompt},{"role": "user", "content": apply_text(all_chunk_text)}]
            
    results = pipe(messages, max_new_tokens=256)
    return results



if __name__ == "__main__":
    document_folder = "/home/amy_pu/ml-climate/src/climate_reports/ccrm_2022_olmocr/results/"
    vector_store = setup_vector_store(document_folder, glob="output_1f175a880ed5a09c6f77e3658258c7740416decf.jsonl")

    # retrieval_query = f"Retrieve all information regarding the company's tracking and disclosure of their greenhouse gas (GHG) emissions, footprints, trajectories. Also retrieve information regarding any climate targets or pledges, or any  commitments towards decarbonizing the value chain or reducing emissions."
    # retrieval_query = f"Retrieve all information regarding the company's emission reduction targets, any target setting approaches or timelines, any 'net-zero' pledges, any plan for immediate action to reduce emissions along the entire value chain paired with a longer-term vision for deep decarbonisation, and coverage of all emission sources and greenhouse gasses. This information should include short, medium, and long-term targets for reducing emissions. Some companies do not commit to absolute GHG-related targets, but rather focus on emission intensity targets or targets associated with decarbonisation indicators, such as renewable energy target."
    # docs = vector_store.similarity_search(retrieval_query, k=10)
    # for i in range(len(docs)):
    #     print(docs[i].page_content)
    #     print("\n\n\n")
    # print(len(docs))

    model, tokenizer = climategpt_7b_setup()
    # model, tokenizer = climategpt_13b_setup()
    # model, tokenizer = climatellmama_8b_setup()
    # model, tokenizer = led_base_setup()
    # model, tokenizer = qwen_setup()
    # model, tokenizer = gemma_setup()
    # model, tokenizer = ministral_8b_it_setup()
    # model, tokenizer = mistral_7b_it_setup()
    # model, tokenizer = climte_nlp_longformer_detect_evidence_setup()
    query_df = load_jsonl("CCRM/questions.jsonl")
    queries = query_df.question.to_list()
    k_docs = query_df.k_docs.to_list()
    for i, query in enumerate(queries):
        docs = vector_store.similarity_search(query, k=k_docs[i])
        full_chat = query_model(model, tokenizer, query, docs)
        # print(full_chat)
        llm_response = full_chat[0]['generated_text'][3]['content']
        print("QUESTION ", i)
        print("\n")
        print(llm_response)
        print("\n\n")

    # query = "Read the following sustainability report and answer the question: What is the company's revenue for the year? If this information is not found in the report, please output 'none.'\n\n────────────────────\n### Output\nReturn **only** this JSON object:\n\n```json\n{\n  \"revenue\": \"<amount>\",\n  \"reasoning\": \"<1-3 sentences citing concrete evidence (page/section refs if possible)>\"\n}\n```"

    # docs = vector_store.similarity_search(query, k=4)
    # full_chat = query_model(model, tokenizer, query, docs)
    # # print(full_chat)
    # llm_response = full_chat[0]['generated_text'][3]['content']
    # print(llm_response)
