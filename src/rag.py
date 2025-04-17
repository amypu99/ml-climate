# for environment, 
# pip install -U langchain sentence-transformers faiss-cpu openpyxl pacmap datasets ragatouille
# pip install -U langchain-community
# pip install jq
# pip install chromadb
# pip install -U langchain-core


import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain.vectorstores import Chroma


def setup_vector_store(document_folder):
    loader = DirectoryLoader(document_folder, glob='output_1f175a880ed5a09c6f77e3658258c7740416decf.jsonl', show_progress=True, loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'content_key': 'text', 'json_lines': True,})
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


def climategpt_setup():
    model_name = "eci-io/climategpt-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model, tokenizer


def apply_prompt(chunk_text, question):
    full_prompt = (
        f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
    )

    return full_prompt

prompt_template = """
Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\n"
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

def apply_prompt(chunk_text, question):
        full_prompt = (
            f"Context information is below.\n---------------------\n{chunk_text}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {question}\nAnswer:"
        )

        return full_prompt


def query_model(query, docs):

    model, tokenizer = climategpt_setup()
    pipe = pipeline("text-generation", model=model, max_new_tokens=200, torch_dtype=torch.bfloat16, device_map='cuda', tokenizer=tokenizer)
    pipe.model = pipe.model.to('cuda')

    all_chunk_text = format_docs(docs)
    full_prompt = apply_prompt(all_chunk_text, query)
    messages = [{"role": "user", "content": full_prompt},]
            
    results = pipe(messages, max_new_tokens=256)
    print(results)



if __name__ == "__main__":
    document_folder = "/home/amy_pu/ml-climate/src/climate_reports/ccrm_2022_olmocr/results/"
    vector_store = setup_vector_store(document_folder)

    query = "What is the company’s revenue?"
    docs = vector_store.similarity_search("What is the company’s revenue?")
    # for i in range(len(docs)):
    #     print(docs[i].page_content)
    #     print("\n\n\n")
    # print(len(docs))
    query_model(query, docs)
