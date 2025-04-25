from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch


def climategpt_7b_setup():
    model_name = "eci-io/climategpt-7b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    max_seq_len = 3800
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len, "name": "climategpt-7b"}
    return model_params

def climategpt_13b_setup():
    model_name = "eci-io/climategpt-13b"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    max_seq_len = 3000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len, "name": "climategpt-13b"}
    return model_params

def qwen_setup():
    model_name = "Qwen/Qwen2.5-7B-Instruct-1M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    max_seq_len = 2000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len, "name": "qwen"}
    return model_params

def gemma_setup():
    model_name = "google/gemma-7b-itM"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    max_seq_len = 7000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len}
    return model_params

def ministral_8b_it_setup():
    model_name = "mistralai/Ministral-8B-Instruct-2410"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    max_seq_len = 20000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len, "name": "ministral-8B"}
    return model_params

def mistral_7b_it_setup():
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    max_seq_len = 7000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len, "name": "mistral_7b"}
    return model_params

def climatellmama_8b_setup():
    model_name = "suayptalha/ClimateLlama-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    max_seq_len = 4000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len}
    return model_params


def longformer_setup():
    model_name = "allenai/longformer-base-4096"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16,
        device_map="auto",)
    max_seq_len = 10000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len}
    return model_params
# cannot use pipeline with following model
def led_base_setup():
    model_name = "allenai/led-base-16384"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    max_seq_len = 8000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len}
    return model_params

def climte_nlp_longformer_detect_evidence_setup():
    model_name = "climate-nlp/longformer-base-4096-1-detect-evidence"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,torch_dtype=torch.bfloat16,
        device_map="auto",)

    max_seq_len = 10000
    model_params = {"model": model, "tokenizer": tokenizer, "max_seq_len": max_seq_len}
    return model_params
