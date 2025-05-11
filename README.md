# 🌿 Automating Corporate Greenwashing Detection with NLP

This project presents an end-to-end NLP pipeline to assess corporate climate claims at scale using large language models (LLMs) and retrieval-augmented generation (RAG). Inspired by expert-led evaluations like the **Corporate Climate Responsibility Monitor (CCRM)** and **Transition Pathways (TP)**, we design a scalable system that extracts structured climate disclosures from unstructured PDF reports—automatically scoring firms across transparency and integrity dimensions.

📝 [Read the full paper](./Climate_Final_Paper.pdf)

---

## ✨ Highlights

- 🔎 High-fidelity OCR + Recursive Chunking + RAG for handling 100+ page reports  
- 🤖 Evaluated 3 domain-adapted LLMs: ClimateGPT, Mistral-8B, and Qwen-1M  
- 📈 Achieved up to **26% question-level accuracy** using rubric-grounded prompting  
- 🗃️ Released a labeled dataset of 227 firm-year climate evaluations  
- 📦 Open-source pipeline for automated CCRM-style assessments  

---

## 📊 Dataset Overview

The [final dataset](./src/merge_data/updated_final_interpolated.csv) includes:

- **227 firm-year entries**  
- **CCRM Labels (2022–2024)**: Transparency and integrity scores (6 levels)  
- **TP Binary Answers**: 23 yes/no climate-readiness questions  
- **Metadata**: Sector, country, year, report provenance, etc. 

All reports were manually retrieved and OCR'd using `olmOCR`, then parsed into LangChain documents for LLM ingestion.

---

## 🛠️ Pipeline Overview

### 1. Text Extraction (olmOCR)

Processed sustainability reports, converted via olmOCR, are available at `./src/climate_reports/all`.

### 2. Recursive Chunking & Embedding

Documents are chunked at ~2,200 tokens with overlap, embedded via MiniLM, and stored for retrieval.

### 3. RAG Prompting & Inference
Prompts include the full CCRM scoring rubric to ensure grounding and reduce hallucinations. Prompt templates are located in `./src/questions`. Model setup is located at `./src/model_setup.py` and rag pipeline is defined in `./src/run_rag.py`.

### 4. Evaluation

Model outputs for both CCRM and TPI assessments are saved in `./src/all_results/combined`. Evaluation scripts and analysis results are in `./src/eval`. Labels for CCRM and TPI are located at `.src/labels/merge_data/updated_final_interpolated.csv`

## 📌 Lessons & Limitations

- ✅ **Embedding rubrics into prompts boosts label fidelity and reduces noise**
- 🚫 Qwen and Mistral underperformed without domain tuning
- ⚠️ OCR errors in tabular data can reduce numeric accuracy
- 🧊 Chunked retrieval beats naive summarization or truncation

---

## 🧪 Future Directions

- Fine-tuning LLMs on rubric-scored climate data  
- Incorporating charts/tables via multimodal models  
- Expanding to real-time monitoring with FAISS indexing  

---

## 📚 Citation

If you use this repo or dataset, please cite:

```bibtex
@article{2025automatingcorporategreenwashing,
  title={Automating Corporate Greenwashing Detection Using Natural Language Processing},
  author={Lin, Nicole and Pu, Amy},
  journal={Columbia University},
  year={2025}
}

