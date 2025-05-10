# 🌿 Automating Corporate Greenwashing Detection with NLP

This project presents an end-to-end NLP pipeline to assess corporate climate claims at scale using large language models (LLMs) and retrieval-augmented generation (RAG). Inspired by expert-led evaluations like the **Corporate Climate Responsibility Monitor (CCRM)** and **Transition Pathways (TP)**, we design a scalable system that extracts structured climate disclosures from unstructured PDF reports—automatically scoring firms across transparency and integrity dimensions.

📝 [Read the full paper](./Climate_Final_Paper.pdf)

---

## ✨ Highlights

- 🔎 High-fidelity OCR + Recursive Chunking + RAG for handling 100+ page reports  
- 🤖 Evaluated 3 domain-adapted LLMs: ClimateGPT, Mistral-8B, and Qwen-1M  
- 📈 Achieved up to **26% question-level accuracy** using rubric-grounded prompting  
- 🗃️ Released a labeled dataset of 236 firm-year climate evaluations  
- 📦 Open-source pipeline for automated CCRM-style assessments  

---

## 📊 Dataset Overview

The final dataset includes:

- **236 firm-year entries**  
- **CCRM Labels (2022–2024)**: Transparency and integrity scores (6 levels)  
- **TP Binary Answers**: 23 yes/no climate-readiness questions  
- **Metadata**: Sector, country, year, and report provenance  

All reports were manually retrieved and OCR'd using `olmOCR`, then parsed into LangChain documents for LLM ingestion.

---


