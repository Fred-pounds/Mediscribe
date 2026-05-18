---
title: MediScribe AI
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
---

# MediScribe AI

AI-powered medical documentation assistant for the **Gemma 4 for Good** hackathon.

Record a doctor-patient consultation via your browser mic. MediScribe transcribes it, repairs ASR errors, extracts structured clinical data, and generates a professional SOAP note and patient summary — powered by Gemma 4.

## Features

- **Browser mic recording** — no software install needed
- **Transcript repair + speaker labelling** via Gemma 4
- **Structured symptom extraction** via Gemma 4 function calling
- **RAG-grounded SOAP notes** with ICD-10 codes and WHO drug references
- **Multimodal document analysis** — upload lab results or prescriptions
- **Patient records** stored in SQLite

## Setup (local)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your Google AI Studio API key
```

Get a free key at https://aistudio.google.com

### 3. Run

```bash
python app.py
```

## Hugging Face Spaces

Set `GEMINI_API_KEY` as a Space secret in Settings → Variables and secrets.

## Project Structure

```
├── app.py                      # Gradio UI + app logic
├── agents/
│   ├── symptom_agent.py        # Symptom extractor (Gemma 4 function calling)
│   └── cloud_agents.py         # SOAP, summary, transcript repair, document analysis
├── transcription/
│   └── transcriber.py          # faster-whisper batch transcription
├── rag/
│   ├── retriever.py            # ChromaDB + sentence-transformers RAG
│   └── data/                   # ICD-10 codes + WHO essential medicines
├── database/
│   └── db.py                   # SQLite helpers
└── requirements.txt
```

## Architecture

```
Browser Mic
  └─► faster-whisper (CPU)      → raw transcript
        └─► Gemma 4 26B (API)   → cleaned transcript + speaker labels
              ├─► Gemma 4 function calling → structured symptom JSON
              ├─► ChromaDB RAG  → ICD-10 codes + drug dosages
              └─► Gemma 4 reasoning mode → SOAP note + patient summary
                    └─► SQLite  → patient records
```
