# Hospital Copilot

AI-powered medical documentation assistant for the **Gemma 4 for Good** hackathon.

Listens to doctor-patient consultations and automatically generates SOAP notes, patient summaries, symptom extractions, and Twi (Akan) translations — reducing paperwork and language barriers in Ghanaian healthcare.

## Features

- **Live transcription** via faster-whisper (runs on CPU)
- **Symptom extraction** via Gemma 4 E2B (local, Ollama, CPU)
- **SOAP note generation** via Gemma 4 27B (Google AI Studio)
- **Patient summary** in plain English
- **English ↔ Twi translation** for Ghanaian patients
- **Patient records** stored in local SQLite

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama and pull Gemma 4

```bash
# Install Ollama: https://ollama.com
ollama pull gemma4:e2b
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Google AI Studio API key
```

Get a free API key at https://aistudio.google.com

### 4. Run

```bash
python app.py
```

Open http://localhost:7860 in your browser.

## Project Structure

```
hosptial_copilot/
├── app.py                      # Gradio UI + app logic
├── agents/
│   ├── symptom_agent.py        # Local Gemma 4 (Ollama) symptom extractor
│   └── cloud_agents.py         # Cloud Gemma 4: SOAP, summary, translation
├── transcription/
│   └── transcriber.py          # faster-whisper live mic transcription
├── database/
│   └── db.py                   # SQLite helpers
├── requirements.txt
└── .env.example
```

## Architecture

```
Microphone
  └─► faster-whisper (local, CPU)   → raw transcript
        ├─► Gemma 4 E2B via Ollama  → symptom JSON (local CPU)
        └─► Gemma 4 27B via API     → SOAP note + summary + Twi translation
              └─► SQLite            → patient records
                    └─► Gradio UI   → doctor dashboard
```
