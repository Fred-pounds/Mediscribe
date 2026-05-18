# MediScribe AI
## An Offline-First Multilingual Clinical Assistant Powered by Gemma 4

**Track:** Health & Sciences

---

## Overview

In many clinics across Africa, doctors spend more time documenting than treating. MediScribe AI is a desktop AI assistant that listens to a doctor-patient consultation, transcribes it in real time, and automatically generates a structured SOAP note, a plain-language patient summary, and a structured symptom record — all reviewed and approved by the doctor before anything is saved.

The system was built specifically for Ghanaian healthcare contexts, targeting clinics where internet access may be intermittent and where patients may speak languages other than English.

---

## The Problem

Healthcare workers in developing regions face crushing administrative workloads. Manual note-taking after every consultation reduces patient interaction time, increases burnout, and introduces inconsistencies in medical records. Language barriers between English-trained doctors and local-language-speaking patients add further friction. Many cloud-first AI tools are impractical where connectivity is unreliable.

---

## What We Built

MediScribe AI is a single Python application with a Gradio web UI. The consultation workflow is:

1. The doctor registers or selects a patient.
2. Consultation starts — the microphone opens and audio begins streaming.
3. faster-whisper transcribes speech in real time in 3-second blocks on CPU.
4. After the consultation ends, Gemma 4 (cloud) repairs ASR errors and labels each turn as Doctor or Patient.
5. The doctor clicks **Generate Notes**. The system:
   - Extracts structured symptoms via **local Gemma 4 E2B** (Ollama, CPU) — chief complaint, symptom list, duration, severity, vitals, medications, allergies, follow-up actions
   - Falls back to **cloud Gemma 4 function calling** if local extraction fails, returning a guaranteed-valid structured schema
   - Runs **semantic RAG retrieval** against a local ChromaDB knowledge base to surface relevant ICD-10 codes and WHO essential medicines dosages
   - Generates a **SOAP note** using cloud Gemma 4 with reasoning/thinking mode enabled, grounded by the RAG context
   - Generates a **plain-language patient summary**
6. The doctor reviews both outputs in editable panels before approving.
7. Everything is saved to local SQLite — transcript, SOAP note, summary, and structured symptom JSON.

An optional document upload allows the doctor to attach a photo or PDF of a lab result, prescription, or X-ray. Gemma 4's multimodal capability reads the document and automatically includes the findings in the SOAP note.

---

## Why Gemma 4

We used Gemma 4 across every AI-powered step of the pipeline:

| Task | Model | Where |
|---|---|---|
| Symptom extraction (primary) | Gemma 4 E2B | Local CPU via Ollama |
| Symptom extraction (fallback) | Gemma 4 26B — function calling | Google AI Studio API |
| Transcript repair + speaker labelling | Gemma 4 26B | Google AI Studio API |
| SOAP note generation | Gemma 4 26B — reasoning mode | Google AI Studio API |
| Patient summary | Gemma 4 26B | Google AI Studio API |
| Medical document analysis | Gemma 4 26B — multimodal | Google AI Studio API |

The local Gemma 4 E2B model (quantized, running via Ollama on CPU) handles the privacy-sensitive symptom extraction step, keeping structured clinical data local when possible. The cloud Gemma 4 model handles the tasks requiring stronger reasoning — particularly SOAP note generation, which uses the model's built-in thinking mode to reason through the clinical picture before writing the note.

Gemma 4's native function calling was used to implement a guaranteed-valid structured output schema for symptom extraction — eliminating JSON parsing failures that plagued earlier prompt-only approaches.

---

## System Architecture

**Frontend:** Gradio web UI running locally on port 7860. Two main tabs — Live Consultation and Patient Records. No external server required.

**Speech-to-Text:** faster-whisper (`small` model, CPU, int8 quantized) with sounddevice for microphone streaming. Audio is processed in 3-second chunks with VAD filtering. Full session audio is saved as WAV after the consultation ends.

**Symptom Extraction Agent (`agents/symptom_agent.py`):** Calls local Gemma 4 E2B via Ollama with a structured JSON prompt. On any failure (model unavailable, invalid JSON, malformed response), automatically falls back to cloud Gemma 4 function calling with a defined schema, guaranteeing a valid structured output.

**Cloud Agent (`agents/cloud_agents.py`):** Wraps the Google GenAI SDK. Implements transcript repair, SOAP generation (with `ThinkingConfig`), patient summary, function-calling symptom extraction, and multimodal document analysis. Temperature is set to 0.3 for clinical outputs; 1.0 when thinking mode is active (required by the API).

**RAG Pipeline (`rag/retriever.py`):** `all-MiniLM-L6-v2` sentence-transformers embeddings + ChromaDB persistent vector store. Two collections: 90+ ICD-10 codes (Ghana-relevant and general) and 40+ WHO Essential Medicines entries. The top-5 ICD codes and top-3 drug matches are retrieved per consultation and injected into the SOAP note prompt as grounding context.

**Database (`database/db.py`):** SQLite with four tables — `patients`, `sessions`, `notes`, and `symptoms`. Stores the full cleaned transcript, SOAP note, English summary, and structured symptom JSON per session.

---

## Key Features

### Real-Time Transcription with Post-Processing
faster-whisper streams transcription as the consultation proceeds. After the session ends, Gemma 4 repairs ASR errors (medical terminology, drug names, run-on sentences) and labels each turn as Doctor or Patient in a single pass.

### Dual-Mode Symptom Extraction
Local-first extraction via Gemma 4 E2B keeps sensitive data off the network whenever possible. Automatic cloud fallback via function calling ensures the pipeline never silently fails.

### RAG-Grounded SOAP Notes
ICD-10 codes and drug dosage references are retrieved semantically before SOAP note generation. This grounds the model's clinical output in verifiable reference data rather than relying purely on parametric knowledge.

### Multimodal Document Analysis
Doctors can upload a photo or PDF of a lab result, prescription, or report. Gemma 4 reads it and its findings are automatically included as context in the SOAP note generation step.

### Human-in-the-Loop Validation
Every generated output is shown to the doctor in a readable panel with an editable fallback before anything is committed to the database. Doctors approve; the AI drafts.

### Local Storage
All patient records, transcripts, SOAP notes, and symptom data are stored in a local SQLite database. No patient data leaves the device except for the API calls to generate notes.

---

## Technical Challenges

### Reliable Structured Output from a Local Small Model
Gemma 4 E2B running on CPU occasionally produces malformed JSON or misses required fields. We implemented a two-tier extraction strategy: local Ollama first with JSON validation, cloud function calling as a typed-schema fallback. This eliminated silent data loss in the pipeline.

### ASR Quality on Medical Vocabulary
faster-whisper on CPU struggles with drug names, medical abbreviations, and Ghanaian proper names. We addressed this by adding a dedicated Gemma 4 repair pass after the consultation ends, correcting the transcript before any clinical information is extracted.

### Thinking Mode Compatibility
Gemma 4's reasoning mode requires `temperature=1.0` and is not supported on all model variants. We implemented a graceful fallback that detects API errors related to `ThinkingConfig` and retries without it, so SOAP generation never fails silently.

### RAG Grounding for Clinical Accuracy
SOAP notes generated without reference context showed inconsistent ICD code suggestions and occasionally incorrect drug dosages. Adding RAG retrieval with ChromaDB significantly improved specificity and reduced hallucinated medication instructions.

---

## What Is Not Yet Built

Twi/English translation is planned (NLLB-200) but not yet implemented — stubs exist in `cloud_agents.py`. Speaker diarization is partially scaffolded (session audio is saved as WAV) but not yet wired up. The system currently requires internet access for SOAP generation and transcript repair; a fully offline mode would require a larger local model than E2B.

---

## Impact

MediScribe AI reduces the documentation burden on doctors by automating the most time-consuming parts of post-consultation admin: writing SOAP notes, summarizing for patients, and coding diagnoses. Because it runs locally and saves to a local database, it is viable in clinics with unreliable connectivity. The human-in-the-loop design keeps the doctor fully in control — the AI is a drafter, not an authority.

---

## Stack Summary

| Component | Technology |
|---|---|
| UI | Gradio (Python, port 7860) |
| Speech-to-Text | faster-whisper small, CPU, int8 |
| Local AI | Gemma 4 E2B via Ollama |
| Cloud AI | Gemma 4 26B-IT via Google AI Studio |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | ChromaDB (local, persistent) |
| Database | SQLite |
| Language | Python 3.11 |
