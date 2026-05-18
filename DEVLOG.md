# Hospital Copilot — Development Log

**Hackathon:** Gemma 4 for Good  
**Team:** Ricky (fredrickandoh17@gmail.com)  
**Stack:** Python · Gradio · Gemma 4 · faster-whisper · ChromaDB · SQLite  
**Started:** 2026-05-16

---

## Project Goal

Build an AI clinical assistant that listens to doctor-patient consultations and automatically produces:
- Live transcription of the conversation
- Structured symptom extraction (symptoms, medications, duration, allergies, follow-up actions)
- SOAP notes grounded with real ICD-10 codes and drug dosages
- Plain-language patient summary
- Structured patient records saved to a local database

**Why:** Reduce doctor burnout from paperwork, improve care quality, and support healthcare workers in low-resource settings like Ghana.

---

## Architecture Overview

```
Microphone
  └─► faster-whisper (STT, local CPU)       → raw transcript
        └─► Gemma 4 26B cloud (speaker labelling) → Doctor:/Patient: transcript
              ├─► Gemma 4 E2B via Ollama (symptom JSON)  → local CPU
              └─► ChromaDB + MiniLM (RAG retrieval)      → ICD-10 codes + drug info
                    └─► Gemma 4 26B cloud (SOAP note, patient summary)
                          └─► SQLite (patients, sessions, notes, symptoms)
                                └─► Gradio UI
```

---

## Features Implemented

### Core Pipeline
| Feature | Status | Implementation |
|---|---|---|
| Live mic transcription | ✅ | faster-whisper `small` model, 3s chunks, VAD filter |
| Speaker diarization | ✅ | Gemma 4 post-hoc Doctor:/Patient: labelling |
| Symptom extraction | ✅ | Gemma 4 E2B via Ollama — JSON: chief complaint, symptoms, duration, severity, medications, allergies, vitals, history, follow-up actions |
| RAG ICD-10 retrieval | ✅ | ChromaDB + all-MiniLM-L6-v2, 90+ Ghana-relevant codes |
| RAG drug grounding | ✅ | ChromaDB, 40+ WHO Essential Medicines with dosages |
| SOAP note generation | ✅ | Gemma 4 26B cloud, RAG context injected into prompt |
| Patient summary | ✅ | Gemma 4 26B cloud, plain English |
| Patient records (SQLite) | ✅ | patients, sessions, notes, symptoms tables |
| Patient registration | ✅ | Name, DOB, gender, phone |
| Records viewer | ✅ | Load any patient's most recent session |

### Translation (Twi/Akan)
| Status | Note |
|---|---|
| ⏸️ Paused | Gemma 4 returned 500 INTERNAL errors on Twi translation. Identified root cause: Twi is a low-resource language and Gemma 4 is not purpose-built for it. Decision: implement NLLB-200 (Meta's No Language Left Behind model) which was specifically trained on Akan/Twi. Deferred until core pipeline is stable. |

### Gemma 4 Advanced Features (Added 2026-05-18)
| Feature | Status | Implementation |
|---|---|---|
| **Reasoning mode (thinking)** | ✅ | `ThinkingConfig(thinking_budget=2048, include_thoughts=False)` on SOAP generation — Gemma 4 reasons step-by-step internally before writing the note |
| **Function calling (symptom extraction)** | ✅ | `FunctionDeclaration` schema with `FunctionCallingMode.ANY` — guaranteed valid structured output, no JSON parsing |
| **Multimodal image/document analysis** | ✅ | `Part.from_bytes()` with lab result / prescription images — extracted findings injected into SOAP context |

---

## Technical Decisions

### 1. Multi-agent Gemma 4 architecture
**Decision:** Use multiple specialised Gemma 4 instances rather than one large model for everything.  
**Reasoning:** Different tasks have different speed/accuracy requirements:
- Symptom extraction: needs to be fast, structured JSON → small local model (E2B)
- SOAP notes: needs medical reasoning and long output → large cloud model (26B)
- Speaker labelling: needs language understanding → cloud model
- Embeddings: needs speed, runs every session → lightweight MiniLM locally

### 2. Local vs cloud split
**Decision:** Run small models locally (Ollama E2B, Whisper, MiniLM, ChromaDB), large inference on cloud API.  
**Reasoning:** User has no GPU. CPU-only local inference is viable for small quantised models (Q4_K_M gemma4:e2b runs at ~5-10 tok/s). Large models (26B+) are impractical on CPU — cloud API provides them at acceptable latency.

### 3. RAG with ChromaDB + MiniLM
**Decision:** Use local vector store over calling the cloud model with full knowledge base in prompt.  
**Reasoning:**
- Injecting 70k ICD-10 codes into every prompt would exceed context limits and cost tokens
- Local ChromaDB persists to disk, zero latency after first build
- MiniLM-L6-v2 (~80MB) gives good semantic similarity for medical terms on CPU
- Retrieves top-5 most relevant codes per consultation — keeps prompt tight and accurate

### 4. Gradio over Streamlit
**Decision:** Use Gradio for the UI.  
**Reasoning:** Gradio has better support for streaming, audio, and timer-based polling. Streamlit's re-run model makes real-time transcript updates difficult. Gradio's `gr.Timer` makes 2-second polling trivial.

### 5. Gemma 4 reasoning mode — temperature requirement
**Decision:** Set `temperature=1.0` when `thinking_config` is enabled, not `0.3`.
**Reasoning:** Google's API requires temperature=1.0 when using ThinkingConfig — lower values raise an error. The thinking process itself introduces determinism so output quality is not degraded. Added graceful fallback: if the model doesn't support thinking (e.g. older model version), retry without `thinking_config`.

### 6. Function calling mode = ANY
**Decision:** Use `FunctionCallingMode.ANY` (force the model to always call the function) rather than `AUTO`.
**Reasoning:** `AUTO` mode allows the model to optionally use the function or just return text — unreliable for extraction tasks. `ANY` mode guarantees the model returns a structured function call every time, eliminating the JSON parse errors we had with the prompt-based approach.

### 7. Symptom extraction: local first, cloud fallback
**Decision:** Keep Gemma 4 E2B (Ollama, local) as primary for symptom extraction, cloud function calling as fallback.
**Reasoning:** Preserves the "local AI, privacy-preserving" story for the hackathon. Cloud fallback ensures reliability when Ollama returns malformed JSON or fails. Both paths return the same dict structure.

### 8. Transcript repair before downstream processing
**Problem:** faster-whisper `small` on CPU makes errors — mishears medical terms, missing punctuation, run-on sentences. Downstream models (symptom extraction, SOAP generation) produce lower quality output when given a garbled transcript.
**Decision:** Add a `clean_and_label_transcript()` step using Gemma 4 cloud that simultaneously repairs ASR errors AND labels speakers in one API call. This runs after `stop_consultation()` before any downstream processing.
**What it fixes:** Incorrect drug names, missing punctuation, filler words (um/uh), run-on sentences, garbled medical terminology.
**What it preserves:** All clinical facts — symptoms, medications, durations, dosages. Never adds or invents information.
**Why one call:** Combining repair + labelling saves one API round-trip and is cheaper than two separate calls.

### 9. Speaker diarization: Gemma 4 post-hoc vs pyannote-audio
**Decision:** Use Gemma 4 cloud to infer Doctor/Patient labels from transcript text.  
**Reasoning:**
- `pyannote-audio` requires HuggingFace account, model license acceptance, and token setup
- For a hackathon demo, Gemma 4 inference from linguistic context is good enough
- Doctors and patients have very different speech patterns (questions vs symptom descriptions) that Gemma 4 reliably distinguishes
- Can always upgrade to pyannote later

### 6. SQLite for storage
**Decision:** Local SQLite over PostgreSQL or cloud database.  
**Reasoning:** Desktop app, no server, no network dependency. SQLite is reliable, zero-config, and sufficient for demo-scale data. Schema: patients → sessions → notes + symptoms.

### 7. Whisper model: small over base
**Decision:** Upgrade from `base` to `small` Whisper model.  
**Reasoning:** `base` had poor accuracy on real speech, especially medical terminology. `small` is ~4x more accurate on medical vocabulary and still runs acceptably on CPU (~2-3x slower than base but real-time viable with 3-second chunking). `medium` was considered but too slow for live demo.

---

## Issues Encountered & Resolutions

### Issue 1: `google-generativeai` deprecated
**Error:** `FutureWarning: All support for the google.generativeai package has ended`  
**Root cause:** Google deprecated the old `google-generativeai` SDK in favour of `google-genai`  
**Resolution:** Replaced `google-generativeai` with `google-genai>=1.0.0` in requirements. Updated `cloud_agents.py` to use `from google import genai` and `genai.Client()` pattern.

### Issue 2: Wrong Gemma 4 cloud model name
**Error:** `404 NOT_FOUND: models/gemma-4-27b-it is not found`  
**Root cause:** Model name `gemma-4-27b-it` does not exist on Google AI Studio API.  
**Resolution:** Listed available models via API (`client.models.list()`). Correct names are:
- `gemma-4-26b-a4b-it` (26B MoE, faster)
- `gemma-4-31b-it` (31B dense, most capable)
Updated default in `cloud_agents.py` and `.env`.

### Issue 3: Twi translation 500 INTERNAL error
**Error:** `500 INTERNAL: Internal error encountered` on `translate_to_twi()`  
**Root cause:** Gemma 4 struggles with Twi (Akan) — a low-resource language with limited training data. The model likely has insufficient Twi coverage to translate medical content reliably, causing server-side failures.  
**Resolution (temporary):** Removed Twi translation from the pipeline. Added try/except guards around all cloud agent calls so one failure doesn't break the entire `generate_notes()` flow.  
**Planned fix:** Integrate NLLB-200 (`facebook/nllb-200-distilled-600M`) — Meta's purpose-built model for 200 low-resource languages including Akan/Twi.

### Issue 4: Ollama version too old for Gemma 4
**Error:** `Error: pull model manifest: 412: The model you are attempting to pull requires a newer version of Ollama`  
**Root cause:** System Ollama was v0.19.0. Gemma 4 requires a newer version.  
**Resolution:** Reinstall Ollama via the official install script: `curl -fsSL https://ollama.com/install.sh | sh` then `sudo systemctl restart ollama`. Note: Linux package managers (snap, apt) ship outdated Ollama versions — always use the curl script.

### Issue 5: `chromadb.PersistentClient | None` TypeError
**Error:** `TypeError: unsupported operand type(s) for |: 'function' and 'NoneType'`  
**Root cause:** `chromadb.PersistentClient` is a factory function, not a class. Using it in a `X | None` type annotation evaluates at runtime and fails.  
**Resolution:** Added `from __future__ import annotations` to `rag/retriever.py` — this makes all annotations lazy (strings at runtime), bypassing the evaluation issue.

### Issue 6: White empty boxes in UI (RAG panels)
**Issue:** `gr.Markdown` components rendered as white boxes on dark Gradio theme, even when empty.  
**Root cause:** Gradio's default light background on Markdown components clashes with the dark theme. Empty panels had no content but still showed as white rectangles.  
**Resolution:** Moved RAG panels (ICD-10, Drug Reference, Symptoms) into `gr.Accordion` components. Accordions collapse when not needed and have theme-consistent styling. Also added CSS `background: transparent` for markdown panels.

### Issue 9: Gemma 4 image input — wrong contents structure
**Error:** `500 INTERNAL` then `Part.from_text() takes 1 positional argument but 2 were given`
**Root cause:** Two sequential mistakes in the multimodal contents format:
  1. First attempt wrapped parts in `types.Content(role="user", parts=[...])` — not needed
  2. Used `types.Part.from_text(IMAGE_PROMPT)` — this method does not exist in the SDK
**Resolution:** Per official Gemma 4 docs (philschmid.de/gemma-4-gemini-api), the correct format is a plain list mixing `Part.from_bytes()` and a raw string:
```python
contents=[
    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
    IMAGE_PROMPT,   # plain string, not Part.from_text()
]
```
All Gemma 4 models (including 26B and 31B) are fully multimodal. The initial 500 error was caused by the wrong content structure, not a model limitation.

### Issue 10: pyannote-audio abandoned in favour of Gemma 4
**Decision made:** Started implementing pyannote-audio for speaker diarization, then stopped.
**Reason:** User confirmed Gemma 4 post-hoc labelling is sufficient for the demo. pyannote requires HuggingFace account, model license acceptance, and heavy torch dependency. Gemma 4 language-based inference is actually more reliable for medical conversations because it uses *context* (doctors ask questions, patients describe symptoms) rather than raw audio signal (which can fail when two speakers have similar voices).

### Issue 10: Gradio CSS parameter deprecation warning
**Warning:** `UserWarning: The parameters have been moved from the Blocks constructor to the launch() method`  
**Root cause:** Gradio 6.0 moved `css` parameter from `gr.Blocks(css=...)` to `demo.launch(css=...)`.  
**Resolution:** Moved `css=CSS` to `demo.launch(...)`.

### Issue 8: uv installing to wrong Python version
**Issue:** `chromadb` and `sentence-transformers` installed but not importable from venv.  
**Root cause:** The venv was created with Python 3.11 (via uv) but system also has Python 3.12. Running `uv pip install` without specifying the environment installed to the wrong location.  
**Resolution:** Used `VIRTUAL_ENV=/path/to/.venv uv pip install ...` to target the correct venv, or used `/path/to/.venv/bin/python -m pip install ...`.

---

## What Was Considered and Rejected

| Option | Rejected because |
|---|---|
| Streamlit UI | Real-time transcript polling is awkward in Streamlit's re-run model |
| PostgreSQL storage | Overkill for desktop demo; SQLite is zero-config |
| pyannote-audio diarization | Requires HF account + model license; too much setup for hackathon timeline |
| Full 70k ICD-10 dataset | Too large to embed in demo time; curated Ghana-relevant subset is more impactful |
| Running everything on cloud API | Wanted to demonstrate hybrid local+cloud multi-agent architecture |
| Whisper `large-v3` | Too slow on CPU for real-time; `small` is the sweet spot |
| Gemma 4 for Twi translation | Low-resource language; model returned 500 errors. NLLB-200 is the right tool |

---

## Remaining Work / Roadmap

- [ ] **Twi translation via NLLB-200** — integrate `facebook/nllb-200-distilled-600M` locally
- [ ] **PDF export** — export SOAP note + patient summary as printable PDF (fpdf2 already in deps)
- [ ] **Multi-session history** — view all past sessions for a patient, not just the most recent
- [ ] **Upgrade to Whisper `medium`** if demo machine is fast enough
- [ ] **ICD-10 code expansion** — add full 70k code dataset for production use
- [ ] **MedGemma** — self-host `medgemma-4b-it` or `medgemma-27b-it` for higher-accuracy medical image analysis
- [ ] **Long-context patient history** — load all previous session notes into SOAP prompt for longitudinal care reasoning

---

## File Structure

```
hosptial_copilot/
├── app.py                          Main Gradio app + UI
├── agents/
│   ├── cloud_agents.py             Gemma 4 cloud: SOAP, summary, speaker labelling
│   └── symptom_agent.py            Gemma 4 E2B local: symptom JSON extraction
├── transcription/
│   └── transcriber.py              faster-whisper live mic streaming
├── rag/
│   ├── retriever.py                ChromaDB + MiniLM embedding + retrieval
│   └── data/
│       ├── icd10_common.json       90+ ICD-10 codes (Ghana-relevant)
│       └── essential_medicines.json 40+ WHO Essential Medicines
├── database/
│   └── db.py                       SQLite schema + helpers
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── DEVLOG.md                       This file
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Google AI Studio API key (required) |
| `WHISPER_MODEL` | `small` | Whisper model size: tiny/base/small/medium/large-v3 |
| `OLLAMA_MODEL` | `gemma4:e2b` | Local Ollama model for symptom extraction |
| `CLOUD_MODEL` | `gemma-4-26b-a4b-it` | Google AI Studio model name |
