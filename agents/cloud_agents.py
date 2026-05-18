"""
cloud_agents.py — Gemma 4 Cloud Inference Pipeline
====================================================
All calls to Gemma 4 (gemma-4-26b-a4b-it) via Google AI Studio go through this module.

Gemma 4 features used:
  - Text generation        : SOAP note generation, patient summary, transcript repair
  - Thinking / reasoning   : SOAP notes use ThinkingConfig so the model reasons through
                             the clinical picture before writing (use_thinking=True)
  - Native function calling: extract_symptoms_cloud() uses a typed FunctionDeclaration
                             schema to guarantee structured clinical JSON output
  - Multimodal (vision)    : analyze_medical_document() passes image/PDF bytes alongside
                             a text prompt so Gemma 4 reads uploaded lab results
"""

import os
import base64
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

CLOUD_MODEL = os.getenv("CLOUD_MODEL", "gemma-4-26b-a4b-it")

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _call(prompt: str, use_thinking: bool = False) -> str:
    try:
        config = types.GenerateContentConfig(temperature=0.3)
        if use_thinking:
            config = types.GenerateContentConfig(
                temperature=1.0,  # required when thinking is enabled
                thinking_config=types.ThinkingConfig(
                    include_thoughts=False,   # reason internally, return only final answer
                    thinking_budget=2048,
                ),
            )
        response = _client.models.generate_content(
            model=CLOUD_MODEL,
            contents=prompt,
            config=config,
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemma API error: {e}") from e


# ── Function calling — symptom schema ────────────────────────────────────────

SYMPTOM_SCHEMA = types.FunctionDeclaration(
    name="record_symptoms",
    description="Record all structured clinical information extracted from the consultation transcript.",
    parameters={
        "type": "object",
        "properties": {
            "chief_complaint":      {"type": "string",  "description": "Main reason for the visit"},
            "symptoms":             {"type": "array",   "items": {"type": "string"}, "description": "List of reported symptoms"},
            "duration":             {"type": "string",  "description": "How long symptoms have been present"},
            "severity":             {"type": "string",  "enum": ["mild", "moderate", "severe"], "description": "Overall severity"},
            "associated_symptoms":  {"type": "array",   "items": {"type": "string"}},
            "medications_mentioned":{"type": "array",   "items": {"type": "string"}, "description": "Drugs or treatments mentioned"},
            "allergies":            {"type": "array",   "items": {"type": "string"}},
            "vitals_mentioned": {
                "type": "object",
                "properties": {
                    "temperature":    {"type": "string"},
                    "blood_pressure": {"type": "string"},
                    "pulse":          {"type": "string"},
                    "weight":         {"type": "string"},
                },
            },
            "relevant_history":     {"type": "string",  "description": "Past medical history mentioned"},
            "follow_up_actions":    {"type": "array",   "items": {"type": "string"}, "description": "Next steps, tests, referrals"},
        },
        "required": ["chief_complaint", "symptoms"],
    },
)

_SYMPTOM_TOOL = types.Tool(function_declarations=[SYMPTOM_SCHEMA])


def extract_symptoms_cloud(transcript: str) -> dict:
    """
    Use cloud Gemma 4 function calling to extract structured symptoms.
    Returns a guaranteed-valid dict — no JSON parsing errors.
    """
    if not transcript.strip():
        return {}
    try:
        response = _client.models.generate_content(
            model=CLOUD_MODEL,
            contents=f"Extract all clinical information from this consultation transcript:\n\n{transcript}",
            config=types.GenerateContentConfig(
                tools=[_SYMPTOM_TOOL],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["record_symptoms"],
                    )
                ),
                temperature=0.1,
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.function_call:
                return dict(part.function_call.args)
    except Exception as e:
        print(f"[FunctionCalling] Cloud extraction failed: {e}")
    return {}


# ── Transcript repair + speaker labelling ─────────────────────────────────────

REPAIR_PROMPT = """You are a medical transcription editor. You will receive a raw speech-to-text transcript of a doctor-patient consultation. The transcript may contain:
- Misheared words or garbled medical terms
- Missing punctuation and sentence breaks
- Run-on sentences
- Filler words (um, uh, like, you know)
- Incorrectly transcribed drug names, symptoms, or medical terminology
- Words run together without spaces

Your job is to:
1. REPAIR the transcript — fix obvious errors, correct medical terminology, add punctuation, split run-on sentences, remove filler words
2. LABEL each speaker — prefix each turn with "Doctor:" or "Patient:"
   - Doctors: ask clinical questions, give diagnoses, prescribe medications, explain treatment
   - Patients: describe symptoms, answer questions, mention their history, express concerns
3. Start a new labelled line each time the speaker changes
4. Do NOT add, invent, or remove any clinical facts — only fix language/transcription errors
5. Keep all mentioned symptoms, medications, durations, and instructions intact

Raw transcript:
{transcript}

Cleaned and labelled transcript:"""


def clean_and_label_transcript(transcript: str) -> str:
    """
    Repair ASR errors and add Doctor/Patient speaker labels in one Gemma 4 call.
    Falls back to raw transcript on failure.
    """
    if not transcript.strip():
        return transcript
    try:
        return _call(REPAIR_PROMPT.format(transcript=transcript))
    except Exception as e:
        print(f"[TranscriptRepair] Failed ({e}), using raw transcript.")
        return transcript


def label_speakers(transcript: str) -> str:
    """Alias kept for backwards compatibility — now delegates to clean_and_label."""
    return clean_and_label_transcript(transcript)


# ── SOAP Note (with reasoning mode) ──────────────────────────────────────────

SOAP_PROMPT = """You are an experienced medical scribe and clinician. Generate a professional SOAP note from the following doctor-patient consultation transcript.

{rag_context}

Think carefully about the clinical picture before writing. Format with these exact sections:

**S - Subjective**
(Patient's reported complaints, history, and symptoms in their own words)

**O - Objective**
(Observable, measurable findings: vitals, physical exam findings, lab values if mentioned)

**A - Assessment**
(Clinical impression and working diagnosis. Include the most likely ICD-10 code.)

**P - Plan**
(Medications with correct dosages from the reference above, investigations ordered, referrals, follow-up schedule, patient education)

Transcript:
{transcript}

SOAP Note:"""


def generate_soap_note(transcript: str, rag_context: str = "") -> str:
    if not transcript.strip():
        return "No transcript available."
    context_block = f"\nClinical Reference:\n{rag_context}\n" if rag_context else ""
    try:
        return _call(
            SOAP_PROMPT.format(transcript=transcript, rag_context=context_block),
            use_thinking=True,
        )
    except RuntimeError as e:
        if "thinking" in str(e).lower() or "ThinkingConfig" in str(e):
            # model doesn't support thinking — retry without it
            print("[Reasoning] Thinking not supported on this model, retrying without.")
            return _call(
                SOAP_PROMPT.format(transcript=transcript, rag_context=context_block),
                use_thinking=False,
            )
        raise


# ── Patient Summary ───────────────────────────────────────────────────────────

SUMMARY_PROMPT = """You are a compassionate medical communicator. Write a clear, friendly patient summary from this consultation that:
- Uses simple, non-technical language
- Explains what was discussed and decided
- Lists medications and dosages prescribed
- States next steps and follow-up plan
- Is encouraging and reassuring in tone

Transcript:
{transcript}

Patient Summary:"""


def generate_patient_summary(transcript: str) -> str:
    if not transcript.strip():
        return "No transcript available."
    return _call(SUMMARY_PROMPT.format(transcript=transcript))


# ── Medical image / document analysis ────────────────────────────────────────

IMAGE_PROMPT = """You are a medical document analyst. Carefully examine this medical document (lab result, prescription, X-ray report, or clinical record).

Extract ALL clinical information present and structure it clearly:

**Document Type:** (lab result / prescription / imaging report / other)

**Key Findings:**
(List every test, value, measurement, or finding with its result and reference range if shown)

**Abnormal Values:**
(Highlight any results outside normal range)

**Medications / Dosages:**
(Any drugs, doses, or treatment instructions visible)

**Clinical Notes:**
(Any doctor notes, diagnoses, or instructions on the document)

**Summary for SOAP Note:**
(One paragraph summarising what this document adds to the clinical picture)"""


def analyze_medical_document(file_path: str) -> str:
    """
    Extract clinical data from an uploaded image or PDF using Gemma 4 multimodal.
    Contents format: [Part.from_bytes(...), "text string"] — per official Gemma 4 docs.
    """
    suffix = Path(file_path).suffix.lower()
    mime_map = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".pdf":  "application/pdf",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    try:
        response = _client.models.generate_content(
            model=CLOUD_MODEL,
            contents=[
                types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                IMAGE_PROMPT,
            ],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Image analysis failed: {e}") from e


# ── Translation stubs (disabled — NLLB-200 planned) ──────────────────────────

def translate_to_twi(english_text: str) -> str:
    return ""


def translate_to_english(twi_text: str) -> str:
    return ""
