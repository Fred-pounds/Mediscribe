import os
import json
import ollama
from agents.cloud_agents import extract_symptoms_cloud

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")

SYMPTOM_PROMPT = """You are a medical symptom extraction AI. Extract all clinical information from this transcript into valid JSON only.

Return ONLY valid JSON — no markdown, no explanation, no code fences:
{{
  "chief_complaint": "main reason for visit",
  "symptoms": ["list", "of", "symptoms"],
  "duration": "how long symptoms have been present",
  "severity": "mild | moderate | severe",
  "associated_symptoms": ["other symptoms"],
  "medications_mentioned": ["drugs or treatments mentioned"],
  "allergies": ["any allergies mentioned"],
  "vitals_mentioned": {{
    "temperature": null,
    "blood_pressure": null,
    "pulse": null,
    "weight": null
  }},
  "relevant_history": "past medical history",
  "follow_up_actions": ["follow-up steps, tests, referrals"]
}}

Transcript:
{transcript}"""


def _extract_via_ollama(transcript: str) -> dict:
    """Primary: local Gemma 4 E2B via Ollama."""
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": SYMPTOM_PROMPT.format(transcript=transcript)}],
        options={"temperature": 0.1},
    )
    raw = response["message"]["content"].strip()

    # strip markdown fences if present
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)
    # must be a dict with at least chief_complaint to be valid
    if not isinstance(result, dict) or "chief_complaint" not in result:
        raise ValueError("Invalid symptom structure from Ollama")
    return result


def extract_symptoms(transcript: str) -> dict:
    """
    Extract structured symptoms from transcript.
    Tries local Gemma 4 E2B (Ollama) first — fast, private.
    Falls back to cloud Gemma 4 function calling on any failure — guaranteed valid schema.
    """
    if not transcript.strip():
        return {}

    try:
        result = _extract_via_ollama(transcript)
        print("[Symptoms] Extracted via local Gemma 4 E2B (Ollama)")
        return result
    except Exception as e:
        print(f"[Symptoms] Ollama failed ({e}), falling back to cloud function calling...")

    try:
        result = extract_symptoms_cloud(transcript)
        print("[Symptoms] Extracted via cloud Gemma 4 function calling")
        return result
    except Exception as e:
        print(f"[Symptoms] Cloud fallback also failed: {e}")
        return {"error": str(e)}


def format_symptoms_for_display(symptoms: dict) -> str:
    if not symptoms or "error" in symptoms:
        return "_No symptoms extracted._"

    lines = []
    if cc := symptoms.get("chief_complaint"):
        lines.append(f"**Chief Complaint:** {cc}")
    if s := symptoms.get("symptoms"):
        lines.append(f"**Symptoms:** {', '.join(s)}")
    if d := symptoms.get("duration"):
        lines.append(f"**Duration:** {d}")
    if sev := symptoms.get("severity"):
        lines.append(f"**Severity:** {sev}")
    if assoc := symptoms.get("associated_symptoms"):
        lines.append(f"**Associated:** {', '.join(assoc)}")
    if meds := symptoms.get("medications_mentioned"):
        lines.append(f"**Medications:** {', '.join(meds)}")
    if allerg := symptoms.get("allergies"):
        lines.append(f"**Allergies:** {', '.join(allerg)}")

    vitals = symptoms.get("vitals_mentioned") or {}
    vital_parts = [f"{k}: {v}" for k, v in vitals.items() if v]
    if vital_parts:
        lines.append(f"**Vitals:** {', '.join(vital_parts)}")
    if hist := symptoms.get("relevant_history"):
        lines.append(f"**History:** {hist}")
    if followup := symptoms.get("follow_up_actions"):
        actions = "\n".join(f"- {a}" for a in followup)
        lines.append(f"**Follow-up Actions:**\n{actions}")

    return "\n\n".join(lines) if lines else "_No structured data found._"
