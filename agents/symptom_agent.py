"""
symptom_agent.py — Structured Clinical Data Extraction
=======================================================
Extracts symptoms, vitals, medications, and follow-up actions from a consultation
transcript using Gemma 4 native function calling.

Gemma 4 feature used:
  - Function calling: extract_symptoms_cloud() sends a typed FunctionDeclaration
    schema (SYMPTOM_SCHEMA in cloud_agents.py) so Gemma 4 returns a validated
    JSON object rather than free-form text — eliminating parsing errors.
"""

from agents.cloud_agents import extract_symptoms_cloud


def extract_symptoms(transcript: str) -> dict:
    """Extract structured symptoms via Gemma 4 function calling."""
    if not transcript.strip():
        return {}
    try:
        result = extract_symptoms_cloud(transcript)
        print("[Symptoms] Extracted via cloud Gemma 4 function calling")
        return result
    except Exception as e:
        print(f"[Symptoms] Extraction failed: {e}")
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
