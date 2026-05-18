import threading
from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from database.db import (
    init_db,
    create_patient,
    get_all_patients,
    get_patient,
    create_session,
    update_transcript,
    close_session,
    save_note,
    save_symptoms,
    get_sessions_for_patient,
    get_note_for_session,
    get_symptoms_for_session,
)
from transcription.transcriber import LiveTranscriber
from agents.symptom_agent import extract_symptoms, format_symptoms_for_display
from agents.cloud_agents import generate_soap_note, generate_patient_summary, clean_and_label_transcript, analyze_medical_document
from rag.retriever import (
    ensure_kb,
    retrieve_icd_codes,
    retrieve_drug_info,
    format_icd_context,
    format_drug_context,
)

# ── Startup ───────────────────────────────────────────────────────────────────

init_db()
ensure_kb()

# ── State ─────────────────────────────────────────────────────────────────────

_transcriber: LiveTranscriber | None = None
_transcript_parts: list[str] = []
_labelled_transcript: str = ""
_document_analysis: str = ""
_current_session_id: int | None = None
_transcript_lock = threading.Lock()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _patient_choices() -> list[str]:
    patients = get_all_patients()
    return [f"{p['id']} — {p['name']}" for p in patients] if patients else []


def _parse_patient_choice(choice: str) -> int:
    return int(choice.split("—")[0].strip())


def _full_transcript() -> str:
    with _transcript_lock:
        return " ".join(_transcript_parts)


def _format_icd_panel(codes: list[dict]) -> str:
    if not codes:
        return "_No ICD-10 suggestions._"
    lines = ["### Suggested ICD-10 Codes\n"]
    for c in codes:
        lines.append(f"- **{c['code']}** — {c['description']} *(confidence: {c['score']})*")
    return "\n".join(lines)


def _format_drug_panel(drugs: list[dict]) -> str:
    if not drugs:
        return "_No drug references matched._"
    lines = ["### Drug Reference\n"]
    for d in drugs:
        lines.append(
            f"**{d['name']}** ({d['class']})\n"
            f"- Adult dose: {d['adult_dose']}\n"
            f"- Indications: {d['indications']}\n"
            f"- Caution: {d['contraindications']}\n"
        )
    return "\n".join(lines)


# ── Tab 1: Live Consultation ──────────────────────────────────────────────────

def register_patient(name, dob, gender, phone):
    if not name.strip():
        return gr.update(), "Please enter a patient name."
    pid = create_patient(name.strip(), dob, gender, phone)
    choices = _patient_choices()
    new_val = next((c for c in choices if c.startswith(str(pid))), choices[-1] if choices else None)
    return gr.update(choices=choices, value=new_val), f"Patient '{name}' registered (ID {pid})."


def start_consultation(patient_choice, doctor_name):
    global _transcriber, _transcript_parts, _current_session_id

    if not patient_choice:
        return "No patient selected.", "", gr.update(interactive=False), gr.update(interactive=True)

    pid = _parse_patient_choice(patient_choice)
    _current_session_id = create_session(pid, doctor_name or "Doctor")

    with _transcript_lock:
        _transcript_parts.clear()
    global _labelled_transcript, _document_analysis
    _labelled_transcript = ""
    _document_analysis = ""

    def on_text(text):
        with _transcript_lock:
            _transcript_parts.append(text)
    _transcriber = LiveTranscriber(on_text=on_text)
    _transcriber.start()

    return (
        "Recording... speak clearly.",
        "",
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def poll_transcript():
    return _full_transcript()


def stop_consultation():
    global _transcriber, _labelled_transcript

    if _transcriber:
        _transcriber.stop()
        _transcriber = None

    raw = _full_transcript()
    if not raw:
        return "No audio captured.", "", gr.update(interactive=False), gr.update(interactive=True)

    if _current_session_id:
        update_transcript(_current_session_id, raw)

    _labelled_transcript = clean_and_label_transcript(raw)

    return (
        "Consultation ended. Transcript cleaned ✓  Click 'Generate Notes' to proceed.",
        _labelled_transcript,
        gr.update(interactive=False),
        gr.update(interactive=True),
    )


def upload_document(file):
    """Analyse an uploaded medical document with Gemma 4 vision."""
    global _document_analysis
    if file is None:
        _document_analysis = ""
        return "_No document uploaded._"
    try:
        result = analyze_medical_document(file.name)
        _document_analysis = result
        return result
    except Exception as e:
        _document_analysis = ""
        return f"_Document analysis failed: {e}_"


def generate_notes():
    """RAG retrieval → cloud agents → save to DB."""
    # Use labelled transcript if available, fall back to raw
    transcript = _labelled_transcript or _full_transcript()
    if not transcript:
        return "No transcript available.", "No transcript available.", "_No symptoms._", "_No ICD codes._", "_No drug info._", "", ""

    # 1. Extract symptoms locally (Gemma 4 E2B via Ollama)
    symptoms = extract_symptoms(transcript)
    symptoms_md = format_symptoms_for_display(symptoms)

    # 2. RAG retrieval
    chief = symptoms.get("chief_complaint", "")
    sym_list = symptoms.get("symptoms", [])
    meds_list = symptoms.get("medications_mentioned", [])
    rag_query = f"{chief} {' '.join(sym_list)}".strip() or transcript[:300]

    icd_codes = retrieve_icd_codes(rag_query, n=5)
    drug_info = retrieve_drug_info(meds_list, n=3) if meds_list else []

    doc_section = f"\nUploaded Medical Document (lab result / prescription / report):\n{_document_analysis}\n" if _document_analysis else ""
    rag_context = "\n".join(filter(None, [
        format_icd_context(icd_codes),
        format_drug_context(drug_info),
        doc_section,
    ]))

    # 3. Cloud agents
    try:
        soap = generate_soap_note(transcript, rag_context=rag_context)
    except Exception as e:
        soap = f"_SOAP note generation failed: {e}_"

    try:
        summary_en = generate_patient_summary(transcript)
    except Exception as e:
        summary_en = f"_Summary generation failed: {e}_"

    # 4. Persist
    if _current_session_id:
        save_note(_current_session_id, soap, summary_en, summary_twi="")
        save_symptoms(_current_session_id, symptoms)
        close_session(_current_session_id)

    return soap, summary_en, symptoms_md, _format_icd_panel(icd_codes), _format_drug_panel(drug_info), soap, summary_en


# ── Tab 2: Patient Records ────────────────────────────────────────────────────

def load_patient_records(patient_choice):
    if not patient_choice:
        return "Select a patient.", "", "", ""

    pid = _parse_patient_choice(patient_choice)
    patient = get_patient(pid)
    if not patient:
        return "Patient not found.", "", "", ""

    sessions = get_sessions_for_patient(pid)
    if not sessions:
        return f"No sessions found for {patient['name']}.", "", "", ""

    latest = sessions[0]
    sid = latest["id"]
    note = get_note_for_session(sid)
    symptoms = get_symptoms_for_session(sid)

    session_info = (
        f"**Patient:** {patient['name']}  |  **DOB:** {patient.get('dob', 'N/A')}  |  "
        f"**Gender:** {patient.get('gender', 'N/A')}\n\n"
        f"**Session:** {latest['date']}  |  **Doctor:** {latest.get('doctor', 'N/A')}"
    )

    soap = note["soap_note"] if note else "_No SOAP note found._"
    summary = note["summary_en"] if note else "_No summary found._"
    symptoms_md = format_symptoms_for_display(symptoms)

    return session_info, soap, summary, symptoms_md


# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container { font-family: 'Segoe UI', system-ui, sans-serif; }

#header-banner {
    background: linear-gradient(135deg, #1a6eb5, #0d4f8a);
    color: white;
    padding: 20px 28px;
    border-radius: 12px;
    margin-bottom: 20px;
}
#header-banner h1 { margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px; }
#header-banner p  { margin: 5px 0 0; opacity: 0.85; font-size: 0.95rem; }

/* Fix markdown panels — transparent so they inherit theme bg */
.gr-markdown, .svelte-1ed2p3z, [data-testid="markdown"] {
    background: transparent !important;
}

/* Note cards — rendered SOAP/summary display */
.note-card {
    border: 1px solid #2d4a6e;
    border-radius: 8px;
    padding: 16px 20px !important;
    min-height: 200px;
    font-size: 0.92rem;
    line-height: 1.7;
}
.note-card h1, .note-card h2, .note-card h3 {
    color: #4a9eff;
    margin-top: 12px;
    font-size: 1rem;
}
.note-card strong { color: #7ec8ff; }
.note-card p { margin: 6px 0; }
.note-card ul, .note-card ol { padding-left: 20px; margin: 4px 0; }

/* Status bar */
.status-bar p { font-weight: 600; color: #4a9eff; font-size: 1rem; }

/* RAG accordion open panel */
.rag-content {
    border-left: 3px solid #1a6eb5;
    padding: 10px 14px;
    border-radius: 0 6px 6px 0;
    font-size: 0.9rem;
}

/* Tighten up accordion headers */
.gr-accordion .label-wrap { font-weight: 600 !important; }

/* Recording pulse indicator */
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.recording p { animation: pulse 1.4s ease-in-out infinite; color: #ff4444 !important; font-weight: 700; }
"""

# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Hospital Copilot") as demo:

    gr.HTML("""
    <div id="header-banner">
        <h1>🏥 Hospital Copilot</h1>
        <p>AI-powered medical documentation &nbsp;·&nbsp; Gemma 4 &nbsp;·&nbsp; RAG-grounded &nbsp;·&nbsp; Ghana</p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: Live Consultation ──────────────────────────────────────
        with gr.Tab("🎙️ Live Consultation"):
            with gr.Row(equal_height=False):

                # Left column — patient panel
                with gr.Column(scale=1, min_width=280):
                    with gr.Group():
                        gr.Markdown("#### 👤 Select Patient")
                        patient_dd  = gr.Dropdown(label="Patient", choices=_patient_choices(), interactive=True)
                        doctor_name = gr.Textbox(label="Doctor", placeholder="Dr. Mensah")

                    with gr.Accordion("➕ Register New Patient", open=False):
                        reg_name   = gr.Textbox(label="Full Name",    placeholder="Kofi Agyeman")
                        reg_dob    = gr.Textbox(label="Date of Birth", placeholder="1985-03-15")
                        reg_gender = gr.Radio(["Male", "Female", "Other"], label="Gender", value="Male")
                        reg_phone  = gr.Textbox(label="Phone",         placeholder="+233 24 000 0000")
                        reg_btn    = gr.Button("Register Patient", variant="primary")
                        reg_status = gr.Markdown()

                    reg_btn.click(
                        register_patient,
                        inputs=[reg_name, reg_dob, reg_gender, reg_phone],
                        outputs=[patient_dd, reg_status],
                    )

                # Right column — consultation
                with gr.Column(scale=3):
                    status_txt = gr.Markdown("_Ready. Select a patient and click Start._", elem_classes=["status-bar"])

                    with gr.Row():
                        start_btn = gr.Button("▶ Start Consultation", variant="primary", scale=1)
                        stop_btn  = gr.Button("⏹ End Consultation",   variant="stop",    scale=1, interactive=False)

                    live_transcript = gr.Textbox(
                        label="Transcript (cleaned & speaker-labelled after consultation ends)",
                        lines=8, max_lines=16,
                        interactive=False,
                        placeholder="Transcript streams here as you speak. After you click End Consultation, Gemma 4 cleans and labels it automatically.",
                    )
                    timer = gr.Timer(value=2)
                    timer.tick(poll_transcript, outputs=live_transcript)

                    with gr.Accordion("🩺 Extracted Symptoms", open=False):
                        symptoms_live = gr.Markdown("_Will populate after Generate Notes._")

            gr.Markdown("---")

            with gr.Accordion("📎 Upload Medical Document (Lab Result / Prescription / Report)", open=False):
                gr.Markdown(
                    "_Optional — upload a photo or PDF of a lab result, prescription, or any medical document. "
                    "Gemma 4 will read it and include the findings in the SOAP note automatically._"
                )
                with gr.Row():
                    doc_upload = gr.File(
                        label="Upload document",
                        file_types=[".jpg", ".jpeg", ".png", ".webp", ".pdf"],
                        scale=1,
                    )
                    doc_analyse_btn = gr.Button("🔍 Analyse Document", variant="secondary", scale=0)
                doc_result = gr.Markdown("_No document uploaded._")
                doc_analyse_btn.click(upload_document, inputs=[doc_upload], outputs=[doc_result])

            generate_btn = gr.Button("⚡ Generate Notes from Transcript", variant="primary", size="lg")

            # RAG panels — inside accordions so they don't show as white boxes
            with gr.Row():
                with gr.Accordion("🏷️ ICD-10 Suggestions", open=True):
                    icd_panel = gr.Markdown("_Click Generate Notes to see suggestions._")
                with gr.Accordion("💊 Drug Reference", open=True):
                    drug_panel = gr.Markdown("_Click Generate Notes to see drug info._")

            gr.Markdown("### 📋 Generated Notes")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🗒️ SOAP Note")
                    soap_out = gr.Markdown(
                        "_SOAP note will appear here after generating._",
                        elem_classes=["note-card"],
                    )
                    with gr.Accordion("✏️ Edit SOAP Note", open=False):
                        soap_edit = gr.Textbox(lines=18, interactive=True, show_label=False)

                with gr.Column():
                    gr.Markdown("#### 📄 Patient Summary")
                    summary_en_out = gr.Markdown(
                        "_Patient summary will appear here after generating._",
                        elem_classes=["note-card"],
                    )
                    with gr.Accordion("✏️ Edit Summary", open=False):
                        summary_edit = gr.Textbox(lines=10, interactive=True, show_label=False)

            start_btn.click(
                start_consultation,
                inputs=[patient_dd, doctor_name],
                outputs=[status_txt, live_transcript, stop_btn, start_btn],
            )
            stop_btn.click(
                stop_consultation,
                outputs=[status_txt, live_transcript, stop_btn, start_btn],
            )
            generate_btn.click(
                generate_notes,
                outputs=[soap_out, summary_en_out, symptoms_live, icd_panel, drug_panel, soap_edit, summary_edit],
            )

        # ── Tab 2: Patient Records ────────────────────────────────────────
        with gr.Tab("📁 Patient Records"):
            with gr.Row():
                records_patient_dd = gr.Dropdown(
                    label="Select Patient", choices=_patient_choices(), interactive=True, scale=3,
                )
                load_btn = gr.Button("Load Records", variant="primary", scale=1)

            session_info_md = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 🗒️ SOAP Note")
                    rec_soap = gr.Markdown("_Load a patient to see their SOAP note._", elem_classes=["note-card"])
                with gr.Column():
                    gr.Markdown("#### 📄 Patient Summary")
                    rec_summary = gr.Markdown("_Load a patient to see their summary._", elem_classes=["note-card"])

            with gr.Accordion("🩺 Extracted Symptoms", open=False):
                rec_symptoms = gr.Markdown()

            load_btn.click(
                load_patient_records,
                inputs=[records_patient_dd],
                outputs=[session_info_md, rec_soap, rec_summary, rec_symptoms],
            )
            reg_btn.click(
                lambda: gr.update(choices=_patient_choices()),
                outputs=[records_patient_dd],
            )

        # ── Tab 3: About ──────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
## Hospital Copilot — Gemma 4 for Good

**Reducing doctor burnout. Improving care quality. Built for Ghana.**

### How it works
1. **Live Transcription** — faster-whisper converts speech to text in real time on CPU
2. **Symptom Extraction** — Gemma 4 E2B (local, Ollama) extracts structured clinical JSON
3. **RAG Retrieval** — sentence-transformers + ChromaDB matches ICD-10 codes and drug dosages
4. **SOAP Note Generation** — Gemma 4 26B (cloud) writes a grounded, accurate medical note
5. **Patient Summary** — plain-language summary the patient can take home
6. **Structured Records** — everything saved to local SQLite

### RAG Knowledge Base
| Collection | Entries | Source |
|---|---|---|
| ICD-10 codes | 90+ | Ghana-relevant + general conditions |
| Essential medicines | 40+ | WHO Essential Medicines List |

### Technology Stack
| Component | Model | Where |
|---|---|---|
| Speech-to-Text | faster-whisper (base) | Local CPU |
| Symptom Extraction | Gemma 4 E2B (Q4_K_M) | Local CPU via Ollama |
| Embeddings | all-MiniLM-L6-v2 | Local CPU |
| Vector Store | ChromaDB | Local disk |
| SOAP / Summary | Gemma 4 26B-IT | Google AI Studio API |
| Storage | SQLite | Local |
| UI | Gradio | Desktop |
            """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
