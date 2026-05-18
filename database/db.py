import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "hospital_copilot.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS patients (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                dob         TEXT,
                gender      TEXT,
                phone       TEXT,
                language    TEXT DEFAULT 'en',
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id  INTEGER NOT NULL REFERENCES patients(id),
                doctor      TEXT,
                date        TEXT DEFAULT (datetime('now')),
                transcript  TEXT,
                status      TEXT DEFAULT 'open'
            );

            CREATE TABLE IF NOT EXISTS notes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                soap_note   TEXT,
                summary_en  TEXT,
                summary_twi TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS symptoms (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL REFERENCES sessions(id),
                data        TEXT,
                created_at  TEXT DEFAULT (datetime('now'))
            );
        """)


# --- Patient helpers ---

def create_patient(name: str, dob: str = "", gender: str = "", phone: str = "", language: str = "en") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO patients (name, dob, gender, phone, language) VALUES (?, ?, ?, ?, ?)",
            (name, dob, gender, phone, language),
        )
        return cur.lastrowid


def get_all_patients() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM patients ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def get_patient(patient_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM patients WHERE id = ?", (patient_id,)).fetchone()
        return dict(row) if row else None


# --- Session helpers ---

def create_session(patient_id: int, doctor: str = "Dr. Unknown") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO sessions (patient_id, doctor) VALUES (?, ?)",
            (patient_id, doctor),
        )
        return cur.lastrowid


def update_transcript(session_id: int, transcript: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET transcript = ? WHERE id = ?",
            (transcript, session_id),
        )


def close_session(session_id: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET status = 'closed' WHERE id = ?",
            (session_id,),
        )


def get_sessions_for_patient(patient_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions WHERE patient_id = ? ORDER BY date DESC",
            (patient_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# --- Notes helpers ---

def save_note(session_id: int, soap_note: str, summary_en: str, summary_twi: str) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO notes (session_id, soap_note, summary_en, summary_twi) VALUES (?, ?, ?, ?)",
            (session_id, soap_note, summary_en, summary_twi),
        )
        return cur.lastrowid


def get_note_for_session(session_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM notes WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None


# --- Symptom helpers ---

def save_symptoms(session_id: int, symptoms: dict):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO symptoms (session_id, data) VALUES (?, ?)",
            (session_id, json.dumps(symptoms)),
        )


def get_symptoms_for_session(session_id: int) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT data FROM symptoms WHERE session_id = ? ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return json.loads(row["data"]) if row else {}
