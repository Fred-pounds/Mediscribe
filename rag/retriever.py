from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DATA_DIR = Path(__file__).parent / "data"
DB_DIR = Path(__file__).parent.parent / "chroma_db"

EMBED_MODEL = "all-MiniLM-L6-v2"

_client: chromadb.PersistentClient | None = None
_icd_col = None
_drug_col = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(DB_DIR))
    return _client


def _embedding_fn():
    return SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def build_knowledge_base(force: bool = False):
    """Embed ICD-10 codes and medicines into ChromaDB. Runs once; skipped if DB exists."""
    client = _get_client()
    ef = _embedding_fn()

    existing = [c.name for c in client.list_collections()]

    # ── ICD-10 ──────────────────────────────────────────────────────────────
    if "icd10" not in existing or force:
        if "icd10" in existing:
            client.delete_collection("icd10")
        col = client.create_collection("icd10", embedding_function=ef)
        with open(DATA_DIR / "icd10_common.json") as f:
            records = json.load(f)
        col.add(
            ids=[r["code"] for r in records],
            documents=[f"{r['description']} {r['keywords']}" for r in records],
            metadatas=[{"code": r["code"], "description": r["description"]} for r in records],
        )
        print(f"[RAG] Indexed {len(records)} ICD-10 codes")

    # ── Medicines ────────────────────────────────────────────────────────────
    if "medicines" not in existing or force:
        if "medicines" in existing:
            client.delete_collection("medicines")
        col = client.create_collection("medicines", embedding_function=ef)
        with open(DATA_DIR / "essential_medicines.json") as f:
            records = json.load(f)
        col.add(
            ids=[str(i) for i in range(len(records))],
            documents=[
                f"{r['name']} {r['class']} {r['indications']}"
                for r in records
            ],
            metadatas=records,
        )
        print(f"[RAG] Indexed {len(records)} essential medicines")


def _icd_collection():
    global _icd_col
    if _icd_col is None:
        _icd_col = _get_client().get_collection("icd10", embedding_function=_embedding_fn())
    return _icd_col


def _drug_collection():
    global _drug_col
    if _drug_col is None:
        _drug_col = _get_client().get_collection("medicines", embedding_function=_embedding_fn())
    return _drug_col


def retrieve_icd_codes(query: str, n: int = 5) -> list[dict]:
    """Return top-n ICD-10 codes matching the clinical query."""
    if not query.strip():
        return []
    results = _icd_collection().query(query_texts=[query], n_results=n)
    codes = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        codes.append({
            "code": meta["code"],
            "description": meta["description"],
            "score": round(1 - dist, 3),
        })
    return codes


def retrieve_drug_info(drug_names: list[str], n: int = 3) -> list[dict]:
    """Return drug info for each named medication. Falls back to closest match."""
    if not drug_names:
        return []
    query = ", ".join(drug_names)
    results = _drug_collection().query(query_texts=[query], n_results=n)
    drugs = []
    for meta in results["metadatas"][0]:
        drugs.append({
            "name": meta["name"],
            "class": meta["class"],
            "adult_dose": meta["adult_dose"],
            "indications": meta["indications"],
            "contraindications": meta["contraindications"],
            "notes": meta.get("notes", ""),
        })
    return drugs


def format_icd_context(codes: list[dict]) -> str:
    """Format ICD codes as text context for injection into prompts."""
    if not codes:
        return ""
    lines = ["Relevant ICD-10 codes to consider:"]
    for c in codes:
        lines.append(f"  {c['code']} — {c['description']}")
    return "\n".join(lines)


def format_drug_context(drugs: list[dict]) -> str:
    """Format drug info as text context for injection into prompts."""
    if not drugs:
        return ""
    lines = ["Relevant medication reference:"]
    for d in drugs:
        lines.append(
            f"  {d['name']} ({d['class']}): {d['adult_dose']}. "
            f"Indications: {d['indications']}."
        )
    return "\n".join(lines)


def ensure_kb():
    """Called at app startup — builds KB only if it doesn't exist yet."""
    client = _get_client()
    existing = [c.name for c in client.list_collections()]
    if "icd10" not in existing or "medicines" not in existing:
        print("[RAG] Building knowledge base for the first time...")
        build_knowledge_base()
    else:
        print("[RAG] Knowledge base ready.")
