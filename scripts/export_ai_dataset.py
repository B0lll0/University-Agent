"""Export all MSA University raw documents, RAG chunks, Q&A JSONL, CSV, structured JSON, ChromaDB, and Pi scripts."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.dataset_builder import (  # noqa: E402
    MSA_BUILDINGS,
    MSA_GATE_DIRECTIONS,
    build_chromadb_payload,
    export_csv_bytes,
    export_jsonl_bytes,
)
from backend.msa_curricula import MSA_FACULTIES_DATA, MSA_TUITION_FEES, TOP_5_MSA_FAQS, build_study_plans_markdown  # noqa: E402

README_TEMPLATE = """# Ai_Datset — MSA University Offline Edge AI Campus Assistant RAG Dataset

Complete bilingual (English & Arabic) RAG dataset and persistent ChromaDB vector store for **MSA University (6th of October City)** plus **AI & Microprocessors course lectures, labs, tutorials, and notes** imported from `C:\\campus_assistant`, prepared for offline Edge AI deployment (Raspberry Pi 3/4/5 + Ollama + ChromaDB + Whisper + Flask).

## Live Dataset Summary

- **Total Sources:** {sources_count} documents (10 official MSA University campus, tuition & admission guides + {course_sources_count} AI, Microprocessors & Project PDFs)
- **RAG Chunks:** {chunks_count} overlapping text chunks
- **Q&A Pairs:** {qa_count} bilingual & course Q&A records
- **ChromaDB Vectors:** {vectors_count} live vectors

## Folder Structure

- `raw_documents/` — Clean UTF-8 Markdown source files for MSA University (Overview, Academic Rules, Campus Buildings, Rooms & Locations, Food Court, Top 5 Internet FAQs, Study Plans for all 11 faculties, and Arabic Guide)
- `exports/` — RAG-ready structured datasets (including all imported AI & Microprocessors lectures, labs, and tutorials):
  - `rag_chunks_chromadb.json` — {chunks_count} overlapping RAG chunks with metadata ready for `nomic-embed-text` + ChromaDB
  - `qa_dataset.jsonl` — {qa_count} instruction/response Q&A pairs for RAG evaluation or LLM fine-tuning
  - `campus_dataset.csv` — UTF-8 BOM Excel-compatible combined spreadsheet (Arabic-safe)
  - `dataset_store.json` — Complete combined Flask store (`sources`, `chunks`, `qa_pairs`)
  - `msa_structured_knowledge.json` — Structured JSON of all 11 faculties, 27 majors, 4/5-year plans, buildings, Gate 1–4 directions, and Top 5 FAQs
  - `msa_top_5_faqs.json` — Top 5 frequently asked MSA University questions
- `chroma_db/` — Pre-built **Persistent ChromaDB** vector database ({vectors_count} live vectors) ready to query immediately
- `scripts/` — Ready-to-run RAG ingestion and Raspberry Pi deployment scripts:
  - `ingest_chromadb.py` — Ingest `exports/rag_chunks_chromadb.json` into ChromaDB using Ollama `nomic-embed-text`
  - `import_campus_assistant.py` — Re-import MSA docs + `C:\\campus_assistant` course PDFs into `data/dataset_store.json` and ChromaDB
  - `deploy_raspberry_pi.sh` — Turnkey Raspberry Pi 3/4/5 installer (Flask + ChromaDB + Ollama + espeak-ng)
  - `kiosk_start.sh` — 7-inch touch display Chromium Kiosk Mode launcher

## Quick RAG Usage (ChromaDB + Ollama)

```bash
pip install chromadb requests
ollama pull nomic-embed-text
ollama pull phi3:mini
python scripts/ingest_chromadb.py --input exports/rag_chunks_chromadb.json --db ./chroma_db_nomic
```
"""


def export_dataset(target_dir: Path) -> None:
    raw_dir = target_dir / "raw_documents"
    exp_dir = target_dir / "exports"
    scr_dir = target_dir / "scripts"
    chroma_dst = target_dir / "chroma_db"

    for folder in (raw_dir, exp_dir, scr_dir):
        folder.mkdir(parents=True, exist_ok=True)

    store = json.loads((ROOT / "data" / "dataset_store.json").read_text(encoding="utf-8"))
    chunks = store.get("chunks", [])
    qa_pairs = store.get("qa_pairs", [])

    for md_file in (ROOT / "data" / "msa_knowledge").glob("*.md"):
        shutil.copy2(md_file, raw_dir / md_file.name)
    (raw_dir / "MSA_Study_Plans_All_Faculties.md").write_text(build_study_plans_markdown(), encoding="utf-8")

    (exp_dir / "rag_chunks_chromadb.json").write_text(
        json.dumps(build_chromadb_payload(chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (exp_dir / "qa_dataset.jsonl").write_bytes(export_jsonl_bytes(qa_pairs))
    (exp_dir / "campus_dataset.csv").write_bytes(export_csv_bytes(chunks, qa_pairs))
    (exp_dir / "dataset_store.json").write_text(
        json.dumps(store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (exp_dir / "msa_structured_knowledge.json").write_text(
        json.dumps(
            {
                "faculties": MSA_FACULTIES_DATA,
                "tuition_fees_2026_2027": MSA_TUITION_FEES,
                "buildings": MSA_BUILDINGS,
                "gate_directions": MSA_GATE_DIRECTIONS,
                "top_5_internet_faqs": TOP_5_MSA_FAQS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (exp_dir / "msa_tuition_fees_2026_2027.json").write_text(
        json.dumps(MSA_TUITION_FEES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (exp_dir / "msa_top_5_faqs.json").write_text(
        json.dumps(TOP_5_MSA_FAQS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for script_name in ("ingest_chromadb.py", "import_campus_assistant.py", "deploy_raspberry_pi.sh", "kiosk_start.sh"):
        src = ROOT / "scripts" / script_name
        if src.exists():
            shutil.copy2(src, scr_dir / script_name)

    chroma_src = ROOT / "data" / "chroma_db"
    if chroma_src.exists():
        shutil.rmtree(chroma_dst, ignore_errors=True)
        shutil.copytree(chroma_src, chroma_dst)

    sources = store.get("sources", [])
    course_sources = max(0, len(sources) - 10)
    readme_body = README_TEMPLATE.format(
        sources_count=len(sources),
        course_sources_count=course_sources,
        chunks_count=len(chunks),
        qa_count=len(qa_pairs),
        vectors_count=len(chunks) + len(qa_pairs),
    )
    (target_dir / "README.md").write_text(readme_body, encoding="utf-8")

    file_count = sum(1 for p in target_dir.rglob("*") if p.is_file())
    print(f"SUCCESS: Exported {file_count} files ({len(chunks)} chunks, {len(qa_pairs)} Q&A pairs + ChromaDB) to {target_dir}")


if __name__ == "__main__":
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"E:\5th THE LAST DANCE\Ai_Datset")
    export_dataset(out_path)
