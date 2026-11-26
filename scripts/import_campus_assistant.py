r"""Import all course PDFs and project data from C:\campus_assistant into the live RAG store and ChromaDB."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.dataset_builder import (  # noqa: E402
    build_from_sample_documents,
    chunk_sections,
    extract_qa_pairs,
)
from backend.parsers import ALLOWED_EXTENSIONS, detect_language, parse_uploaded_file  # noqa: E402
from backend.rag_engine import EdgeRAGEngine  # noqa: E402

DATA_DIR = ROOT / "data"
STORE_PATH = DATA_DIR / "dataset_store.json"
CHROMA_DIR = DATA_DIR / "chroma_db"


def _classify_course(file_path: Path) -> tuple[str, str]:
    parts_low = [p.lower() for p in file_path.parts]
    if "ai" in parts_low:
        sub = next((p.title() for p in file_path.parts if p.lower() in {"lectures", "labs", "tutorials", "notes"}), "Course")
        return f"Artificial Intelligence ({sub})", "Computer Science & AI"
    if "micro" in parts_low:
        sub = next((p.title() for p in file_path.parts if p.lower() in {"lectures", "labs", "tutorials", "notes"}), "Course")
        return f"Microprocessors & Embedded Systems ({sub})", "Engineering & Embedded"
    return "MSA Edge AI Project Proposal", "Engineering & Embedded"


def import_folder(source_root: Path, preserve_store: Path | None = None) -> None:
    # 1. Start with all 10 official MSA University bilingual knowledge documents
    base_chunks, base_qa, base_sources = build_from_sample_documents(chunk_size=768, chunk_overlap=96)
    seen_chunk_ids = {c["id"] for c in base_chunks}
    seen_qa_ids = {q["id"] for q in base_qa}
    seen_source_names = {s["filename"] for s in base_sources}

    # 2. Discover all supported files in C:\campus_assistant\campus_assistant\dataset + top-level PDF
    candidate_files: List[Path] = []
    dataset_dir = source_root / "campus_assistant" / "dataset"
    if dataset_dir.exists():
        candidate_files.extend(
            sorted(p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)
        )
    for top_pdf in sorted(source_root.glob("*.pdf")):
        candidate_files.append(top_pdf)

    print(f"Found {len(candidate_files)} files in {source_root}...")
    imported_files = 0
    skipped_files: List[str] = []

    for fpath in candidate_files:
        try:
            raw = fpath.read_bytes()
            sections = parse_uploaded_file(fpath.name, raw)
            if not sections:
                skipped_files.append(f"{fpath.name} (no extractable text / scanned image)")
                continue

            course_name, department = _classify_course(fpath)
            chunks = chunk_sections(
                sections,
                source_file=fpath.name,
                course_name=course_name,
                department=department,
                chunk_size=900,
                chunk_overlap=110,
            )
            # Extract concise explicit/section Q&A pairs (cap per PDF to keep Q&A crisp)
            qa_pairs, _ = extract_qa_pairs(chunks[:8], use_ollama=False)

            new_chunks = [c for c in chunks if c["id"] not in seen_chunk_ids]
            new_qa = [q for q in qa_pairs if q["id"] not in seen_qa_ids]
            for c in new_chunks:
                seen_chunk_ids.add(c["id"])
            for q in new_qa:
                seen_qa_ids.add(q["id"])

            base_chunks.extend(new_chunks)
            base_qa.extend(new_qa)
            base_sources.append(
                {
                    "filename": fpath.name,
                    "course": course_name,
                    "department": department,
                    "chunks_count": len(new_chunks),
                    "qa_count": len(new_qa),
                    "language": detect_language(" ".join(c["text"] for c in new_chunks[:5])),
                    "uploaded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            seen_source_names.add(fpath.name)
            imported_files += 1
        except Exception as exc:
            skipped_files.append(f"{fpath.name} ({exc})")

    # 2b. Preserve any previously imported external course PDFs (e.g. AI & Micro lectures) from backup store
    preserved_files = 0
    if preserve_store and preserve_store.exists():
        prev = json.loads(preserve_store.read_text(encoding="utf-8"))
        missing_sources = [s for s in prev.get("sources", []) if s.get("filename") not in seen_source_names]
        missing_names = {s["filename"] for s in missing_sources}
        if missing_names:
            for c in prev.get("chunks", []):
                if c.get("source_file") in missing_names and c.get("id") not in seen_chunk_ids:
                    seen_chunk_ids.add(c["id"])
                    base_chunks.append(c)
            for q in prev.get("qa_pairs", []):
                if q.get("source_file") in missing_names and q.get("id") not in seen_qa_ids:
                    seen_qa_ids.add(q["id"])
                    base_qa.append(q)
            for s in missing_sources:
                base_sources.append(s)
                seen_source_names.add(s["filename"])
                preserved_files += 1

    # 3. Save merged dataset_store.json
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = {
        "sources": base_sources,
        "chunks": base_chunks,
        "qa_pairs": base_qa,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    STORE_PATH.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Saved merged store: {len(base_sources)} sources ({imported_files} freshly parsed, {preserved_files} preserved from backup), "
        f"{len(base_chunks)} chunks, {len(base_qa)} Q&A pairs."
    )
    if skipped_files:
        print(f"Skipped {len(skipped_files)} file(s): {', '.join(skipped_files[:8])}")

    # 4. Rebuild Persistent ChromaDB with Ollama nomic-embed-text (or local 384d fallback)
    rag = EdgeRAGEngine(CHROMA_DIR)
    status = rag.rebuild_index(base_chunks, base_qa)
    print(f"ChromaDB indexed: {status}")


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\campus_assistant")
    backup = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(r"E:\5th THE LAST DANCE\Ai_Datset\exports\dataset_store.json")
    import_folder(src, preserve_store=backup)
